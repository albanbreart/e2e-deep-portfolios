"""Open Source Asset Pricing (Chen & Zimmerman, 2022) data access.

This is one of the predictor datasets listed in the project instructions
("Chen-Zimmerman data: monthly long-short return for 205 predictors").  We use
three artefacts from it:

``SignalDoc.csv``
    Metadata for 331 documented signals: economic category, the sample period
    of the *original paper*, the sign of the published effect, and Chen &
    Zimmerman's own replication-quality grade.  The original sample end year is
    what lets us measure post-publication decay in :mod:`xsdp.analysis.decay`.

``signed_predictors_dl_wide.zip``
    The firm-level panel: one row per (``permno``, ``yyyymm``), one column per
    signal, already sign-flipped so that "high" always means "predicted to earn
    a high return".  This is the feature matrix of the whole project.

``PredictorPortsFull.csv``
    The authors' own long-short portfolio returns.  We never train on these;
    they are the external benchmark that our model has to beat.

The raw archive is ~2.4 GB, so :func:`download_firm_char` streams it to disk
and :func:`firm_char_to_parquet` converts it out-of-core.  Both are idempotent:
re-running them is a no-op once the target file exists.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import polars as pl
import requests

log = logging.getLogger(__name__)

_CHUNK = 1 << 22  # 4 MiB


def _resolve_url(name: str) -> str:
    """Ask the ``openassetpricing`` package for a direct download link.

    The package resolves a Google Drive file id into a ``usercontent`` URL that
    already carries the large-file confirmation token, which is why we go
    through it rather than hard-coding a link that would rot.
    """
    import openassetpricing as oap

    return oap.OpenAP()._get_url(name)


def _stream_download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest``, resuming is not attempted but partials are."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        step = max(total // 20, 1)
        next_mark = step
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(_CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if done >= next_mark:
                    log.info("  %s: %.0f%% (%.2f GB)", dest.name, 100 * done / total, done / 1e9)
                    next_mark += step
    tmp.replace(dest)
    return dest


def download_firm_char(raw_dir: Path) -> Path:
    """Fetch the wide firm-level signal archive (~2.4 GB) into ``raw_dir``."""
    dest = raw_dir / "signed_predictors_dl_wide.zip"
    if dest.exists():
        log.info("firm-char archive already present at %s", dest)
        return dest
    log.info("downloading firm-level characteristics archive (~2.4 GB)")
    return _stream_download(_resolve_url("firm_char"), dest)


def firm_char_to_parquet(zip_path: Path, out_path: Path, *, batch_rows: int = 500_000) -> Path:
    """Convert the wide CSV inside ``zip_path`` to a compact Parquet file.

    The CSV is 8.35 GB uncompressed with 211 columns, of which 209 are signals
    that are mostly missing -- no pandas call survives it.  We stream it: PyArrow
    reads decompressed blocks straight out of the zip member and each record
    batch is written to Parquet immediately, so peak memory is one batch rather
    than one dataset, and no 8 GB temp file ever touches the disk.

    The schema is declared rather than inferred.  Block-wise inference on a
    sparse file is a classic source of "column X has type int64 in block 3 and
    double in block 4" failures, and every signal is a real number anyway.
    ``float32`` halves the file with no meaningful loss: these are ranks,
    ratios and z-scores, not currency amounts.
    """
    if out_path.exists():
        log.info("firm-char parquet already present at %s", out_path)
        return out_path

    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(".parquet.tmp")

    with zipfile.ZipFile(zip_path) as zf:
        member = zf.namelist()[0]
        with zf.open(member) as probe:
            header = probe.readline().decode("utf8").strip().split(",")

        id_cols = ["permno", "yyyymm"]
        signal_cols = [c for c in header if c not in id_cols]
        column_types = {c: pa.float32() for c in signal_cols}
        column_types["permno"] = pa.int32()
        column_types["yyyymm"] = pa.int32()
        schema = pa.schema([(c, column_types[c]) for c in header])

        log.info("streaming %s -> %s (%d signals)", member, out_path.name, len(signal_cols))
        with zf.open(member) as fh:
            reader = pacsv.open_csv(
                fh,
                read_options=pacsv.ReadOptions(block_size=1 << 26),
                convert_options=pacsv.ConvertOptions(column_types=column_types),
            )
            writer = pq.ParquetWriter(tmp_out, schema, compression="zstd")
            rows = 0
            try:
                for batch in reader:
                    writer.write_batch(batch.select(header))
                    rows += batch.num_rows
                    if rows % (20 * batch_rows) < batch.num_rows:
                        log.info("  %.1fM rows", rows / 1e6)
            finally:
                writer.close()
                reader.close()

    tmp_out.replace(out_path)
    log.info("wrote %s: %d rows, %d signals", out_path, rows, len(signal_cols))
    return out_path


def load_signal_doc(external_dir: Path) -> pl.DataFrame:
    """Signal metadata, cached to ``external_dir/cz_signal_doc.parquet``."""
    cache = external_dir / "cz_signal_doc.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    import openassetpricing as oap

    doc = oap.OpenAP().dl_signal_doc("polars")
    external_dir.mkdir(parents=True, exist_ok=True)
    doc.write_parquet(cache)
    return doc


#: Portfolio files we use, and why each one is here.
PORTFOLIO_SETS: dict[str, str] = {
    # The headline long-short return of every predictor: our investable universe.
    "op": "cz_predictor_ports",
    # Value-weighted deciles: lets us look inside a spread instead of only at it.
    "deciles_vw": "cz_deciles_vw",
    # The same predictors rebuilt on stocks above the NYSE 20th percentile of
    # market cap. This is the capacity test -- an anomaly that survives here is
    # one a fund could actually trade.
    "ex_nyse_p20_me": "cz_ports_large",
    # And rebuilt excluding sub-$5 stocks, where quoted spreads are fiction.
    "ex_price5": "cz_ports_price5",
}


def load_cz_portfolios(external_dir: Path, which: str = "op") -> pl.DataFrame:
    """Download (and cache) one of Chen & Zimmerman's portfolio return files.

    ``which`` is a key of :data:`PORTFOLIO_SETS`.  The ``op`` file holds the
    headline long-short return of each predictor; the liquidity-screened
    variants hold the same predictors rebuilt on a tradable subset, which is how
    we test capacity without ever touching stock-level data.

    We stream the file to disk ourselves rather than going through
    ``openassetpricing.dl_port``, which hands the Google Drive URL straight to a
    dataframe reader; that call blocks indefinitely on the larger archives.
    """
    if which not in PORTFOLIO_SETS:
        raise KeyError(f"unknown portfolio set {which!r}; have {sorted(PORTFOLIO_SETS)}")
    cache = external_dir / f"{PORTFOLIO_SETS[which]}.parquet"
    if cache.exists():
        log.info("%s cache hit", cache.name)
        return pl.read_parquet(cache)

    external_dir.mkdir(parents=True, exist_ok=True)
    url = _resolve_url(which)
    raw = external_dir / f"{PORTFOLIO_SETS[which]}.download"
    if not raw.exists():
        log.info("downloading portfolio set %r", which)
        _stream_download(url, raw)

    ports = _read_portfolio_file(raw)
    ports.write_parquet(cache)
    raw.unlink(missing_ok=True)
    log.info("wrote %s: %d rows, %d signals",
             cache.name, ports.height, ports["signalname"].n_unique())
    return ports


def _read_portfolio_file(path: Path) -> pl.DataFrame:
    """Read a downloaded portfolio file, which may be a bare CSV or a zip."""
    with path.open("rb") as fh:
        magic = fh.read(2)
    if magic == b"PK":
        with zipfile.ZipFile(path) as zf:
            member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            data = zf.read(member)
        return pl.read_csv(data, null_values=["NA", ""], infer_schema_length=20_000)
    return pl.read_csv(path, null_values=["NA", ""], infer_schema_length=20_000)


