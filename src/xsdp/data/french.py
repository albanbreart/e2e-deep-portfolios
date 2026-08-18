"""Fama-French factors from Ken French's data library.

The library is publicly downloadable with no account, which matters here: it is
what lets the whole project run without a single paid subscription.  We use it
for two things only -- the risk-free rate, and the benchmark factors that the
alpha regressions in :mod:`xsdp.metrics` need.

The files are fixed-width-ish CSVs with a prose header and several stacked
tables ("monthly", then "annual"), so the parser below finds the monthly block
by looking for the first run of ``YYYYMM`` keys and stops at the first line that
is not one.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FILES = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom": "F-F_Momentum_Factor_CSV.zip",
}


def _download_csv(name: str) -> str:
    url = f"{BASE}/{FILES[name]}"
    log.info("downloading %s", url)
    r = requests.get(url, timeout=(30, 180), headers={"User-Agent": "xsdp/1.0"})
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        return zf.read(zf.namelist()[0]).decode("latin-1")


def _parse_monthly(text: str) -> pd.DataFrame:
    """Extract the monthly block: rows whose first field is a 6-digit YYYYMM."""
    rows, started = [], False
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            if started:
                break
            continue
        key = parts[0]
        if len(key) == 6 and key.isdigit():
            started = True
            rows.append(parts)
        elif started:
            break
    if not rows:
        raise ValueError("no monthly block found in Ken French file")

    header = None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == len(rows[0]) and not parts[0]:
            header = [c.lower().replace("-", "") for c in parts[1:]]
            break
    if header is None:
        header = [f"f{i}" for i in range(len(rows[0]) - 1)]

    df = pd.DataFrame(rows, columns=["yyyymm", *header])
    df["yyyymm"] = df["yyyymm"].astype("int32")
    for c in header:
        # Ken French reports percent and codes missing as -99.99.
        df[c] = pd.to_numeric(df[c], errors="coerce").where(lambda s: s > -99, other=pd.NA) / 100.0
    return df


def load_ff_factors(external_dir: Path, *, force: bool = False) -> pd.DataFrame:
    """FF5 + momentum + risk-free rate, monthly, cached as Parquet."""
    cache = external_dir / "ff_factors_monthly.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    ff5 = _parse_monthly(_download_csv("ff5"))
    mom = _parse_monthly(_download_csv("mom"))
    mom = mom.rename(columns={mom.columns[1]: "umd"})[["yyyymm", "umd"]]

    out = ff5.merge(mom, on="yyyymm", how="left")
    out = out.rename(columns={"mktrf": "mktrf", "mkt_rf": "mktrf", "mktrf ": "mktrf"})
    if "mktrf" not in out.columns:                       # header variant
        first = [c for c in out.columns if c.startswith("mkt")]
        if first:
            out = out.rename(columns={first[0]: "mktrf"})

    external_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    log.info("wrote %s: %d months (%d-%d)", cache, len(out),
             out["yyyymm"].min(), out["yyyymm"].max())
    return out
