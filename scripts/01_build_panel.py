#!/usr/bin/env python
"""Build the anomaly-allocation panel and its feature matrix.

    python scripts/01_build_panel.py [--force]

Writes ``data/processed/panel.parquet`` and prints the sample-construction table
that goes into the report's data section, so the reader can see exactly how many
stock-months each screen removed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsdp.config import load_config  # noqa: E402
from xsdp.data.build_anomaly_panel import build_anomaly_panel, feature_columns  # noqa: E402
from xsdp.features.transforms import build_features  # noqa: E402

log = logging.getLogger("panel")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    cfg.paths.mkdirs()

    path = build_anomaly_panel(cfg, force=args.force)
    panel = pl.read_parquet(path)
    raw_cols = feature_columns(panel)

    panel, feature_cols = build_features(cfg, panel, raw_cols)
    out = cfg.paths.processed / f"panel_features{cfg.suffix}.parquet"
    panel.write_parquet(out, compression="zstd")

    months = panel["yyyymm"]
    log.info("panel: %d rows | %d months (%d-%d) | %d features",
             panel.height, months.n_unique(), months.min(), months.max(), len(feature_cols))
    per_month = panel.group_by("yyyymm").len()["len"]
    log.info("anomalies per month: min %d, median %d, max %d",
             per_month.min(), int(per_month.median()), per_month.max())
    log.info("wrote %s", out)

    (cfg.paths.processed / f"feature_cols{cfg.suffix}.txt").write_text("\n".join(feature_cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
