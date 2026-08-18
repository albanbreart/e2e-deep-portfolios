#!/usr/bin/env python
"""Run the walk-forward horse race and write out-of-sample predictions.

    python scripts/02_run_experiments.py                 # the full slate
    python scripts/02_run_experiments.py --models ridge gbrt xs_sharpe
    python scripts/02_run_experiments.py --smoke          # 3 folds, 1 seed

This is the expensive step.  On an Apple M-series laptop the full slate takes a
few hours; ``--smoke`` gives a runnable end-to-end check in a few minutes and is
what the tests exercise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsdp.config import load_config  # noqa: E402
from xsdp.experiment import DEFAULT_SPECS, run_walk_forward  # noqa: E402

log = logging.getLogger("experiments")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of model names (default: all)")
    ap.add_argument("--smoke", action="store_true", help="3 folds, 1 seed, few epochs")
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    cfg.paths.mkdirs()

    panel_path = cfg.paths.processed / f"panel_features{cfg.suffix}.parquet"
    if not panel_path.exists():
        log.error("%s missing -- run scripts/01_build_panel.py first", panel_path)
        return 1
    panel = pl.read_parquet(panel_path)
    feature_cols = (cfg.paths.processed / f"feature_cols{cfg.suffix}.txt").read_text().split("\n")
    feature_cols = [c for c in feature_cols if c]

    specs = DEFAULT_SPECS
    if args.models:
        wanted = set(args.models)
        specs = [s for s in specs if s.name in wanted]
        missing = wanted - {s.name for s in specs}
        if missing:
            log.error("unknown models: %s", sorted(missing))
            return 1

    max_folds = args.max_folds
    if args.smoke:
        cfg = replace(cfg, train=replace(cfg.train, n_ensemble=1, max_epochs=8, patience=3))
        max_folds = max_folds or 3

    log.info("panel %d rows, %d features, models: %s",
             panel.height, len(feature_cols), [s.name for s in specs])
    run_walk_forward(cfg, panel, feature_cols, specs, max_folds=max_folds)
    log.info("predictions written to %s", cfg.paths.outputs / f"predictions{cfg.suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
