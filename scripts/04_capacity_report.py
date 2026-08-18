#!/usr/bin/env python
"""Compare the headline result with the large-cap capacity run, side by side.

    python scripts/04_capacity_report.py

Published anomaly returns are concentrated in small, illiquid stocks. Rebuilding
every anomaly on stocks above the NYSE 20th percentile of market capitalisation
costs 30% of the gross premium before a single basis point of trading cost. The
question this answers is whether the *ranking of objectives* -- the paper's
actual claim -- survives that restriction, or whether it was an artefact of an
untradable universe.

Reads both prediction sets and writes ``report/tables/t10_capacity.{csv,tex}``
plus a comparison figure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsdp import analysis as A  # noqa: E402
from xsdp.config import load_config  # noqa: E402
from xsdp.experiment import load_predictions  # noqa: E402
from xsdp.viz import plots as P  # noqa: E402
from xsdp.viz import style as S  # noqa: E402

log = logging.getLogger("capacity")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="configs/default.yaml")
    ap.add_argument("--variant", default="configs/capacity.yaml")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", stream=sys.stdout)
    base_cfg = load_config(args.base)
    var_cfg = load_config(args.variant)
    S.use_report_style()

    frames = []
    for label, cfg in (("all stocks", base_cfg), ("large caps only", var_cfg)):
        preds = load_predictions(cfg)
        if not preds:
            log.error("no predictions for %r (suffix %r) -- run scripts/02_run_experiments.py "
                      "with the matching config first", label, cfg.suffix)
            return 1
        econ = A.economic_table(preds, cfg)
        econ.insert(0, "universe", label)
        frames.append(econ[["universe", "model", "ret_gross", "ret_net",
                            "sharpe_gross", "sharpe_net", "t_net", "turnover"]])
        log.info("%s: %d models, %d months", label, len(preds),
                 int(econ["n_months"].iloc[0]) if "n_months" in econ else -1)

    both = pd.concat(frames, ignore_index=True)

    wide = both.pivot(index="model", columns="universe", values="sharpe_net")
    wide["delta"] = wide["large caps only"] - wide["all stocks"]
    wide = wide.sort_values("all stocks", ascending=False).reset_index()

    rho = wide[["all stocks", "large caps only"]].corr(method="spearman").iloc[0, 1]
    log.info("rank correlation of net Sharpe across the two universes: %.3f", rho)

    tables = base_cfg.paths.tables
    tables.mkdir(parents=True, exist_ok=True)
    for df, name, cap in (
        (both, "t10_capacity_full", "Performance in both universes"),
        (wide, "t10_capacity", "Net Sharpe ratio: all stocks vs large caps only"),
    ):
        out = tables / name
        rounded = df.copy()
        for c in rounded.columns:
            if rounded[c].dtype.kind == "f":
                rounded[c] = rounded[c].round(3)
        rounded.to_csv(out.with_suffix(".csv"), index=False)
        rounded.to_latex(out.with_suffix(".tex"), index=False, escape=True,
                         caption=cap, label=f"tab:{name}")
        log.info("wrote %s", out.with_suffix(".csv").name)

    fig = P.capacity_scatter(wide, rho=rho)
    S.save(fig, base_cfg.paths.figures / "f9_capacity")
    log.info("wrote f9_capacity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
