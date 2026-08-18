#!/usr/bin/env python
"""Turn predictions into the report's tables and figures.

    python scripts/03_evaluate.py

Reads ``outputs/predictions/*.parquet`` and writes every table to
``report/tables`` (CSV and LaTeX) and every figure to ``report/figures`` (PDF for
the paper, PNG for the README).  Nothing here refits a model, so it is cheap to
re-run while writing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsdp import analysis as A  # noqa: E402
from xsdp.config import load_config  # noqa: E402
from xsdp.experiment import load_predictions  # noqa: E402
from xsdp.metrics import monthly_ic  # noqa: E402
from xsdp.viz import plots as P  # noqa: E402
from xsdp.viz import style as S  # noqa: E402

log = logging.getLogger("evaluate")

ROUNDING = {
    "oos_r2_pct": 3, "ic_mean": 4, "ic_t": 2, "ic_ir": 2, "ic_hit": 2,
    "ret_gross": 4, "ret_net": 4, "vol": 4, "sharpe_gross": 2, "sharpe_net": 2,
    "sharpe_net_se": 2, "t_net": 2, "max_dd": 3, "skew": 2, "turnover": 3,
    "cost_drag": 4, "hit_rate": 2, "breakeven_bps": 0, "alpha_ann": 4,
    "alpha_t": 2, "r2": 3, "switching_ann": 4, "carrying_ann": 4,
    "gross_ann": 4, "net_ann": 4, "beta_bench": 2, "info_ratio": 2,
    "t_active": 2, "active_ann": 4, "mean_tilt": 5, "mean_turnover": 3, "tilt": 5, "mean": 5,
    "cost_mult": 2, "cost_bps": 1, "spearman_vs_sharpe_net": 3, "p_value": 3,
}


def _write(df: pd.DataFrame, path: Path, *, caption: str = "") -> None:
    """Write a table as CSV and as a LaTeX fragment the report can \\input."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rounded = df.copy()
    for col, nd in ROUNDING.items():
        if col in rounded:
            rounded[col] = pd.to_numeric(rounded[col], errors="coerce").round(nd)
    rounded.to_csv(path.with_suffix(".csv"), index=False)
    rounded.to_latex(path.with_suffix(".tex"), index=False, escape=True,
                     caption=caption or path.stem.replace("_", " "),
                     label=f"tab:{path.stem}", longtable=False)
    log.info("wrote %s (%d rows)", path.with_suffix(".csv").name, len(rounded))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scheme", default="tilt", choices=["tilt", "neutral", "quantile"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    cfg.paths.mkdirs()
    S.use_report_style()

    preds = load_predictions(cfg)
    if not preds:
        log.error("no predictions in %s -- run scripts/02_run_experiments.py",
                  cfg.paths.outputs / f"predictions{cfg.suffix}")
        return 1
    log.info("loaded %d models: %s", len(preds), sorted(preds))

    panel = pl.read_parquet(cfg.paths.processed / f"panel_features{cfg.suffix}.parquet").to_pandas()
    # Variant runs (the capacity checks) write into their own subdirectory so a
    # re-run can never silently overwrite the headline tables and figures.
    variant = cfg.suffix.lstrip("_")
    tables = cfg.paths.tables / variant if variant else cfg.paths.tables
    figures = cfg.paths.figures / variant if variant else cfg.paths.figures
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    # ---- tables -----------------------------------------------------------
    _write(A.statistical_table(preds), tables / "t1_statistical",
           caption="Out-of-sample predictive accuracy")

    econ = A.economic_table(preds, cfg, schemes=(args.scheme,))
    _write(econ, tables / "t2_economic",
           caption="Allocation performance, gross and net of transaction costs")

    _write(A.accuracy_versus_value(A.statistical_table(preds), econ),
           tables / "t1b_accuracy_vs_value",
           caption="Rank correlation between forecast accuracy and net Sharpe ratio")

    profile = A.turnover_profile(preds, cfg, scheme=args.scheme)
    _write(profile, tables / "t3_turnover",
           caption="Allocation turnover and the two cost components")

    rets = A.strategy_returns(preds, cfg, scheme=args.scheme)
    rets.to_csv(cfg.paths.outputs / f"strategy_returns{cfg.suffix}.csv")

    _write(A.alpha_vs_benchmark(rets), tables / "t4_vs_benchmark",
           caption="Skill beyond the equal-weighted benchmark")

    ff_path = cfg.paths.external / "ff_factors_monthly.parquet"
    if ff_path.exists():
        ff = pd.read_parquet(ff_path).set_index("yyyymm")
        _write(A.alpha_table(rets, ff), tables / "t5_alpha",
               caption="Alpha against the Fama-French five factors plus momentum")

    _write(A.decay_analysis(preds, cfg), tables / "t6_subperiods",
           caption="Performance by sub-period")

    cs = A.cost_sensitivity(preds, cfg, scheme=args.scheme)
    _write(cs, tables / "t7_cost_sensitivity",
           caption="Net Sharpe ratio as the cost model is scaled")

    best = econ[econ["model"] != "equal_weight"].iloc[0]["model"]
    tilt = A.allocation_tilt_analysis(preds, cfg, best)
    if not tilt.empty:
        _write(tilt, tables / "t8_tilt_by_cost",
               caption="Average allocation tilt by anomaly carrying cost")
    cats = A.category_analysis(preds, cfg, best, panel)
    if not cats.empty:
        _write(cats, tables / "t9_tilt_by_category",
               caption="Average allocation tilt by anomaly category")

    # ---- figures ----------------------------------------------------------
    order = [m for m in econ["model"] if m != "equal_weight"][:4]
    top = ["equal_weight", *order]

    S.save(P.cumulative_returns(
        rets, highlight=top,
        subtitle="Walk-forward out of sample, net of measured anomaly-specific costs"),
        figures / "f1_cumulative")

    S.save(P.cost_sensitivity(
        cs, x="cost_bps", highlight=order[:3],
        subtitle="Both cost components scale together; the dotted line is our base case"),
        figures / "f2_cost_sensitivity")

    S.save(P.turnover_vs_sharpe(profile), figures / "f3_turnover")

    ic = pd.DataFrame({name: monthly_ic(df.dropna(subset=["score", "ret_next"]))
                       for name, df in preds.items()})
    S.save(P.rolling_ic(ic[order], subtitle="Signal strength over time"),
           figures / "f4_rolling_ic")

    S.save(P.drawdown(rets, models=top[:3],
                      subtitle="Net of costs, the experience of holding the allocation"),
           figures / "f5_drawdown")

    S.save(P.cost_decomposition(profile), figures / "f6_cost_decomposition")
    S.save(P.turnover_bars(profile), figures / "f7_turnover_bars")
    if not tilt.empty:
        S.save(P.tilt_by_cost(tilt), figures / "f8_tilt_by_cost")

    log.info("tables -> %s", tables)
    log.info("figures -> %s", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
