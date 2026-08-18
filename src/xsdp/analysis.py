"""From prediction frames to the tables the report is built from.

Each function returns a tidy DataFrame and writes nothing; the scripts decide
where output lands.  That means every table in the paper can be regenerated in a
notebook without touching the filesystem, which is what makes the results
checkable.

One convention runs through all of it: **the 1/N allocation is a model.**  It
appears as a row in every table under the name ``equal_weight``, costed exactly
like the others.  An allocator who cannot beat it has learned nothing, and
burying the benchmark in the text instead of the table is how that gets hidden.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .metrics import (
    factor_regression,
    ic_summary,
    monthly_ic,
    oos_r2,
    performance_summary,
    subperiod_table,
)
from .portfolio import backtest, benchmark

log = logging.getLogger(__name__)

TARGET = "ret_next"


def statistical_table(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Out-of-sample R^2 and information coefficients, one row per model."""
    rows = []
    for name, df in predictions.items():
        d = df.dropna(subset=["score", TARGET])
        ic = monthly_ic(d, target=TARGET)
        rows.append({
            "model": name,
            "n_obs": len(d),
            "oos_r2_pct": 100 * oos_r2(d[TARGET], _scale_to_target(d)),
            **ic_summary(ic),
        })
    return pd.DataFrame(rows).sort_values("ic_mean", ascending=False).reset_index(drop=True)


def _scale_to_target(d: pd.DataFrame) -> np.ndarray:
    """Put scores on the scale of returns before computing an R^2.

    Losses like the Sharpe ratio or the IC are scale-free, so the models trained
    on them emit a ranking with no natural units; comparing that raw output to a
    return would give a meaningless, hugely negative R^2.  We apply the single
    OLS rescaling any user of the signal would apply, which leaves the ranking
    untouched.  The report states what this means: the R^2 of a scale-free model
    is a *best-case* number, and the honest comparison between objectives is the
    IC and the portfolio columns.
    """
    x = d["score"].to_numpy(dtype=float)
    y = d[TARGET].to_numpy(dtype=float)
    if x.std() == 0:
        return np.full_like(y, y.mean())
    beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    return beta * (x - x.mean()) + y.mean()


def _with_benchmark(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Any one prediction frame, used to build the 1/N benchmark on the same universe."""
    return next(iter(predictions.values()))


def economic_table(predictions: dict[str, pd.DataFrame], cfg, *,
                   schemes: tuple[str, ...] = ("tilt",)) -> pd.DataFrame:
    """Portfolio performance for every model plus the 1/N benchmark."""
    rows = []
    bench = benchmark(_with_benchmark(predictions), cfg)
    rows.append({"model": "equal_weight", "scheme": "equal",
                 **performance_summary(bench.monthly, label="equal_weight")})
    for name, df in predictions.items():
        for scheme in schemes:
            res = backtest(df, cfg, scheme=scheme)
            rows.append({"model": name, "scheme": scheme,
                         **performance_summary(res.monthly, label=f"{name}/{scheme}")})
    out = pd.DataFrame(rows).drop(columns=["label"])
    return out.sort_values("sharpe_net", ascending=False).reset_index(drop=True)


def strategy_returns(predictions: dict[str, pd.DataFrame], cfg, *,
                     scheme: str = "tilt") -> pd.DataFrame:
    """Net monthly returns of every allocation, aligned on one index."""
    series = {"equal_weight": benchmark(_with_benchmark(predictions), cfg).monthly["net"]}
    for name, df in predictions.items():
        series[name] = backtest(df, cfg, scheme=scheme).monthly["net"]
    return pd.DataFrame(series).sort_index()


def alpha_table(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Annualised alpha and HAC t-stat of each allocation against FF5 + momentum."""
    rows = []
    for name in returns.columns:
        reg = factor_regression(returns[name], factors)
        if reg.empty:
            continue
        row = {"model": name, "alpha_ann": reg.loc["const", "coef"],
               "alpha_t": reg.loc["const", "t"], "r2": reg.attrs["r2"]}
        for f in reg.index.drop("const"):
            row[f"beta_{f}"] = reg.loc[f, "coef"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("alpha_t", ascending=False).reset_index(drop=True)


def alpha_vs_benchmark(returns: pd.DataFrame, *, bench: str = "equal_weight") -> pd.DataFrame:
    """Each model regressed on the 1/N benchmark: is there any skill *beyond* it?

    The sharpest test in the paper.  A model can post a fine Sharpe ratio purely
    by being a levered version of equal weight; the intercept here is what is
    left after that explanation is exhausted.
    """
    import statsmodels.api as sm

    from .metrics import MONTHS, newey_west_tstat

    rows = []
    for name in returns.columns:
        if name == bench:
            continue
        df = returns[[name, bench]].dropna()
        if len(df) < 24:
            continue
        fit = sm.OLS(df[name], sm.add_constant(df[[bench]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 6})
        # The information ratio is the *active* return per unit of active risk.
        # It cannot be read off the regression residuals: an OLS with an
        # intercept forces their mean to zero by construction, so that route
        # returns 0.000 for every model no matter how good it is.
        active = (df[name] - df[bench]).to_numpy()
        sd_active = active.std(ddof=1)
        rows.append({
            "model": name,
            "alpha_ann": float(fit.params["const"] * MONTHS),
            "alpha_t": float(fit.tvalues["const"]),
            "beta_bench": float(fit.params[bench]),
            "r2": float(fit.rsquared),
            "active_ann": float(active.mean() * MONTHS),
            "info_ratio": float(active.mean() / sd_active * np.sqrt(MONTHS))
            if sd_active > 0 else np.nan,
            "t_active": newey_west_tstat(active),
        })
    return pd.DataFrame(rows).sort_values("alpha_t", ascending=False).reset_index(drop=True)


def cost_sensitivity(predictions: dict[str, pd.DataFrame], cfg, *,
                     multipliers=None, scheme: str = "tilt") -> pd.DataFrame:
    """Net Sharpe as a function of a scaling of the whole cost model.

    The most informative single exhibit in the paper: it shows where each model
    crosses zero, and whether the cost-aware objective's advantage is real or an
    artefact of one particular cost assumption.  Both cost components -- the
    per-anomaly carry and the switching charge -- scale together, so a reader who
    believes spreads are half what we assume can read their answer off directly.
    """
    multipliers = multipliers or cfg.costs.sensitivity_multipliers
    rows = []
    bench_df = _with_benchmark(predictions)
    for m in multipliers:
        b = benchmark(bench_df, cfg, cost_multiplier=float(m)).monthly["net"]
        rows.append({"model": "equal_weight", "cost_mult": float(m),
                     "cost_bps": float(m) * cfg.costs.spread_bps,
                     "sharpe_net": _sharpe(b), "ret_net": float(b.mean() * 12)})
    for name, df in predictions.items():
        for m in multipliers:
            net = backtest(df, cfg, scheme=scheme, cost_multiplier=float(m)).monthly["net"]
            rows.append({"model": name, "cost_mult": float(m),
                         "cost_bps": float(m) * cfg.costs.spread_bps,
                         "sharpe_net": _sharpe(net), "ret_net": float(net.mean() * 12)})
    return pd.DataFrame(rows)


def _sharpe(r: pd.Series) -> float:
    s = r.std(ddof=1)
    return float("nan") if s == 0 else float(r.mean() / s * np.sqrt(12))


def turnover_profile(predictions: dict[str, pd.DataFrame], cfg,
                     scheme: str = "tilt") -> pd.DataFrame:
    """Allocation turnover, the two cost components, and what survives."""
    rows = []
    b = benchmark(_with_benchmark(predictions), cfg).monthly
    rows.append({"model": "equal_weight", "turnover": float(b["turnover"].mean()),
                 "switching_ann": float(b["switching"].mean() * 12),
                 "carrying_ann": float(b["carrying"].mean() * 12),
                 "gross_ann": float(b["gross"].mean() * 12),
                 "net_ann": float(b["net"].mean() * 12)})
    for name, df in predictions.items():
        m = backtest(df, cfg, scheme=scheme).monthly
        rows.append({"model": name, "turnover": float(m["turnover"].mean()),
                     "switching_ann": float(m["switching"].mean() * 12),
                     "carrying_ann": float(m["carrying"].mean() * 12),
                     "gross_ann": float(m["gross"].mean() * 12),
                     "net_ann": float(m["net"].mean() * 12)})
    return pd.DataFrame(rows).sort_values("net_ann", ascending=False).reset_index(drop=True)


def decay_analysis(predictions: dict[str, pd.DataFrame], cfg,
                   breaks: tuple[int, ...] = (201501,)) -> pd.DataFrame:
    """Sub-period performance: has the edge survived, or was it an artefact?"""
    frames = []
    for name, df in predictions.items():
        t = subperiod_table(backtest(df, cfg).monthly, breaks=breaks)
        t.insert(0, "model", name)
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def allocation_tilt_analysis(predictions: dict[str, pd.DataFrame], cfg,
                             model: str, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """What kind of anomaly does the model actually overweight?

    Averages the model's tilt away from 1/N across anomalies grouped by their
    measured turnover.  If the cost-aware objective is doing what it claims, its
    tilt should be systematically negative on the expensive-to-carry sleeves --
    and that is a mechanism, not just a performance number.
    """
    df = predictions[model].dropna(subset=["score"])
    if "turnover_raw" not in df.columns:
        return pd.DataFrame()

    res = backtest(df, cfg, scheme="tilt")
    parts = []
    for t, w in res.weights.items():
        g = df[df["yyyymm"] == t].set_index("signalname")
        common = w.index.intersection(g.index)
        parts.append(pd.DataFrame({
            "yyyymm": t,
            "signalname": common,
            "weight": w.reindex(common).to_numpy(),
            "equal": 1.0 / len(w),
            "turnover_raw": g.loc[common, "turnover_raw"].to_numpy(),
        }))
    if not parts:
        return pd.DataFrame()

    allw = pd.concat(parts, ignore_index=True)
    allw["tilt"] = allw["weight"] - allw["equal"]
    allw["cost_quintile"] = allw.groupby("yyyymm")["turnover_raw"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop") + 1
    )
    out = allw.groupby("cost_quintile").agg(
        mean_tilt=("tilt", "mean"),
        mean_turnover=("turnover_raw", "mean"),
        n=("tilt", "size"),
    ).reset_index()
    out.insert(0, "model", model)
    return out


def category_analysis(predictions: dict[str, pd.DataFrame], cfg, model: str,
                      panel: pd.DataFrame) -> pd.DataFrame:
    """Average tilt by the anomaly's economic category."""
    cats = [c for c in panel.columns if c.startswith("cat_economic_")]
    if not cats:
        return pd.DataFrame()
    df = predictions[model]
    res = backtest(df, cfg, scheme="tilt")
    rows = []
    lookup = panel.drop_duplicates("signalname").set_index("signalname")[cats]
    for w in res.weights.values():
        tilt = w - 1.0 / len(w)
        sub = lookup.reindex(w.index).fillna(0.0)
        for c in cats:
            sel = sub[c] > 0
            if sel.any():
                rows.append({"category": c.replace("cat_economic_", ""),
                             "tilt": float(tilt[sel.to_numpy()].mean())})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).groupby("category")["tilt"].agg(["mean", "count"]).reset_index()
    out.insert(0, "model", model)
    return out.sort_values("mean", ascending=False).reset_index(drop=True)


def accuracy_versus_value(statistical: pd.DataFrame, economic: pd.DataFrame
                          ) -> pd.DataFrame:
    """Does forecast accuracy order the models by how much money they make?

    Returns the rank correlation between each statistical metric and net Sharpe
    ratio, with its p-value.  This exists so the claim in the paper is a computed
    number that moves when the data does, rather than a sentence asserted once and
    never re-checked.  With ten models the test has very little power, which is
    itself part of the answer and is reported alongside.
    """
    from scipy import stats as st

    econ = economic[economic["model"] != "equal_weight"][["model", "sharpe_net"]]
    merged = statistical.merge(econ, on="model")
    rows = []
    for metric in ("ic_mean", "ic_t", "oos_r2_pct"):
        if metric not in merged:
            continue
        rho, pval = st.spearmanr(merged[metric], merged["sharpe_net"])
        rows.append({"metric": metric, "spearman_vs_sharpe_net": float(rho),
                     "p_value": float(pval), "n_models": int(len(merged))})
    return pd.DataFrame(rows)
