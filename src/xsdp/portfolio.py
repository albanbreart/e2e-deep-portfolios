"""Turning scores into an allocation across anomalies, and that into net P&L.

This module is deliberately separate from the models.  A prediction is not a
strategy: the map from "which anomaly looks good" to "how much capital it gets"
involves a benchmark, concentration limits, and — above all — the cost of getting
from last month's allocation to this one *and* the cost of simply carrying each
sleeve for another month.  Most of the gap between a headline information
coefficient and a tradable Sharpe ratio lives here.

The accounting convention throughout:

*   ``w[t]`` is the allocation set at the **close of month t**, using only data
    dated ``t`` or earlier.
*   It earns ``r[t+1]``, the anomaly returns of month ``t+1``.
*   Switching cost is charged at ``t`` on the distance between the new weights
    and the weights *drifted* from ``t-1``.  Drift matters: an allocation left
    completely untouched still changes as the sleeves earn different returns, and
    charging turnover against the stale ``w[t-1]`` would invent trades that never
    happened.
*   Carrying cost is charged at ``t`` on ``|w[t]|``, at each anomaly's own
    measured monthly turnover times the per-stock spread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Weight construction
# --------------------------------------------------------------------------- #
def equal_weights(n: int) -> np.ndarray:
    """The 1/N benchmark. Hard to beat, and the thing every table compares to."""
    return np.full(n, 1.0 / n) if n else np.zeros(0)


def tilt_weights(scores: np.ndarray, *, tilt_gross: float = 1.0) -> np.ndarray:
    """Equal weight plus a dollar-neutral tilt proportional to the demeaned rank.

    Ranks rather than raw scores, so the allocation is invariant to any monotone
    transformation of a model's output and two models are compared on the
    *ordering* they produce rather than on the calibration of their level.
    """
    n = len(scores)
    if n == 0:
        return np.zeros(0)
    base = equal_weights(n)
    if n < 2:
        return base
    r = pd.Series(scores).rank(method="average").to_numpy()
    dev = r - r.mean()
    denom = np.abs(dev).sum()
    if denom == 0:
        return base
    return base + tilt_gross * dev / denom


def neutral_weights(scores: np.ndarray, *, gross: float = 2.0) -> np.ndarray:
    """Pure dollar-neutral long-short across anomalies (no 1/N base)."""
    n = len(scores)
    if n < 2:
        return np.zeros(n)
    r = pd.Series(scores).rank(method="average").to_numpy()
    dev = r - r.mean()
    denom = np.abs(dev).sum()
    return np.zeros(n) if denom == 0 else gross * dev / denom


def quantile_weights(scores: np.ndarray, *, n_quantiles: int = 5,
                     gross: float = 1.0) -> np.ndarray:
    """Long-only: equal weight the best quintile of anomalies, nothing else."""
    n = len(scores)
    w = np.zeros(n)
    if n < n_quantiles:
        return equal_weights(n)
    r = pd.Series(scores).rank(method="first").to_numpy()
    top = r > n * (1 - 1.0 / n_quantiles)
    if top.sum():
        w[top] = gross / top.sum()
    return w


def cap_and_renormalise(w: np.ndarray, *, max_weight: float, budget: float = 1.0,
                        max_iter: int = 50) -> np.ndarray:
    """Enforce a per-name cap while preserving the total budget.

    Naively clipping and rescaling does not converge: rescaling pushes the
    just-clipped names back over the cap and the iteration oscillates.  The fix
    is to treat a capped name as *fixed* at the cap and redistribute the
    remaining budget only across names still free, which converges monotonically
    because the free set shrinks every pass.
    """
    w = np.asarray(w, dtype=float).copy()
    n = len(w)
    if n == 0 or n * max_weight <= budget:      # cap infeasible: leave as is
        return w

    free = np.ones(n, dtype=bool)
    for _ in range(max_iter):
        breach = free & (w > max_weight + 1e-12)
        if not breach.any():
            break
        w[breach] = max_weight
        free &= ~breach
        remaining = budget - w[~free].sum()
        free_sum = w[free].sum()
        if free_sum <= 0 or remaining <= 0:
            break
        w[free] *= remaining / free_sum
    return w


SCHEMES = {
    "equal": lambda s, **k: equal_weights(len(s)),
    "tilt": tilt_weights,
    "neutral": neutral_weights,
    "quantile": quantile_weights,
}


# --------------------------------------------------------------------------- #
# Costs and the backtest loop
# --------------------------------------------------------------------------- #
def drift_weights(w_prev: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Weights at the end of a month, given start-of-month weights and returns.

    Position values grow by ``1 + r`` while the account's net asset value grows
    by ``1 + w'r``.  Re-expressing positions as a fraction of the new NAV gives
    the allocation an untouched portfolio would be holding a month later; the
    distance between *that* and the new target is the only turnover traded.
    """
    grown = w_prev * (1.0 + r)
    nav = 1.0 + float(w_prev @ r)
    return grown if abs(nav) < 1e-8 else grown / nav


def _next_month(yyyymm: int) -> int:
    y, m = divmod(yyyymm, 100)
    return (y + 1) * 100 + 1 if m == 12 else y * 100 + m + 1


@dataclass
class BacktestResult:
    monthly: pd.DataFrame       # index = month the P&L is EARNED (t+1)
    weights: dict[int, pd.Series]

    @property
    def net(self) -> pd.Series:
        return self.monthly["net"]


def backtest(
    predictions: pd.DataFrame,
    cfg,
    *,
    scheme: str | None = None,
    cost_multiplier: float = 1.0,
    vol_target: float | None = ...,
) -> BacktestResult:
    """Run the month-by-month allocation backtest.

    ``predictions`` must contain ``yyyymm``, ``signalname``, ``score``,
    ``ret_next`` and ``carry_cost`` (the per-anomaly monthly carrying rate, in
    decimal, built by :mod:`xsdp.data.build_anomaly_panel`).

    ``cost_multiplier`` scales *both* cost components at once, which is what the
    sensitivity exhibit sweeps: it lets a reader who disbelieves our spread
    assumption read their own answer off the chart.
    """
    pc, cc = cfg.portfolio, cfg.costs
    scheme = scheme or pc.scheme
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; have {sorted(SCHEMES)}")
    if vol_target is ...:
        vol_target = pc.vol_target_annual

    required = {"yyyymm", "signalname", "score", "ret_next"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"predictions missing columns: {sorted(missing)}")

    switch_rate = 2.0 * cc.switch_bps / 1e4 * cost_multiplier
    kwargs = {}
    if scheme == "tilt":
        kwargs["tilt_gross"] = pc.tilt_gross
    elif scheme == "neutral":
        kwargs["gross"] = pc.gross_leverage * 2.0
    elif scheme == "quantile":
        kwargs["n_quantiles"] = pc.n_quantiles

    months = np.sort(predictions["yyyymm"].unique())
    prev_w = pd.Series(dtype=float)
    rows, weight_book = [], {}

    for t in months:
        g = predictions[predictions["yyyymm"] == t]
        w = SCHEMES[scheme](g["score"].to_numpy(), **kwargs)

        if pc.max_weight and scheme in ("tilt", "equal", "quantile"):
            w = cap_and_renormalise(w, max_weight=pc.max_weight, budget=float(np.sum(w)))

        w = pd.Series(w, index=g["signalname"].to_numpy())

        idx = w.index.union(prev_w.index)
        delta = w.reindex(idx).fillna(0.0) - prev_w.reindex(idx).fillna(0.0)
        switching = switch_rate * float(delta.abs().sum())
        turnover = float(delta.abs().sum())

        carry_rate = (
            g.set_index("signalname")["carry_cost"].reindex(w.index).fillna(0.0)
            if "carry_cost" in g else pd.Series(0.0, index=w.index)
        )
        carrying = float((carry_rate * w.abs()).sum()) * cost_multiplier

        r = g.set_index("signalname")["ret_next"].reindex(w.index).fillna(0.0)
        gross_ret = float((w * r).sum())

        short_notional = float(w[w < 0].abs().sum())
        borrow = short_notional * cc.short_fee_bps_annual / 1e4 / 12.0

        cost = switching + carrying + borrow
        rows.append({"formation_yyyymm": int(t), "yyyymm": _next_month(int(t)),
                     "gross": gross_ret, "cost": cost, "net": gross_ret - cost,
                     "switching": switching, "carrying": carrying,
                     "turnover": turnover, "n": len(w)})
        weight_book[int(t)] = w
        prev_w = pd.Series(drift_weights(w.to_numpy(), r.to_numpy()), index=w.index)

    monthly = pd.DataFrame(rows).set_index("yyyymm")
    if vol_target:
        monthly = _apply_vol_target(monthly, vol_target)
    return BacktestResult(monthly, weight_book)


def _apply_vol_target(monthly: pd.DataFrame, target_annual: float,
                      *, window: int = 36, min_periods: int = 12,
                      max_leverage: float = 3.0) -> pd.DataFrame:
    """Scale the book to a constant ex-ante volatility, using a causal estimate.

    The scaling factor for month ``t`` uses only returns realised strictly before
    ``t`` (``shift(1)`` on a trailing window), so it is implementable.  Without
    the shift this is one of the sneakiest look-aheads in the backtesting canon.
    """
    out = monthly.copy()
    realised = out["net"].rolling(window, min_periods=min_periods).std().shift(1) * np.sqrt(12)
    lev = (target_annual / realised).clip(upper=max_leverage).fillna(1.0)
    out["leverage"] = lev
    for col in ("gross", "cost", "net", "turnover", "switching", "carrying"):
        out[col] = out[col] * lev
    return out


def benchmark(predictions: pd.DataFrame, cfg, **kwargs) -> BacktestResult:
    """The 1/N allocation over the same universe, costed identically."""
    eq = predictions.copy()
    eq["score"] = 0.0
    return backtest(eq, cfg, scheme="equal", **kwargs)
