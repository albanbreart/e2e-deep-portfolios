"""Evaluation metrics for predictions and for portfolios.

Two families, kept apart on purpose:

*Statistical* metrics say whether the model forecasts returns.  We report the
out-of-sample :math:`R^2` in the Gu-Kelly-Xiu form, which benchmarks against a
**zero** forecast rather than the historical mean.  That choice is not cosmetic:
the historical mean of an individual stock's return is a famously noisy
predictor, so benchmarking against it flatters any model, and a "positive
:math:`R^2`" measured that way can coexist with a useless strategy.

*Economic* metrics say whether the forecasts are worth money.  Sharpe ratios
come with Newey-West standard errors (returns are mildly autocorrelated and
strongly heteroskedastic), and every headline figure has a net-of-cost twin.
The break-even cost — the level of transaction costs at which the strategy stops
making money — is the single number a trading desk will actually ask for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MONTHS = 12


# --------------------------------------------------------------------------- #
# Statistical
# --------------------------------------------------------------------------- #
def oos_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Gu-Kelly-Xiu out-of-sample R^2: benchmark is a zero forecast."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    denom = np.sum(y[ok] ** 2)
    return float("nan") if denom == 0 else float(1.0 - np.sum((y[ok] - yhat[ok]) ** 2) / denom)


def monthly_ic(df: pd.DataFrame, *, score: str = "score", target: str = "ret_next",
               method: str = "spearman") -> pd.Series:
    """Cross-sectional information coefficient, one number per month."""
    return (
        df.groupby("yyyymm")[[score, target]]
        .apply(lambda g: g[score].corr(g[target], method=method))
        .rename("ic")
    )


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """Mean IC, its Newey-West t-statistic, and the information ratio of the IC."""
    ic = ic.dropna()
    if ic.empty:
        return {"ic_mean": np.nan, "ic_t": np.nan, "ic_ir": np.nan, "ic_hit": np.nan}
    t = newey_west_tstat(ic.to_numpy())
    return {
        "ic_mean": float(ic.mean()),
        "ic_t": t,
        "ic_ir": float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else np.nan,
        "ic_hit": float((ic > 0).mean()),
    }


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def newey_west_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """t-statistic of the mean of ``x`` with a Newey-West HAC standard error."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    var = float(e @ e) / n
    for lag in range(1, lags + 1):
        cov = float(e[lag:] @ e[:-lag]) / n
        var += 2 * (1 - lag / (lags + 1)) * cov      # Bartlett kernel
    var = max(var, 1e-18)
    return float(x.mean() / np.sqrt(var / n))


def sharpe_se(returns: np.ndarray) -> float:
    """Standard error of the annualised Sharpe ratio (Lo, 2002; iid-adjusted)."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 12 or r.std(ddof=1) == 0:
        return float("nan")
    sr_m = r.mean() / r.std(ddof=1)
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    var = (1 + 0.5 * sr_m**2 - g3 * sr_m + (g4 - 3) / 4 * sr_m**2) / n
    return float(np.sqrt(max(var, 0)) * np.sqrt(MONTHS))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss of the compounded equity curve."""
    equity = (1 + returns.fillna(0)).cumprod()
    return float((equity / equity.cummax() - 1).min())


# --------------------------------------------------------------------------- #
# Economic
# --------------------------------------------------------------------------- #
def performance_summary(monthly: pd.DataFrame, *, label: str = "") -> dict[str, float]:
    """Headline table row for one strategy: gross and net, risk, cost, capacity."""
    net, gross = monthly["net"].dropna(), monthly["gross"].dropna()
    out = {
        "label": label,
        "n_months": len(net),
        "ret_gross": float(gross.mean() * MONTHS),
        "ret_net": float(net.mean() * MONTHS),
        "vol": float(net.std(ddof=1) * np.sqrt(MONTHS)),
        "sharpe_gross": _sharpe(gross),
        "sharpe_net": _sharpe(net),
        "sharpe_net_se": sharpe_se(net.to_numpy()),
        "t_net": newey_west_tstat(net.to_numpy()),
        "max_dd": max_drawdown(net),
        "skew": float(stats.skew(net)),
        "turnover": float(monthly["turnover"].mean()),
        "cost_drag": float(monthly["cost"].mean() * MONTHS),
        "hit_rate": float((net > 0).mean()),
    }
    out["breakeven_bps"] = _breakeven_bps(monthly)
    return out


def _sharpe(r: pd.Series) -> float:
    s = r.std(ddof=1)
    return float("nan") if s == 0 else float(r.mean() / s * np.sqrt(MONTHS))


def _breakeven_bps(monthly: pd.DataFrame) -> float:
    """One-way cost (bps) that would drive the average net return to zero.

    Reported because it is cost-assumption-free: whatever a reader believes
    about spreads and impact, they can compare it to this number themselves.
    """
    turnover = monthly["turnover"].mean()
    return float("nan") if turnover <= 0 else float(monthly["gross"].mean() / turnover * 1e4)


def factor_regression(returns: pd.Series, factors: pd.DataFrame,
                      cols: list[str] | None = None) -> pd.DataFrame:
    """Regress strategy returns on a factor model; HAC t-stats on every coefficient.

    Alpha here answers the question a fund allocator asks: is this new, or is it
    a repackaging of exposures I can buy for a few basis points?
    """
    import statsmodels.api as sm

    cols = cols or [c for c in ("mktrf", "smb", "hml", "rmw", "cma", "umd") if c in factors]
    df = pd.concat([returns.rename("y"), factors[cols]], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()
    X = sm.add_constant(df[cols])
    fit = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    out = pd.DataFrame({"coef": fit.params, "t": fit.tvalues, "p": fit.pvalues})
    out.loc["const", "coef"] *= MONTHS      # report alpha annualised
    out.attrs["r2"] = fit.rsquared
    out.attrs["n"] = int(fit.nobs)
    return out


def subperiod_table(monthly: pd.DataFrame, *, breaks: tuple[int, ...] = (200001, 201001)
                    ) -> pd.DataFrame:
    """Net Sharpe by sub-period, the cheapest test of whether an edge decayed."""
    edges = [monthly.index.min()] + list(breaks) + [monthly.index.max() + 1]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        block = monthly[(monthly.index >= lo) & (monthly.index < hi)]
        if len(block) < 12:
            continue
        rows.append({
            "period": f"{lo // 100}-{(hi - 1) // 100}",
            "n_months": len(block),
            "ret_net": float(block["net"].mean() * MONTHS),
            "sharpe_net": _sharpe(block["net"]),
            "t_net": newey_west_tstat(block["net"].to_numpy()),
            "max_dd": max_drawdown(block["net"]),
        })
    return pd.DataFrame(rows)
