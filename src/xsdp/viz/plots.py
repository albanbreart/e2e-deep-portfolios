"""The report's figures.

Each function takes tidy data and returns a Matplotlib figure; nothing is saved
here, and nothing reads from disk.  That keeps the figures testable and lets the
same code serve the PDF report and the README.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from . import style as S


def _month_axis(ax, index: pd.Index) -> np.ndarray:
    """Turn a YYYYMM index into a decimal-year x-axis with readable ticks."""
    years = index // 100 + (index % 100 - 1) / 12
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    return np.asarray(years, dtype=float)


def cumulative_returns(returns: pd.DataFrame, *, highlight: list[str] | None = None,
                       title: str = "Cumulative net-of-cost return",
                       subtitle: str | None = None, log_scale: bool = True):
    """Compounded equity curves, direct-labelled, others pushed to gray.

    Log scale by default: on a linear axis a strategy that compounds for thirty
    years makes its first two decades invisible, and those decades are where the
    interesting differences between models usually are.
    """
    fig, ax = plt.subplots(figsize=S.FIGSIZE)
    x = _month_axis(ax, returns.index)
    highlight = highlight or list(returns.columns)[:6]

    ends = []
    for name in returns.columns:
        if name in highlight:
            continue
        ax.plot(x, (1 + returns[name].fillna(0)).cumprod(), color=S.MUTED, lw=0.9,
                alpha=0.55, zorder=2)

    for i, name in enumerate(highlight):
        if name not in returns:
            continue
        eq = (1 + returns[name].fillna(0)).cumprod()
        c = S.colour(i)
        ax.plot(x, eq, color=c, lw=1.6, zorder=3)
        ends.append((eq.iloc[-1], name, c))

    for y, name, c in _declutter(ends):
        S.label_line_end(ax, x[-1], y, name, c)

    if log_scale:
        S.log_return_axis(ax)
    ax.axhline(1.0, color=S.RULE, lw=0.9, zorder=1)
    ax.set_xlabel("")
    ax.set_ylabel("Growth of $1")
    # Room on the right for the direct labels, expressed in data units so it
    # survives `bbox_inches="tight"` (which crops to the axes, not to overflow).
    ax.set_xlim(x[0], x[-1] + 0.16 * (x[-1] - x[0]))
    S.limit_ticks_to_data(ax, x[0], x[-1])
    S.title(ax, title, subtitle)
    return fig


def _declutter(ends: list[tuple[float, str, str]], *, min_gap_frac: float = 0.095):
    """Nudge overlapping end-labels apart in log space so none collide."""
    if not ends:
        return ends
    ends = sorted(ends, key=lambda t: t[0])
    ys = np.array([e[0] for e in ends], dtype=float)
    span = np.log(ys.max() / max(ys.min(), 1e-9)) or 1.0
    gap = span * min_gap_frac
    log_y = np.log(np.maximum(ys, 1e-9))
    for i in range(1, len(log_y)):
        if log_y[i] - log_y[i - 1] < gap:
            log_y[i] = log_y[i - 1] + gap
    return [(float(np.exp(ly)), name, c) for ly, (_, name, c) in zip(log_y, ends, strict=False)]


def cost_sensitivity(cs: pd.DataFrame, *, x: str = "cost_bps",
                     highlight: list[str] | None = None,
                     benchmark: str = "equal_weight",
                     base_case: float | None = 25.0,
                     title: str = "Net Sharpe ratio vs transaction costs",
                     subtitle: str | None = None):
    """Where each model's edge dies.

    The x-axis is the assumption and the y-axis is the answer, so a reader who
    disbelieves our spread estimate can substitute their own and read off the
    consequence.

    With eleven models this cannot be an eleven-colour chart: the categorical
    palette carries eight slots and past that identity by colour breaks down.
    So only the models the argument turns on are coloured and labelled, the
    benchmark is drawn in ink, and the rest collapse into an unlabelled gray
    band that shows the range they occupy.
    """
    fig, ax = plt.subplots(figsize=S.FIGSIZE)
    models = list(dict.fromkeys(cs["model"]))
    highlight = highlight or [m for m in models if m != benchmark][:3]

    rest = [m for m in models if m not in highlight and m != benchmark]
    if rest:
        block = cs[cs["model"].isin(rest)].pivot_table(index=x, columns="model",
                                                       values="sharpe_net")
        ax.fill_between(block.index, block.min(axis=1), block.max(axis=1),
                        color=S.MUTED, alpha=0.22, lw=0, zorder=2)
        S.label_line_end(ax, block.index.max(), float(block.iloc[-1].median()),
                         f"{len(rest)} others", S.MUTED)

    ends = []
    if benchmark in models:
        b = cs[cs["model"] == benchmark].sort_values(x)
        ax.plot(b[x], b["sharpe_net"], color=S.INK_2, lw=1.8, ls=(0, (5, 2)),
                marker="o", markersize=3.2, zorder=4)
        ends.append((float(b["sharpe_net"].iloc[-1]), "1/N benchmark", S.INK_2))

    for i, m in enumerate(highlight):
        b = cs[cs["model"] == m].sort_values(x)
        c = S.colour(i)
        ax.plot(b[x], b["sharpe_net"], color=c, lw=1.7, marker="o",
                markersize=3.2, zorder=5)
        ends.append((float(b["sharpe_net"].iloc[-1]), m, c))

    for y, name, c in _declutter_linear(ends):
        S.label_line_end(ax, cs[x].max(), y, name, c)

    S.zero_line(ax)
    if base_case is not None:
        ax.axvline(base_case, color=S.RULE, lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.set_xlabel("Assumed one-way stock trading cost (basis points)")
    ax.set_ylabel("Annualised Sharpe ratio, net")
    lo, hi = float(cs[x].min()), float(cs[x].max())
    ax.set_xlim(lo - 0.02 * (hi - lo), hi + 0.26 * (hi - lo))
    S.limit_ticks_to_data(ax, lo, hi, integer=False)
    S.title(ax, title, subtitle)
    return fig


def tilt_by_cost(tilt: pd.DataFrame, *,
                 title: str = "The model learns to avoid expensive anomalies",
                 subtitle: str = "Average allocation weight minus 1/N, by anomaly carrying cost"):
    """The mechanism, not just the outcome.

    If the cost-aware objective is doing what it claims, its allocation should
    tilt *away* from anomalies whose books are expensive to carry.  A diverging
    palette is right here because the quantity has a meaningful zero: overweight
    versus underweight relative to equal weight.
    """
    fig, ax = plt.subplots(figsize=S.FIGSIZE_WIDE)
    x = np.arange(len(tilt))
    vals = tilt["mean_tilt"].to_numpy()
    colours = [S.POS if v >= 0 else S.NEG for v in vals]
    ax.bar(x, vals, width=0.62, color=colours, zorder=3,
           edgecolor=S.SURFACE, linewidth=1.0)
    S.zero_line(ax)
    labels = [f"Q{int(q)}\n{t:.2f}" for q, t in
              zip(tilt["cost_quintile"], tilt["mean_turnover"], strict=False)]
    ax.set_xticks(x, labels)
    ax.set_xlabel("Carrying-cost quintile (1 = cheapest)   ·   mean monthly turnover below")
    ax.set_ylabel("Mean tilt vs 1/N")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.3f}"))
    S.title(ax, title, subtitle)
    return fig


def _declutter_linear(ends, *, min_gap_frac: float = 0.10):
    if not ends:
        return ends
    ends = sorted(ends, key=lambda t: t[0])
    ys = np.array([e[0] for e in ends], dtype=float)
    gap = (ys.max() - ys.min() or 1.0) * min_gap_frac
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    return [(float(y), n, c) for y, (_, n, c) in zip(ys, ends, strict=False)]


def turnover_vs_sharpe(profile: pd.DataFrame, *,
                       title: str = "The cost of churn",
                       subtitle: str = "Gross Sharpe rewards trading; net Sharpe pays for it"):
    """One labelled point per model: turnover on x, net Sharpe on y.

    A scatter puts every pair of colours on screen at once, which the palette
    cannot guarantee to separate beyond three slots, so identity here is carried
    by a text label on every point and colour is purely decorative emphasis.
    """
    fig, ax = plt.subplots(figsize=S.FIGSIZE)
    best = profile["net_ann"].idxmax()
    for i, row in profile.iterrows():
        c = S.colour(0) if i == best else S.MUTED
        ax.scatter(row["turnover"], row["net_ann"], s=46, color=c, zorder=3,
                   edgecolor=S.SURFACE, linewidth=1.4)
        ax.annotate(row["model"], (row["turnover"], row["net_ann"]),
                    xytext=(6, 0), textcoords="offset points", fontsize=8,
                    color=S.INK if i == best else S.INK_2, va="center")
    S.zero_line(ax)
    ax.set_xlabel("Average monthly turnover (fraction of gross book)")
    ax.set_ylabel("Annualised net return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.margins(x=0.12, y=0.16)
    S.title(ax, title, subtitle)
    return fig


def rolling_ic(ic: pd.DataFrame, *, window: int = 36,
               title: str = "Rolling information coefficient",
               subtitle: str | None = None):
    """36-month rolling mean IC per model -- the picture of signal decay."""
    fig, ax = plt.subplots(figsize=S.FIGSIZE)
    roll = ic.rolling(window, min_periods=window // 2).mean().dropna(how="all")
    x = _month_axis(ax, roll.index)
    ends = []
    for i, name in enumerate(roll.columns):
        c = S.colour(i)
        ax.plot(x, roll[name], color=c, lw=1.5, zorder=3)
        last = roll[name].dropna()
        if len(last):
            ends.append((float(last.iloc[-1]), name, c))
    for y, name, c in _declutter_linear(ends):
        S.label_line_end(ax, x[-1], y, name, c)
    S.zero_line(ax)
    ax.set_ylabel(f"{window}-month mean rank IC")
    ax.set_xlim(x[0], x[-1] + 0.16 * (x[-1] - x[0]))
    S.limit_ticks_to_data(ax, x[0], x[-1])
    S.title(ax, title, subtitle)
    return fig


def drawdown(returns: pd.DataFrame, *, models: list[str] | None = None,
             title: str = "Drawdown", subtitle: str | None = None):
    """Underwater curves -- what an investor actually experiences."""
    fig, ax = plt.subplots(figsize=S.FIGSIZE_WIDE)
    models = models or list(returns.columns)[:3]
    x = _month_axis(ax, returns.index)
    for i, name in enumerate(models):
        eq = (1 + returns[name].fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.fill_between(x, dd, 0, color=S.colour(i), alpha=0.18, lw=0, zorder=2)
        ax.plot(x, dd, color=S.colour(i), lw=1.3, zorder=3, label=name)
    S.zero_line(ax)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Peak-to-trough")
    ax.margins(x=0.01)
    if len(models) > 1:
        ax.legend(loc="lower left", ncols=min(len(models), 3))
    S.title(ax, title, subtitle)
    return fig


def grouped_bars(df: pd.DataFrame, *, index: str, columns: str, values: str,
                 title: str, subtitle: str | None = None, ylabel: str = "",
                 percent: bool = False):
    """Generic grouped bar chart with a 2px surface gap between adjacent bars."""
    wide = df.pivot(index=index, columns=columns, values=values)
    fig, ax = plt.subplots(figsize=S.FIGSIZE)
    n_groups, n_series = wide.shape
    total = 0.82
    w = total / n_series
    x = np.arange(n_groups)
    for i, name in enumerate(wide.columns):
        ax.bar(x + i * w - total / 2 + w / 2, wide[name], width=w * 0.9,
               color=S.colour(i), label=str(name), zorder=3,
               edgecolor=S.SURFACE, linewidth=1.0)
    S.zero_line(ax)
    ax.set_xticks(x, [str(v) for v in wide.index])
    ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    if n_series > 1:
        ax.legend(ncols=min(n_series, 4), loc="best")
    S.title(ax, title, subtitle)
    return fig


def feature_importance(imp: pd.Series, *, top_n: int = 25,
                       title: str = "Which characteristics the model uses",
                       subtitle: str | None = None):
    """Horizontal bars, sorted -- the only ordering a reader can scan."""
    top = imp.sort_values(ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.9, max(3.0, 0.18 * len(top))))
    ax.barh(np.arange(len(top)), top.to_numpy(), color=S.colour(0),
            height=0.72, zorder=3)
    ax.set_yticks(np.arange(len(top)), top.index, fontsize=7.5)
    ax.grid(axis="x", color=S.GRID, lw=0.7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Relative importance")
    S.title(ax, title, subtitle)
    return fig


def sharpe_by_bucket(df: pd.DataFrame, *, models: list[str] | None = None,
                     title: str = "Where the alpha lives",
                     subtitle: str = "Net Sharpe by market-cap quintile (1 = smallest tradable)"):
    """Net Sharpe per size quintile -- the capacity question, in one chart."""
    sub = df if models is None else df[df["model"].isin(models)]
    return grouped_bars(sub, index="size_quintile", columns="model",
                        values="sharpe_net", title=title, subtitle=subtitle,
                        ylabel="Annualised Sharpe, net")


def cost_decomposition(profile: pd.DataFrame, *,
                       title: str = "Where the gross return goes",
                       subtitle: str = "Annualised, out of sample. The bar is what survives."):
    """Gross return split into the two costs and what is left.

    This is the mechanism exhibit.  It shows that the cost-aware objectives do
    not win by picking cheaper anomalies -- their carrying cost is much like
    everyone else's -- but by *rebalancing less*, which shows up as a switching
    bar roughly half the size.
    """
    df = profile.sort_values("net_ann", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.9, max(3.4, 0.34 * len(df) + 0.6)))

    ax.barh(y, df["net_ann"], height=0.62, color=S.colour(0), zorder=3,
            edgecolor=S.SURFACE, linewidth=1.0, label="Net return")
    left = df["net_ann"].clip(lower=0)
    ax.barh(y, df["switching_ann"], left=left, height=0.62, color=S.colour(1),
            zorder=3, edgecolor=S.SURFACE, linewidth=1.0, label="Switching cost")
    ax.barh(y, df["carrying_ann"], left=left + df["switching_ann"], height=0.62,
            color=S.colour(3), zorder=3, edgecolor=S.SURFACE, linewidth=1.0,
            label="Carrying cost")

    S.zero_line(ax, axis="x")
    ax.set_yticks(y, df["model"])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Annualised return")
    ax.grid(axis="x", color=S.GRID, lw=0.7)
    ax.grid(axis="y", visible=False)
    # Below the axes: inside the plot the legend would sit on top of the bars,
    # and there is no empty quadrant in a stacked horizontal bar chart.
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    S.title(ax, title, subtitle)
    return fig


def turnover_bars(profile: pd.DataFrame, *,
                  title: str = "The cost-aware objectives rebalance half as often",
                  subtitle: str = "Mean monthly allocation turnover, out of sample"):
    """The single number that explains the result."""
    df = profile.sort_values("turnover").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.9, max(2.8, 0.32 * len(df))))
    y = np.arange(len(df))
    colours = [S.colour(0) if m.startswith("xs_") and m.split("_")[-1] in ("sharpe", "utility")
               else (S.INK_2 if m == "equal_weight" else S.MUTED) for m in df["model"]]
    ax.barh(y, df["turnover"], height=0.62, color=colours, zorder=3,
            edgecolor=S.SURFACE, linewidth=1.0)
    for i, v in enumerate(df["turnover"]):
        ax.annotate(f"{v:.2f}", (v, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=S.INK_2)
    ax.set_yticks(y, df["model"])
    ax.set_xlabel("Fraction of the book traded per month")
    ax.grid(axis="x", color=S.GRID, lw=0.7)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.10)
    S.title(ax, title, subtitle)
    return fig


def capacity_scatter(wide: pd.DataFrame, *, rho: float | None = None,
                     title: str = "Does the edge survive in tradable stocks?",
                     subtitle: str = "Net Sharpe ratio, same models, two universes"):
    """Each model once: unrestricted universe on x, large-cap-only on y.

    A scatter puts every pair of colours on screen at once, which a categorical
    palette cannot guarantee to separate beyond three slots, so identity is
    carried by a text label on every point and colour marks only the two
    cost-aware objectives and the benchmark.  The 45-degree line is the
    reference: points below it lost ground when the universe was restricted to
    stocks a fund could actually trade at size.
    """
    x, y = wide["all stocks"], wide["large caps only"]
    fig, ax = plt.subplots(figsize=S.FIGSIZE)

    lo = float(min(x.min(), y.min())) - 0.08
    hi = float(max(x.max(), y.max())) + 0.08
    ax.plot([lo, hi], [lo, hi], color=S.RULE, lw=0.9, ls=(0, (4, 3)), zorder=1)

    # Label placement: points that sit close together get their labels pushed
    # apart vertically, otherwise the near-ties (xs_utility/xs_sharpe,
    # pls/ridge) overprint each other and the chart loses the identities it
    # exists to show.
    pts = sorted(wide.to_dict("records"), key=lambda r: r["large caps only"])
    span = (hi - lo) or 1.0
    label_y, last = [], -1e9
    for row in pts:
        y_lab = max(row["large caps only"], last + 0.045 * span)
        label_y.append(y_lab)
        last = y_lab

    for row, y_lab in zip(pts, label_y, strict=False):
        m = row["model"]
        if m in ("xs_utility", "xs_sharpe"):
            c, ink = S.colour(0), S.INK
        elif m == "equal_weight":
            c, ink = S.INK_2, S.INK
        else:
            c, ink = S.MUTED, S.INK_2
        ax.scatter(row["all stocks"], row["large caps only"], s=48, color=c,
                   zorder=3, edgecolor=S.SURFACE, linewidth=1.4)
        ax.annotate(m, (row["all stocks"], y_lab), xytext=(7, 0),
                    textcoords="offset points", fontsize=7.8, color=ink,
                    va="center", annotation_clip=False)

    S.zero_line(ax)
    S.zero_line(ax, axis="x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Net Sharpe, all stocks")
    ax.set_ylabel("Net Sharpe, large caps only")
    if rho is not None:
        ax.annotate(f"rank correlation {rho:+.2f}", xy=(0.98, 0.04),
                    xycoords="axes fraction", ha="right", fontsize=8, color=S.INK_2)
    S.title(ax, title, subtitle)
    return fig
