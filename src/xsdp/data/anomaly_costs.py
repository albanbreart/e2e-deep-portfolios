"""Deriving each anomaly's trading cost from the firm-level signals.

The allocation problem this project solves is: how much capital to put on each of
~200 published anomalies, month by month.  Getting the transaction costs right is
what makes the problem interesting, and anomaly-level costs have two distinct
components that behave very differently:

**Switching cost.**  Moving capital from one anomaly to another means unwinding
one stock book and building another.  It is paid on ``|Δw|`` -- the change in the
allocation weight -- and it is what a turnover penalty on the allocation
punishes.

**Carrying cost.**  This is the one that is easy to miss and that dominates in
practice.  Even holding an allocation *perfectly constant*, the anomaly itself
rebalances every month: short-term reversal completely rewrites its book monthly,
while book-to-market barely moves.  So an anomaly has an intrinsic cost of being
*held*, proportional to its own internal turnover.

That second number is not published, but it is recoverable: Chen & Zimmerman
distribute the firm-level signal used to build each portfolio, so we can
reconstruct the decile book month by month and measure how much of it turns over.
This module does exactly that, once, and caches the result.

The estimate is deliberately conservative in one direction and stated plainly:
we compute ``Σ|w_t - w_{t-1}|`` without drifting last month's weights by realised
returns, because stock returns are precisely what we do not have.  Drift would
lower every anomaly's measured turnover slightly.  Since the number is used to
*rank and price* anomalies relative to each other, and the omission applies
uniformly, the comparison it supports is unaffected.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

ID_COLS = ("permno", "yyyymm")


def _turnover_series_for_signal(df: pl.DataFrame, col: str, *, n_quantiles: int = 10
                                ) -> pl.DataFrame:
    """Month-by-month turnover of a long-short decile book built on one signal.

    Returns a frame of ``(yyyymm, turnover, rank_autocorr)``, one row per month
    in which a full rebalance could be measured.  Producing the *series* rather
    than a single average is what makes the cost estimate usable as a feature:
    the model at time ``t`` gets an expanding-window average of everything
    observed strictly before ``t``, so nothing about the future leaks in.

    The book is equal-weighted within each extreme decile and scaled to gross
    exposure 2 (one dollar long, one dollar short), matching how Chen &
    Zimmerman build the portfolios we are going to trade.
    """
    d = df.drop_nulls(col)
    if d.height == 0:
        return pl.DataFrame(schema={"yyyymm": pl.Int32, "turnover": pl.Float64,
                                    "rank_autocorr": pl.Float64})

    d = d.with_columns(
        (pl.col(col).rank("ordinal").over("yyyymm") /
         pl.len().over("yyyymm")).alias("_u"),
        pl.len().over("yyyymm").alias("_n"),
    ).filter(pl.col("_n") >= 2 * n_quantiles)
    if d.height == 0:
        return pl.DataFrame(schema={"yyyymm": pl.Int32, "turnover": pl.Float64,
                                    "rank_autocorr": pl.Float64})

    q = 1.0 / n_quantiles
    d = d.with_columns(
        pl.when(pl.col("_u") > 1 - q).then(1.0)
        .when(pl.col("_u") <= q).then(-1.0)
        .otherwise(0.0).alias("_side")
    ).with_columns(
        (pl.col("_side") /
         pl.col("_side").abs().sum().over("yyyymm").clip(lower_bound=1.0) * 2.0).alias("w")
    ).select(["permno", "yyyymm", "w", "_u"])

    months, out = d["yyyymm"].unique().sort().to_list(), []
    prev_w = prev_u = prev_month = None
    for m in months:
        cur = d.filter(pl.col("yyyymm") == m)
        permnos = cur["permno"].to_list()
        w = dict(zip(permnos, cur["w"].to_list(), strict=False))
        u = dict(zip(permnos, cur["_u"].to_list(), strict=False))
        if prev_w is not None and _is_next_month(prev_month, m):
            keys = set(w) | set(prev_w)
            turn = sum(abs(w.get(k, 0.0) - prev_w.get(k, 0.0)) for k in keys)
            common = list(set(u) & set(prev_u))
            ac = float("nan")
            if len(common) > 30:
                a = np.array([u[k] for k in common])
                b = np.array([prev_u[k] for k in common])
                if a.std() > 0 and b.std() > 0:
                    ac = float(np.corrcoef(a, b)[0, 1])
            out.append({"yyyymm": int(m), "turnover": float(turn), "rank_autocorr": ac})
        prev_w, prev_u, prev_month = w, u, m

    if not out:
        return pl.DataFrame(schema={"yyyymm": pl.Int32, "turnover": pl.Float64,
                                    "rank_autocorr": pl.Float64})
    return pl.DataFrame(out).with_columns(pl.col("yyyymm").cast(pl.Int32))


def _is_next_month(a: int, b: int) -> bool:
    return (b // 100) * 12 + (b % 100) - ((a // 100) * 12 + (a % 100)) == 1


def compute_turnover_series(
    signals_path: Path,
    out_path: Path,
    *,
    start_yyyymm: int = 192601,
    end_yyyymm: int = 202312,
    chunk: int = 12,
    force: bool = False,
) -> pl.DataFrame:
    """Measure every signal's decile-book turnover, month by month, and cache it.

    Signals are processed in small column groups because the source file is
    ~1.4 GB and 209 columns wide; Parquet is columnar, so reading a dozen
    columns at a time is cheap and keeps peak memory in the hundreds of
    megabytes.
    """
    if out_path.exists() and not force:
        log.info("turnover-series cache hit: %s", out_path)
        return pl.read_parquet(out_path)

    schema = pl.scan_parquet(signals_path).collect_schema().names()
    signal_cols = [c for c in schema if c not in ID_COLS]
    log.info("measuring monthly decile turnover for %d signals", len(signal_cols))

    frames = []
    for i in range(0, len(signal_cols), chunk):
        block = signal_cols[i:i + chunk]
        df = (
            pl.scan_parquet(signals_path)
            .select([*ID_COLS, *block])
            .filter((pl.col("yyyymm") >= start_yyyymm) & (pl.col("yyyymm") <= end_yyyymm))
            .collect()
        )
        for col in block:
            series = _turnover_series_for_signal(df.select([*ID_COLS, col]), col)
            if series.height:
                frames.append(series.with_columns(pl.lit(col).alias("signalname")))
        log.info("  %d/%d signals", min(i + chunk, len(signal_cols)), len(signal_cols))

    out = pl.concat(frames).select(["signalname", "yyyymm", "turnover", "rank_autocorr"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    log.info("wrote %s: %d signal-months, %d signals",
             out_path, out.height, out["signalname"].n_unique())
    return out


def causal_turnover(series: pl.DataFrame, *, min_months: int = 24) -> pl.DataFrame:
    """Expanding-window mean turnover, using only months strictly before ``t``.

    This is the form the model and the cost model are allowed to see.  A
    full-sample average would be a look-ahead: in 1990 nobody had measured how
    much a signal published in 2005 churns.
    """
    s = series.sort(["signalname", "yyyymm"])
    return s.with_columns(
        pl.col("turnover").shift(1).cum_sum().over("signalname").alias("_cs"),
        pl.col("turnover").shift(1).cum_count().over("signalname").alias("_cn"),
    ).with_columns(
        pl.when(pl.col("_cn") >= min_months)
        .then(pl.col("_cs") / pl.col("_cn"))
        .otherwise(None)
        .alias("turnover_causal")
    ).drop(["_cs", "_cn"])


def attach_holding_period(turnover: pl.DataFrame, signal_doc: pl.DataFrame) -> pl.DataFrame:
    """Adjust raw turnover for anomalies that hold positions longer than a month.

    Several anomalies are documented with a 6- or 12-month holding period.  Those
    are implemented as overlapping portfolios: a twelfth of the book is rolled
    each month, not the whole thing.  Measuring the signal's month-on-month decile
    churn therefore over-states what the traded portfolio actually pays, by
    roughly the holding period, so we divide it out.
    """
    doc = signal_doc.select(
        pl.col("Acronym").alias("signalname"),
        pl.col("Portfolio Period").cast(pl.Float64).alias("holding_months"),
        pl.col("Stock Weight").alias("stock_weight"),
        pl.col("LS Quantile").cast(pl.Float64).alias("ls_quantile"),
    )
    out = turnover.join(doc, on="signalname", how="left")
    return out.with_columns(
        (pl.col("turnover_causal") /
         pl.col("holding_months").fill_null(1.0).clip(lower_bound=1.0)).alias("turnover_monthly")
    )


def cost_rate(turnover: pl.DataFrame, *, spread_bps: float = 25.0,
              switch_bps: float = 25.0) -> pl.DataFrame:
    """Translate turnover into the two cost rates the loss function consumes.

    ``carry_bps``
        Cost of *holding* one unit of the anomaly for one month, paid on the
        allocation weight.  Equal to the anomaly's own monthly turnover times the
        per-stock one-way cost.
    ``switch_bps``
        Cost of *changing* the allocation weight by one unit, paid on ``|Δw|``.
        Building or unwinding a gross-2 long-short book costs roughly twice the
        one-way per-stock cost.
    """
    return turnover.with_columns(
        (pl.col("turnover_monthly") * spread_bps).alias("carry_bps"),
        pl.lit(2.0 * switch_bps).alias("switch_bps"),
    )
