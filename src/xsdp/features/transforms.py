"""Cross-sectional feature transforms.

Every transform here is computed **within a single month's cross-section**.
That is not a stylistic choice: a transform fitted on the pooled sample (a
global ``StandardScaler``, say) uses the mean and variance of the *test* period
to normalise the *training* period, which is a subtle but real look-ahead.  It
also happens to be the wrong economics — a book-to-market of 1.2 means
something different in 1975 than in 1999, and only its rank among contemporaries
is comparable through time.

The default pipeline is the one used by Gu, Kelly and Xiu (2020):

    rank within month -> map to [-1, 1] -> fill missing with 0 (the median)

which is scale-free, robust to the fat tails and reporting errors that riddle
accounting data, and leaves a well-behaved input for both trees and networks.
"""

from __future__ import annotations

import logging

import polars as pl

log = logging.getLogger(__name__)


def rank_normalise(
    panel: pl.DataFrame,
    signal_cols: list[str],
    *,
    group: str = "yyyymm",
    fill: float = 0.0,
) -> pl.DataFrame:
    """Map each signal to its within-month rank, rescaled to ``[-1, 1]``.

    Missing values are left out of the ranking (so they do not distort the
    ranks of observed names) and then set to ``fill``, i.e. the cross-sectional
    median.  This is the "impute to the median" convention; the alternative of
    dropping incomplete rows would throw away most of the panel, since almost
    no stock-month has all ~150 signals populated.
    """
    exprs = []
    for c in signal_cols:
        r = pl.col(c).rank("average").over(group)
        n = pl.col(c).is_not_null().sum().over(group)
        scaled = pl.when(n > 1).then(2.0 * (r - 1) / (n - 1) - 1.0).otherwise(0.0)
        exprs.append(scaled.fill_null(fill).cast(pl.Float32).alias(c))
    return panel.with_columns(exprs)


def add_missingness_indicators(
    panel: pl.DataFrame, signal_cols: list[str], *, min_var: float = 0.01
) -> tuple[pl.DataFrame, list[str]]:
    """Add ``<signal>_isna`` flags for signals whose *absence* is informative.

    Whether a firm reports a given item is itself a signal (young firms, firms
    without analyst coverage, firms between filings).  We only keep flags that
    actually vary — a signal that is never missing contributes a constant column.
    """
    added: list[str] = []
    exprs = []
    for c in signal_cols:
        frac = panel[c].is_null().mean()
        if min_var <= frac <= 1 - min_var:
            name = f"{c}_isna"
            exprs.append(pl.col(c).is_null().cast(pl.Float32).alias(name))
            added.append(name)
    if exprs:
        panel = panel.with_columns(exprs)
    log.info("added %d missingness indicators", len(added))
    return panel, added


def winsorise(panel: pl.DataFrame, cols: list[str], *, group: str = "yyyymm",
              lower: float = 0.01, upper: float = 0.99) -> pl.DataFrame:
    """Clip columns at within-month quantiles (used for raw, un-ranked inputs)."""
    exprs = []
    for c in cols:
        lo = pl.col(c).quantile(lower).over(group)
        hi = pl.col(c).quantile(upper).over(group)
        exprs.append(pl.col(c).clip(lo, hi).alias(c))
    return panel.with_columns(exprs)


def demean_target(panel: pl.DataFrame, col: str = "ret_next",
                  out: str = "ret_next_dm", group: str = "yyyymm") -> pl.DataFrame:
    """Subtract the equal-weighted cross-sectional mean from the target.

    The equal-weighted average anomaly return is what the 1/N benchmark earns,
    and no model input predicts it.  Training on the demeaned target therefore
    aligns the loss with what the allocation can actually control -- the tilt
    away from 1/N -- instead of spending capacity on a common component.
    """
    return panel.with_columns(
        (pl.col(col) - pl.col(col).mean().over(group)).alias(out)
    )


def _is_binary(panel: pl.DataFrame, col: str) -> bool:
    vals = panel[col].drop_nulls().unique()
    return vals.len() <= 2


def build_features(cfg, panel: pl.DataFrame, feature_cols: list[str]
                   ) -> tuple[pl.DataFrame, list[str]]:
    """The full feature pipeline, returning the panel and the feature names.

    Continuous features are ranked within the month; indicator features (the
    category dummies, the value-weighting flag) are left alone, because ranking a
    0/1 column turns it into a number whose value depends on how many other
    assets share the category -- interpretable neither to a reader nor to a tree.
    """
    continuous = [c for c in feature_cols if not _is_binary(panel, c)]
    binary = [c for c in feature_cols if c in feature_cols and c not in continuous]

    panel, flags = add_missingness_indicators(panel, continuous)
    panel = rank_normalise(panel, continuous)
    panel = panel.with_columns([pl.col(c).cast(pl.Float32).fill_null(0.0) for c in binary])
    if "ret_next_dm" not in panel.columns:
        panel = demean_target(panel)

    out_cols = sorted(continuous) + sorted(binary) + sorted(flags)
    log.info("feature matrix: %d columns (%d continuous, %d indicator, %d missingness)",
             len(out_cols), len(continuous), len(binary), len(flags))
    return panel, out_cols
