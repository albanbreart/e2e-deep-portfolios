"""Assemble the modelling panel: one row per (anomaly, month), features + target.

The asset universe is the set of **published cross-sectional anomalies**, and the
allocation problem is how much capital to put on each of them, month by month.

Three construction decisions carry most of the weight, and each is a place where
a careless version of this project would quietly cheat:

1.  **The universe is point-in-time by publication date.**  An anomaly enters in
    the January *after* the year its paper appeared.  An investor in 1980 could
    not allocate to an anomaly documented in 2005, and a backtest that lets them
    is measuring the profitability of a time machine.  This is also what makes
    "years since publication" an honest feature rather than a leak: by
    construction it is never negative.

2.  **Only variables knowable at time t become features.**  Chen & Zimmerman's
    replication-quality grades and the 2025 citation counts are excluded, even
    though they are sitting right there in the metadata file and are strongly
    related to future returns -- precisely because that relationship is
    hindsight.  What survives is the strategy's own definition (holding period,
    quantile, weighting), the effect size reported in the original paper, and
    everything derivable from returns realised before t.

3.  **Costs are anomaly-specific and measured, not assumed.**  Each anomaly's
    intrinsic monthly turnover is reconstructed from the firm-level signals
    (:mod:`xsdp.data.anomaly_costs`) on an expanding window, so holding a
    monthly-reset signal like ``MaxRet`` is correctly nine times dearer than
    holding ``BM``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

ID_COLS = ("signalname", "yyyymm")
#: Columns the backtester and metrics need, which are never model inputs.
META_COLS = (
    "ret", "ret_next", "ret_next_dm", "carry_cost", "switch_cost",
    "turnover_raw", "n_available",
)

#: Metadata that is genuinely known at the time an anomaly enters the universe.
#: Deliberately excludes citation counts and Chen-Zimmerman's replication grade,
#: both of which are assessments made in the 2020s.
_DOC_FIELDS = {
    "Acronym": "signalname",
    "Year": "pub_year",
    "SampleEndYear": "sample_end_year",
    "SampleStartYear": "sample_start_year",
    "T-Stat": "orig_tstat",
    "Return": "orig_ret",
    "Portfolio Period": "holding_months",
    "LS Quantile": "ls_quantile",
    "Stock Weight": "stock_weight",
    "Cat.Economic": "cat_economic",
    "Cat.Data": "cat_data",
    "Cat.Form": "cat_form",
}


def _month_index(col: str = "yyyymm") -> pl.Expr:
    return (pl.col(col) // 100) * 12 + (pl.col(col) % 100)


def build_anomaly_panel(cfg, *, force: bool = False) -> Path:
    """Build ``data/processed/anomaly_panel.parquet`` and return its path."""
    out = cfg.paths.processed / f"anomaly_panel{cfg.suffix}.parquet"
    if out.exists() and not force:
        log.info("anomaly panel cache hit: %s", out)
        return out

    rets = _load_long_short_returns(cfg, cfg.universe.portfolio_set)
    doc = _load_metadata(cfg)
    rets = _restrict_to_post_publication(rets, doc, cfg)
    panel = _add_return_features(rets)
    panel = _add_cross_sectional_features(panel)
    panel = _attach_metadata(panel, doc)
    panel = _attach_costs(panel, cfg)
    panel = _add_target(panel, cfg)

    out.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(out, compression="zstd")
    log.info("wrote %s: %d rows, %d anomalies, %d months (%d-%d)",
             out, panel.height, panel["signalname"].n_unique(),
             panel["yyyymm"].n_unique(), panel["yyyymm"].min(), panel["yyyymm"].max())
    return out


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _load_long_short_returns(cfg, which: str = "op") -> pl.DataFrame:
    """The long-short return of every predictor, one row per (anomaly, month)."""
    from .openap import PORTFOLIO_SETS

    path = cfg.paths.external / f"{PORTFOLIO_SETS[which]}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `python scripts/00_pull_data.py`.")

    df = pl.read_parquet(path)
    df = df.rename({c: c.lower() for c in df.columns if c.lower() in ("port", "date", "ret")})
    if "port" in df.columns:
        df = df.filter(pl.col("port").cast(pl.Utf8).str.to_uppercase() == "LS")

    date = pl.col("date").cast(pl.Utf8).str.replace_all("-", "")
    df = df.with_columns(date.str.slice(0, 6).cast(pl.Int32).alias("yyyymm"))

    # Chen & Zimmerman report returns in PERCENT. Everything downstream --
    # Sharpe ratios, cost rates, the loss function -- assumes decimals, so this
    # single division is the difference between a plausible table and one with
    # Sharpe ratios in the hundreds.
    breadth = [c for c in ("Nlong", "Nshort") if c in df.columns]
    out = df.select(
        ["signalname", "yyyymm", (pl.col("ret") / 100.0).alias("ret"),
         *[pl.col(c).cast(pl.Float64) for c in breadth]]
    ).drop_nulls("ret")
    if breadth:
        out = out.with_columns(
            (pl.col("Nlong") + pl.col("Nshort")).log1p().alias("log_breadth")
        ).drop(breadth)
    return out.unique(subset=["signalname", "yyyymm"]).sort(["signalname", "yyyymm"])


def _load_metadata(cfg) -> pl.DataFrame:
    path = cfg.paths.external / "cz_signal_doc.parquet"
    doc = pl.read_parquet(path)
    have = {k: v for k, v in _DOC_FIELDS.items() if k in doc.columns}
    doc = doc.select([pl.col(k).alias(v) for k, v in have.items()])
    numeric = ["pub_year", "sample_end_year", "sample_start_year", "orig_tstat",
               "orig_ret", "holding_months", "ls_quantile"]
    return doc.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in numeric if c in doc.columns]
    ).unique(subset=["signalname"])


def _restrict_to_post_publication(rets: pl.DataFrame, doc: pl.DataFrame, cfg) -> pl.DataFrame:
    """Drop every month before an anomaly was publicly known.

    An anomaly becomes investable in the January following its publication year.
    Anomalies with no recorded publication year are dropped rather than guessed.
    """
    before, n_sig = rets.height, rets["signalname"].n_unique()
    j = rets.join(doc.select(["signalname", "pub_year"]), on="signalname", how="inner")
    j = j.drop_nulls("pub_year").filter(
        pl.col("yyyymm") >= ((pl.col("pub_year") + 1) * 100 + 1).cast(pl.Int32)
    )
    u = cfg.universe
    lo = int(u.start[:4]) * 100 + int(u.start[5:7])
    hi = int(u.end[:4]) * 100 + int(u.end[5:7])
    j = j.filter((pl.col("yyyymm") >= lo) & (pl.col("yyyymm") <= hi))
    log.info("post-publication screen: %d -> %d anomaly-months (%d -> %d anomalies)",
             before, j.height, n_sig, j["signalname"].n_unique())
    return j.drop("pub_year").sort(["signalname", "yyyymm"])


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def _add_return_features(df: pl.DataFrame) -> pl.DataFrame:
    """Trailing performance, risk and drawdown, all using data through month t.

    Every window ends at ``t`` inclusive and the target is ``t+1``, so none of
    these can see the return they are asked to predict.  ``r_12_2`` skips the
    most recent month, the standard momentum construction that avoids
    contaminating a medium-term signal with short-term reversal.
    """
    g = "signalname"
    logret = (1.0 + pl.col("ret")).log()

    out = df.with_columns(logret.alias("_lr"))
    for w in (3, 6, 12, 24, 36, 60):
        out = out.with_columns(
            (pl.col("_lr").rolling_sum(w, min_samples=w).over(g).exp() - 1).alias(f"r_{w}")
        )
    out = out.with_columns(
        pl.col("ret").alias("r_1"),
        ((pl.col("_lr").rolling_sum(12, min_samples=12).over(g)
          - pl.col("_lr")).exp() - 1).alias("r_12_2"),
    )
    for w in (12, 36, 60):
        out = out.with_columns(
            pl.col("ret").rolling_std(w, min_samples=w // 2).over(g).alias(f"vol_{w}")
        )
    for w in (36, 60):
        out = out.with_columns(
            (pl.col("ret").rolling_mean(w, min_samples=w // 2).over(g)
             / pl.col(f"vol_{w}").clip(lower_bound=1e-6) * np.sqrt(12)).alias(f"sharpe_{w}")
        )
    out = out.with_columns(
        pl.col("ret").rolling_skew(36).over(g).alias("skew_36"),
        (pl.col("_lr").cum_sum().over(g)
         - pl.col("_lr").cum_sum().over(g).rolling_max(60, min_samples=12).over(g)
         ).alias("dd_60"),
    )
    return out.drop("_lr")


def _add_cross_sectional_features(df: pl.DataFrame) -> pl.DataFrame:
    """How each anomaly relates to the average anomaly, and how crowded the field is.

    ``beta_idx`` and ``corr_idx`` are computed against the equal-weighted index of
    every anomaly *available that month*, which is itself the benchmark the model
    has to beat.  A high-beta anomaly adds little diversification; a low-beta one
    that still earns a premium is the one an allocator wants.
    """
    idx = df.group_by("yyyymm").agg(
        pl.col("ret").mean().alias("idx_ret"),
        pl.len().alias("n_available"),
    )
    out = df.join(idx, on="yyyymm", how="left").sort(["signalname", "yyyymm"])

    g = "signalname"
    w = 60
    out = out.with_columns(
        pl.rolling_cov("ret", "idx_ret", window_size=w, min_periods=24).over(g).alias("_cov"),
        pl.col("idx_ret").rolling_std(w, min_samples=24).over(g).alias("_sd_idx"),
        pl.col("ret").rolling_std(w, min_samples=24).over(g).alias("_sd_own"),
    )
    return out.with_columns(
        (pl.col("_cov") / (pl.col("_sd_idx") ** 2).clip(lower_bound=1e-12)).alias("beta_idx"),
        (pl.col("_cov") / (pl.col("_sd_idx") * pl.col("_sd_own")).clip(lower_bound=1e-12)
         ).alias("corr_idx"),
        (pl.col("idx_ret").rolling_sum(12, min_samples=12).over(g)).alias("idx_r_12"),
    ).drop(["_cov", "_sd_idx", "_sd_own"])


def _attach_metadata(df: pl.DataFrame, doc: pl.DataFrame) -> pl.DataFrame:
    """Publication age, the original paper's effect size, and the strategy's shape."""
    out = df.join(doc, on="signalname", how="left")
    year = pl.col("yyyymm") // 100
    out = out.with_columns(
        (year - pl.col("pub_year")).cast(pl.Float64).alias("yrs_since_pub"),
        (year - pl.col("sample_end_year")).cast(pl.Float64).alias("yrs_since_sample_end"),
        (pl.col("sample_end_year") - pl.col("sample_start_year")).cast(pl.Float64)
        .alias("orig_sample_years"),
        (pl.col("stock_weight").cast(pl.Utf8).str.to_uppercase() == "VW")
        .cast(pl.Float64).alias("is_value_weighted"),
    )
    for col in ("cat_economic", "cat_data", "cat_form"):
        if col not in out.columns:
            continue
        levels = [v for v in out[col].drop_nulls().unique().to_list() if isinstance(v, str)]
        # Keep the common categories; rare ones fold into an implicit "other".
        counts = out.group_by(col).len().sort("len", descending=True)
        keep = [v for v in counts[col].to_list()[:10] if v in levels]
        out = out.with_columns([
            (pl.col(col) == lv).cast(pl.Float64).alias(f"{col}_{_slug(lv)}") for lv in keep
        ])
    # Raw calendar levels are dropped: with only ~200 assets, `pub_year` and the
    # original sample dates come close to naming each anomaly, and a network
    # given them can memorise which specific anomalies paid off rather than
    # learning what *kind* of anomaly pays off. The relative versions
    # (`yrs_since_pub`, `yrs_since_sample_end`, `orig_sample_years`) carry the
    # economics without the identifier.
    drop = ("cat_economic", "cat_data", "cat_form", "stock_weight",
            "pub_year", "sample_start_year", "sample_end_year")
    return out.drop([c for c in drop if c in out.columns])


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(s)).strip("_").lower()


def _attach_costs(df: pl.DataFrame, cfg) -> pl.DataFrame:
    """Per-anomaly carrying and switching costs from the measured turnover series."""
    from .anomaly_costs import causal_turnover

    path = cfg.paths.processed / "turnover_series.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `python scripts/00_pull_data.py`.")

    series = causal_turnover(pl.read_parquet(path))
    out = df.join(
        series.select(["signalname", "yyyymm", "turnover_causal"]),
        on=["signalname", "yyyymm"], how="left",
    )
    # Forward-fill within an anomaly, then fall back to the cross-sectional
    # median for the handful of anomalies whose signal is not in the wide file.
    out = out.sort(["signalname", "yyyymm"]).with_columns(
        pl.col("turnover_causal").forward_fill().over("signalname")
    )
    med = out["turnover_causal"].median()
    out = out.with_columns(pl.col("turnover_causal").fill_null(med))

    # Anomalies held longer than a month roll only part of the book each month.
    out = out.with_columns(
        (pl.col("turnover_causal")
         / pl.col("holding_months").fill_null(1.0).clip(lower_bound=1.0)).alias("turnover_monthly")
    )
    cc = cfg.costs
    # Two copies on purpose. `turnover_raw` is kept out of the feature set and
    # stays in physical units, because the cost model and every table that
    # reports "how much does this anomaly churn" need the actual number.
    # `turnover_monthly` goes on to be rank-normalised with the other features,
    # which is what the models should see.
    return out.with_columns(
        pl.col("turnover_monthly").alias("turnover_raw"),
        (pl.col("turnover_monthly") * cc.spread_bps / 1e4).alias("carry_cost"),
        pl.lit(2.0 * cc.switch_bps / 1e4).alias("switch_cost"),
    ).drop("turnover_causal")


def _add_target(df: pl.DataFrame, cfg) -> pl.DataFrame:
    """Next month's return, joined by explicit month index so gaps cannot shift it."""
    nxt = df.select(
        pl.col("signalname"),
        (_month_index() - 1).alias("_mi"),
        pl.col("ret").alias("ret_next"),
    )
    out = (
        df.with_columns(_month_index().alias("_mi"))
        .join(nxt, on=["signalname", "_mi"], how="inner")
        .drop("_mi")
    )
    out = out.with_columns(
        (pl.col("ret_next") - pl.col("ret_next").mean().over("yyyymm")).alias("ret_next_dm")
    )
    counts = out.group_by("yyyymm").len().filter(
        pl.col("len") >= cfg.universe.min_assets_per_month
    )
    out = out.join(counts.select("yyyymm"), on="yyyymm", how="inner")

    ordered = [c for c in ID_COLS] + [c for c in META_COLS if c in out.columns]
    features = sorted(c for c in out.columns if c not in ordered)
    return out.select(ordered + features)


def feature_columns(panel: pl.DataFrame) -> list[str]:
    """The model-input columns of a built panel, in a stable order."""
    return [c for c in panel.columns if c not in ID_COLS + META_COLS]
