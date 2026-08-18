"""Shared fixtures: a synthetic anomaly panel with a *known* signal.

Testing a research pipeline against real data is circular -- you cannot tell a
bug from a market that simply refuses to cooperate.  So the fixtures here build a
panel in which the answer is known by construction: one feature genuinely
predicts next month's return with a given information coefficient, the rest are
noise, and each synthetic anomaly carries a known trading cost.

A pipeline that cannot recover a planted signal is broken.  A pipeline that
recovers a signal from pure noise is worse than broken, so ``ic=0.0`` is
exercised too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsdp.config import (  # noqa: E402
    Config,
    CostConfig,
    PortfolioConfig,
    SplitConfig,
    TrainConfig,
)


def make_synthetic_panel(
    *,
    n_months: int = 240,
    n_assets: int = 120,
    n_features: int = 12,
    ic: float = 0.05,
    start_yyyymm: int = 200001,
    cost_spread: float = 3.0,
    seed: int = 0,
) -> tuple[pl.DataFrame, list[str]]:
    """A balanced anomaly panel where feature 0 predicts the target.

    Returns are ``ic * feature + sqrt(1 - ic^2) * noise`` scaled to a 4% monthly
    dispersion, which is the order of magnitude of real long-short anomaly
    returns.  Carrying costs are spread log-uniformly across assets by a factor
    of ``cost_spread``, mirroring the ~500x range measured on the real data, so
    that cost-aware and cost-blind objectives actually have something to disagree
    about.
    """
    rng = np.random.default_rng(seed)
    months, y, m = [], start_yyyymm // 100, start_yyyymm % 100
    for _ in range(n_months):
        months.append(y * 100 + m)
        m += 1
        if m == 13:
            y, m = y + 1, 1

    rows = n_months * n_assets
    feats = rng.normal(size=(rows, n_features)).astype(np.float32)
    noise = rng.normal(size=rows).astype(np.float32)
    ret_next = (ic * feats[:, 0] + np.sqrt(max(1 - ic**2, 0.0)) * noise) * 0.04

    # A per-asset carrying cost, constant through time (as the real one nearly is).
    turnover = np.exp(rng.uniform(np.log(0.02), np.log(0.02 * 10**cost_spread), n_assets))
    turnover = np.tile(turnover, n_months).astype(np.float32)

    data = {
        "signalname": np.tile(np.array([f"A{i:03d}" for i in range(n_assets)]), n_months),
        "yyyymm": np.repeat(np.asarray(months, dtype=np.int32), n_assets),
        "ret": ret_next.astype(np.float32),
        "ret_next": ret_next.astype(np.float32),
        "turnover_raw": turnover,
        "carry_cost": (turnover * 25.0 / 1e4).astype(np.float32),
        "switch_cost": np.full(rows, 2 * 25.0 / 1e4, dtype=np.float32),
        "n_available": np.full(rows, n_assets, dtype=np.int32),
    }
    feature_cols = [f"f{i:02d}" for i in range(n_features)]
    for i, c in enumerate(feature_cols):
        data[c] = feats[:, i]

    panel = pl.DataFrame(data).with_columns(
        (pl.col("ret_next") - pl.col("ret_next").mean().over("yyyymm")).alias("ret_next_dm")
    )
    return panel, feature_cols


@pytest.fixture
def synthetic_panel():
    return make_synthetic_panel()


@pytest.fixture
def noise_panel():
    return make_synthetic_panel(ic=0.0, seed=99)


@pytest.fixture
def cfg(tmp_path):
    """A config sized for tests: short windows, few epochs, no vol targeting."""
    c = Config(
        split=SplitConfig(train_start="2000-01-31", first_test_year=2012,
                          val_years=3, embargo_months=6),
        train=TrainConfig(n_ensemble=1, max_epochs=6, patience=3, batch_months=12,
                          device="cpu"),
        portfolio=PortfolioConfig(vol_target_annual=None, max_weight=0.5),
        costs=CostConfig(spread_bps=25.0, switch_bps=25.0, short_fee_bps_annual=0.0),
    )
    object.__setattr__(c.paths, "root", tmp_path)
    return c


@pytest.fixture
def free_cfg(cfg):
    """The same config with every transaction cost switched off."""
    from dataclasses import replace
    return replace(cfg, costs=CostConfig(spread_bps=0.0, switch_bps=0.0,
                                         short_fee_bps_annual=0.0))
