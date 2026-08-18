"""Allocation construction, the two-part cost model, and causal vol targeting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xsdp.portfolio import (
    backtest,
    benchmark,
    cap_and_renormalise,
    drift_weights,
    equal_weights,
    neutral_weights,
    quantile_weights,
    tilt_weights,
)


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
def test_equal_weights_sum_to_one():
    w = equal_weights(37)
    assert abs(w.sum() - 1.0) < 1e-12
    assert np.allclose(w, w[0])


def test_tilt_keeps_the_budget_and_centres_on_equal_weight():
    scores = np.random.default_rng(0).normal(size=200)
    w = tilt_weights(scores, tilt_gross=1.0)
    assert abs(w.sum() - 1.0) < 1e-10          # still fully invested
    assert abs(np.abs(w - 1.0 / 200).sum() - 1.0) < 1e-10   # tilt has the stated L1


def test_zero_tilt_is_exactly_equal_weight():
    scores = np.random.default_rng(1).normal(size=50)
    assert np.allclose(tilt_weights(scores, tilt_gross=0.0), equal_weights(50))


def test_tilt_is_invariant_to_monotone_transforms():
    s = np.random.default_rng(2).normal(size=120)
    assert np.allclose(tilt_weights(s), tilt_weights(np.exp(3 * s)))


def test_tilt_overweights_the_highest_score():
    s = np.arange(60, dtype=float)
    w = tilt_weights(s)
    assert w[-1] == w.max() and w[0] == w.min()
    assert w[-1] > 1 / 60 > w[0]


def test_neutral_weights_are_dollar_neutral():
    w = neutral_weights(np.random.default_rng(3).normal(size=150), gross=2.0)
    assert abs(w.sum()) < 1e-10
    assert abs(np.abs(w).sum() - 2.0) < 1e-10


def test_quantile_weights_hold_only_the_best_fifth():
    w = quantile_weights(np.arange(100, dtype=float), n_quantiles=5)
    assert (w > 0).sum() == 20
    assert abs(w.sum() - 1.0) < 1e-10
    assert w[0] == 0 and w[-1] > 0


def test_cap_binds_and_preserves_the_budget():
    w = cap_and_renormalise(np.array([0.5, 0.3, 0.15, 0.05]), max_weight=0.30, budget=1.0)
    assert w.max() <= 0.30 + 1e-9
    assert abs(w.sum() - 1.0) < 1e-9


def test_cap_left_alone_when_infeasible():
    w = np.array([0.4, 0.35, 0.25])
    out = cap_and_renormalise(w, max_weight=0.30, budget=1.0)
    assert np.allclose(out, w)          # 3 * 0.30 < 1.0 -> cannot be satisfied


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
def test_drift_is_identity_under_flat_returns():
    w = np.array([0.5, 0.5])
    assert np.allclose(drift_weights(w, np.zeros(2)), w)


def test_drift_moves_weight_to_the_winner_and_stays_invested():
    out = drift_weights(np.array([0.5, 0.5]), np.array([0.20, 0.0]))
    assert out[0] > 0.5 > out[1]
    assert abs(out.sum() - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def _predictions(n_months=36, n_assets=60, *, constant_signal: bool,
                 carry: float | np.ndarray = 0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n_assets)
    carry_arr = np.full(n_assets, carry) if np.isscalar(carry) else np.asarray(carry)
    frames = []
    for t in range(n_months):
        frames.append(pd.DataFrame({
            "yyyymm": 200001 + t,
            "signalname": [f"A{i:03d}" for i in range(n_assets)],
            "score": base if constant_signal else rng.normal(size=n_assets),
            "ret_next": rng.normal(scale=0.03, size=n_assets),
            "carry_cost": carry_arr,
            "turnover_raw": carry_arr * 400,
        }))
    return pd.concat(frames, ignore_index=True)


def test_constant_signal_barely_trades_after_entry(cfg):
    res = backtest(_predictions(constant_signal=True), cfg, cost_multiplier=0.0,
                   vol_target=None)
    assert res.monthly["turnover"].iloc[0] > 0.5        # entering from cash
    assert res.monthly["turnover"].iloc[1:].max() < 0.2  # only drift afterwards


def test_random_signal_churns_the_allocation(cfg):
    res = backtest(_predictions(constant_signal=False), cfg, cost_multiplier=0.0,
                   vol_target=None)
    assert res.monthly["turnover"].iloc[1:].mean() > 0.5


def test_switching_cost_only_ever_reduces_net_return(cfg):
    preds = _predictions(constant_signal=False)
    free = backtest(preds, cfg, cost_multiplier=0.0, vol_target=None)
    dear = backtest(preds, cfg, cost_multiplier=4.0, vol_target=None)
    assert (dear.monthly["switching"] > 0).all()
    assert dear.monthly["net"].mean() < free.monthly["net"].mean()
    assert np.allclose(free.monthly["gross"], dear.monthly["gross"])


def test_carrying_cost_is_charged_even_when_nothing_trades(cfg):
    """The distinguishing feature of this cost model, checked directly."""
    preds = _predictions(constant_signal=True, carry=0.002)
    res = backtest(preds, cfg, vol_target=None)
    later = res.monthly.iloc[1:]
    assert later["turnover"].max() < 0.2      # essentially no trading
    assert (later["carrying"] > 0).all()      # yet it still costs money
    assert later["carrying"].mean() > later["switching"].mean()


def test_expensive_assets_cost_more_to_hold(cfg):
    n = 60
    cheap = _predictions(constant_signal=True, n_assets=n, carry=0.0002)
    dear = _predictions(constant_signal=True, n_assets=n, carry=0.0040)
    a = backtest(cheap, cfg, vol_target=None).monthly["carrying"].mean()
    b = backtest(dear, cfg, vol_target=None).monthly["carrying"].mean()
    assert b > 10 * a


def test_pnl_is_dated_to_the_month_it_is_earned(cfg):
    res = backtest(_predictions(n_months=3, constant_signal=True), cfg,
                   cost_multiplier=0.0, vol_target=None)
    assert list(res.monthly.index) == [200002, 200003, 200004]
    assert list(res.monthly["formation_yyyymm"]) == [200001, 200002, 200003]


def test_benchmark_is_the_equal_weighted_allocation(cfg):
    preds = _predictions(constant_signal=False)
    res = benchmark(preds, cfg, vol_target=None)
    w = res.weights[200001]
    assert np.allclose(w.to_numpy(), 1.0 / len(w))


def test_benchmark_turns_over_far_less_than_any_active_model(cfg):
    preds = _predictions(constant_signal=False)
    active = backtest(preds, cfg, vol_target=None).monthly["turnover"].iloc[1:].mean()
    passive = benchmark(preds, cfg, vol_target=None).monthly["turnover"].iloc[1:].mean()
    assert passive < active


def test_vol_targeting_uses_no_contemporaneous_information(cfg):
    """Changing one month's return must not change any earlier leverage."""
    preds = _predictions(n_months=80, constant_signal=False)
    a = backtest(preds, cfg, vol_target=0.10)

    bumped = preds.copy()
    bumped.loc[bumped["yyyymm"] == bumped["yyyymm"].max(), "ret_next"] *= 20.0
    b = backtest(bumped, cfg, vol_target=0.10)

    common = a.monthly.index.intersection(b.monthly.index)[:-1]
    assert np.allclose(a.monthly.loc[common, "leverage"], b.monthly.loc[common, "leverage"])
