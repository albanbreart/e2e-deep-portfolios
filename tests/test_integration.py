"""End-to-end runs on synthetic data, where the right answer is known.

These are the tests that would catch every serious bug this project could have.
Three matter most:

``test_pipeline_recovers_a_planted_signal``
    If a feature really does predict returns, the whole chain -- features, folds,
    model, allocation, costs -- has to turn that into a positive out-of-sample
    Sharpe ratio.  A pipeline that fails here is broken.

``test_pipeline_finds_nothing_in_pure_noise``
    And if nothing predicts returns, the same chain must produce a t-statistic
    indistinguishable from zero.  A pipeline that fails *this* is worse than
    broken: it manufactures alpha, and every result it ever produces is
    worthless.

``test_cost_aware_model_trades_less``
    The mechanism the project claims.  Same architecture, same data, same folds;
    only the objective differs.
"""

from __future__ import annotations

from conftest import make_synthetic_panel
from xsdp.analysis import cost_sensitivity, economic_table, statistical_table
from xsdp.experiment import ModelSpec, run_walk_forward
from xsdp.features.transforms import build_features
from xsdp.metrics import newey_west_tstat
from xsdp.portfolio import backtest, benchmark

FAST_SPECS = [
    ModelSpec("ridge", "baseline"),
    ModelSpec("xs_ic", "net", estimator="xs_tiny", loss="ic"),
    ModelSpec("xs_sharpe", "net", estimator="xs_tiny", loss="sharpe"),
]


def _run(panel, feature_cols, cfg, specs=None):
    panel, feats = build_features(cfg, panel, feature_cols)
    return run_walk_forward(cfg, panel, feats, specs or FAST_SPECS,
                            out_dir=cfg.paths.root / "preds")


def test_pipeline_recovers_a_planted_signal(cfg):
    panel, feats = make_synthetic_panel(n_months=180, n_assets=100, ic=0.10)
    preds = _run(panel, feats, cfg)
    for name, df in preds.items():
        res = backtest(df, cfg, cost_multiplier=0.0, vol_target=None)
        active = res.monthly["net"] - benchmark(df, cfg, cost_multiplier=0.0,
                                                vol_target=None).monthly["net"]
        t = newey_west_tstat(active.to_numpy())
        assert t > 1.5, f"{name} failed to recover an ic=0.10 signal (t={t:.2f})"


def test_pipeline_finds_nothing_in_pure_noise(cfg):
    panel, feats = make_synthetic_panel(n_months=180, n_assets=100, ic=0.0, seed=7)
    preds = _run(panel, feats, cfg)
    for name, df in preds.items():
        res = backtest(df, cfg, cost_multiplier=0.0, vol_target=None)
        active = res.monthly["net"] - benchmark(df, cfg, cost_multiplier=0.0,
                                                vol_target=None).monthly["net"]
        t = newey_west_tstat(active.to_numpy())
        assert abs(t) < 3.0, f"{name} manufactured alpha from noise (t={t:.2f})"


def test_cost_aware_model_trades_less_than_the_cost_blind_one(cfg):
    """Same architecture, same folds, different objective."""
    panel, feats = make_synthetic_panel(n_months=180, n_assets=100, ic=0.06, seed=3)
    preds = _run(panel, feats, cfg, specs=[
        ModelSpec("blind", "net", estimator="xs_tiny", loss="ic"),
        ModelSpec("aware", "net", estimator="xs_tiny", loss="sharpe"),
    ])
    turn = {k: backtest(v, cfg).monthly["turnover"].mean() for k, v in preds.items()}
    assert turn["aware"] < turn["blind"], turn


def test_every_test_month_is_predicted_exactly_once(cfg):
    panel, feats = make_synthetic_panel(n_months=180, n_assets=60)
    preds = _run(panel, feats, cfg, specs=[ModelSpec("ridge", "baseline")])
    counts = preds["ridge"].groupby(["yyyymm", "signalname"]).size()
    assert counts.max() == 1


def test_predictions_never_cover_the_training_period(cfg):
    panel, feats = make_synthetic_panel(n_months=180, n_assets=60)
    preds = _run(panel, feats, cfg, specs=[ModelSpec("ridge", "baseline")])
    assert preds["ridge"]["yyyymm"].min() >= cfg.split.first_test_year * 100 + 1


def test_analysis_tables_are_well_formed(cfg):
    panel, feats = make_synthetic_panel(n_months=180, n_assets=80, ic=0.06)
    preds = _run(panel, feats, cfg)

    stat = statistical_table(preds)
    assert set(stat["model"]) == set(preds)
    assert stat["ic_mean"].notna().all()

    econ = economic_table(preds, cfg)
    assert "equal_weight" in set(econ["model"])       # the benchmark is always a row
    assert econ["sharpe_net"].notna().all()
    assert (econ["turnover"] >= 0).all()

    cs = cost_sensitivity(preds, cfg, multipliers=(0.0, 4.0))
    for model in cs["model"].unique():
        block = cs[cs["model"] == model].set_index("cost_mult")
        assert block.loc[4.0, "ret_net"] <= block.loc[0.0, "ret_net"] + 1e-12
