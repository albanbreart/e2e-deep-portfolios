"""Non-neural benchmarks.

A deep model is only interesting relative to what a careful analyst would do
without one.  The ladder below is deliberately the standard one from the
empirical asset-pricing literature, so that the deep results can be read against
numbers a reader already has intuitions about:

``mean``
    Predict the cross-sectional mean, i.e. no view.  Anchors the OOS R^2.
``ols``
    Pooled OLS on all characteristics.  With ~150 correlated predictors and a
    signal-to-noise ratio around 1%, it overfits badly -- which is the point.
``ridge`` / ``enet``
    Shrinkage, with the penalty chosen on the validation block.  Usually the
    strongest linear competitor.
``pcr`` / ``pls``
    Dimension reduction before regression: the answer to "are the 150 signals
    really 150 things, or five things measured 30 ways?"
``gbrt``
    Gradient-boosted trees (LightGBM).  Captures interactions and
    non-linearities without any of the neural machinery, and in much of the
    published evidence it is the model to beat.

Every model exposes the same two methods, so the walk-forward driver treats them
interchangeably with the networks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline

log = logging.getLogger(__name__)


@dataclass
class Fitted:
    """A fitted baseline plus whatever hyper-parameter the validation chose."""

    name: str
    model: object
    chosen: dict

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:                      # the 'mean' benchmark
            return np.zeros(len(X), dtype=np.float32)
        return np.asarray(self.model.predict(X), dtype=np.float32).ravel()


def _validate(candidates: dict, X_tr, y_tr, X_va, y_va, name: str) -> Fitted:
    """Fit each candidate on train, keep the one with the lowest validation MSE."""
    best, best_mse, best_key = None, np.inf, None
    for key, model in candidates.items():
        model.fit(X_tr, y_tr)
        mse = float(np.mean((y_va - np.asarray(model.predict(X_va)).ravel()) ** 2))
        if mse < best_mse:
            best, best_mse, best_key = model, mse, key
    log.info("    %s: chose %s (val MSE %.6g)", name, best_key, best_mse)
    return Fitted(name, best, {"param": best_key, "val_mse": best_mse})


def _clip_grid(values: tuple[int, ...], max_k: int) -> list[int]:
    """Keep the component counts that the design matrix can actually support."""
    kept = sorted({min(v, max_k) for v in values if max_k >= 1})
    return kept or [1]


def fit_baseline(name: str, X_tr, y_tr, X_va, y_va, *, seed: int = 0) -> Fitted:
    """Fit one baseline, selecting its hyper-parameter on the validation block."""
    if name == "mean":
        return Fitted("mean", None, {})

    if name == "ols":
        return Fitted("ols", LinearRegression().fit(X_tr, y_tr), {})

    if name == "ols_h":
        # Huber loss, GKX's robust linear benchmark.
        m = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500).fit(X_tr, y_tr)
        return Fitted("ols_h", m, {})

    if name == "ridge":
        grid = {a: Ridge(alpha=a) for a in (1e1, 1e2, 1e3, 1e4, 1e5)}
        return _validate(grid, X_tr, y_tr, X_va, y_va, name)

    if name == "enet":
        grid = {
            (a, l1): ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000, random_state=seed)
            for a in (1e-5, 1e-4, 1e-3)
            for l1 in (0.1, 0.5, 0.9)
        }
        return _validate(grid, X_tr, y_tr, X_va, y_va, name)

    # A component count can never exceed the rank of the design matrix; the
    # grids below are clipped so the same config works on a 25-signal smoke test
    # and on the full 150-signal panel.
    max_k = min(X_tr.shape[1], X_tr.shape[0] - 1)

    if name == "pcr":
        grid = {
            k: make_pipeline(PCA(n_components=k, random_state=seed), LinearRegression())
            for k in _clip_grid((3, 5, 10, 20, 40), max_k)
        }
        return _validate(grid, X_tr, y_tr, X_va, y_va, name)

    if name == "pls":
        grid = {
            k: PLSRegression(n_components=k, scale=False)
            for k in _clip_grid((1, 2, 3, 5, 10), max_k)
        }
        return _validate(grid, X_tr, y_tr, X_va, y_va, name)

    if name == "gbrt":
        return _fit_gbrt(X_tr, y_tr, X_va, y_va, seed=seed)

    raise KeyError(f"unknown baseline {name!r}")


def _fit_gbrt(X_tr, y_tr, X_va, y_va, *, seed: int = 0) -> Fitted:
    """LightGBM with early stopping on the validation block.

    Shallow trees and heavy row/column subsampling: with a signal-to-noise ratio
    this low, depth buys variance rather than signal.
    """
    import lightgbm as lgb

    params = dict(
        objective="huber", alpha=0.02, learning_rate=0.03, num_leaves=31,
        max_depth=4, min_child_samples=200, subsample=0.7, subsample_freq=1,
        colsample_bytree=0.6, reg_lambda=1.0, n_estimators=2000,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    model = lgb.LGBMRegressor(**params)
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    # LightGBM 4.7 renamed the validation arguments; support both so the repo
    # runs on whatever version the reader has pinned.
    try:
        model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va,
                  eval_metric="l2", callbacks=callbacks)
    except TypeError:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  eval_metric="l2", callbacks=callbacks)
    return Fitted("gbrt", model, {"best_iteration": int(model.best_iteration_ or 0)})


BASELINES = ("mean", "ols", "ols_h", "ridge", "enet", "pcr", "pls", "gbrt")
