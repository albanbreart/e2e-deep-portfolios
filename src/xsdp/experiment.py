"""The walk-forward driver: fits every model on identical folds and collects
out-of-sample predictions.

Everything a reader needs in order to believe a comparison is enforced here
rather than left to the caller's discipline:

*   All models see the **same folds**, produced once by :func:`xsdp.split.walk_forward`
    and checked by :func:`xsdp.split.assert_no_leakage`.
*   All models see the **same features**, in the same order.
*   Hyper-parameters are chosen on the fold's own validation block, so a model
    that needs heavy tuning is not being handed a free look at the test period.
*   Predictions are stamped with the fold that produced them, so a reviewer can
    check that every test month is covered exactly once.

The output of a run is one tidy frame per model — ``yyyymm``, ``permno``,
``score``, plus the columns the backtester needs — which is deliberately the
same interface a live signal would have.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from .models.baselines import fit_baseline
from .models.dataset import PanelTensors
from .models.trainer import predict as nn_predict
from .models.trainer import train_ensemble
from .split import Fold, assert_no_leakage, walk_forward

log = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """One entry in the horse race."""

    name: str                       # label used in tables and filenames
    kind: str                       # "baseline" | "net"
    estimator: str = ""             # baseline name, or architecture name
    loss: str = "mse"               # nets only
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.estimator:
            self.estimator = self.name


#: The headline experiment. The first block is the classical ladder; the second
#: isolates the two things this project claims matter — seeing the cross-section
#: (``xs`` vs ``nn3``) and optimising the traded objective (``sharpe`` vs ``mse``).
DEFAULT_SPECS: list[ModelSpec] = [
    # The classical ladder.
    ModelSpec("ols_h", "baseline"),
    ModelSpec("ridge", "baseline"),
    ModelSpec("enet", "baseline"),
    ModelSpec("pls", "baseline"),
    ModelSpec("gbrt", "baseline"),
    # Does seeing the cross-section help?  nn3 vs xs, same objective.
    ModelSpec("nn3_mse", "net", estimator="nn3", loss="mse"),
    ModelSpec("xs_mse", "net", estimator="xs_small", loss="mse"),
    # Does optimising the ordering help?  xs_ic vs xs_mse.
    ModelSpec("xs_ic", "net", estimator="xs_small", loss="ic"),
    # Does optimising net-of-cost P&L help?  xs_sharpe vs xs_ic.
    ModelSpec("xs_sharpe", "net", estimator="xs_small", loss="sharpe"),
    ModelSpec("xs_utility", "net", estimator="xs_small", loss="utility"),
]


def run_walk_forward(
    cfg,
    panel: pl.DataFrame,
    feature_cols: list[str],
    specs: list[ModelSpec] | None = None,
    *,
    target: str = "ret_next_dm",
    id_col: str = "signalname",
    out_dir: Path | None = None,
    max_folds: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Fit and predict across all folds; return one prediction frame per model."""
    specs = specs or DEFAULT_SPECS
    out_dir = out_dir or cfg.paths.outputs / f"predictions{cfg.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    months = panel["yyyymm"].unique().sort().to_numpy()
    folds = walk_forward(months, cfg)
    assert_no_leakage(folds)
    if max_folds:
        folds = folds[:max_folds]
    log.info("walk-forward: %d folds, test %d..%d",
             len(folds), folds[0].test[0], folds[-1].test[-1])

    tensors = PanelTensors(panel, feature_cols, cfg, target=target, id_col=id_col)
    meta = _meta_frame(panel, id_col)

    collected: dict[str, list[pd.DataFrame]] = {s.name: [] for s in specs}
    timings: list[dict] = []

    for i, fold in enumerate(folds, 1):
        log.info("[fold %d/%d] %s", i, len(folds), fold)
        cache = _fold_arrays(tensors, fold)
        for spec in specs:
            t0 = time.time()
            scores = _run_one(spec, tensors, fold, cfg, cache)
            index = tensors.month_frame(fold.test).to_pandas()
            index["score"] = scores
            index["fold"] = fold.label
            collected[spec.name].append(index)
            timings.append({"fold": fold.label, "model": spec.name,
                            "seconds": round(time.time() - t0, 1)})
            log.info("    %-11s done in %5.1fs", spec.name, time.time() - t0)

    results: dict[str, pd.DataFrame] = {}
    for name, parts in collected.items():
        df = pd.concat(parts, ignore_index=True).merge(meta, on=["yyyymm", id_col], how="left")
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        results[name] = df
        log.info("wrote predictions for %s: %d rows", name, len(df))

    (out_dir / "timings.json").write_text(json.dumps(timings, indent=2))
    (out_dir / "specs.json").write_text(json.dumps([asdict(s) for s in specs], indent=2))
    return results


def _meta_frame(panel: pl.DataFrame, id_col: str) -> pd.DataFrame:
    """The non-feature columns the backtester and the metrics need."""
    cols = [c for c in ("yyyymm", id_col, "ret", "ret_next", "carry_cost",
                        "turnover_raw", "n_available") if c in panel.columns]
    return panel.select(cols).to_pandas()


def _fold_arrays(tensors: PanelTensors, fold: Fold) -> dict:
    """Materialise the pooled train/val arrays once, shared by every baseline."""
    X_tr, y_tr, _ = tensors.flat(fold.train)
    X_va, y_va, _ = tensors.flat(fold.val)
    X_te, _, _ = tensors.flat(fold.test)
    return {"X_tr": X_tr, "y_tr": y_tr, "X_va": X_va, "y_va": y_va, "X_te": X_te}


def _run_one(spec: ModelSpec, tensors: PanelTensors, fold: Fold, cfg, cache: dict) -> np.ndarray:
    if spec.kind == "baseline":
        fitted = fit_baseline(spec.estimator, cache["X_tr"], cache["y_tr"],
                              cache["X_va"], cache["y_va"], seed=cfg.train.seed)
        return fitted.predict(cache["X_te"])

    if spec.kind == "net":
        models, _ = train_ensemble(
            tensors, fold.train, fold.val, cfg,
            arch=spec.estimator, loss_name=spec.loss, **spec.kwargs,
        )
        return nn_predict(models, tensors, fold.test, cfg)

    raise ValueError(f"unknown model kind {spec.kind!r}")


def load_predictions(cfg, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Read back prediction frames written by a previous run."""
    d = cfg.paths.outputs / f"predictions{cfg.suffix}"
    files = sorted(d.glob("*.parquet"))
    out = {}
    for f in files:
        if names is None or f.stem in names:
            out[f.stem] = pd.read_parquet(f)
    return out
