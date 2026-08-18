"""Walk-forward evaluation protocol.

The single most common way to produce a beautiful backtest that loses money is
to select a model using data that would not have existed at the time.  This
module makes that impossible by construction: it emits a sequence of
``(train, validation, test)`` month blocks in which every index in ``test``
strictly post-dates every index in ``train`` and ``validation``.

Three properties are worth spelling out, because reviewers look for them:

*Expanding, not rolling.*  At each refit the training set is everything from the
sample start up to the validation block.  A real fund would not throw away the
1970s, and the extra history is what lets a deep network estimate ~10^5
parameters without collapsing.  A rolling window is available via
``SplitConfig.expanding = False`` and is reported as a robustness check.

*A validation block, not cross-validation.*  Model selection (early stopping,
hyper-parameters) happens on the most recent years before the test block, never
on interleaved folds, because interleaved folds put the future on both sides of
the fit.

*An embargo.*  Accounting signals are built from data with up to a 12-month
publication lag, so an observation in the last months of the training window
shares its underlying fundamentals with the first months of the test window.
``embargo_months`` drops that overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    """One walk-forward refit: three disjoint, ordered blocks of YYYYMM months."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.test[0]}-{self.test[-1]}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Fold(train={self.train[0]}..{self.train[-1]} n={len(self.train)}, "
            f"val={self.val[0]}..{self.val[-1]} n={len(self.val)}, "
            f"test={self.label} n={len(self.test)})"
        )


def _to_month_index(yyyymm: np.ndarray) -> np.ndarray:
    return (yyyymm // 100) * 12 + (yyyymm % 100)


def walk_forward(months: np.ndarray, cfg) -> list[Fold]:
    """Generate the walk-forward folds for a sorted array of YYYYMM months."""
    months = np.sort(np.unique(np.asarray(months)))
    mi = _to_month_index(months)
    sc = cfg.split

    folds: list[Fold] = []
    test_year = sc.first_test_year
    last_year = months[-1] // 100

    while test_year <= last_year:
        test_lo = test_year * 100 + 1
        test_hi = (test_year + sc.test_step_years - 1) * 100 + 12
        test_mask = (months >= test_lo) & (months <= test_hi)
        if not test_mask.any():
            break

        val_hi_mi = _to_month_index(np.array([test_lo]))[0] - 1
        val_lo_mi = val_hi_mi - sc.val_years * 12 + 1
        val_mask = (mi >= val_lo_mi) & (mi <= val_hi_mi)

        train_hi_mi = val_lo_mi - 1 - sc.embargo_months
        train_lo_mi = (
            _to_month_index(np.array([int(sc.train_start[:4]) * 100 + int(sc.train_start[5:7])]))[0]
            if sc.expanding
            else train_hi_mi - sc.train_years * 12 + 1
        )
        train_mask = (mi >= train_lo_mi) & (mi <= train_hi_mi)

        if train_mask.sum() >= 60 and val_mask.sum() >= 24:
            folds.append(
                Fold(months[train_mask].copy(), months[val_mask].copy(), months[test_mask].copy())
            )
        test_year += sc.test_step_years

    if not folds:
        raise ValueError("walk_forward produced no folds; check SplitConfig against the sample")
    return folds


def assert_no_leakage(folds: list[Fold]) -> None:
    """Fail loudly if any fold has a train/val month at or after a test month."""
    for f in folds:
        assert f.train.max() < f.val.min(), f"train overlaps val in {f.label}"
        assert f.val.max() < f.test.min(), f"val overlaps test in {f.label}"
    all_test = np.concatenate([f.test for f in folds])
    assert len(all_test) == len(np.unique(all_test)), "test blocks overlap each other"
