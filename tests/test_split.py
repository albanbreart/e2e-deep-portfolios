"""The walk-forward protocol must be incapable of leaking the future."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_synthetic_panel
from xsdp.split import assert_no_leakage, walk_forward


def test_folds_are_strictly_ordered(cfg):
    panel, _ = make_synthetic_panel(n_months=240)
    folds = walk_forward(panel["yyyymm"].unique().sort().to_numpy(), cfg)
    assert folds
    assert_no_leakage(folds)
    for f in folds:
        assert f.train.max() < f.val.min()
        assert f.val.max() < f.test.min()


def test_embargo_gap_is_respected(cfg):
    panel, _ = make_synthetic_panel(n_months=240)
    folds = walk_forward(panel["yyyymm"].unique().sort().to_numpy(), cfg)
    for f in folds:
        def mi(x):
            return (x // 100) * 12 + (x % 100)
        gap = mi(f.val.min()) - mi(f.train.max()) - 1
        assert gap >= cfg.split.embargo_months


def test_test_blocks_tile_without_overlap(cfg):
    panel, _ = make_synthetic_panel(n_months=240)
    folds = walk_forward(panel["yyyymm"].unique().sort().to_numpy(), cfg)
    all_test = np.concatenate([f.test for f in folds])
    assert len(all_test) == len(np.unique(all_test))


def test_raises_when_sample_too_short(cfg):
    with pytest.raises(ValueError):
        walk_forward(np.array([202001, 202002, 202003]), cfg)
