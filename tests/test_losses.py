"""The cost-aware losses must behave like the economics they encode."""

from __future__ import annotations

import pytest
import torch

from xsdp.losses import (
    build_weights,
    drift,
    masked_huber,
    masked_mse,
    mean_variance_loss,
    net_sharpe_loss,
    normalise_weights,
    portfolio_returns,
    rank_ic_loss,
    simplex_weights,
    tilt_weights,
)
from xsdp.models.nets import build_net


@pytest.fixture
def batch():
    torch.manual_seed(0)
    T, N, K = 24, 80, 15
    mask = (torch.rand(T, N) > 0.15).float()
    y = torch.randn(T, N) * 0.04 * mask
    switch = torch.full((T, N), 0.005)
    # A wide spread of carrying costs, as in the real data.
    carry = torch.rand(N).mul(0.004).add(0.00005).expand(T, N).contiguous()
    x = torch.randn(T, N, K)
    return x, y, mask, switch, carry


# --------------------------------------------------------------------------- #
# Weight schemes
# --------------------------------------------------------------------------- #
def test_tilt_weights_sum_to_one_and_ignore_padding(batch):
    _, _, mask, _, _ = batch
    w = tilt_weights(torch.randn_like(mask), mask, tilt_gross=1.0)
    assert torch.allclose(w.sum(-1), torch.ones(w.shape[0]), atol=1e-5)
    assert (w * (1 - mask)).abs().max() == 0


def test_zero_tilt_is_equal_weight(batch):
    _, _, mask, _, _ = batch
    w = tilt_weights(torch.randn_like(mask), mask, tilt_gross=0.0)
    n = mask.sum(-1, keepdim=True)
    assert torch.allclose(w, mask / n, atol=1e-6)


def test_neutral_weights_are_dollar_neutral(batch):
    _, _, mask, _, _ = batch
    w = normalise_weights(torch.randn_like(mask), mask, gross=2.0)
    assert w.sum(-1).abs().max() < 1e-5
    assert torch.allclose(w.abs().sum(-1), torch.full((w.shape[0],), 2.0), atol=1e-4)


def test_simplex_weights_are_a_long_only_allocation(batch):
    _, _, mask, _, _ = batch
    w = simplex_weights(torch.randn_like(mask), mask)
    assert (w >= 0).all()
    assert torch.allclose(w.sum(-1), torch.ones(w.shape[0]), atol=1e-5)
    assert (w * (1 - mask)).abs().max() == 0


@pytest.mark.parametrize("scheme", ["tilt", "neutral", "simplex"])
def test_weights_are_invariant_to_affine_rescaling(batch, scheme):
    _, _, mask, _, _ = batch
    s = torch.randn_like(mask)
    a = build_weights(s, mask, scheme=scheme)
    b = build_weights(3 * s + 7, mask, scheme=scheme) if scheme != "simplex" \
        else build_weights(s + 7, mask, scheme=scheme)
    assert torch.allclose(a, b, atol=1e-4)


# --------------------------------------------------------------------------- #
# Cost accounting
# --------------------------------------------------------------------------- #
def test_drift_is_the_identity_under_zero_returns(batch):
    _, _, mask, _, _ = batch
    w = tilt_weights(torch.randn_like(mask), mask)
    assert torch.allclose(drift(w, torch.zeros_like(w), mask), w, atol=1e-6)


def test_holding_a_book_costs_no_switching(batch):
    """Repeating the same weights must produce ~zero switching turnover."""
    _, y, mask, switch, carry = batch
    mask = torch.ones_like(mask)
    w = tilt_weights(torch.randn_like(mask[:1]), mask[:1]).repeat(mask.shape[0], 1)
    _, _, to = portfolio_returns(w, torch.zeros_like(y), mask, switch, carry * 0)
    assert to[1:].max() < 1e-3


def test_carrying_is_charged_on_a_completely_static_book(batch):
    """The term that makes this problem different from stock selection."""
    _, y, mask, switch, carry = batch
    mask = torch.ones_like(mask)
    w = tilt_weights(torch.randn_like(mask[:1]), mask[:1]).repeat(mask.shape[0], 1)
    _, cost, to = portfolio_returns(w, torch.zeros_like(y), mask, switch, carry)
    assert to[1:].max() < 1e-3          # nothing traded
    assert (cost[1:] > 0).all()          # yet it still costs


def test_costs_scale_with_the_carrying_rate(batch):
    _, y, mask, switch, carry = batch
    w = tilt_weights(torch.randn_like(mask), mask)
    _, cheap, _ = portfolio_returns(w, y, mask, switch, carry)
    _, dear, _ = portfolio_returns(w, y, mask, switch, carry * 5)
    assert (dear >= cheap - 1e-9).all()


# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #
def test_perfect_foresight_beats_noise(batch):
    _, y, mask, switch, carry = batch
    good, _ = net_sharpe_loss(y, y, mask, switch, carry)
    bad, _ = net_sharpe_loss(torch.randn_like(y), y, mask, switch, carry)
    assert good < bad


def test_higher_costs_never_improve_the_objective(batch):
    _, y, mask, switch, carry = batch
    s = torch.randn_like(y)
    cheap, _ = net_sharpe_loss(s, y, mask, switch * 0, carry * 0)
    dear, _ = net_sharpe_loss(s, y, mask, switch * 10, carry * 10)
    assert dear >= cheap


def test_cost_aware_objective_prefers_the_cheap_asset_all_else_equal(batch):
    """The central economic claim, isolated.

    Two signals with identical predictive content, one pointing at cheap assets
    and one at expensive ones.  A cost-blind objective cannot tell them apart; the
    net-Sharpe objective must prefer the cheap one.
    """
    _, y, mask, switch, carry = batch
    cheap_side = (carry[0] < carry[0].median()).float()      # [N]
    toward_cheap = y + 0.02 * cheap_side
    toward_dear = y + 0.02 * (1 - cheap_side)

    blind_a, _ = rank_ic_loss(toward_cheap, y, mask)
    blind_b, _ = rank_ic_loss(toward_dear, y, mask)
    aware_a, _ = net_sharpe_loss(toward_cheap, y, mask, switch, carry)
    aware_b, _ = net_sharpe_loss(toward_dear, y, mask, switch, carry)

    assert abs(float(blind_a) - float(blind_b)) < 0.15    # roughly indifferent
    assert float(aware_a) < float(aware_b)                # strictly prefers cheap


def test_turnover_penalty_punishes_churn_more_than_stability(batch):
    _, y, mask, switch, carry = batch
    torch.manual_seed(1)
    stable = y[:1].repeat(y.shape[0], 1) * 0.3 + y * 0.7
    churn = y * 0.7 + torch.randn_like(y) * 0.3

    free_s, _ = net_sharpe_loss(stable, y, mask, switch * 0, carry * 0, turnover_penalty=0.0)
    free_c, _ = net_sharpe_loss(churn, y, mask, switch * 0, carry * 0, turnover_penalty=0.0)
    pen_s, _ = net_sharpe_loss(stable, y, mask, switch, carry, turnover_penalty=0.02)
    pen_c, _ = net_sharpe_loss(churn, y, mask, switch, carry, turnover_penalty=0.02)
    assert (pen_c - free_c) > (pen_s - free_s)


@pytest.mark.parametrize("loss_name", ["sharpe", "utility", "mse", "huber", "ic"])
def test_every_loss_produces_finite_gradients(batch, loss_name):
    x, y, mask, switch, carry = batch
    model = build_net("xs_small", x.shape[-1])
    scores = model(x, mask)
    if loss_name == "sharpe":
        loss, _ = net_sharpe_loss(scores, y, mask, switch, carry)
    elif loss_name == "utility":
        loss, _ = mean_variance_loss(scores, y, mask, switch, carry)
    elif loss_name == "mse":
        loss, _ = masked_mse(scores, y, mask)
    elif loss_name == "huber":
        loss, _ = masked_huber(scores, y, mask)
    else:
        loss, _ = rank_ic_loss(scores, y, mask)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_masked_losses_ignore_padding(batch):
    _, y, mask, _, _ = batch
    pred = torch.randn_like(y)
    a, _ = masked_mse(pred * mask, y * mask, mask)
    b, _ = masked_mse(pred * mask + (1 - mask) * 1e6, y * mask, mask)
    assert torch.allclose(a, b)


def test_network_is_permutation_equivariant(batch):
    x, _, mask, _, _ = batch
    model = build_net("xs", x.shape[-1]).eval()
    perm = torch.randperm(x.shape[1])
    with torch.no_grad():
        a = model(x, mask)[:, perm]
        b = model(x[:, perm], mask[:, perm])
    assert float((a - b).abs().max()) < 1e-5
