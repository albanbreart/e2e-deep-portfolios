"""Objective functions, including the cost-aware portfolio losses.

Standard supervised learning on returns minimises squared error.  That objective
is misaligned with what the fund is paid for, in three separate ways:

1.  **It weights observations by return variance, not by economic value.**  A
    volatile micro-cap dominates the gradient even though the strategy can only
    take a small position in it.
2.  **It is indifferent to the cross-sectional ordering**, which is the only
    thing a dollar-neutral book actually monetises.
3.  **It knows nothing about turnover.**  Two models with identical MSE, one
    producing stable rankings and one producing rankings that reshuffle every
    month, are worth very different amounts of money after costs.

The functions below let the network optimise the quantity we actually care
about — the Sharpe ratio of a portfolio's return *after* the cost of trading
into it — which is differentiable end-to-end because portfolio construction is
just a normalisation and turnover is just an L1 distance.

Shapes throughout: ``[T, N]`` where ``T`` is a window of consecutive months and
``N`` the padded cross-section, with ``mask[t, i] = 1`` when stock ``i`` is in
the investable universe in month ``t``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def smooth_abs(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """``|x|`` with a rounded kink, so the turnover gradient is defined at zero.

    Plain ``abs`` works with Adam in practice (the subgradient at 0 is a measure
    zero event), but the smoothed version makes the loss surface better behaved
    when a position is genuinely being held flat month after month.
    """
    return torch.sqrt(x * x + eps * eps) - eps


def normalise_weights(scores: torch.Tensor, mask: torch.Tensor,
                      *, gross: float = 2.0) -> torch.Tensor:
    """Dollar-neutral weights with fixed gross exposure (the ``neutral`` scheme).

    Demeaning within the month enforces neutrality; dividing by the L1 norm fixes
    gross leverage.  Both are differentiable, so the network trains straight
    through them, and both make the output invariant to the scale and location of
    the scores -- the network cannot cheat by inflating its predictions.
    """
    scores = scores * mask
    n = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    mean = scores.sum(dim=-1, keepdim=True) / n
    dev = (scores - mean) * mask
    l1 = dev.abs().sum(dim=-1, keepdim=True).clamp(min=EPS)
    return gross * dev / l1


def tilt_weights(scores: torch.Tensor, mask: torch.Tensor,
                 *, tilt_gross: float = 1.0) -> torch.Tensor:
    """Equal weight plus a dollar-neutral tilt -- the default allocation scheme.

    The benchmark an allocator has to beat is 1/N, which is famously hard to
    beat, so the model is asked for the *deviation* from it rather than for the
    allocation itself.  This also conditions training well: at initialisation the
    network's scores are noise, the tilt is small and random, and the portfolio
    starts life as the equal-weighted benchmark rather than as a random book.

    ``tilt_gross`` is the L1 size of the tilt.  Weights sum to one by
    construction, and may go slightly negative for anomalies the model dislikes.
    """
    n = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    base = mask / n
    scores = scores * mask
    dev = (scores - scores.sum(dim=-1, keepdim=True) / n) * mask
    l1 = dev.abs().sum(dim=-1, keepdim=True).clamp(min=EPS)
    return base + tilt_gross * dev / l1


def simplex_weights(scores: torch.Tensor, mask: torch.Tensor,
                    *, temperature: float = 1.0) -> torch.Tensor:
    """Long-only weights summing to one (the ``simplex`` scheme).

    A softmax over the valid entries.  Useful as a robustness check: a fund that
    cannot short an anomaly sleeve still wants to know whether the model's
    ordering is worth anything.
    """
    logits = scores / max(temperature, EPS)
    logits = logits.masked_fill(mask == 0, float("-inf"))
    w = torch.softmax(logits, dim=-1)
    return torch.nan_to_num(w, nan=0.0)


WEIGHT_SCHEMES = {"neutral": normalise_weights, "tilt": tilt_weights, "simplex": simplex_weights}


def build_weights(scores: torch.Tensor, mask: torch.Tensor, *, scheme: str = "tilt",
                  **kwargs) -> torch.Tensor:
    """Dispatch to a weight scheme by name."""
    if scheme not in WEIGHT_SCHEMES:
        raise KeyError(f"unknown scheme {scheme!r}; have {sorted(WEIGHT_SCHEMES)}")
    return WEIGHT_SCHEMES[scheme](scores, mask, **kwargs)


def drift(weights: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Weights carried to the end of the month by returns alone (no trading)."""
    grown = weights * (1.0 + returns) * mask
    nav = 1.0 + (weights * returns * mask).sum(dim=-1, keepdim=True)
    return grown / nav.clamp(min=0.1)


def portfolio_returns(
    weights: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    switch_cost: torch.Tensor,
    carry_cost: torch.Tensor,
    *,
    prev_weights: torch.Tensor | None = None,
    smooth: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gross return, total cost and switching turnover for each month of a window.

    Two costs are charged, and separating them is the economic content of this
    project:

    *Switching* is paid on ``|w[t] - drifted(w[t-1])|`` at rate ``switch_cost``.
    Charging against the **drifted** book rather than the stale one matters: an
    allocation left untouched still changes as the anomalies earn different
    returns, and billing that as a trade would invent turnover that never
    happened.

    *Carrying* is paid on ``|w[t]|`` at rate ``carry_cost``, which is the
    anomaly's own internal monthly turnover times the per-stock spread.  This is
    the term that makes the problem non-trivial: an anomaly can be expensive
    even if the allocator never touches it.

    The first month of a window is charged against ``prev_weights`` when the
    caller carries state across windows, and otherwise treated as an entry from
    cash -- which slightly over-states cost at the boundary, an effect that is
    second-order for windows of two years or more.
    """
    T = weights.shape[0]
    zero = torch.zeros_like(weights[:1])
    prev = zero if prev_weights is None else prev_weights.unsqueeze(0)

    if T > 1:
        drifted = drift(weights[:-1], returns[:-1], mask[:-1])
        prior = torch.cat([prev, drifted], dim=0)
    else:
        prior = prev

    delta = weights - prior
    abs_delta = smooth_abs(delta) if smooth else delta.abs()

    turnover = (abs_delta * mask).sum(dim=-1)
    switching = (switch_cost * abs_delta * mask).sum(dim=-1)
    carrying = (carry_cost * weights.abs() * mask).sum(dim=-1)
    gross = (weights * returns * mask).sum(dim=-1)
    return gross, switching + carrying, turnover


def net_sharpe_loss(
    scores: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    switch_cost: torch.Tensor,
    carry_cost: torch.Tensor,
    *,
    scheme: str = "tilt",
    turnover_penalty: float = 0.0,
    **scheme_kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Negative annualised Sharpe ratio of the net-of-cost portfolio return.

    ``turnover_penalty`` adds an explicit L1 charge on top of the modelled cost.
    It is not double-counting: it is a *regulariser* that expresses the
    uncertainty in the cost model itself, and sweeping it traces out the
    efficient frontier between raw predictive power and implementability.
    """
    w = build_weights(scores, mask, scheme=scheme, **scheme_kwargs)
    g, c, to = portfolio_returns(w, returns, mask, switch_cost, carry_cost)
    net = g - c - turnover_penalty * to

    mu, sd = net.mean(), net.std(unbiased=True).clamp(min=EPS)
    sharpe = mu / sd * (12 ** 0.5)
    stats = {
        "sharpe_net": sharpe.detach(),
        "ret_gross": g.mean().detach() * 12,
        "cost": c.mean().detach() * 12,
        "turnover": to.mean().detach(),
    }
    return -sharpe, stats


def mean_variance_loss(
    scores: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    switch_cost: torch.Tensor,
    carry_cost: torch.Tensor,
    *,
    risk_aversion: float = 10.0,
    scheme: str = "tilt",
    turnover_penalty: float = 0.0,
    **scheme_kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Negative mean-variance utility of net returns.

    Preferred over the Sharpe loss for short windows: the Sharpe ratio's
    denominator is a sample standard deviation estimated on ``T`` points, so with
    ``T`` small the gradient is dominated by noise in the variance estimate.
    Mean-variance utility has the same optimum for a given risk level but a much
    better-conditioned gradient.
    """
    w = build_weights(scores, mask, scheme=scheme, **scheme_kwargs)
    g, c, to = portfolio_returns(w, returns, mask, switch_cost, carry_cost)
    net = g - c - turnover_penalty * to
    utility = net.mean() - 0.5 * risk_aversion * net.var(unbiased=True)
    stats = {
        "sharpe_net": (net.mean() / net.std(unbiased=True).clamp(min=EPS) * (12 ** 0.5)).detach(),
        "ret_gross": g.mean().detach() * 12,
        "cost": c.mean().detach() * 12,
        "turnover": to.mean().detach(),
    }
    return -utility, stats


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
               ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Plain squared error over valid entries -- the baseline objective."""
    n = mask.sum().clamp(min=1.0)
    loss = (((pred - target) ** 2) * mask).sum() / n
    return loss, {"mse": loss.detach()}


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 *, delta: float = 0.02) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Huber loss over valid entries.

    Monthly stock returns have extreme tails; under squared error a handful of
    +300% months drive the fit.  Gu, Kelly and Xiu make the same argument for
    their "OLS-H" benchmark, with ``delta`` around 2% per month.
    """
    n = mask.sum().clamp(min=1.0)
    loss = (F.huber_loss(pred, target, reduction="none", delta=delta) * mask).sum() / n
    return loss, {"huber": loss.detach()}


def rank_ic_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
                 ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Negative mean cross-sectional Pearson correlation (a soft IC objective).

    A middle ground between MSE and the full portfolio loss: it optimises the
    ordering, which is what a long-short book monetises, but ignores costs.
    Included so the ablation can separate "ordering matters" from
    "costs matter".
    """
    def _demean(x):
        n = mask.sum(-1, keepdim=True).clamp(min=1.0)
        return (x - (x * mask).sum(-1, keepdim=True) / n) * mask

    p, t = _demean(pred), _demean(target)
    num = (p * t).sum(-1)
    den = (p.pow(2).sum(-1).clamp(min=EPS).sqrt() * t.pow(2).sum(-1).clamp(min=EPS).sqrt())
    ic = (num / den)
    return -ic.mean(), {"ic": ic.mean().detach()}
