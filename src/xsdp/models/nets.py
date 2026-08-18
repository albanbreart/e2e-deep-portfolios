"""Neural architectures for the cross-section of returns.

Two families, and the contrast between them is one of the paper's results.

:class:`AssetMLP`
    The Gu-Kelly-Xiu (2020) benchmark: a feed-forward network applied to each
    asset-month independently.  It cannot see the rest of the cross-section, so
    everything it knows about "attractive relative to peers" has to arrive
    through the features, which is exactly why the features are pre-ranked.

:class:`CrossSectionalNet`
    A permutation-equivariant network over the *whole month at once*.  It is a
    Set Transformer (Lee et al., 2019): assets exchange information through a
    small set of learned "inducing points", which costs :math:`O(N m)` instead of
    the :math:`O(N^2)` of full self-attention — the difference between a model
    that fits on a laptop GPU and one that does not, since a monthly
    cross-section runs to a couple of hundred names here and to thousands in the
    stock-level version of the same problem.

    Permutation equivariance is the right inductive bias here and is worth
    stating precisely: relabelling the assets permutes the outputs identically.
    The network therefore cannot learn anything about an asset from *where it sits
    in the input*, only from its characteristics and from the distribution of its
    peers — which is what an asset-pricing model should do.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _masked_softmax(logits: torch.Tensor, key_mask: torch.Tensor | None) -> torch.Tensor:
    """Softmax over the last axis, ignoring padded keys.

    Padded entries are pushed to ``-inf`` before the softmax rather than zeroed
    after it, so the surviving weights still sum to one.  ``logits`` is
    ``[..., heads, queries, keys]`` while ``key_mask`` is ``[..., keys]``, so the
    mask gains a head axis and a query axis before broadcasting.  A month in
    which every key is padded would give a whole row of ``-inf`` and hence NaN;
    ``nan_to_num`` turns that into "attend to nothing", which the caller's own
    mask then discards.
    """
    if key_mask is not None:
        logits = logits.masked_fill(key_mask[..., None, None, :] == 0, float("-inf"))
    out = torch.softmax(logits, dim=-1)
    return torch.nan_to_num(out, nan=0.0)


class MAB(nn.Module):
    """Multihead attention block: queries ``Q`` attend over keys/values ``K``."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.h = n_heads
        self.dk = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim)
        )
        self.drop = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        *lead, n, d = x.shape
        return x.view(*lead, n, self.h, self.dk).transpose(-3, -2)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                key_mask: torch.Tensor | None = None) -> torch.Tensor:
        Q, K, V = self._split(self.q(q)), self._split(self.k(kv)), self._split(self.v(kv))
        att = _masked_softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.dk), key_mask)
        ctx = (att @ V).transpose(-3, -2).flatten(-2)
        h = self.ln1(q + self.drop(self.o(ctx)))
        return self.ln2(h + self.drop(self.ff(h)))


class ISAB(nn.Module):
    """Induced set-attention block: ``O(N m)`` all-to-all mixing over the month."""

    def __init__(self, dim: int, n_inducing: int = 32, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(n_inducing, dim) * 0.02)
        self.mab_in = MAB(dim, n_heads, dropout)     # inducing points read the month
        self.mab_out = MAB(dim, n_heads, dropout)    # assets read the summary

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-2]
        i = self.inducing.expand(*lead, *self.inducing.shape)
        summary = self.mab_in(i, x, key_mask=mask)   # [..., m, d]
        return self.mab_out(x, summary)              # [..., N, d]


class AssetMLP(nn.Module):
    """Per-asset feed-forward network (the Gu-Kelly-Xiu 'NN1'-'NN5' family).

    ``hidden=(32, 16, 8)`` reproduces GKX's NN3.  Batch-norm is deliberately
    absent: with a masked, ragged cross-section its running statistics are
    contaminated by padding, and LayerNorm gives the same conditioning without
    that failure mode.
    """

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (32, 16, 8),
                 dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        d = n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1) * mask


class CrossSectionalNet(nn.Module):
    """Set Transformer over the monthly cross-section.

    The output is a raw score per asset; :func:`xsdp.losses.normalise_weights`
    turns it into a dollar-neutral book.  Keeping normalisation out of the model
    means the same architecture serves both the "predict then optimise" and the
    end-to-end experiments -- only the loss changes.
    """

    def __init__(self, n_features: int, dim: int = 64, n_blocks: int = 2,
                 n_inducing: int = 32, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_features, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList(
            [ISAB(dim, n_inducing, n_heads, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.embed(x) * mask.unsqueeze(-1)
        for block in self.blocks:
            h = block(h, mask) * mask.unsqueeze(-1)
        return self.head(h).squeeze(-1) * mask


#: Registered architectures.  The sizes are deliberately modest: the panel has
#: ~200 assets and ~50k rows, so a network with a million parameters would be
#: fitting noise, and the capacity ladder below is what the appendix sweeps.
ARCHITECTURES: dict[str, dict] = {
    "nn1": {"cls": AssetMLP, "kwargs": {"hidden": (16,)}},
    "nn3": {"cls": AssetMLP, "kwargs": {"hidden": (32, 16, 8)}},
    "nn5": {"cls": AssetMLP, "kwargs": {"hidden": (32, 16, 8, 4, 2)}},
    "xs_tiny": {"cls": CrossSectionalNet,
                "kwargs": {"dim": 16, "n_blocks": 1, "n_inducing": 8, "n_heads": 2}},
    "xs_small": {"cls": CrossSectionalNet,
                 "kwargs": {"dim": 32, "n_blocks": 1, "n_inducing": 16, "n_heads": 4}},
    "xs": {"cls": CrossSectionalNet,
           "kwargs": {"dim": 48, "n_blocks": 2, "n_inducing": 24, "n_heads": 4}},
    "xs_deep": {"cls": CrossSectionalNet,
                "kwargs": {"dim": 64, "n_blocks": 3, "n_inducing": 32, "n_heads": 4}},
}


def build_net(name: str, n_features: int, **overrides) -> nn.Module:
    """Instantiate a registered architecture by name."""
    if name not in ARCHITECTURES:
        raise KeyError(f"unknown architecture {name!r}; have {sorted(ARCHITECTURES)}")
    spec = ARCHITECTURES[name]
    kwargs = {**spec["kwargs"], **overrides}
    return spec["cls"](n_features, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
