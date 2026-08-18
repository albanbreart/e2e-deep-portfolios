"""Training loop for the neural models, shared by every objective.

The loop is objective-agnostic: it takes a loss function with the signature used
in :mod:`xsdp.losses` and does not care whether that loss is a squared error or
the Sharpe ratio of a simulated book.  That is what makes the central comparison
of the project — same architecture, same data, same folds, different objective —
an actually controlled experiment.

Two conventions worth flagging:

*Early stopping is on the validation value of the training objective.*  It would
be tempting to early-stop an MSE model on validation Sharpe, but then the MSE
model is quietly getting some of the end-to-end model's advantage and the
comparison stops being clean.  Each model is stopped on its own terms.

*Results are averaged over random seeds.*  A single neural network fitted to
monthly returns is a high-variance estimator; Gu, Kelly and Xiu average ten.
We average :attr:`TrainConfig.n_ensemble` (default five), which removes most of
the seed noise without which two architectures cannot be told apart.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from ..losses import masked_huber, masked_mse, mean_variance_loss, net_sharpe_loss, rank_ic_loss
from .dataset import PanelTensors
from .nets import build_net, count_parameters

log = logging.getLogger(__name__)

LOSSES: dict[str, Callable] = {
    "mse": masked_mse,
    "huber": masked_huber,
    "ic": rank_ic_loss,
    "sharpe": net_sharpe_loss,
    "utility": mean_variance_loss,
}

#: Objectives that simulate a portfolio and therefore need returns, mask *and*
#: costs; the rest are pointwise and only need predictions and targets.
PORTFOLIO_LOSSES = {"sharpe", "utility"}


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainReport:
    """What happened during one fit, for the appendix and for debugging."""

    best_epoch: int
    best_val: float
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    n_params: int = 0


def _evaluate(model, tensors: PanelTensors, months: np.ndarray, loss_name: str,
              cfg, device, *, window: int) -> float:
    """Mean loss over a block of months, computed in windows to bound memory."""
    model.eval()
    loss_fn = LOSSES[loss_name]
    total, n = 0.0, 0
    with torch.no_grad():
        for chunk in tensors.windows(months, length=window, stride=window):
            w = tensors.window(chunk).to(device)
            scores = model(w.x, w.mask)
            if loss_name in PORTFOLIO_LOSSES:
                loss, _ = loss_fn(scores, w.y, w.mask, w.switch, w.carry,
                                  scheme=cfg.portfolio.scheme)
            else:
                loss, _ = loss_fn(scores, w.y, w.mask)
            total += float(loss) * len(chunk)
            n += len(chunk)
    return total / max(n, 1)


def train_one(
    tensors: PanelTensors,
    train_months: np.ndarray,
    val_months: np.ndarray,
    cfg,
    *,
    arch: str = "xs",
    loss_name: str = "sharpe",
    seed: int = 0,
    window: int | None = None,
    verbose: bool = False,
    **arch_kwargs,
) -> tuple[torch.nn.Module, TrainReport]:
    """Fit one network on ``train_months``, early-stopping on ``val_months``."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    tc = cfg.train
    device = resolve_device(tc.device)
    window = window or tc.batch_months

    model = build_net(arch, tensors.n_features, **arch_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc.max_epochs)
    loss_fn = LOSSES[loss_name]

    # Portfolio losses need contiguous months; pointwise ones do not, but using
    # the same windows keeps memory identical across objectives.
    batches = tensors.windows(train_months, length=window)
    report = TrainReport(best_epoch=-1, best_val=float("inf"),
                         n_params=count_parameters(model))
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t0, bad_epochs = time.time(), 0

    for epoch in range(tc.max_epochs):
        model.train()
        order = np.random.permutation(len(batches))
        epoch_loss = 0.0
        for bi in order:
            w = tensors.window(batches[bi]).to(device)
            scores = model(w.x, w.mask)
            if loss_name in PORTFOLIO_LOSSES:
                loss, _ = loss_fn(scores, w.y, w.mask, w.switch, w.carry,
                                  scheme=cfg.portfolio.scheme)
            else:
                loss, _ = loss_fn(scores, w.y, w.mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.detach())
        sched.step()

        val = _evaluate(model, tensors, val_months, loss_name, cfg, device, window=window)
        report.history.append({"epoch": epoch, "train": epoch_loss / len(batches), "val": val})

        if val < report.best_val - 1e-6:
            report.best_val, report.best_epoch, bad_epochs = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= tc.patience:
                break
        if verbose and epoch % 10 == 0:
            log.info("    epoch %3d train=%+.5f val=%+.5f", epoch, epoch_loss / len(batches), val)

    model.load_state_dict(best_state)
    report.seconds = time.time() - t0
    log.info("    %s/%s seed=%d: best epoch %d, val %+.5f, %d params, %.0fs",
             arch, loss_name, seed, report.best_epoch, report.best_val,
             report.n_params, report.seconds)
    return model, report


def train_ensemble(tensors, train_months, val_months, cfg, **kwargs
                   ) -> tuple[list[torch.nn.Module], list[TrainReport]]:
    """Fit :attr:`TrainConfig.n_ensemble` networks that differ only in seed."""
    models, reports = [], []
    for i in range(cfg.train.n_ensemble):
        m, r = train_one(tensors, train_months, val_months, cfg,
                         seed=cfg.train.seed + i, **kwargs)
        models.append(m)
        reports.append(r)
    return models, reports


@torch.no_grad()
def predict(models: list[torch.nn.Module], tensors: PanelTensors, months: np.ndarray,
            cfg, *, window: int = 12) -> np.ndarray:
    """Ensemble-average predictions for ``months``, in panel (row) order.

    Each member's scores are standardised within the month before averaging.
    Without that, a member whose outputs happen to have a larger scale would
    dominate the ensemble for reasons that have nothing to do with its accuracy.
    """
    device = resolve_device(cfg.train.device)
    for m in models:
        m.eval().to(device)

    chunks = []
    for block in tensors.windows(np.sort(months), length=window, stride=window):
        w = tensors.window(block).to(device)
        acc = torch.zeros_like(w.mask)
        for m in models:
            s = m(w.x, w.mask)
            n = w.mask.sum(-1, keepdim=True).clamp(min=1)
            mu = (s * w.mask).sum(-1, keepdim=True) / n
            sd = (((s - mu) ** 2 * w.mask).sum(-1, keepdim=True) / n).sqrt().clamp(min=1e-6)
            acc += (s - mu) / sd * w.mask
        acc /= len(models)
        acc = acc.cpu().numpy()
        for t in range(len(block)):
            sel = w.mask[t].cpu().numpy() > 0
            chunks.append(acc[t][sel])
    return np.concatenate(chunks)
