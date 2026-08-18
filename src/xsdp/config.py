"""Typed configuration objects, loaded from the YAML files in ``configs/``.

Every number that could plausibly be tuned lives here rather than in the
research code, so that a reviewer can read one file and know exactly what
universe, what sample split and what cost assumption produced a given table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

#: Where data and outputs live. Defaults to the repository, but ``XSDP_ROOT``
#: redirects the whole tree elsewhere -- which is how the smoke tests run against
#: throwaway data without any risk of overwriting a real, expensive panel.
REPO_ROOT = Path(os.environ.get("XSDP_ROOT") or Path(__file__).resolve().parents[2])


@dataclass(frozen=True)
class Paths:
    root: Path = REPO_ROOT
    raw: Path = REPO_ROOT / "data/raw"
    external: Path = REPO_ROOT / "data/external"
    interim: Path = REPO_ROOT / "data/interim"
    processed: Path = REPO_ROOT / "data/processed"
    outputs: Path = REPO_ROOT / "outputs"
    figures: Path = REPO_ROOT / "report/figures"
    tables: Path = REPO_ROOT / "report/tables"

    def mkdirs(self) -> None:
        for f in fields(self):
            getattr(self, f.name).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class UniverseConfig:
    """Which anomaly-months enter the study.

    The asset universe is the set of published cross-sectional anomalies from
    Chen & Zimmerman's Open Source Asset Pricing project.  It is **point-in-time
    by publication date**: an anomaly becomes investable in the January after
    its paper appeared, so the backtest never allocates to a strategy nobody had
    heard of yet.

    ``start`` is 1973 rather than the 1926 the return data reaches back to,
    because before the early 1970s only a handful of anomalies had been
    published and a "cross-section" of six assets is not one.
    """

    start: str = "1973-01-31"
    end: str = "2023-12-31"
    min_assets_per_month: int = 20
    min_history_months: int = 60      # an anomaly needs this much return history
    require_publication_year: bool = True

    #: Which Chen-Zimmermann portfolio construction to allocate across.
    #: ``op`` is the headline set (all stocks).  ``ex_nyse_p20_me`` rebuilds every
    #: anomaly using only stocks above the NYSE 20th percentile of market cap and
    #: ``ex_price5`` only stocks above $5 -- the capacity checks.  Published
    #: anomaly returns are concentrated in small, illiquid names, so what these
    #: answer is whether an edge survives in stocks a fund can trade at size.
    portfolio_set: str = "op"


@dataclass(frozen=True)
class SplitConfig:
    """Walk-forward (expanding-window) evaluation protocol.

    We never use a random train/test split: returns are a time series and a
    random split leaks the future into the past.  The sample is walked forward
    one year at a time, refitting on everything known at that point.

    The dates below are dictated by the data.  The point-in-time universe holds
    only ~13 anomalies in 1990 and does not reach 20 until 1994, so training
    starts there; the first prediction is made for 2008, after fourteen years of
    history, leaving seventeen years out of sample.

    ``embargo_months`` drops the observations immediately before each validation
    block.  Twelve months, because several anomalies are built from annual
    accounting data and a shorter gap would let the same fundamentals sit on
    both sides of the boundary (Lopez de Prado's purging argument).
    """

    train_start: str = "1994-01-31"
    first_test_year: int = 2008
    val_years: int = 4
    test_step_years: int = 1
    embargo_months: int = 12
    expanding: bool = True
    train_years: int = 20        # only used when expanding=False


@dataclass(frozen=True)
class CostConfig:
    """Transaction-cost assumptions, used everywhere including inside the loss.

    Allocating across anomalies has two distinct costs, and conflating them is
    the most common way to get this problem wrong:

    ``spread_bps``
        The one-way cost of trading the underlying stocks.  Multiplied by each
        anomaly's own measured monthly turnover, it gives the cost of *holding*
        that anomaly for a month -- which is why ``MaxRet``, whose book is
        rewritten monthly, is roughly nine times dearer to carry than ``BM``.

    ``switch_bps``
        The one-way cost of *changing* the allocation, paid on the change in
        weight.  Building or unwinding a gross-2 long-short book costs about
        twice the per-stock one-way cost, which :mod:`xsdp.data.build_anomaly_panel`
        applies.

    ``short_fee_bps_annual``
        Stock-borrow cost on the short leg of each anomaly.
    """

    spread_bps: float = 25.0
    switch_bps: float = 25.0
    short_fee_bps_annual: float = 100.0
    #: Grid used by the cost-sensitivity exhibit, as multiples of the above.
    sensitivity_multipliers: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)


@dataclass(frozen=True)
class PortfolioConfig:
    """How model scores become an allocation across anomalies.

    ``scheme``
        ``"tilt"``    equal weight plus a dollar-neutral tilt.  The default, and
                      the honest one: the benchmark an allocator must beat is
                      1/N, so the model is asked for the *deviation* from it.
        ``"simplex"`` long-only weights summing to one (softmax).
        ``"neutral"`` a pure dollar-neutral long-short across anomalies.

    ``tilt_gross``
        L1 size of the tilt relative to the 1/N base.  At 1.0 the tilt is as
        large as the base allocation, which permits zero weights but not shorts
        for a typical cross-section.
    """

    scheme: str = "tilt"
    tilt_gross: float = 1.0
    gross_leverage: float = 1.0
    max_weight: float = 0.10          # per-anomaly cap, fraction of gross
    n_quantiles: int = 5
    vol_target_annual: float | None = 0.10


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    n_ensemble: int = 5             # NN results are averaged over seeds (Gu-Kelly-Xiu)
    max_epochs: int = 120
    patience: int = 15
    batch_months: int = 32          # a "sample" is a whole cross-section, not a stock
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "auto"            # auto -> mps > cuda > cpu


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def suffix(self) -> str:
        """Filename suffix that keeps variant runs from overwriting each other."""
        ps = self.universe.portfolio_set
        return "" if ps == "op" else f"__{ps}"


def _build(cls, payload: dict[str, Any]):
    kwargs = {}
    for f in fields(cls):
        if f.name not in payload:
            continue
        value = payload[f.name]
        if is_dataclass(f.type) if isinstance(f.type, type) else False:
            kwargs[f.name] = _build(f.type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """Load ``configs/default.yaml`` (or ``path``) into a :class:`Config`."""
    if path is None:
        path = REPO_ROOT / "configs/default.yaml"
    path = Path(path)
    if not path.exists():
        return Config()
    payload = yaml.safe_load(path.read_text()) or {}
    sub = {
        "universe": UniverseConfig,
        "split": SplitConfig,
        "costs": CostConfig,
        "portfolio": PortfolioConfig,
        "train": TrainConfig,
    }
    kwargs: dict[str, Any] = {}
    for key, cls in sub.items():
        if key in payload:
            kwargs[key] = _build(cls, payload[key])
    return Config(**kwargs)
