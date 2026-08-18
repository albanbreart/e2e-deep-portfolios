"""Tensor construction: turning a long panel into windows of cross-sections.

A conventional tabular loader treats each asset-month as an independent sample.
That representation cannot express either of the two things this project is
about:

*   **Cross-sectional structure** — the model should be able to see that an
    anomaly's trailing Sharpe ratio is high *relative to its peers this month*,
    which requires the whole month to be present in one forward pass.
*   **Turnover** — the cost of an allocation depends on the allocation held last
    month, which requires *consecutive* months in one backward pass.

So the unit of batching is a **window of ``T`` consecutive months**, padded to
the union of assets appearing in that window.  Padding wastes some memory but
keeps everything dense, which is what makes it fast on a GPU; the mask keeps the
padding out of every statistic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch

log = logging.getLogger(__name__)


@dataclass
class MonthWindow:
    """One training example: ``T`` consecutive months of the cross-section."""

    x: torch.Tensor        # [T, N, K] features, zero-padded
    y: torch.Tensor        # [T, N]    next-month return
    mask: torch.Tensor     # [T, N]    1 where the asset is in the universe
    carry: torch.Tensor    # [T, N]    cost of holding one unit for a month
    switch: torch.Tensor   # [T, N]    cost of changing the weight by one unit
    months: np.ndarray     # [T]       YYYYMM of each row
    assets: np.ndarray     # [N]       column -> asset id

    def to(self, device) -> MonthWindow:
        return MonthWindow(
            self.x.to(device), self.y.to(device), self.mask.to(device),
            self.carry.to(device), self.switch.to(device), self.months, self.assets,
        )


class PanelTensors:
    """Materialises a panel into contiguous arrays once, then slices windows cheaply.

    Holding one ``[n_rows, K]`` feature block plus per-month row offsets uses
    about the same memory as the source Parquet and avoids re-scanning the panel
    every epoch.
    """

    def __init__(self, panel: pl.DataFrame, feature_cols: list[str], cfg,
                 *, target: str = "ret_next_dm", id_col: str = "signalname"):
        panel = panel.sort(["yyyymm", id_col])
        self.feature_cols = feature_cols
        self.id_col = id_col
        self.months = panel["yyyymm"].unique(maintain_order=True).to_numpy()

        self._x = np.nan_to_num(
            panel.select(feature_cols).to_numpy().astype(np.float32), copy=False
        )
        self._y = np.nan_to_num(panel[target].to_numpy().astype(np.float32), copy=False)

        ids = panel[id_col].to_numpy()
        self._asset_codes, self._asset_names = _encode(ids)

        self._carry = (
            panel["carry_cost"].to_numpy().astype(np.float32)
            if "carry_cost" in panel.columns
            else np.full(panel.height, cfg.costs.spread_bps / 1e4, dtype=np.float32)
        )
        switch = 2.0 * cfg.costs.switch_bps / 1e4
        self._switch = np.full(panel.height, switch, dtype=np.float32)

        counts = panel.group_by("yyyymm", maintain_order=True).len()["len"].to_numpy()
        self._offsets = np.concatenate([[0], np.cumsum(counts)])
        self._month_pos = {int(m): i for i, m in enumerate(self.months)}
        self.n_features = len(feature_cols)
        log.info("PanelTensors: %d months, %d rows, %d features, cross-section %d-%d",
                 len(self.months), len(self._y), self.n_features, counts.min(), counts.max())

    def _slice(self, i: int) -> slice:
        return slice(int(self._offsets[i]), int(self._offsets[i + 1]))

    def window(self, month_values: np.ndarray) -> MonthWindow:
        """Build the padded tensors for a specific list of consecutive months."""
        idx = [self._month_pos[int(m)] for m in month_values]
        slices = [self._slice(i) for i in idx]

        codes = np.unique(np.concatenate([self._asset_codes[s] for s in slices]))
        col_of = {c: j for j, c in enumerate(codes)}
        T, N, K = len(idx), len(codes), self.n_features

        x = np.zeros((T, N, K), dtype=np.float32)
        y = np.zeros((T, N), dtype=np.float32)
        mask = np.zeros((T, N), dtype=np.float32)
        carry = np.zeros((T, N), dtype=np.float32)
        switch = np.zeros((T, N), dtype=np.float32)

        for t, s in enumerate(slices):
            cols = np.fromiter((col_of[c] for c in self._asset_codes[s]), dtype=np.int64)
            x[t, cols] = self._x[s]
            y[t, cols] = self._y[s]
            mask[t, cols] = 1.0
            carry[t, cols] = self._carry[s]
            switch[t, cols] = self._switch[s]

        return MonthWindow(
            torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask),
            torch.from_numpy(carry), torch.from_numpy(switch),
            np.asarray(month_values), self._asset_names[codes],
        )

    def windows(self, month_values: np.ndarray, *, length: int, stride: int | None = None
                ) -> list[np.ndarray]:
        """Split a block of months into overlapping windows of ``length`` months."""
        months = np.sort(np.asarray(month_values))
        stride = stride or max(length // 2, 1)
        if len(months) <= length:
            return [months]
        starts = list(range(0, len(months) - length + 1, stride))
        if starts[-1] + length < len(months):
            starts.append(len(months) - length)
        return [months[s:s + length] for s in starts]

    def flat(self, month_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pooled ``(X, y, asset_code)`` arrays -- what trees and linear models want."""
        idx = [self._month_pos[int(m)] for m in np.sort(month_values)]
        rows = np.concatenate([np.arange(self._slice(i).start, self._slice(i).stop)
                               for i in idx])
        return self._x[rows], self._y[rows], self._asset_codes[rows]

    def month_frame(self, month_values: np.ndarray) -> pl.DataFrame:
        """``(yyyymm, asset)`` index for the given months, in tensor order."""
        idx = [self._month_pos[int(m)] for m in np.sort(month_values)]
        out = []
        for i in idx:
            s = self._slice(i)
            out.append(pl.DataFrame({
                "yyyymm": np.full(s.stop - s.start, self.months[i], dtype=np.int32),
                self.id_col: self._asset_names[self._asset_codes[s]],
            }))
        return pl.concat(out)


def _encode(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map asset identifiers to contiguous integer codes and back."""
    names, codes = np.unique(values, return_inverse=True)
    return codes.astype(np.int32), names
