"""Rotation nulls for a label against a forward target.

Rotating the label keeps its run lengths and the target's own dependence (overlapping windows, volatility
clustering) intact, so only the alignment between the two is broken.

- ``DayLattice`` rotates by whole trading days and keeps each bar's intraday slot, so the label's time-of-day
  pattern stays aligned with the target's and the clock cannot pass for regime information (layers L1/L2).
- ``bar_shift_offsets`` rotates the stitched out-of-sample sequence by arbitrary bar counts (the L0 diagnostic).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DayLattice:
    day_pos: np.ndarray  # 0..D-1 per out-of-sample bar
    slot: np.ndarray  # intraday slot per out-of-sample bar
    grid: np.ndarray  # (D, J) state code at (day, slot); -1 where no bar or unknown state

    @classmethod
    def build(cls, trading_day: np.ndarray, slot: np.ndarray, codes: np.ndarray) -> "DayLattice":
        days, day_pos = np.unique(trading_day, return_inverse=True)
        n_slots = int(slot.max()) + 1
        flat = day_pos * n_slots + slot
        if np.unique(flat).size != flat.size:
            raise ValueError("Two bars share a (trading day, slot); the bar spacing is irregular.")
        grid = np.full((days.size, n_slots), -1, dtype=np.int64)
        grid[day_pos, slot] = codes
        return cls(day_pos.astype(np.int64), slot.astype(np.int64), grid)

    @property
    def n_days(self) -> int:
        return self.grid.shape[0]

    def rotate(self, offset: int) -> np.ndarray:
        """State codes after moving the label ``offset`` trading days, slot by slot (-1 where the slot is empty)."""
        return self.grid[(self.day_pos + offset) % self.n_days, self.slot]


def draw_offsets(n_positions: int, min_offset: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Distinct offsets from [min_offset, n_positions - min_offset], at most ``n_draws`` of them."""
    candidates = np.arange(min_offset, n_positions - min_offset + 1)
    if candidates.size == 0:
        return candidates
    return np.sort(rng.choice(candidates, size=min(n_draws, candidates.size), replace=False))


def mean_dwell(codes: np.ndarray) -> float:
    """Mean run length of known states in a code sequence (-1 breaks a run)."""
    known = codes >= 0
    if not known.any():
        return np.nan
    starts = known & np.r_[True, codes[1:] != codes[:-1]]
    return float(known.sum() / starts.sum())


def episode_starts(codes: np.ndarray) -> np.ndarray:
    """True on the first bar of every run of a known state."""
    return (codes >= 0) & np.r_[True, codes[1:] != codes[:-1]]


def p_value(observed: float, null: np.ndarray) -> float:
    """(1 + #{null >= observed}) / (1 + B) over the finite null draws."""
    finite = null[np.isfinite(null)]
    if not np.isfinite(observed) or finite.size == 0:
        return np.nan
    return float((1 + (finite >= observed).sum()) / (1 + finite.size))
