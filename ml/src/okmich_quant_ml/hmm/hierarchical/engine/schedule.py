"""Exit schedules: when the nodes at a given depth may, must or may not end.

By default every node ends with its learnt exit probability (``ExitMode.LEARNED``). A schedule overrides that per
depth and per bar, for the transition *into* bar t (from bar t-1):

- ``FORBIDDEN``: nodes at that depth cannot end; their children keep moving horizontally;
- ``FORCED``: nodes at that depth end after whichever child is active (and so do all deeper nodes).

``FORCED`` at the first bar of each block and ``FORBIDDEN`` inside the block is the embedded / clock-chunked HHMM
(Adam et al. 2019): the depth-1 state can only change at block boundaries (e.g. days) and its sub-chain restarts from
its entry distribution at each block. The root (depth 0) never ends, so schedules apply to depths >= 1.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Mapping, Optional

import numpy as np


class ExitMode(IntEnum):
    """How the nodes at one depth end on the transition into a bar."""

    LEARNED = 0     # end with the learnt exit probability
    FORBIDDEN = 1   # cannot end
    FORCED = 2      # must end (and every deeper node with them)


class ExitSchedule:
    """Per-depth, per-bar exit modes. Depths not listed stay ``LEARNED``.

    Parameters
    ----------
    modes : {depth: (T,) array of ExitMode values}. Entry t governs the transition from bar t-1 to bar t; entry 0
        (and the first bar of every sequence) is unused because those bars start from the root's entry distribution.
    """

    def __init__(self, modes: Mapping[int, np.ndarray]):
        if not modes:
            raise ValueError("An ExitSchedule needs at least one depth; omit the schedule for all-LEARNED exits")
        lengths = {len(np.asarray(v)) for v in modes.values()}
        if len(lengths) != 1:
            raise ValueError(f"All depths of an ExitSchedule need the same length, got {sorted(lengths)}")
        self.modes: dict[int, np.ndarray] = {}
        for depth, values in modes.items():
            depth = int(depth)
            if depth < 1:
                raise ValueError(f"Schedules apply to depths >= 1 (the root never ends), got depth {depth}")
            arr = np.asarray(values, dtype=np.int64)
            if arr.ndim != 1:
                raise ValueError(f"Depth {depth}: modes must be 1-D, got shape {arr.shape}")
            valid = {int(m) for m in ExitMode}
            if not set(np.unique(arr).tolist()) <= valid:
                raise ValueError(f"Depth {depth}: modes must be ExitMode values {sorted(valid)}")
            self.modes[depth] = arr.copy()
        self.n_bars = lengths.pop()

    @classmethod
    def blocks(cls, block_id: np.ndarray, depth: int = 1) -> "ExitSchedule":
        """Depth-``depth`` nodes end exactly when ``block_id`` changes (FORCED there, FORBIDDEN inside blocks).

        ``block_id`` is any per-bar label, e.g. the trading date of each bar.
        """
        ids = np.asarray(block_id)
        if ids.ndim != 1 or len(ids) == 0:
            raise ValueError("block_id must be a non-empty 1-D array")
        change = np.zeros(len(ids), dtype=bool)
        change[1:] = ids[1:] != ids[:-1]
        return cls.boundaries(change, depth)

    @classmethod
    def boundaries(cls, is_boundary: np.ndarray, depth: int = 1,
                   inside: ExitMode = ExitMode.FORBIDDEN) -> "ExitSchedule":
        """FORCED on bars where ``is_boundary`` is True, ``inside`` (FORBIDDEN or LEARNED) elsewhere."""
        flags = np.asarray(is_boundary, dtype=bool)
        inside = ExitMode(inside)
        if inside is ExitMode.FORCED:
            raise ValueError("inside must be FORBIDDEN or LEARNED")
        modes = np.where(flags, int(ExitMode.FORCED), int(inside)).astype(np.int64)
        return cls({int(depth): modes})

    def compile(self, n_bars: int, max_internal_depth: int) -> tuple[np.ndarray, np.ndarray]:
        """Unique mode patterns and the pattern index of every bar.

        Returns ``patterns`` (P, max_internal_depth) of ExitMode values for depths 1..max_internal_depth, and
        ``index`` (n_bars,) int64. FORCED at a depth is propagated to every deeper depth.
        """
        if self.n_bars != n_bars:
            raise ValueError(f"The schedule covers {self.n_bars} bars but the data has {n_bars}")
        bad = [d for d in self.modes if d > max_internal_depth]
        if bad:
            raise ValueError(f"Schedule depths {bad} have no nodes that can end (deepest such depth is "
                             f"{max_internal_depth})")
        return compile_patterns(self.modes, n_bars, max_internal_depth)


def compile_patterns(modes: Mapping[int, np.ndarray], n_bars: int,
                     max_internal_depth: int) -> tuple[np.ndarray, np.ndarray]:
    """Shared by ``ExitSchedule.compile`` and the all-LEARNED default."""
    width = max(max_internal_depth, 1)
    grid = np.full((n_bars, width), int(ExitMode.LEARNED), dtype=np.int64)
    for depth, values in modes.items():
        grid[:, depth - 1] = values
    for d in range(1, width):                      # FORCED propagates downwards
        grid[:, d] = np.where(grid[:, d - 1] == int(ExitMode.FORCED), int(ExitMode.FORCED), grid[:, d])
    patterns, index = np.unique(grid, axis=0, return_inverse=True)
    return patterns.astype(np.int64), index.reshape(-1).astype(np.int64)


def default_patterns(n_bars: int, max_internal_depth: int) -> tuple[np.ndarray, np.ndarray]:
    """All-LEARNED: one pattern used by every bar."""
    return compile_patterns({}, n_bars, max_internal_depth)


def resolve_schedule(schedule: Optional[ExitSchedule], n_bars: int,
                     max_internal_depth: int) -> tuple[np.ndarray, np.ndarray]:
    if schedule is None:
        return default_patterns(n_bars, max_internal_depth)
    if not isinstance(schedule, ExitSchedule):
        raise TypeError(f"schedule must be an ExitSchedule, got {type(schedule).__name__}")
    return schedule.compile(n_bars, max_internal_depth)
