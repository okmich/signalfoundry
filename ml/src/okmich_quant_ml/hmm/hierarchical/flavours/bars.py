"""Two-scale HHMMs on bar features: joint multi-scale and embedded (clock-chunked) multi-timeframe.

Joint multi-scale
-----------------
A slow *backdrop* state on top and a fast state underneath, on one clock, learned together. The slow feature columns
are emitted with parameters tied at depth 1 (one distribution per backdrop state), the fast columns per leaf, so each
level is informed by features at its own speed. The backdrop can change on any bar, but only when its fast sub-chain
ends (exit), so the two levels interact through the exits. ``predict_proba_level(X, level=1)`` is the backdrop
posterior.

Embedded multi-timeframe (Adam et al. 2019; Oelschlager & Adam 2020)
-------------------------------------------------------------------
A coarse state per block (e.g. a trading day) owns its own fine HMM over the block's bars. The coarse state can
change only at block boundaries, and the fine chain restarts from its entry distribution at each block: an exit
schedule that is FORCED at block starts and FORBIDDEN inside (``block_schedule``). Optional coarse columns (a
block-level feature) are tied at depth 1; put the value on exactly one bar per block and NaN elsewhere so it counts
once.
Filtering gives P(coarse state) updating through the day from the block's bars so far: the causal read the papers
do not show.

Coarse-feature timing (causality). A value may sit only on a bar at which it is known:
- a statistic of *earlier* blocks (yesterday's return or realised vol, a weekly average up to yesterday) may sit on
  today's first bar;
- a statistic of *today's* block (today's range, today's realised vol) is known only at today's last bar, so put it
  there. On the first bar it would hand the filter the day's outcome at the open.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..engine import CovarianceType, ExitSchedule, GaussianEmission, HHMM, LeafEmission, StateNode, StudentTEmission


class EmissionFamily(StrEnum):
    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"


class BlockUnit(StrEnum):
    DAY = "day"
    WEEK = "week"


def _emission(family: EmissionFamily, columns: Sequence[int], covariance_type: CovarianceType,
              tie_level: Optional[int]) -> LeafEmission:
    family = EmissionFamily(family)
    if family is EmissionFamily.GAUSSIAN:
        return GaussianEmission(list(columns), covariance_type=covariance_type, tie_level=tie_level)
    return StudentTEmission(list(columns), covariance_type=covariance_type, tie_level=tie_level)


def _names(prefix: str, n: int, names: Optional[Sequence[str]]) -> list[str]:
    if names is None:
        return [f"{prefix}{k}" for k in range(n)]
    names = list(names)
    if len(names) != n:
        raise ValueError(f"Expected {n} names, got {len(names)}")
    return names


def joint_multiscale(slow_columns: Sequence[int], fast_columns: Sequence[int], n_top: int = 2, n_fast: int = 2, *,
                     family: EmissionFamily = EmissionFamily.GAUSSIAN,
                     covariance_type: CovarianceType = CovarianceType.DIAG, allow_top_reentry: bool = False,
                     top_names: Optional[Sequence[str]] = None, **hhmm_kwargs) -> HHMM:
    """Backdrop (``n_top`` states, slow columns tied at depth 1) over a fast chain (``n_fast`` leaves each).

    ``allow_top_reentry=False`` makes the backdrop change whenever it ends (no exit-and-re-enter of the same state,
    which would duplicate the fast chain's own moves and blur identification).
    """
    if n_top < 2 or n_fast < 1:
        raise ValueError("joint_multiscale needs n_top >= 2 and n_fast >= 1")
    if set(slow_columns) & set(fast_columns):
        raise ValueError("slow_columns and fast_columns must not overlap")
    names = _names("backdrop", n_top, top_names)
    tops = [StateNode(n, [StateNode.leaf(f"{n}.fast{j}") for j in range(n_fast)]) for n in names]
    mask = None if allow_top_reentry else ~np.eye(n_top, dtype=bool)
    root = StateNode("root", tops, transition_mask=mask)
    emissions = [_emission(family, slow_columns, covariance_type, 1),
                 _emission(family, fast_columns, covariance_type, None)]
    return HHMM(root, emissions, **hhmm_kwargs)


def embedded_hhmm(fine_columns: Sequence[int], n_coarse: int = 2, n_fine: int = 3, *,
                  coarse_columns: Optional[Sequence[int]] = None, family: EmissionFamily = EmissionFamily.GAUSSIAN,
                  covariance_type: CovarianceType = CovarianceType.DIAG, coarse_names: Optional[Sequence[str]] = None,
                  block_column: Optional[int] = None, **hhmm_kwargs) -> HHMM:
    """Coarse block states, each owning an ``n_fine``-state HMM over the bars.

    Fit and decode with ``schedule=block_schedule(index, ...)``, or give ``block_column``: a column of X holding the
    block id (e.g. ``block_ids(index, ...)``), from which the model builds the schedule itself. That is what a
    pipeline passing only a feature matrix (``HmmModelWrapper``) needs. The coarse states never end on their own
    (their exits are masked); the schedule forces them to end at each block start, where the root may move to
    another coarse state or stay.
    """
    if n_coarse < 2 or n_fine < 1:
        raise ValueError("embedded_hhmm needs n_coarse >= 2 and n_fine >= 1")
    names = _names("coarse", n_coarse, coarse_names)
    coarse = [StateNode(n, [StateNode.leaf(f"{n}.fine{j}") for j in range(n_fine)], exit_mask=[False] * n_fine)
              for n in names]
    emissions = [_emission(family, fine_columns, covariance_type, None)]
    if coarse_columns:
        if set(coarse_columns) & set(fine_columns):
            raise ValueError("coarse_columns and fine_columns must not overlap")
        emissions.append(_emission(family, coarse_columns, covariance_type, 1))
    return HHMM(StateNode("root", coarse), emissions, schedule_column=block_column, schedule_depth=1, **hhmm_kwargs)


def block_ids(index: pd.DatetimeIndex, unit: BlockUnit = BlockUnit.DAY, tz: Optional[str] = None,
              roll_hour: int = 0) -> np.ndarray:
    """Block label of each bar: the date (or ISO week) of the bar's time in ``tz``, with the day starting at
    ``roll_hour`` (e.g. tz="America/New_York", roll_hour=17 for the FX day)."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if not 0 <= roll_hour < 24:
        raise ValueError(f"roll_hour must be in 0..23, got {roll_hour}")
    t = index
    if tz is not None:
        t = t.tz_convert(tz) if t.tz is not None else t.tz_localize("UTC").tz_convert(tz)
    # Wall-clock arithmetic: the day starts at roll_hour local time on both sides of a DST change (subtracting a
    # duration from an aware timestamp would move the roll by an hour across the change).
    wall = t.tz_localize(None) if t.tz is not None else t
    shifted = wall - pd.Timedelta(hours=roll_hour)
    if BlockUnit(unit) is BlockUnit.DAY:
        return shifted.normalize().values.astype("datetime64[D]").astype(np.int64)
    iso = shifted.isocalendar()
    return (iso["year"].to_numpy(dtype=np.int64) * 100 + iso["week"].to_numpy(dtype=np.int64))


def block_schedule(index: pd.DatetimeIndex, unit: BlockUnit = BlockUnit.DAY, tz: Optional[str] = None,
                   roll_hour: int = 0, depth: int = 1) -> ExitSchedule:
    """Exit schedule for the embedded HHMM: depth-``depth`` states end exactly at each new block."""
    return ExitSchedule.blocks(block_ids(index, unit, tz, roll_hour), depth=depth)
