"""Measure, rather than trust, whether a feature is look-ahead-free.

A feature is causal iff its value at bar ``t`` depends only on bars ``<= t``. The test is exact and model-free:
compute the feature on ``raw`` and on ``raw[:cut]``; on the shared first ``cut`` rows the two must be identical. Any
row that differs was computed with information from after it — a whole-series percentile, a ``pd.qcut`` bin edge, a
centred window, a backward fill, a negative shift. The registry's ``causal`` flag is a claim; this is the measurement.

What one cut can see: a leak of whole-series reach (a full-sample statistic, a ``qcut``) changes rows far from the
cut and is caught at any cut; a leak of bounded reach (``shift(-k)``, a centred window) always straddles the cut and
is caught too. A leak whose reach is data-dependent and often zero — a backward fill across gaps, a pivot that is
only confirmed later — shows only when the cut lands where it is active. Audit such features at several cuts.

Measured 2026-10-07 on the FXPIG pool: 7 catalogue entries flagged ``causal=True`` failed it (whole-series doji
caps in the absorption-ratio family, whole-series volume bins in the binned-MFI family), and the pipeline's
dataset builder carried three of them. ``test_catalogue_is_lookahead_free`` keeps that from happening again.

    >>> report = truncation_audit(ohlcv, cut=len(ohlcv) * 2 // 3)
    >>> report[report.status == AuditStatus.LEAK]       # must be empty

The end cut keeps the START fixed, so it cannot see the other way backtest and live disagree: a value that depends on
where the history begins. A cumsum from the first bar is the case in point — causal, but an integrated level that a
live process with a shorter history computes differently, and one that earns spurious-regression scores in a
screen. ``start_cut_audit`` drops the first ``start`` bars instead and, after ``settle`` bars of history, requires
the two runs to agree. It also names any output column that is an input column passed straight through.

    >>> report = start_cut_audit(ohlcv, start=300, settle=1200)
    >>> report[report.status == AuditStatus.ANCHORED]   # must be empty
"""
from __future__ import annotations

import enum
import warnings
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from ..registry import FeatureEntry, FeatureRegistry
from ._binder import build_binding, call_feature, derived_overrides, numeric_columns

#: Relative / absolute tolerance for "identical". Truncating the END of the input cannot change the arithmetic that
#: produced an earlier value, so any difference beyond floating-point noise is information from the future.
RTOL = 1e-7
ATOL = 1e-12
#: Minimum shared finite rows before a column counts as checked.
MIN_SHARED = 50
#: Start-cut tolerance, relative to the column's standard deviation over the compared rows. Exact equality is the wrong
#: test here: a new start legitimately perturbs a long-memory recursion (an EMA forgets its seed geometrically) and the
#: running-sum rounding of a rolling window (measured: ~1e-8 of a std on Bollinger %B). A level that never forgets its
#: start sits orders of magnitude above this.
START_RTOL = 1e-6
#: Share of compared rows that may still differ before a column counts as ANCHORED. A rank over a tie-rich series flips
#: a few ties when the rounding changes (measured: 0.6% of rows for dv2's percent rank); an anchored level differs on
#: every row.
START_MAX_SHARE = 0.01


class AuditStatus(enum.StrEnum):
    CAUSAL = "causal"                # every shared row identical
    LEAK = "leak"                    # at least one shared row changed when later bars were appended
    UNCHECKABLE = "uncheckable"      # too few shared finite rows (warm-up longer than the cut)
    BLOCKED = "blocked"              # could not be bound or called (cross-market, event-indexed, error)
    START_FREE = "start_free"        # after the settle period the value no longer depends on where the history starts
    ANCHORED = "anchored"            # still depends on the first bar after the settle period (cumsum level, expanding)


def compare_truncated(full: pd.Series, part: pd.Series, cut: int) -> tuple[int, int, float]:
    """``(shared_rows, changed_rows, max_abs_diff)`` over the first ``cut`` rows of both series."""
    a = np.asarray(full, dtype=float)[:cut]
    b = np.asarray(part, dtype=float)[:cut]
    both = np.isfinite(a) & np.isfinite(b)
    one_sided = np.isfinite(a) ^ np.isfinite(b)          # NaN in one run, a number in the other: also a change
    diff = both & ~np.isclose(a, b, rtol=RTOL, atol=ATOL)
    changed = int(diff.sum() + one_sided.sum())
    max_abs = float(np.max(np.abs(a[both] - b[both]))) if both.any() else float("nan")
    return int(both.sum()), changed, max_abs


def check_function(fn: Callable[[pd.DataFrame], object], raw: pd.DataFrame, cut: int,
                   name: str = "feature") -> pd.DataFrame:
    """Truncation-check any ``fn(frame) -> Series | DataFrame | tuple``. One row per output column."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full, _ = numeric_columns(fn(raw.copy()), name, len(raw))
        part, _ = numeric_columns(fn(raw.iloc[:cut].copy()), name, cut)
    return _rows(name, full, part, cut, causal_flag=None)


def _rows(entry: str, full: dict[str, pd.Series], part: dict[str, pd.Series], cut: int,
          causal_flag: bool | None) -> pd.DataFrame:
    rows = []
    for col, series in full.items():
        if col not in part:
            rows.append({"entry": entry, "column": col, "status": AuditStatus.BLOCKED.value, "shared_rows": 0,
                         "changed_rows": 0, "changed_share": float("nan"), "max_abs_diff": float("nan"),
                         "causal_flag": causal_flag, "reason": "column absent from the truncated run"})
            continue
        shared, changed, max_abs = compare_truncated(series, part[col], cut)
        if changed > 0:
            status = AuditStatus.LEAK
        elif shared < MIN_SHARED:
            status = AuditStatus.UNCHECKABLE
        else:
            status = AuditStatus.CAUSAL
        rows.append({"entry": entry, "column": col, "status": status.value, "shared_rows": shared,
                     "changed_rows": changed, "changed_share": changed / max(1, shared), "max_abs_diff": max_abs,
                     "causal_flag": causal_flag, "reason": ""})
    return pd.DataFrame(rows)


def truncation_audit(raw: pd.DataFrame, cut: int, entries: Iterable[FeatureEntry] | None = None,
                     overrides: dict[str, dict] | None = None) -> pd.DataFrame:
    """Truncation-check every bindable catalogue entry (or ``entries``) on ``raw``.

    ``raw`` needs ``open/high/low/close`` and ideally ``tick_volume``/``volume`` and ``spread``. Returns one row per
    output column with an ``AuditStatus``; entries that cannot be bound or called get one BLOCKED row with the reason.
    Data-derived kwargs are computed once on ``raw`` and held fixed for the truncated run, so the only thing that
    differs between the two runs is the data after ``cut``.
    """
    if not 0 < cut < len(raw):
        raise ValueError(f"cut must be inside the frame (0 < cut < {len(raw)}), got {cut}")
    fixed = {**derived_overrides(raw), **(overrides or {})}
    frames = []
    for e in (entries if entries is not None else FeatureRegistry()):
        qn = e.qualified_name
        b = build_binding(qn, e.module, e.name, raw, overrides=fixed.get(qn))
        if b.fn is None:
            frames.append(_blocked(qn, e.causal, f"{b.category}: {b.blocker}"))
            continue
        try:
            full, why = call_feature(b, raw)
            part, _ = call_feature(b, raw.iloc[:cut])
        except Exception as ex:
            frames.append(_blocked(qn, e.causal, f"error: {type(ex).__name__}: {ex}"))
            continue
        if not full:
            frames.append(_blocked(qn, e.causal, why or "no per-bar output"))
            continue
        frames.append(_rows(qn, full, part, cut, causal_flag=e.causal))
    return pd.concat(frames, ignore_index=True)


def _blocked(entry: str, causal_flag: bool, reason: str) -> pd.DataFrame:
    return pd.DataFrame([{"entry": entry, "column": entry, "status": AuditStatus.BLOCKED.value, "shared_rows": 0,
                          "changed_rows": 0, "changed_share": float("nan"), "max_abs_diff": float("nan"),
                          "causal_flag": causal_flag, "reason": reason}])


def compare_start_cut(full: pd.Series, part: pd.Series, start: int, settle: int) -> tuple[int, int, float]:
    """``(shared_rows, changed_rows, max_rel_diff)`` on the rows ``part`` computed with ``settle``+ bars of history.

    ``full`` covers the whole frame and ``part`` the frame from row ``start``; differences are scaled by the column's
    standard deviation over the compared rows (absolute when that is zero).
    """
    a = np.asarray(full, dtype=float)[start + settle:]
    b = np.asarray(part, dtype=float)[settle:]
    both = np.isfinite(a) & np.isfinite(b)
    one_sided = np.isfinite(a) ^ np.isfinite(b)
    if not both.any():
        return 0, int(one_sided.sum()), float("nan")
    scale = float(np.std(a[both]))
    rel = np.abs(a[both] - b[both]) / (scale if scale > 0 else 1.0)
    return int(both.sum()), int((rel > START_RTOL).sum() + one_sided.sum()), float(rel.max())


def passthrough_of(series: pd.Series, raw: pd.DataFrame) -> str:
    """Name of the ``raw`` column that ``series`` reproduces exactly on its finite rows, or ``""``."""
    a = np.asarray(series, dtype=float)
    finite = np.isfinite(a)
    if finite.sum() < MIN_SHARED:
        return ""
    for col in raw.columns:
        if pd.api.types.is_numeric_dtype(raw[col]) and np.array_equal(a[finite], raw[col].to_numpy(float)[finite]):
            return col
    return ""


def check_function_start(fn: Callable[[pd.DataFrame], object], raw: pd.DataFrame, start: int, settle: int,
                         name: str = "feature") -> pd.DataFrame:
    """Start-cut-check any ``fn(frame) -> Series | DataFrame | tuple``. One row per output column."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full, _ = numeric_columns(fn(raw.copy()), name, len(raw))
        part, _ = numeric_columns(fn(raw.iloc[start:].copy()), name, len(raw) - start)
    return _start_rows(name, full, part, raw, start, settle, causal_flag=None)


def _start_rows(entry: str, full: dict[str, pd.Series], part: dict[str, pd.Series], raw: pd.DataFrame, start: int,
                settle: int, causal_flag: bool | None) -> pd.DataFrame:
    rows = []
    for col, series in full.items():
        through = passthrough_of(series, raw)
        if col not in part:
            rows.append({"entry": entry, "column": col, "status": AuditStatus.BLOCKED.value, "shared_rows": 0,
                         "changed_rows": 0, "changed_share": float("nan"), "max_rel_diff": float("nan"),
                         "passthrough": through, "causal_flag": causal_flag, "reason": "column absent from the cut run"})
            continue
        shared, changed, max_rel = compare_start_cut(series, part[col], start, settle)
        share = changed / max(1, shared)
        if shared < MIN_SHARED:
            status = AuditStatus.UNCHECKABLE
        elif share > START_MAX_SHARE:
            status = AuditStatus.ANCHORED
        else:
            status = AuditStatus.START_FREE
        rows.append({"entry": entry, "column": col, "status": status.value, "shared_rows": shared,
                     "changed_rows": changed, "changed_share": share, "max_rel_diff": max_rel, "passthrough": through,
                     "causal_flag": causal_flag, "reason": ""})
    return pd.DataFrame(rows)


def start_cut_audit(raw: pd.DataFrame, start: int, settle: int, entries: Iterable[FeatureEntry] | None = None,
                    overrides: dict[str, dict] | None = None) -> pd.DataFrame:
    """Start-cut-check every bindable catalogue entry (or ``entries``) on ``raw``: computed on ``raw`` and on
    ``raw[start:]``, the rows that have ``settle`` bars of history in the cut run must agree.

    ``settle`` must outlast the longest legitimate warm-up and the forgetting time of the longest recursion in the
    catalogue, or those show up as ANCHORED. Returns one row per output column (with ``passthrough`` naming an input
    column reproduced exactly); entries that cannot be bound or called get one BLOCKED row with the reason.
    """
    if start <= 0 or start + settle + MIN_SHARED > len(raw):
        raise ValueError(f"need 0 < start and start + settle + {MIN_SHARED} <= {len(raw)}, got start={start}, "
                         f"settle={settle}")
    fixed = {**derived_overrides(raw), **(overrides or {})}
    frames = []
    for e in (entries if entries is not None else FeatureRegistry()):
        qn = e.qualified_name
        b = build_binding(qn, e.module, e.name, raw, overrides=fixed.get(qn))
        if b.fn is None:
            frames.append(_blocked(qn, e.causal, f"{b.category}: {b.blocker}"))
            continue
        try:
            full, why = call_feature(b, raw)
            part, _ = call_feature(b, raw.iloc[start:])
        except Exception as ex:
            frames.append(_blocked(qn, e.causal, f"error: {type(ex).__name__}: {ex}"))
            continue
        if not full:
            frames.append(_blocked(qn, e.causal, why or "no per-bar output"))
            continue
        frames.append(_start_rows(qn, full, part, raw, start, settle, causal_flag=e.causal))
    return pd.concat(frames, ignore_index=True)
