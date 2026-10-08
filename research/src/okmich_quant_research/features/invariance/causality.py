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


class AuditStatus(enum.StrEnum):
    CAUSAL = "causal"                # every shared row identical
    LEAK = "leak"                    # at least one shared row changed when later bars were appended
    UNCHECKABLE = "uncheckable"      # too few shared finite rows (warm-up longer than the cut)
    BLOCKED = "blocked"              # could not be bound or called (cross-market, event-indexed, error)


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
