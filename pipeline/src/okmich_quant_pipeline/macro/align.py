"""No-lookahead asof-merge — broadcast availability-stamped features onto bars.

Pure, IO-free. ``attach_exogenous`` is **cadence-agnostic**: it only ever reads
``available_from_utc`` + ``value`` from the feature frame, so daily, weekly, monthly, or
irregular event-driven series all attach through the same path. There is no assumption that
features share a cadence or a release schedule — a single forward-fill reconciles whatever
mix is present.

Algorithm
---------
1. Validate ``bars`` index is a sorted, UTC-tz-aware ``DatetimeIndex``.
2. Pivot the long feature frame to wide, **indexed by ``available_from_utc``** (one column per
   feature). Heterogeneous release times produce interleaved NaNs across rows — exactly what
   the next step reconciles.
3. ``ffill`` down the columns: each feature carries its last-known value forward to every
   later release instant. This is causal — forward-fill propagates *past* values forward in
   time only — and it is what lets a single merge serve features of different cadence.
4. One ``merge_asof(direction="backward")``: every bar gets, per feature, the most recent value
   whose ``available_from_utc <= bar_timestamp``. Never a future observation.

Worked example: a bar at ``D+1 10:00 UTC`` gets VIX(D) (released D 22:00) and CREDIT(D-1)
(released D 22:00), but NOT VIX(D+1) / CREDIT(D) which are not public until D+1 22:00.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_bars(bars: pd.DataFrame) -> None:
    """Reject anything that would make the asof-merge silently wrong."""
    idx = bars.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError(f"bars must have a DatetimeIndex; got {type(idx).__name__}")
    if idx.tz is None:
        raise ValueError("bars index must be tz-aware UTC; got tz-naive (refusing to assume a timezone)")
    # Require canonical UTC (not merely a zero-offset tz) so the merge keys are unambiguous;
    # callers normalize via tz_convert("UTC") upstream.
    if str(idx.tz) != "UTC":
        raise ValueError(f"bars index must be UTC; got {idx.tz}")
    if not idx.is_monotonic_increasing:
        raise ValueError("bars index must be sorted ascending")


def attach_exogenous(bars: pd.DataFrame, features: pd.DataFrame, *, prefix: str = "macro_",
                     max_staleness: pd.Timedelta | None = None) -> pd.DataFrame:
    """Attach availability-stamped daily/weekly/irregular features onto intraday bars.

    Parameters
    ----------
    bars
        Intraday frame with a sorted, UTC-tz-aware ``DatetimeIndex`` (a ``DatasetBuilder``
        output satisfies this).
    features
        Long frame with columns ``feature``, ``value``, ``available_from_utc`` (tz-aware UTC).
        Cadence is irrelevant; each feature may release on its own schedule.
    prefix
        Prepended to every attached column (default ``"macro_"``).
    max_staleness
        If set, a bar whose most recent available macro observation is older than this is given
        NaN macro values instead of silently carrying a stale value forward. Guards the
        operational case where the macro store stopped refreshing (e.g. a dead fetch job).
        ``None`` (default) preserves the carry-forward behaviour.

    Returns
    -------
    pd.DataFrame
        ``bars`` with one ``{prefix}{feature}`` column per feature. Same index, same row count
        and order. Bars before a feature's first release (or inside its warmup) get NaN.
    """
    _validate_bars(bars)
    result = bars.copy()
    if features.empty:
        return result

    # aggfunc="last" only disambiguates a duplicate (availability, feature) pair; by construction
    # each observation maps to a unique availability per feature, so there are none in practice.
    wide = features.pivot_table(index="available_from_utc", columns="feature", values="value", aggfunc="last").sort_index()
    wide = wide.ffill()
    wide.columns = [f"{prefix}{col}" for col in wide.columns]
    macro_cols = list(wide.columns)

    # merge_asof requires both keys at the same datetime resolution. Parquet bar indices are
    # often datetime64[us] while the stamps are [ns]; normalize both to ns so any input
    # resolution (s/ms/us/ns) merges cleanly.
    left = result.reset_index()
    ts_col = left.columns[0]
    left[ts_col] = left[ts_col].dt.as_unit("ns")
    right = wide.reset_index()
    right["available_from_utc"] = right["available_from_utc"].dt.as_unit("ns")
    merged = pd.merge_asof(left, right, left_on=ts_col, right_on="available_from_utc", direction="backward")

    if max_staleness is not None:
        # age = bar_ts − most-recent available macro instant. NaT age (pre-history, no match)
        # compares False and is already NaN from the merge, so it is left untouched.
        age = merged[ts_col] - merged["available_from_utc"]
        merged.loc[age > max_staleness, macro_cols] = np.nan

    # Single bulk assignment — merge_asof preserves left/bar order (bars are validated-sorted),
    # so this is positionally aligned, avoids per-column fragmentation, and needs no index round-trip.
    result[macro_cols] = merged[macro_cols].to_numpy()
    return result
