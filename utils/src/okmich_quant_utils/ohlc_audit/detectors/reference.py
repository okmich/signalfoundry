"""D13: compare a symbol with an independent reference series stamped in true UTC (e.g. FXPIG crypto vs Binance).

Two checks, both only where the reference has data:
- Bad prices: the rolling median of |log(price / reference)| rises well above its own long-run level.
- Clock: per week, the whole-hour lag that best aligns the returns must be 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..findings import Finding, group_positions
from ..profile import ReferenceSpec
from ..symbol_data import SymbolData
from ..enums import Action, FindingKind, Severity
from .alignment import MAX_LAG_H, _lag_windows, _masked_corr


def reference_check(sd: SymbolData, ref: ReferenceSpec, rel_x: float = 3.0, floor_bp: float = 15.0, severe_bp: float = 50.0,
                    severe_bars: int = 12) -> list[Finding]:
    """Divergence of at least ``severe_bp`` or ``severe_bars`` is CRITICAL (drop). A milder one is MAJOR (inspect): a
    broker quoting a smoother price source with normal tick flow is not necessarily wrong for that broker."""
    r = ref.load()
    px = pd.Series(sd.df["close"].to_numpy(dtype=float), index=sd.utc)
    px = px[~px.index.duplicated(keep=False)].sort_index()      # bars sharing a UTC slot are D1's; don't attribute them here
    joined = pd.concat([px.rename("x"), r.rename("r")], axis=1, join="inner").dropna()
    if len(joined) < 2016:
        return []
    dev = (np.log(joined["x"]) - np.log(joined["r"])).abs()
    level = dev.rolling(6, center=True, min_periods=3).median()
    day = joined.index.floor("D")
    base = pd.Series(dev.groupby(day).median().rolling(7, min_periods=2).median().shift(1).reindex(day).to_numpy(), index=dev.index).bfill()
    bad = (level > np.maximum(rel_x * base, floor_bp * 1e-4)).to_numpy()
    out = []
    for a, b in group_positions(np.where(bad)[0], max_gap=6):
        u0, u1 = joined.index[a], joined.index[b] + sd.bar
        n = int(((sd.utc >= u0) & (sd.utc < u1)).sum())
        peak = float(dev.iloc[a:b + 1].max() * 1e4)
        severe = peak >= severe_bp or n >= severe_bars
        if not severe and n < 3:
            continue                                    # 1-2 bar blips at the edge of a divergence: not a finding
        out.append(sd.from_utc(FindingKind.BAD_PRICES, Severity.CRITICAL if severe else Severity.MAJOR, "D13", u0, u1, n,
                               Action.DROP if severe else Action.NONE, max_dev_bp=round(peak, 1), base_bp=round(float(base.iloc[a] * 1e4), 2),
                               reference=ref.note or ref.path.name))
    out += _weekly_lag(sd, px, r)
    return out


def _weekly_lag(sd: SymbolData, px: pd.Series, ref: pd.Series) -> list[Finding]:
    lo, hi = max(px.index[0], ref.index[0]), min(px.index[-1], ref.index[-1])
    if hi <= lo:
        return []
    idx = pd.date_range(lo.floor("D"), hi.ceil("D"), freq=sd.bar)
    X = np.log(px.reindex(idx)).diff().to_numpy()
    Y = np.log(ref.reindex(idx)).diff().to_numpy()
    step = 60 // sd.profile.bar_minutes
    wk = idx.tz_localize(None).to_period("W-SAT")
    bounds = np.r_[0, np.where(wk[1:] != wk[:-1])[0] + 1, len(idx)]
    out = []
    for s0, e0 in zip(bounds[:-1], bounds[1:], strict=True):
        a = X[s0:e0]
        if np.count_nonzero(~np.isnan(a)) < 600:
            continue
        c = _masked_corr(a, _lag_windows(Y, s0, e0, step), 300)
        if np.isnan(c).all():
            continue
        best = int(np.nanargmax(c))
        if best != MAX_LAG_H and c[best] - np.nan_to_num(c[MAX_LAG_H], nan=-1) > 0.05:
            u0, u1 = idx[s0], idx[e0 - 1] + sd.bar
            n = int(((sd.utc >= u0) & (sd.utc < u1)).sum())
            out.append(sd.from_utc(FindingKind.CLOCK_MISALIGNED, Severity.CRITICAL, "D13", u0, u1, n, Action.DROP, lag_hours=best - MAX_LAG_H,
                                   corr_lag0=round(float(c[MAX_LAG_H]), 3), corr_best=round(float(c[best]), 3), method="reference"))
    return out
