"""D8a clock alignment against peers (broker level), D8b weekly open/close stamps (per symbol).

Clock errors on MT5 servers are whole hours, so lags are searched in whole hours (-12 h .. +12 h). The 13.5 h replay
is a copy, not a shift; D6 catches it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..findings import Finding, runs
from ..symbol_data import SymbolData
from ..enums import Action, AssetClass, FindingKind, Severity, SlotState

MAX_LAG_H = 12


# ---- D8b ------------------------------------------------------------------------------------------------------------

def event_stamps(sd: SymbolData) -> list[Finding]:
    """D8b: a trading week whose close (and open) sits a whole number of hours off the calendar."""
    grid, state = sd.grid, sd.grid.to_numpy()
    weekends = [(a, b) for a, b in runs(state == SlotState.CLOSED.value) if b - a >= sd.bars_per_day]
    if len(weekends) < 2:
        return []
    soft = ~np.isin(state, [SlotState.OPEN.value, SlotState.THIN.value, SlotState.CLOSED.value])
    u = np.unique(sd.utc.asi8)
    out = []
    for (_, b0), (a1, _) in zip(weekends[:-1], weekends[1:], strict=True):
        exp_open, exp_close = grid.index[b0], grid.index[a1 - 1]
        if soft[b0:min(b0 + sd.bars_per_day, a1)].any() or soft[max(a1 - sd.bars_per_day, b0):a1].any():
            continue                                    # holiday or early close at either end: stamps legitimately move
        lo, hi = (exp_open - pd.Timedelta(hours=MAX_LAG_H)).value, (exp_close + pd.Timedelta(hours=MAX_LAG_H)).value
        seg = u[(u >= lo) & (u <= hi)]
        if len(seg) < sd.bars_per_day:
            continue
        k_close = (seg[-1] - exp_close.value) / 6e10    # minutes
        k_open = (seg[0] - exp_open.value) / 6e10
        whole = abs(k_close) >= 55 and abs(k_close - 60 * round(k_close / 60)) <= 5
        if whole and -5 <= k_open - k_close <= 60:
            n = int(np.searchsorted(u, seg[-1], side="right") - np.searchsorted(u, seg[0]))
            out.append(sd.from_utc(FindingKind.EVENT_STAMP_DRIFT, Severity.CRITICAL, "D8b", pd.Timestamp(seg[0], tz="UTC"),
                                   pd.Timestamp(seg[-1], tz="UTC") + sd.bar, n, Action.DROP, shift_hours=round(k_close / 60),
                                   expected_close_utc=str(exp_close), actual_close_utc=str(pd.Timestamp(seg[-1], tz="UTC"))))
    return out


# ---- D8a ------------------------------------------------------------------------------------------------------------

def _masked_corr(a: np.ndarray, B: np.ndarray, min_n: int) -> np.ndarray:
    m = ~np.isnan(a)[None, :] & ~np.isnan(B)
    n = m.sum(1)
    a0, B0 = np.where(m, a[None, :], 0.0), np.where(m, B, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = a0.sum(1) / n, B0.sum(1) / n
        cov = (a0 * B0).sum(1) / n - ma * mb
        va, vb = (a0 * a0).sum(1) / n - ma ** 2, (B0 * B0).sum(1) / n - mb ** 2
        c = cov / np.sqrt(va * vb)
    c[n < min_n] = np.nan
    return c


def _lag_windows(Y: np.ndarray, s: int, e: int, step: int) -> np.ndarray:
    """Rows k = -MAX..+MAX hours: Y[s - k*step : e - k*step] (NaN-padded at the edges)."""
    pad = MAX_LAG_H * step
    lo, hi = s - pad, e + pad
    seg = np.full(hi - lo, np.nan)
    a, b = max(lo, 0), min(hi, len(Y))
    seg[a - lo:b - lo] = Y[a:b]
    win = sliding_window_view(seg, e - s)[::step]       # row j starts at lo + j*step  ->  lag k = MAX - j
    return win[::-1]                                     # row i <-> lag k = i - MAX


def _fx_peers(names: list[str]) -> dict[str, list[str]]:
    """Pairs sharing the quote currency; same-base pairs are added only when fewer than two share the quote.

    Same-base pairs often share the feed defect (2026-09: AUDUSD, AUDCAD and AUDJPY were shifted together), so their
    zero-lag votes would mask each other. Same-quote pairs see the shifted leg against an independent one.
    """
    out = {}
    for s in names:
        base, quote = s[:3], s[3:]
        same_quote = [p for p in names if p != s and p[3:] == quote]
        same_base = [p for p in names if p != s and p[:3] == base]
        out[s] = same_quote if len(same_quote) >= 2 else same_quote + same_base
    return out


def _class_peers(sds: dict[str, SymbolData]) -> dict[str, list[str]]:
    by = {}
    for s, sd in sds.items():
        by.setdefault(sd.spec.asset_class, []).append(s)
    peers = {}
    for s, sd in sds.items():
        ac = sd.spec.asset_class
        if ac is AssetClass.METAL or ac is AssetClass.US_INDEX:
            peers[s] = [p for p in by.get(ac, []) if p != s]
        elif ac is AssetClass.EU_INDEX:
            peers[s] = list(by.get(AssetClass.US_INDEX, []))
    return {s: p for s, p in peers.items() if p}


def _vote(rows: list[np.ndarray], min_corr: float, min_gain: float) -> tuple[int, int, int]:
    """(lag in hours agreed by >= 60% of ALL peers with data, else 0; n_votes; n_agree).

    Abstaining peers count against agreement: weakly co-moving pairs (EURGBP vs the EUR crosses) can produce two
    stray votes on one day, and those must not decide alone.
    """
    votes = []
    for c in rows:
        if np.isnan(c).all():
            continue
        c0, best = c[MAX_LAG_H], int(np.nanargmax(c))
        if c[best] >= min_corr and c[best] - (c0 if np.isfinite(c0) else -1) >= min_gain and best != MAX_LAG_H:
            votes.append(best - MAX_LAG_H)
        elif np.isfinite(c0) and c0 >= min_corr:
            votes.append(0)
    if len(votes) < 2 and not (len(votes) == 1 and len(rows) == 1):
        return 0, len(votes), 0
    vals, counts = np.unique(votes, return_counts=True)
    k = int(vals[np.argmax(counts)])
    agree = int(counts.max())
    n_peers = sum(1 for c in rows if not np.isnan(c).all())
    if k != 0 and agree / max(n_peers, 1) >= 0.6:
        return k, len(votes), agree
    return 0, len(votes), agree


def peer_alignment(sds: dict[str, SymbolData]) -> list[Finding]:
    """D8a: FX day by day on signed returns vs same-currency pairs; metals / indices week by week on |return| ranks."""
    if not sds:
        return []
    any_sd = next(iter(sds.values()))
    step = 60 // any_sd.profile.bar_minutes
    lo = min(sd.grid.index[0] for sd in sds.values())
    hi = max(sd.grid.index[-1] for sd in sds.values())
    idx = pd.date_range(lo, hi, freq=f"{any_sd.profile.bar_minutes}min")
    R = {s: sd.returns_grid().reindex(idx).to_numpy() for s, sd in sds.items()}
    out = []
    fx = [s for s, sd in sds.items() if sd.spec.asset_class is AssetClass.FX and len(s) == 6]
    day_key = (idx.tz_convert("America/New_York") + pd.Timedelta(hours=7)).normalize()   # trading day starts 17:00 NY
    out += _align(sds, R, idx, day_key, _fx_peers(fx), step, signed=True, min_corr=0.3, min_gain=0.1, min_n=150)
    week_key = idx.tz_convert("America/New_York").tz_localize(None).to_period("W-SAT").start_time
    out += _align(sds, R, idx, week_key, _class_peers(sds), step, signed=False, min_corr=0.2, min_gain=0.05, min_n=600)
    return out


def _align(sds, R, idx, key, peers, step, signed, min_corr, min_gain, min_n) -> list[Finding]:
    if not peers:
        return []
    names = set(peers) | {p for ps in peers.values() for p in ps}
    # |return| ranks over the full series are a monotone transform of |r|, so per-window correlations keep their meaning
    Z = {s: R[s] if signed else pd.Series(np.abs(R[s])).rank(pct=True).to_numpy() for s in names}
    key = pd.Index(key)
    bounds = np.r_[0, np.where(key[1:] != key[:-1])[0] + 1, len(key)]
    flagged: dict[str, list[tuple[int, int, int, int, int]]] = {}
    for s0, e0 in zip(bounds[:-1], bounds[1:], strict=True):
        X = {s: Z[s][s0:e0] for s in names if np.count_nonzero(~np.isnan(Z[s][s0:e0])) >= min_n}
        for s, ps in peers.items():
            if s not in X:
                continue
            rows = []
            for p in ps:
                if p not in X:
                    continue
                rows.append(_masked_corr(X[s], _lag_windows(Z[p], s0, e0, step), min_n // 2))
            k, nv, na = _vote(rows, min_corr, min_gain)
            if k:
                flagged.setdefault(s, []).append((s0, e0, k, nv, na))
    out = []
    for s, items in flagged.items():
        sd = sds[s]
        cur = None
        for s0, e0, k, nv, na in items + [(None, None, None, 0, 0)]:
            if cur and s0 is not None and k == cur[2] and s0 <= cur[1] + 1:
                cur = (cur[0], e0, k, cur[3] + nv, cur[4] + na)
                continue
            if cur:
                u0, u1 = idx[cur[0]], idx[cur[1] - 1] + sd.bar
                n = int(((sd.utc >= u0) & (sd.utc < u1)).sum())
                sev = Severity.CRITICAL if len(peers[s]) > 1 else Severity.MAJOR
                out.append(sd.from_utc(FindingKind.CLOCK_MISALIGNED, sev, "D8a", u0, u1, n, Action.DROP, lag_hours=cur[2],
                                       method="signed returns, same-currency pairs" if signed else "|return| ranks, class peers",
                                       votes=cur[3], agreeing=cur[4]))
            cur = (s0, e0, k, nv, na) if s0 is not None else None
    return out
