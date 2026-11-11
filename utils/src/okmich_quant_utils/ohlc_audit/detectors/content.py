"""D6 repeated blocks, D7 two-level alternation, D9 spikes, D10 stale / zero volume, D11 spread."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..findings import Finding, group_positions, runs
from ..symbol_data import SymbolData
from ..enums import Action, FindingKind, Severity, SlotState

REPEAT_MIN_BARS = 4
REPEAT_MAX_LAG = 576            # 2 days of 5m bars: catches the 13.5 h replay and the 1 h splice


def repeated_blocks(sd: SymbolData) -> list[Finding]:
    """D6: >= 4 consecutive non-flat bars whose OHLC equals the bars L positions earlier (L <= 576)."""
    df = sd.df
    h = pd.util.hash_pandas_object(df[["open", "high", "low", "close"]], index=False).to_numpy()
    nonflat = (df["high"] > df["low"]).to_numpy()
    hits: list[tuple[int, int, int]] = []
    for lag in range(1, min(REPEAT_MAX_LAG, len(h) - 1) + 1):
        eq = (h[lag:] == h[:-lag]) & nonflat[lag:]
        if not eq.any():
            continue
        for a, b in runs(eq):
            if b - a >= REPEAT_MIN_BARS:
                hits.append((a + lag, b + lag, lag))
    if not hits:
        return []
    hits.sort()
    merged = [list(hits[0])]
    for a, b, lag in hits[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b, lag])
    out = []
    for a, b, lag in merged:
        tlag = sd.raw[a] - sd.raw[a - lag]
        out.append(sd.from_positions(FindingKind.REPEATED_BLOCK, Severity.CRITICAL, "D6", a, b, Action.DROP, bars=b - a, lag_bars=lag,
                                     time_lag=str(tlag), source_raw=str(sd.raw[a - lag])))
    return out


def prior_median(values: np.ndarray, utc: pd.DatetimeIndex, days: int = 1) -> np.ndarray:
    """Robust local scale without a rolling median: per-day medians, rolled over ``days`` days, from strictly earlier days."""
    day = utc.floor("D")
    v = pd.Series(np.asarray(values, dtype=float), index=day)
    med = v.groupby(level=0).median()
    prev = med.rolling(days, min_periods=1).median().shift(1)
    return prev.reindex(day).to_numpy()


def alternation(sd: SymbolData) -> list[Finding]:
    """D7 (per symbol): close alternates between two levels bar by bar. Severity is set at broker level (audit.py)."""
    c = pd.Series(sd.df["close"].to_numpy(dtype=float))
    d1, d2 = c.diff(), c.diff(2)
    scale = pd.Series(prior_median(d1.abs().to_numpy(), sd.utc))
    hit = (d1.abs() > 6 * scale) & (d2.abs() < 0.3 * d1.abs()) & (np.sign(d1) != np.sign(d1.shift(1)))
    dense = hit.rolling(6).sum() >= 3
    out = []
    for a, b in group_positions(np.where(dense.to_numpy())[0], max_gap=6):
        a0 = max(a - 5, 0)
        n_hits = int(hit.iloc[a0:b + 1].sum())
        x = float((d1.abs() / scale).iloc[a0:b + 1].max())
        out.append(sd.from_positions(FindingKind.INTERLEAVE, Severity.MINOR, "D7", a0, b + 1, Action.NONE, hits=n_hits, x_scale=round(x, 1)))
    return out


def spikes(sd: SymbolData) -> list[Finding]:
    """D9 (per symbol): a reverting close jump > 25x, or a wick > 40x, the local median range. Market events are merged later."""
    df = sd.df
    rng = (df["high"] - df["low"]).to_numpy(dtype=float)
    med = prior_median(rng, sd.utc)
    c = df["close"].to_numpy(dtype=float)
    dc = np.r_[np.nan, np.diff(c)]
    nxt = np.r_[dc[1:], np.nan]
    gap_before = np.r_[True, np.diff(sd.utc.asi8) != sd.bar.value]
    with np.errstate(invalid="ignore", divide="ignore"):
        jump = np.abs(dc) / med
        revert = (np.sign(dc) != np.sign(nxt)) & (np.abs(nxt) > 0.7 * np.abs(dc))
        body_hi = np.maximum(df["open"].to_numpy(dtype=float), c)
        body_lo = np.minimum(df["open"].to_numpy(dtype=float), c)
        wick = np.maximum(df["high"].to_numpy(dtype=float) - body_hi, body_lo - df["low"].to_numpy(dtype=float)) / med
    bad = ((jump > 25) & revert & ~gap_before) | (wick > 40)
    out = []
    for i in np.where(bad)[0]:
        out.append(sd.from_positions(FindingKind.SPIKE_SUSPECT, Severity.MINOR, "D9", int(i), int(i) + 1, Action.REFETCH,
                                     jump_x=round(float(np.nan_to_num(jump[i])), 1), wick_x=round(float(np.nan_to_num(wick[i])), 1)))
    return out


def stale_and_volume(sd: SymbolData, max_unchanged: int = 12) -> list[Finding]:
    """D10: close unchanged for more than ``max_unchanged`` bars in open slots; bars with zero tick volume."""
    out = []
    c = sd.df["close"].to_numpy(dtype=float)
    same = np.r_[False, c[1:] == c[:-1]]
    open_bar = np.isin(sd.bar_states(), [SlotState.OPEN.value, SlotState.THIN.value])
    for a, b in runs(same & open_bar):
        if b - a + 1 > max_unchanged:
            out.append(sd.from_positions(FindingKind.STALE_RUN, Severity.MINOR, "D10", a - 1, b, Action.FLAG, bars=b - a + 1))
    if "tick_volume" in sd.df:
        zero = sd.df["tick_volume"].to_numpy(dtype=float) == 0
        for a, b in group_positions(np.where(zero)[0], max_gap=12):
            out.append(sd.from_positions(FindingKind.ZERO_VOLUME, Severity.MINOR, "D10", a, b + 1, Action.FLAG, bars=int(zero[a:b + 1].sum())))
    return out


def spread(sd: SymbolData) -> list[Finding]:
    """D11 (MT5 ``spread`` only). Values are never judged by size: a blown-out spread (silver, news) is the market.

    - Negative spread is impossible: SPREAD_INVALID (MINOR, flag).
    - Zero spread may be real on a raw-spread account, or a recording quirk (FXPIG FX, May-Aug 2023): SPREAD_ZERO (INFO,
      flag), so a study can decide whether to trust it.
    Windows merge across gaps under 1 day.
    """
    if "spread" not in sd.df:
        return []
    s = sd.df["spread"].to_numpy(dtype=float)
    out = []
    for kind, sev, mask in ((FindingKind.SPREAD_INVALID, Severity.MINOR, s < 0), (FindingKind.SPREAD_ZERO, Severity.INFO, s == 0)):
        for a, b in group_positions(np.where(mask)[0], max_gap=sd.bars_per_day):
            out.append(sd.from_positions(kind, sev, "D11", a, b + 1, Action.FLAG, bars=int(mask[a:b + 1].sum())))
    return out
