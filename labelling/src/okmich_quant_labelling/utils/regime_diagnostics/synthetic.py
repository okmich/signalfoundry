"""Synthetic FX-like 5m data for validating the test (the validation in README.md).

Weekday trading days on the 17:00 New York roll; t(3) shocks; persistent log-volatility; intraday seasonality; and
optional state-dependent drift, AR(1) coefficient and volatility multiplier. The state at bar t-1 drives the return
of bar t, so a label known at the close of t carries information about t+1 onward and nothing about bar t itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit
from scipy.signal import lfilter

from .clock import NEW_YORK


def fx_index(n_days: int, start: str = "2019-01-07", spacing: str = "5min") -> pd.DatetimeIndex:
    """UTC bar stamps for ``n_days`` weekday trading days, each running 17:00 New York to 17:00 New York."""
    days = pd.bdate_range(start, periods=n_days)
    opens = pd.DatetimeIndex(days - pd.Timedelta(days=1) + pd.Timedelta(hours=17)).tz_localize(NEW_YORK)
    step = pd.Timedelta(spacing)
    per_day = int(pd.Timedelta(days=1) / step)
    stamps = opens.tz_convert("UTC").asi8[:, None] + np.arange(per_day) * step.value
    return pd.DatetimeIndex(stamps.ravel(), tz="UTC")


def intraday_season(index: pd.DatetimeIndex) -> np.ndarray:
    """Volatility multiplier by New York hour: quiet Asia, London, and the London/New York overlap."""
    hour = index.tz_convert(NEW_YORK).hour.to_numpy()
    season = np.full(hour.size, 0.8)
    season[(hour >= 3) & (hour < 12)] = 1.4
    season[(hour >= 8) & (hour < 11)] = 1.9
    season[(hour >= 12) & (hour < 16)] = 1.1
    return season


def sticky_label(n: int, n_states: int, dwell: float, rng: np.random.Generator) -> np.ndarray:
    """Markov-like state codes with geometric run lengths of mean ``dwell``; each run picks a different state."""
    lengths = rng.geometric(1 / dwell, size=int(3 * n / dwell) + 10)
    shifts = rng.integers(1, n_states, size=lengths.size)
    states = np.cumsum(shifts) % n_states
    return np.repeat(states, lengths)[:n]


@njit(cache=True)
def _ar_filter(shocks: np.ndarray, phi: np.ndarray) -> np.ndarray:
    out = np.empty_like(shocks)
    out[0] = shocks[0]
    for t in range(1, shocks.size):
        out[t] = phi[t] * out[t - 1] + shocks[t]
    return out


def simulate_close(index: pd.DatetimeIndex, rng: np.random.Generator, states: np.ndarray | None = None,
                   drift=None, ar=None, vol_mult=None, base_vol: float = 1e-4, logvol_phi: float = 0.995,
                   logvol_sd: float = 0.4, season: bool = True) -> pd.Series:
    """Close series. ``drift`` (in units of the bar's volatility), ``ar`` and ``vol_mult`` are per-state sequences
    applied to bar t through the state at t-1."""
    n = len(index)
    innovations = rng.standard_normal(n) * logvol_sd * np.sqrt(1 - logvol_phi ** 2)
    scale = base_vol * np.exp(lfilter([1.0], [1.0, -logvol_phi], innovations))
    if season:
        scale = scale * intraday_season(index)
    prev = np.r_[0, states[:-1]] if states is not None else None
    if vol_mult is not None:
        scale = scale * np.asarray(vol_mult, dtype=float)[prev]
    shocks = rng.standard_t(3, n) / np.sqrt(3.0) * scale
    if drift is not None:
        shocks = shocks + np.asarray(drift, dtype=float)[prev] * scale
    phi = np.asarray(ar, dtype=float)[prev] if ar is not None else np.zeros(n)
    returns = _ar_filter(shocks, phi)
    return pd.Series(np.exp(np.cumsum(returns)), index=index, name="close")


def london_label(index: pd.DatetimeIndex) -> pd.Series:
    """1 during London hours (03:00-12:00 New York), else 0: a pure clock proxy."""
    hour = index.tz_convert(NEW_YORK).hour.to_numpy()
    return pd.Series(((hour >= 3) & (hour < 12)).astype(float), index=index, name="london")


def trailing_vol_tercile(close: pd.Series, window: int = 12, burn_in: float = 0.2) -> pd.Series:
    """Tercile of trailing realised volatility, with thresholds fixed on the first ``burn_in`` share of the data
    (NaN during the burn-in), so the label is causal."""
    log_close = np.log(close.to_numpy())
    sq = np.r_[0.0, np.cumsum(np.diff(log_close) ** 2)]
    pos = np.arange(log_close.size)
    trailing = np.where(pos >= window, np.sqrt(sq - sq[np.maximum(pos - window, 0)]), np.nan)
    cut = int(burn_in * trailing.size)
    lo, hi = np.nanquantile(trailing[:cut], [1 / 3, 2 / 3])
    label = np.select([trailing <= lo, trailing <= hi], [0.0, 1.0], 2.0)
    label[:cut] = np.nan
    label[~np.isfinite(trailing)] = np.nan
    return pd.Series(label, index=close.index, name="rv_tercile")
