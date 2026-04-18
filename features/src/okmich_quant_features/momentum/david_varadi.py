import numpy as np
import pandas as pd


def dvo(high: pd.Series, low: pd.Series, close: pd.Series, n_avg: int = 2, pct_lookback: int = 390,
        detrend: bool = True, n_dt: int = 390) -> pd.Series:
    """
    David Varadi Oscillator (DVO) — rank-based, detrended mean-reversion oscillator.

    Algorithm (3 steps):
        1. ratio      = close / hl2                              # detrended per-bar signal
        2. avg_ratio  = SMA(ratio, n_avg)                        # light smoothing
        2b. if detrend: avg_ratio -= SMA(avg_ratio, n_dt)        # remove multi-session drift
        3. dvo[t]     = rank of avg_ratio[t] vs prior N-1 values in [t-N+1 .. t-1],
                        scaled to [0, 100]

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC price series with a shared DatetimeIndex.
    n_avg : int, default 2
        SMA window for smoothing the close/hl2 ratio. n_avg=2 is the classic DV2.
    pct_lookback : int, default 390
        Rolling window for the percent rank. Default 390 = 1 trading week of 5-min bars
        (78 bars/day * 5 days). Primary tuning knob.
    detrend : bool, default True
        If True, subtract an SMA(n_dt) of the smoothed ratio before ranking.
        Removes intraday and multi-session drift. Recommended for intraday data.
    n_dt : int, default 390
        SMA window for detrending. Only used when detrend=True. Default matches pct_lookback.

    Returns
    -------
    pd.Series
        DVO values in [0, 100], centred at 50. Name: "DVO_{n_avg}_{pct_lookback}".

    Notes
    -----
    - Percent rank uses strict less-than on the N-1 prior values (denominator N-1),
      per the canonical DV2 definition.
    - NaNs in the rolling window are skipped when counting; the denominator becomes
      the number of non-NaN prior values.
    - This is distinct from `_percent_rank_hlc` used by `aggregate_m`, which ranks
      close against a stacked H/L/C pool rather than the smoothed ratio's own history.
    """
    if not isinstance(n_avg, (int, np.integer)) or n_avg < 1:
        raise ValueError(f"n_avg must be a positive integer, got {n_avg!r}")
    if not isinstance(pct_lookback, (int, np.integer)) or pct_lookback < 2:
        raise ValueError(f"pct_lookback must be an integer >= 2 (need at least 1 prior value), got {pct_lookback!r}")
    if detrend and (not isinstance(n_dt, (int, np.integer)) or n_dt < 1):
        raise ValueError(f"n_dt must be a positive integer when detrend=True, got {n_dt!r}")

    hl2 = (high + low) / 2.0
    # Guard against malformed bars where hl2 <= 0 (would inject inf/NaN into rolling stats).
    ratio = close / hl2.where(hl2 > 0)
    avg_ratio = ratio.rolling(window=n_avg, min_periods=n_avg).mean()

    if detrend:
        trend = avg_ratio.rolling(window=n_dt, min_periods=n_dt).mean()
        avg_ratio = avg_ratio - trend

    def _pct_rank(window: np.ndarray) -> float:
        current = window[-1]
        prior = window[:-1]
        valid = prior[~np.isnan(prior)]
        if len(valid) == 0 or np.isnan(current):
            return np.nan
        return float(np.sum(valid < current) / len(valid) * 100.0)

    result = avg_ratio.rolling(window=pct_lookback, min_periods=pct_lookback).apply(_pct_rank, raw=True)
    result.name = f"DVO_{n_avg}_{pct_lookback}"
    return result


def dv2(high: pd.Series, low: pd.Series, close: pd.Series, pct_lookback: int = 252) -> pd.Series:
    """
    Classic DV2 — DVO with n_avg=2 and no detrending, daily-style lookback.

    This is the original David Varadi DV2 as published for daily-bar mean reversion.
    Use `dvo(..., detrend=True)` for intraday 5-min bars instead.
    """
    return dvo(high, low, close, n_avg=2, pct_lookback=pct_lookback, detrend=False)


def _percent_rank_hlc(close_series, high_series, low_series, period):
    """
    Percent rank of current close vs prior HLC pool over `period` bars.

    For each bar t >= period - 1, compares close[t] against all (high, low, close)
    values from bars [t - period + 1, t], excluding close[t] itself. Returns the
    percentile rank in [0, 100].

    NaNs in the comparison pool are skipped (denominator = count of non-NaN prior
    values). If close[t] is NaN or all prior values are NaN, the output is NaN.

    Output bars before the first full window (index < period - 1) are NaN.
    """
    if not isinstance(period, (int, np.integer)) or period < 1:
        raise ValueError(f"period must be a positive integer, got {period!r}")

    close_arr = close_series.values
    high_arr = high_series.values
    low_arr = low_series.values

    n = len(close_arr)
    ranks = np.full(n, np.nan)
    hlc_stacked = np.column_stack([high_arr, low_arr, close_arr])

    # First full window of `period` bars ends at index `period - 1`.
    for i in range(period - 1, n):
        current_close = close_arr[i]
        if np.isnan(current_close):
            continue

        window = hlc_stacked[i - period + 1: i + 1]
        hlc_values = window.ravel()[:-1]  # exclude current close itself

        valid = hlc_values[~np.isnan(hlc_values)]
        if len(valid) == 0:
            continue

        ranks[i] = 100.0 * np.sum(valid < current_close) / len(valid)

    return ranks


def aggregate_m(ohlcv: pd.DataFrame, slow_period: int = 252, fast_period: int = 10, current_bar_weight: int = 60,
                trend_weight: int = 50, high_column: str = "high", low_column: str = "low",
                close_column: str = "close") -> pd.Series:
    """
    Calculate David Varadi's Aggregate M++ Mean Reversion Oscillator

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV dataframe with columns: 'open', 'high', 'low', 'close', 'volume'
    slow_period : int, default=252
        Larger lookback period (typically 252 for one trading year)
    fast_period : int, default=10
        Smaller lookback period for short-term signals
    current_bar_weight : int, default=60
        Weight for current bar vs previous bar (0-100%)
        Higher values = more responsive to current price action
    trend_weight : int, default=50
        Weight for trend component (0-100%)
        Higher values = more trend-following emphasis
        Lower values = more mean-reversion emphasis

    Returns:
    --------
    pd.Series
        Aggregate M++ values (0-100 scale)
        Values > 50: Bullish / Trend following favorable
        Values < 50: Bearish / Mean reversion favorable
    """
    if not isinstance(slow_period, (int, np.integer)) or slow_period < 1:
        raise ValueError(f"slow_period must be a positive integer, got {slow_period!r}")
    if not isinstance(fast_period, (int, np.integer)) or fast_period < 1:
        raise ValueError(f"fast_period must be a positive integer, got {fast_period!r}")
    if not 0 <= current_bar_weight <= 100:
        raise ValueError(f"current_bar_weight must be in [0, 100], got {current_bar_weight}")
    if not 0 <= trend_weight <= 100:
        raise ValueError(f"trend_weight must be in [0, 100], got {trend_weight}")

    # Case-tolerant column resolution. Exact match wins first. If only case-insensitive
    # match exists and multiple columns collide on the same lower-case key, raise rather
    # than silently pick one — collisions mean the caller's intent is ambiguous.
    case_groups: dict[str, list] = {}
    for c in ohlcv.columns:
        case_groups.setdefault(str(c).lower(), []).append(c)

    def _resolve(name: str):
        if name in ohlcv.columns:
            return name
        matches = case_groups.get(name.lower(), [])
        if not matches:
            raise ValueError(
                f"DataFrame must contain a '{name}' column (case-insensitive). "
                f"Got: {list(ohlcv.columns)}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous column resolution for '{name}': multiple case variants present {matches}. "
                "Pass the exact column name."
            )
        return matches[0]

    high_col = ohlcv[_resolve(high_column)]
    low_col = ohlcv[_resolve(low_column)]
    close_col = ohlcv[_resolve(close_column)]
    index = ohlcv.index

    hlc_slow = _percent_rank_hlc(close_col, high_col, low_col, slow_period)
    hlc_fast = _percent_rank_hlc(close_col, high_col, low_col, fast_period)

    # Weighted combination of slow and fast ranks. Stays NaN wherever either rank is NaN.
    m = (hlc_slow * trend_weight + hlc_fast * (100 - trend_weight)) / 100

    # Exponential smoothing with current_bar_weight.
    # Gap-tolerant: advance past NaN prefix to find the first valid seed, and treat
    # NaN inputs as honest gaps (output NaN at that bar, but preserve the last valid
    # smoothed state so the stream recovers when inputs recover).
    n = len(close_col)
    agg_m = np.full(n, np.nan)

    back_weight = (100 - current_bar_weight) / 100.0
    current_weight = current_bar_weight / 100.0

    seed_idx = max(slow_period, fast_period) - 1
    while seed_idx < n and np.isnan(m[seed_idx]):
        seed_idx += 1

    if seed_idx < n:
        last_valid = m[seed_idx]
        agg_m[seed_idx] = last_valid
        for i in range(seed_idx + 1, n):
            if np.isnan(m[i]):
                continue  # leave NaN, do not update last_valid
            last_valid = back_weight * last_valid + current_weight * m[i]
            agg_m[i] = last_valid

    return pd.Series(agg_m, index=index)
