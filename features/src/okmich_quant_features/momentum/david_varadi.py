import numpy as np
import pandas as pd


def _resolve_column(ohlcv: pd.DataFrame, name: str) -> str:
    if name in ohlcv.columns:
        return name
    case_groups: dict[str, list[str]] = {}
    for c in ohlcv.columns:
        case_groups.setdefault(str(c).lower(), []).append(c)
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


def _pct_rank_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Sliding percent rank of the last element vs the prior (window-1) elements.

    For each position t >= window-1:
        rank[t] = count(arr[t-window+1 .. t-1] < arr[t]) / (window-1) * 100

    NaN values in the prior window are excluded from numerator and denominator.
    Returns NaN when arr[t] is NaN or all prior elements are NaN.
    Positions t < window-1 are NaN (warm-up).
    """
    n = len(arr)
    result = np.full(n, np.nan)
    if n < window:
        return result

    # sliding_window_view is a zero-copy stride view: shape (n-window+1, window)
    wins = np.lib.stride_tricks.sliding_window_view(arr, window)
    current = wins[:, -1]   # last element of each window: (n-window+1,)
    prior = wins[:, :-1]    # prior window-1 elements: (n-window+1, window-1)

    # NaN < x evaluates to False in NumPy, so the less-than count naturally ignores NaN pool values.
    less = (prior < current[:, None]).sum(axis=1).astype(np.float64)

    if np.isnan(arr).any():
        valid = (~np.isnan(prior)).sum(axis=1)
        safe_valid = np.where(valid > 0, valid, 1)
        ranks = np.where((valid > 0) & ~np.isnan(current), less / safe_valid * 100.0, np.nan)
    else:
        ranks = less / (window - 1) * 100.0

    result[window - 1:] = ranks
    return result


def dvo(high: pd.Series, low: pd.Series, close: pd.Series, n_avg: int = 2, pct_lookback: int = 390,
        detrend: bool = True, n_dt: int = 390) -> pd.Series:
    """David Varadi Oscillator (DVO) — rank-based, detrended mean-reversion oscillator.

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
    - NaNs in the rolling window are skipped; the denominator becomes the count of
      non-NaN prior values.
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

    ranks = _pct_rank_1d(avg_ratio.to_numpy(dtype=float), pct_lookback)
    return pd.Series(ranks, index=avg_ratio.index, name=f"DVO_{n_avg}_{pct_lookback}")


def dv2(high: pd.Series, low: pd.Series, close: pd.Series, pct_lookback: int = 252) -> pd.Series:
    """Classic DV2 — DVO with n_avg=2 and no detrending, daily-style lookback.

    This is the original David Varadi DV2 as published for daily-bar mean reversion.
    Use `dvo(..., detrend=True)` for intraday 5-min bars instead.
    """
    return dvo(high, low, close, n_avg=2, pct_lookback=pct_lookback, detrend=False)


def _percent_rank_hlc(close_series: pd.Series, high_series: pd.Series, low_series: pd.Series, period: int) -> np.ndarray:
    """Percent rank of current close vs the HLC pool over `period` bars.

    For each bar t >= period-1, compares close[t] against the pool:
        { high[t-p+1..t], low[t-p+1..t], close[t-p+1..t-1] }
    i.e. all H and L values in the window plus all prior closes, excluding close[t] itself.
    Pool size = 3*period - 1. Returns the percentile rank in [0, 100].

    NaN values in the pool are excluded (denominator = count of non-NaN pool values).
    Returns NaN when close[t] is NaN or all pool values are NaN.
    Positions t < period-1 are NaN (warm-up).
    """
    if not isinstance(period, (int, np.integer)) or period < 1:
        raise ValueError(f"period must be a positive integer, got {period!r}")

    close_arr = close_series.to_numpy(dtype=float)
    high_arr = high_series.to_numpy(dtype=float)
    low_arr = low_series.to_numpy(dtype=float)
    n = len(close_arr)

    result = np.full(n, np.nan)
    if n < period:
        return result

    # Zero-copy stride views: shape (n-period+1, period)
    h_win = np.lib.stride_tricks.sliding_window_view(high_arr, period)
    l_win = np.lib.stride_tricks.sliding_window_view(low_arr, period)
    c_win = np.lib.stride_tricks.sliding_window_view(close_arr, period)

    current = c_win[:, -1]   # current close: (n-period+1,)
    prior_c = c_win[:, :-1]  # prior closes, excludes close[t]: (n-period+1, period-1)

    # NaN < x evaluates to False in NumPy, so NaN pool values are naturally excluded from the count.
    less_total = (
        (h_win < current[:, None]).sum(axis=1)
        + (l_win < current[:, None]).sum(axis=1)
        + (prior_c < current[:, None]).sum(axis=1)
    ).astype(np.float64)

    pool_size = 3 * period - 1
    has_nan = np.isnan(high_arr).any() or np.isnan(low_arr).any() or np.isnan(close_arr).any()

    if has_nan:
        valid = pool_size - (
            np.isnan(h_win).sum(axis=1)
            + np.isnan(l_win).sum(axis=1)
            + np.isnan(prior_c).sum(axis=1)
        )
        safe_valid = np.where(valid > 0, valid, 1)
        ranks = np.where((valid > 0) & ~np.isnan(current), less_total / safe_valid * 100.0, np.nan)
    else:
        ranks = less_total / pool_size * 100.0

    result[period - 1:] = ranks
    return result


def aggregate_m_components(ohlcv: pd.DataFrame, slow_period: int = 252, fast_period: int = 10,
                           current_bar_weight: int = 60, trend_weight: int = 50,
                           high_column: str = "high", low_column: str = "low",
                           close_column: str = "close") -> pd.DataFrame:
    """Compute all Aggregate M++ components as a DataFrame.

    Returns a DataFrame with four columns:
    - slow_rank : percent rank of close vs HLC pool over slow_period bars
    - fast_rank : percent rank of close vs HLC pool over fast_period bars
    - raw_m     : weighted blend of slow_rank and fast_rank, before EMA smoothing
    - agg_m     : exponentially smoothed raw_m — identical to aggregate_m() output

    Parameters are identical to aggregate_m(). aggregate_m() is a thin wrapper
    over this function that returns only the agg_m column.
    """
    if not isinstance(slow_period, (int, np.integer)) or slow_period < 1:
        raise ValueError(f"slow_period must be a positive integer, got {slow_period!r}")
    if not isinstance(fast_period, (int, np.integer)) or fast_period < 1:
        raise ValueError(f"fast_period must be a positive integer, got {fast_period!r}")
    if not 0 <= current_bar_weight <= 100:
        raise ValueError(f"current_bar_weight must be in [0, 100], got {current_bar_weight}")
    if not 0 <= trend_weight <= 100:
        raise ValueError(f"trend_weight must be in [0, 100], got {trend_weight}")

    high_col = ohlcv[_resolve_column(ohlcv, high_column)]
    low_col = ohlcv[_resolve_column(ohlcv, low_column)]
    close_col = ohlcv[_resolve_column(ohlcv, close_column)]
    index = ohlcv.index

    hlc_slow = _percent_rank_hlc(close_col, high_col, low_col, slow_period)
    hlc_fast = _percent_rank_hlc(close_col, high_col, low_col, fast_period)
    raw_m = (hlc_slow * trend_weight + hlc_fast * (100 - trend_weight)) / 100.0

    n = len(close_col)
    agg_m_arr = np.full(n, np.nan)

    seed_idx = max(slow_period, fast_period) - 1
    while seed_idx < n and np.isnan(raw_m[seed_idx]):
        seed_idx += 1

    if seed_idx < n:
        post_seed = raw_m[seed_idx:]
        if not np.isnan(post_seed).any() and current_bar_weight > 0:
            # Fast path: vectorised EMA via pandas ewm (Cython-backed).
            # ewm(alpha, adjust=False) computes y[0]=x[0], y[i]=alpha*x[i]+(1-alpha)*y[i-1],
            # which is identical to the gap-tolerant loop when there are no NaN gaps.
            # Requires alpha > 0 — current_bar_weight=0 is routed to the loop below.
            ema_vals = (pd.Series(post_seed).ewm(alpha=current_bar_weight / 100.0, adjust=False).mean().to_numpy())
            agg_m_arr[seed_idx:] = ema_vals
        else:
            # Slow path: gap-tolerant loop preserves EMA state across NaN inputs.
            alpha = current_bar_weight / 100.0
            beta = 1.0 - alpha
            last_valid = raw_m[seed_idx]
            agg_m_arr[seed_idx] = last_valid
            for i in range(seed_idx + 1, n):
                if np.isnan(raw_m[i]):
                    continue
                last_valid = beta * last_valid + alpha * raw_m[i]
                agg_m_arr[i] = last_valid

    return pd.DataFrame(
        {"slow_rank": hlc_slow, "fast_rank": hlc_fast, "raw_m": raw_m, "agg_m": agg_m_arr},
        index=index,
    )


def aggregate_m(ohlcv: pd.DataFrame, slow_period: int = 252, fast_period: int = 10, current_bar_weight: int = 60,
                trend_weight: int = 50, high_column: str = "high", low_column: str = "low",
                close_column: str = "close") -> pd.Series:
    """Calculate David Varadi's Aggregate M++ Mean Reversion Oscillator.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        OHLCV dataframe. Must contain high, low, and close columns (case-insensitive).
    slow_period : int, default 252
        Lookback for the slow HLC percent rank (typically one trading year of daily bars).
    fast_period : int, default 10
        Lookback for the fast HLC percent rank (short-term signal).
    current_bar_weight : int, default 60
        EMA weight for the current bar vs the prior smoothed value (0–100).
        Higher = more responsive to current price action.
    trend_weight : int, default 50
        Weight assigned to the slow rank when blending slow and fast ranks (0–100).
        Higher = more trend-following; lower = more mean-reversion.
    high_column, low_column, close_column : str
        Column name overrides. Resolved case-insensitively.

    Returns
    -------
    pd.Series
        Aggregate M++ values on a [0, 100] scale.
    """
    return aggregate_m_components(
        ohlcv, slow_period=slow_period, fast_period=fast_period,
        current_bar_weight=current_bar_weight, trend_weight=trend_weight,
        high_column=high_column, low_column=low_column, close_column=close_column,
    )["agg_m"]
