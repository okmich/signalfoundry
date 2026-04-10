from typing import List, Optional

import numpy as np
import pandas as pd

from ._atr import atr, ttr_ema_ratio, atr_ratio, atr_sma_ratio
from ._volatility import (
    volatility_ratio, parkinson_volatility, volatility_signature,
    garman_klass_volatility, rolling_volatility, realized_volatility,
    realized_volatility_for_windows, realized_volatility_with_bipower_jump_variations,
    rogers_satchell_volatility, yang_zhang_volatility, volume_weighted_volatility,
    realized_volatility_window_with_bipower_jump_variations,
    volatility_of_volatility, vov_normalized, volatility_term_structure,
    gap_risk_ratio, trend_quality_ratio, overnight_ratio, jump_detection,
)
from .quantile_based_volatility import quantile_based_volatility_labeling, optimize_quantile_based_volatility_labels


def core_volatility_features(
        df: pd.DataFrame,
        window: int = 20,
        long_window: int = 40,
        short_window: int = 5,
        # VoV parameters
        vov_window: int = 20,
        # ATR parameters
        atr_period: int = 14,
        atr_short_period: int = 7,
        atr_sma_period: int = 20,
        # Jump detection parameters
        jump_threshold: float = 3.0,
        # Realized volatility parameters (require DatetimeIndex)
        freq_minutes: int = 5,
        trading_hours_per_day: float = 24.0,
        trading_days_per_year: int = 252,
        rv_windows: Optional[List[int]] = None,
        # Column names
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: Optional[str] = None,
) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)

    open_price = df[open_col]
    high_price = df[high_col]
    low_price = df[low_col]
    close_price = df[close_col]

    # numpy arrays for ATR functions (talib-based, require ndarray)
    open_arr = open_price.values
    high_arr = high_price.values
    low_arr = low_price.values
    close_arr = close_price.values

    has_datetime_index = isinstance(df.index, pd.DatetimeIndex)

    # ==================== Close-to-close Volatility ====================
    result["rolling_vol"] = rolling_volatility(close_price, window=window)

    # ==================== OHLC Volatility Estimators ====================
    result["parkinson_vol"] = parkinson_volatility(high_price, low_price, window=window)
    result["garman_klass_vol"] = garman_klass_volatility(open_price, high_price, low_price, close_price, window=window)
    result["yang_zhang_vol"] = yang_zhang_volatility(open_price, high_price, low_price, close_price, window=window)
    result["rogers_satchell_vol"] = rogers_satchell_volatility(open_price, high_price, low_price, close_price,
                                                                window=window)

    # ==================== Volatility of Volatility ====================
    result["vov"] = volatility_of_volatility(high_price, low_price, window=window, vov_window=vov_window)
    result["vov_normalized"] = vov_normalized(high_price, low_price, window=window, vov_window=vov_window)

    # ==================== Volatility Term Structure ====================
    result["vol_term_structure"] = volatility_term_structure(high_price, low_price,
                                                              short_window=short_window, long_window=window)
    result["vol_term_structure_long"] = volatility_term_structure(high_price, low_price,
                                                                   short_window=window, long_window=long_window)

    # ==================== Volatility Ratios ====================
    result["gap_risk_ratio"] = gap_risk_ratio(open_price, high_price, low_price, close_price, window=window)
    result["trend_quality_ratio"] = trend_quality_ratio(open_price, high_price, low_price, close_price, window=window)
    result["overnight_ratio"] = overnight_ratio(high_price, low_price, close_price, window=window)

    # ==================== ATR Features ====================
    atr_vals, atr_norm = atr(high_arr, low_arr, close_arr, period=atr_period)
    result["atr"] = atr_vals
    result["atr_normalized"] = atr_norm
    result["atr_ratio"] = atr_ratio(high_arr, low_arr, close_arr,
                                     short_period=atr_short_period, long_period=long_window)
    result["atr_sma_ratio"] = atr_sma_ratio(high_arr, low_arr, close_arr,
                                              period=atr_period, sma_period=atr_sma_period)
    result["ttr_ema_ratio"] = ttr_ema_ratio(high_arr, low_arr, close_arr, ema_period=window)

    # ==================== Jump Detection ====================
    result["jump"] = jump_detection(close_price, window=window, threshold_sigma=jump_threshold)

    # ==================== Volume-Weighted Volatility ====================
    if volume_col is not None:
        result["volume_weighted_vol"] = volume_weighted_volatility(close_price, df[volume_col], window=window)

    # ==================== Realized Volatility (requires DatetimeIndex) ====================
    if has_datetime_index:
        if rv_windows is None:
            rv_windows = [freq_minutes * window, freq_minutes * long_window]

        rv_df = realized_volatility_window_with_bipower_jump_variations(
            close_price, windows=rv_windows, freq_minutes=freq_minutes,
            trading_hours_per_day=trading_hours_per_day,
            trading_days_per_year=trading_days_per_year,
        )
        for col in rv_df.columns:
            result[col] = rv_df[col]

        result["vol_signature"] = volatility_signature(
            close_price, short_window=rv_windows[0], long_window=rv_windows[-1],
            freq_minutes=freq_minutes,
        )
        result["vol_ratio"] = volatility_ratio(high_price, low_price, close_price,
                                                window=window, freq_minutes=freq_minutes)

    return result