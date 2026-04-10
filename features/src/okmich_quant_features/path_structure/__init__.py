from typing import List, Optional
import warnings

import pandas as pd
import talib
import numpy as np

from ._auto_corr import auto_corr
from ._choppiness_index import choppiness_index
from ._hurst import hurst_exponent
from ._trend_strength import trend_strength, detrended_trend_strength
from ._variance_ratio import variance_ratio
from ._zigzag_density import zigzag_density
from ._stats import kendall_tau, runs_test, shannon_entropy, ljung_box_test, bds_test
from ..momentum import efficiency_ratio
from ..trend.misc import bollinger_band
from ..utils import logit_transformation


def core_path_structure_features(df: pd.DataFrame,
    # Auto-correlation parameters
    ac_window: int = 40, auto_corr_lags: List[int] = None, strict: bool = False,
    # Hurst exponent parameters
    hurst_window: int = 24, hurst_min_window: int = 24, hurst_min_lags: int = 2, hurst_max_lags: Optional[int] = None,
    hurst_detrend: str = "none", hurst_return_confidence: bool = False, hurst_confidence_level: float = 0.95,
    # Trend strength parameters
    trend_strength_window: int = 20, detrended_trend_strength_window: int = 20,
    # Variance ratio parameters
    variance_ratio_window: int = 20, variance_ratio_q: int = 2, variance_ratio_min_periods: Optional[int] = None,
    # Zigzag density parameters
    zigzag_window: Optional[int] = 20, zigzag_threshold: float = 0.02, zigzag_align: str = "causal",
    # Bollinger Band Width parameters
    bbw_window: int = 20,
    # Choppiness Index parameters
    choppiness_window: int = 14,
    # ATR parameters
    atr_window: int = 14,
    # Efficiency Ratio parameters
    efficiency_ratio_window: int = 20,
    # Statistical test parameters
    kendall_tau_window: int = 50, runs_test_window: int = 50, shannon_entropy_window: int = 50, shannon_entropy_bins: int = 20,
    ljung_box_window: int = 50, ljung_box_lags: int = 20, bds_window: int = 50, bds_max_dim: int = 2,
    # Column names
    high_col: str = "high", low_col: str = "low", close_col: str = "close") -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)

    # Extract OHLC columns
    high_price = df[high_col]
    low_price = df[low_col]
    close_price = df[close_col]

    # Compute log returns for statistical tests
    log_returns = np.log(close_price / close_price.shift(1))

    # Set default auto-correlation lags if not provided
    if auto_corr_lags is None:
        auto_corr_lags = [1, 5, 10, 20]

    # ==================== Auto-Correlation Features ====================
    for lag in auto_corr_lags:
        col_name = f"auto_corr_{ac_window}_{lag}"
        result[col_name] = auto_corr(close_price, window=ac_window, lag=lag)

    # ==================== Hurst Exponent ====================
    if hurst_return_confidence:
        hurst_val, hurst_lower, hurst_upper = hurst_exponent(close_price, window=hurst_window, min_window=hurst_min_window,
                                                             min_lags=hurst_min_lags, max_lags=hurst_max_lags,
                                                             detrend=hurst_detrend, return_confidence=True,
                                                             confidence_level=hurst_confidence_level)
        result[f"hurst_{hurst_window}"] = hurst_val
        result[f"hurst_{hurst_window}_lower_ci"] = hurst_lower
        result[f"hurst_{hurst_window}_upper_ci"] = hurst_upper
    else:
        result[f"hurst_{hurst_window}"] = hurst_exponent(close_price, window=hurst_window, min_window=hurst_min_window,
                                                         min_lags=hurst_min_lags, max_lags=hurst_max_lags,
                                                         detrend=hurst_detrend, return_confidence=False)

    # ==================== Trend Strength ====================
    ts_col = f"trend_strength_{trend_strength_window}"
    result[ts_col] = trend_strength(close_price, window=trend_strength_window)

    result[f"detrended_trend_strength_{detrended_trend_strength_window}"] = (
        detrended_trend_strength(close_price, window=detrended_trend_strength_window)
    )

    # ==================== Variance Ratio ====================
    result[f"variance_ratio_{variance_ratio_window}"] = variance_ratio(log_returns, window=variance_ratio_window,
                                                                       q=variance_ratio_q,
                                                                       min_periods=variance_ratio_min_periods)

    # ==================== Zigzag Density ====================
    zigzag_col = (
        f"zigzag_density_{zigzag_window}" if zigzag_window else "zigzag_density"
    )
    zigzag_dens, _ = zigzag_density(close_price, window=zigzag_window, threshold=zigzag_threshold, align=zigzag_align)
    result[zigzag_col] = zigzag_dens

    # ==================== Bollinger Band Width ====================
    _, _, _, _, bbw = bollinger_band(close_price, window=bbw_window)
    bbw_col = f"bbw_{bbw_window}"
    result[bbw_col] = bbw

    # ==================== Choppiness Index ====================
    result[f"choppiness_index_{choppiness_window}"] = choppiness_index(
        high_price, low_price, close_price, window=choppiness_window)

    # ==================== Average True Range ====================
    result[f"atr_{atr_window}"] = talib.ATR(high_price, low_price, close_price, timeperiod=atr_window)

    # ==================== Efficiency Ratio ====================
    er_col = f"efficiency_ratio_{efficiency_ratio_window}"
    result[er_col] = efficiency_ratio(close_price, window=efficiency_ratio_window)

    # ==================== Statistical Tests ====================
    # Kendall's tau (rolling)
    result[f"kendall_tau_{kendall_tau_window}"] = kendall_tau(close_price, window=kendall_tau_window)

    # Runs test (rolling window)
    try:
        result[f"runs_test_{runs_test_window}"] = runs_test(log_returns, window=runs_test_window)
    except (ValueError, RuntimeError) as e:
        if strict:
            raise
        warnings.warn(f"runs_test failed (window={runs_test_window}): {e}; filling with NaN.", stacklevel=2)
        result[f"runs_test_{runs_test_window}"] = np.nan

    # Shannon entropy (rolling window)
    try:
        result[f"shannon_entropy_{shannon_entropy_window}"] = shannon_entropy(
            log_returns, bins=shannon_entropy_bins, window=shannon_entropy_window)
    except (ValueError, RuntimeError) as e:
        if strict:
            raise
        warnings.warn(f"shannon_entropy failed (window={shannon_entropy_window}): {e}; filling with NaN.", stacklevel=2)
        result[f"shannon_entropy_{shannon_entropy_window}"] = np.nan

    # Ljung-Box test (rolling window)
    try:
        lb_stat, lb_pval = ljung_box_test(log_returns, lags=ljung_box_lags, window=ljung_box_window, n_jobs=-1)
        result[f"ljung_box_stat_{ljung_box_window}"] = lb_stat
        result[f"ljung_box_pvalue_{ljung_box_window}"] = lb_pval
    except (ValueError, RuntimeError) as e:
        if strict:
            raise
        warnings.warn(f"ljung_box_test failed (window={ljung_box_window}): {e}; filling with NaN.", stacklevel=2)
        result[f"ljung_box_stat_{ljung_box_window}"] = np.nan
        result[f"ljung_box_pvalue_{ljung_box_window}"] = np.nan

    # BDS test for nonlinear dependence (rolling window)
    try:
        bds_stat, bds_pval = bds_test(log_returns, max_dim=bds_max_dim, window=bds_window, n_jobs=-1)
        result[f"bds_stat_{bds_window}"] = bds_stat
        result[f"bds_pvalue_{bds_window}"] = bds_pval
    except (ValueError, RuntimeError) as e:
        if strict:
            raise
        warnings.warn(f"bds_test failed (window={bds_window}): {e}; filling with NaN.", stacklevel=2)
        result[f"bds_stat_{bds_window}"] = np.nan
        result[f"bds_pvalue_{bds_window}"] = np.nan

    return result


# Complete EDA workflow
def append_path_structure_features(data: pd.DataFrame, window: int = 20, choppiness_window: int = 14,
                                   ac_window: int = 40, auto_corr_lags: List[int] = None,
                                   he_window: int = None, vr_window: int = None, ts_window: int = None,
                                   zigzag_window: int = None, zigzag_threshold: float = 0.02,
                                   high_col: str = "high", low_col: str = "low", close_col: str = "close"):
    if auto_corr_lags is None:
        auto_corr_lags = [1, 5, 10, 20]

    # Resolve actual computation windows so column names match what was computed.
    _he_w = he_window if he_window else window
    _ts_w = ts_window if ts_window else window
    _vr_w = vr_window if vr_window else window
    _ch_w = choppiness_window if choppiness_window else window

    # auto_corr — column names use ac_window (actual computation window), not window
    for lag in auto_corr_lags:
        data[f"auto_corr_{ac_window}_{lag}"] = auto_corr(
            data[close_col], window=ac_window, lag=lag
        )
    if f"auto_corr_{ac_window}_1" in data.columns:
        data[f"auto_corr_{ac_window}_1_logit"] = logit_transformation(
            data[f"auto_corr_{ac_window}_1"]
        )
    if f"auto_corr_{ac_window}_5" in data.columns:
        data[f"auto_corr_{ac_window}_5_logit"] = logit_transformation(
            data[f"auto_corr_{ac_window}_5"]
        )
    # hurst exponent — column name reflects actual window used
    data[f"hurst_{_he_w}"] = hurst_exponent(data[close_col], window=_he_w, min_window=_he_w)
    # trend strength — column name reflects actual window used
    data[f"trend_strength_{_ts_w}"] = trend_strength(data[close_col], _ts_w)
    data[f"detrend_strength_{_ts_w}"] = detrended_trend_strength(data[close_col], window=_ts_w)
    # variance_ratio — must be applied to log returns, not price levels
    _vr_returns = np.log(data[close_col] / data[close_col].shift(1))
    data[f"variance_ratio_{_vr_w}"] = variance_ratio(_vr_returns, _vr_w)
    # zigzag_density
    data[f"zigzag_density_{zigzag_window}"], _ = zigzag_density(
        data[close_col], window=zigzag_window, threshold=zigzag_threshold
    )
    # bbw
    _, _, _, _, data[f"bbw_{window}"] = bollinger_band(data[close_col], window=window)
    data[f"bbw_{window}_logit"] = logit_transformation(data[f"bbw_{window}"])
    # choppiness
    data[f"choppiness_index_{_ch_w}"] = choppiness_index(
        data[high_col], data[low_col], data[close_col], window=_ch_w,
    )
    # atr — column name reflects actual timeperiod used (choppiness_window)
    data[f"atr_{_ch_w}"] = talib.ATR(
        data[high_col], data[low_col], data[close_col], timeperiod=_ch_w,
    )
    # efficiency_ratio
    data[f"efficiency_ratio_{window}"] = efficiency_ratio(data[close_col], window=window)
    data[f"efficiency_ratio_{window}_logit"] = logit_transformation(data[f"efficiency_ratio_{window}"])
    return data
