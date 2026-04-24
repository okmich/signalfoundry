from typing import Optional

import pandas as pd

from ._adx import adx, plus_di, minus_di
from ._core_momentum import roc, roc_smoothed, momentum, macd, log_returns, momentum_acceleration, roc_velocity, \
    momentum_volatility_ratio, rolling_slope, stochastic, mean_adjusted_ratio, rsi, rolling_sharpe, jerk, williams_r, \
    ema_residual, trend_quality, detrended_price, vol_adj_mom_osc
from ._cross_sectional_momentum import imom, csz_mom, peer_rel_z_mom, idiomatic_intraday_mom
from ._efficiency_ratio import efficiency_ratio
from ._misc import rolling_high_proximity, high_proximity, session_high_low_pct, lagged_return_skip, \
    lagged_delta_returns, pr_skip, sustained_velocity, exponential_velocity, velocity_magnitude, velocity_consistency

from ._williamblau import true_strength_index, stochastic_momentum_index, slope_divergence_tsi, \
    directional_trend_index_blau, directional_efficiency_index

from .david_varadi import aggregate_m, aggregate_m_components, dvo, dv2


def core_momentum_features(df: pd.DataFrame, window: int=18, long_window:int=40,
                           # Core momentum parameters
                           roc_window: int = 14, roc_smoothed_window: int = 21, roc_signal_window: int = 9,
                           roc_velocity_window: int = 10, momentum_period: int = 10,
                           momentum_accel_short: int = 5, momentum_accel_long: int = 10,
                           jerk_window: int = 3,
                           # Oscillator parameters
                           stoch_fastk: int = 14,
                           stoch_slowk: int = 3,
                           # MACD parameters
                           macd_signal: int = 9,
                           # Volatility-adjusted parameters
                           mom_vol_ratio_period: int = 14, vol_adj_osc_period: int = 14,
                           # Efficiency ratio
                           efficiency_ratio_window: int = 10,
                           # ADX parameters
                           adx_period: int = 14,
                           # William Blau indicators
                           tsi_r: int = 25, tsi_s: int = 13, tsi_signal: int = 7,
                           sdtsi_slope_period: int = 3, sdtsi_method: str = "diff",
                           smi_k_period: int = 14, smi_r: int = 3, smi_s: int = 3, smi_signal: int = 3,
                           dti_q: int = 2, dti_r: int = 20, dti_s: int = 5, dti_u: int = 3,
                           dti_signal: Optional[int] = None,
                           dei_r: int = 14, dei_s: int = 14, dei_signal: int = 9,
                           # Miscellaneous parameters
                           high_prox_lookback: int = 288,
                           rolling_high_prox_lookback: int = 72,
                           session_window: int = 72,
                           lagged_return_lag: int = 12,
                           pr_skip_lag: int = 1,
                           exp_vel_lookback: int = 10,
                           exp_vel_alpha: float = 0.3,
                           # Aggregate M parameters
                           agg_m_slow: int = 288, agg_m_fast: int = 12, agg_m_current_weight: int = 60,
                           agg_m_trend_weight: int = 50,
                           # Column names
                           open_col: str = "open", high_col: str = "high", low_col: str = "low",
                           close_col: str = "close") -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)

    # Extract OHLC columns
    open_price = df[open_col]
    high_price = df[high_col]
    low_price = df[low_col]
    close_price = df[close_col]

    # ==================== Core Momentum Features ====================
    # Log returns (foundational)
    result["log_returns"] = log_returns(close_price, period=1)
    result["rolling_sharpe"] = rolling_sharpe(result["log_returns"], window=window)
    result["roc"] = roc(close_price, window=roc_window)
    roc_raw, roc_sig, roc_div = roc_smoothed(close_price, roc_window=roc_smoothed_window,
                                             signal_window=roc_signal_window)
    result["roc_raw"] = roc_raw
    result["roc_signal"] = roc_sig
    result["roc_divergence"] = roc_div

    result["roc_velocity"] = roc_velocity(close_price, window=roc_velocity_window)

    result["momentum"] = momentum(close_price, period=momentum_period)
    result["momentum_acceleration"] = momentum_acceleration(close_price, short=momentum_accel_short,
                                                            long=momentum_accel_long)

    # Jerk (3rd derivative)
    result["jerk"] = jerk(close_price, window=jerk_window)
    result["rolling_slope"] = rolling_slope(close_price, window=window)

    # ==================== Oscillators ====================
    # RSI
    result["rsi"] = rsi(close_price, period=window)

    # Stochastic
    stoch_k, stoch_sig = stochastic(close_price, high_price, low_price, fastk_period=stoch_fastk, slowk_period=stoch_slowk)
    result["stochastic_k"] = stoch_k
    result["stochastic_signal"] = stoch_sig

    # Williams %R
    result["williams_r"] = williams_r(high_price, low_price, close_price, period=window)

    # Mean-adjusted ratio
    result["mean_adjusted_ratio"] = mean_adjusted_ratio(close_price, n=window)
    result["ema_residual"] = ema_residual(close_price, period=window)
    result["detrended_price"] = detrended_price(close_price, period=window)
    macd_line, macd_signal_line, macd_hist = macd(close_price, fast=window, slow=long_window, signal=macd_signal)
    result["macd_line"] = macd_line
    result["macd_signal"] = macd_signal_line
    result["macd_hist"] = macd_hist

    # ==================== Volatility-Adjusted Momentum ====================
    result["momentum_volatility_ratio"] = momentum_volatility_ratio(close_price, high_price, low_price,
                                                                    period=mom_vol_ratio_period)
    result["trend_quality"] = trend_quality(close_price, high_price, low_price, period=window)
    result["vol_adj_mom_osc"] = vol_adj_mom_osc(df, n=vol_adj_osc_period)

    # ==================== Efficiency Ratio ====================
    result["efficiency_ratio"] = efficiency_ratio(close_price, window=efficiency_ratio_window)

    # ==================== ADX ====================
    result["adx"] = adx(high_price, low_price, close_price, period=adx_period)
    result["plus_di"] = plus_di(high_price, low_price, close_price, period=adx_period)
    result["minus_di"] = minus_di(high_price, low_price, close_price, period=adx_period)

    # ==================== William Blau Indicators ====================
    # True Strength Index + Slope Divergence
    tsi_val, tsi_sig, tsi_dif, tsi_slope, tsi_bull_div, tsi_bear_div = slope_divergence_tsi(
        close_price, r=tsi_r, s=tsi_s, signal=tsi_signal,
        slope_period=sdtsi_slope_period, method=sdtsi_method)
    result["tsi"] = tsi_val
    result["tsi_signal"] = tsi_sig
    result["tsi_diff"] = tsi_dif
    result["tsi_slope"] = tsi_slope
    result["tsi_bullish_div"] = tsi_bull_div
    result["tsi_bearish_div"] = tsi_bear_div

    # Stochastic Momentum Index
    smi_val, smi_sig, smi_dif = stochastic_momentum_index(high_price, low_price, close_price,
                                                          k_period=smi_k_period, r=smi_r, s=smi_s, signal=smi_signal,
                                                          as_percent=True)
    result["smi"] = smi_val
    result["smi_signal"] = smi_sig
    result["smi_diff"] = smi_dif

    # Directional Trend Index
    dti_val, dti_sig, dti_dif = directional_trend_index_blau(high_price, low_price, q=dti_q, r=dti_r, s=dti_s, u=dti_u,
                                                             signal=dti_signal, as_percent=True)
    result["dti"] = dti_val
    if dti_sig is not None:
        result["dti_signal"] = dti_sig
        result["dti_diff"] = dti_dif

    # Directional Efficiency Index
    dei_val, dei_sig, dei_dif = directional_efficiency_index(high_price, low_price, close_price,
                                                             r=dei_r, s=dei_s, signal=dei_signal, as_percent=True)
    result["dei"] = dei_val
    result["dei_signal"] = dei_sig
    result["dei_diff"] = dei_dif

    # ==================== Proximity & Range Metrics ====================
    result["high_proximity"] = high_proximity(close_price, lookback=high_prox_lookback)
    result["rolling_high_proximity"] = rolling_high_proximity(close_price, lookback=rolling_high_prox_lookback)
    result["session_high_low_pct"] = session_high_low_pct(high_price, low_price, close_price,
                                                          session_window=session_window)

    # ==================== Lagged & Skip Momentum ====================
    result["lagged_return"] = lagged_return_skip(df, lag=lagged_return_lag, skip=pr_skip_lag)
    result["pr_skip"] = pr_skip(df, lag=lagged_return_lag, skip=pr_skip_lag)
    result["lagged_delta_returns"] = lagged_delta_returns(close_price, lag=1)

    # ==================== Velocity-Based Momentum ====================
    result["sustained_velocity"] = sustained_velocity(close_price, lookback=window)
    result["exponential_velocity"] = exponential_velocity(close_price, lookback=exp_vel_lookback, alpha=exp_vel_alpha)
    result["velocity_magnitude"] = velocity_magnitude(close_price, lookback=window)
    result["velocity_consistency"] = velocity_consistency(close_price, lookback=window)

    # ==================== Aggregate Indicators ====================
    result["aggregate_m"] = aggregate_m(
        df, slow_period=agg_m_slow, fast_period=agg_m_fast, current_bar_weight=agg_m_current_weight,
        trend_weight=agg_m_trend_weight, high_column=high_col, low_column=low_col, close_column=close_col)

    return result
