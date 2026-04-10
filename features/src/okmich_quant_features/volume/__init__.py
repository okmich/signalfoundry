import numpy as np
import pandas as pd
from typing import Optional

from ._volume import (
    abnormal_volume,
    discretize_volume,
    volume_profile,
    volume_momentum,
    volume_return_divergence,
    volume_volatility_correlation,
    volume_bin_mfi_persistence,
    mfi_volume_bin_ratio,
    binned_mfi_delta,
    mfi,
    categorized_mfi_trend,
    volume_weighted_rate_of_change,
    volume_surge_rate_of_change,
    vwap_dev_momentum,
    vwap_adjusted_roc,
    tick_volume_phase_zscore,
    tick_volume_how_zscore,
    ad,
)
from ._mfi import market_facilitation_index, mfi_features


# ==========================================================================================================
# # #####################################   Volume Analysis Framework   ####################################
# ==========================================================================================================
#
# Core Philosophy: Volume confirms price action. These features help you distinguish between:
#   - Genuine moves (high volume, high MFI, strong correlation)
#   - False breakouts (low volume, poor MFI, weak correlation)
#   - Accumulation/distribution (abnormal volume, changing profile)
#   - Exhaustion points (volume divergences, climax activity)
#
# Optimal Combination: Use these volume features WITH your volatility features for complete market regime analysis:
#   - High volatility + high volume = genuine trend
#   - High volatility + low volume = false breakout
#   - Low volatility + high volume = accumulation
#   - Low volatility + low volume = consolidation


########################## Most Impactful Use Cases: ####################################
# Breakout Trading: Volume Momentum + Volume Profile combination
# Trend Following: Volume-Volatility Correlation for trend confirmation
# Reversal Trading: Volume-Return Divergence for exhaustion signals
# News/Event Trading: Abnormal Volume spikes for unexpected events
# Range-bound Markets: Low Volume Profile suggests consolidation periods


# # Market Facilitation Index (market_facilitation_index)
#     - Definition: Measures efficiency of price movement per unit volume (Bill Williams).
#     - High Impact Use:
#         - Detects genuine moves versus false breakouts.
#         - Identifies accumulation/distribution and exhaustion points.
#     - Key Insight:
#         - GREEN (Up MFI, Up Volume) → Genuine buying/selling.
#         - FADE (Up MFI, Down Volume) → Weak move, potential reversal.
#         - FAKE (Down MFI, Up Volume) → Traps and false breakouts.
#         - SQUAT (Down MFI, Down Volume) → Consolidation/accumulation.
#     - Best Markets: Intraday FX, Equities, Crypto (5min–1h).
#     - Applications:
#         - Breakout confirmation.
#         - Reversal/exhaustion detection.
#         - Position building during accumulation.
#
# # Discretize Volume (discretize_volume)
#     - Definition: Quantile-based binning of volume into discrete regimes.
#     - High Impact Use:
#         - Essential for MFI analysis and pattern recognition.
#         - Adapts to changing market volumes without assumptions.
#     - Key Insight:
#         - Quartile bins distinguish low, moderate, and high participation.
#     - Best Markets: All liquid markets.
#     - Applications:
#         - Volume clustering.
#         - Breakout confirmation.
#         - Liquidity assessment.
#
# # Volume Momentum (volume_momentum)
#     - Definition: Ratio of fast SMA to slow SMA of volume.
#     - High Impact Use:
#         - Detects acceleration/deceleration in volume relative to history.
#     - Key Insight:
#         - >1.2 → strong momentum, breakout confirmation.
#         - <0.8 → deceleration, potential exhaustion.
#     - Best Markets: Intraday FX, Equities, Crypto.
#     - Applications:
#         - Entry timing.
#         - Divergence detection.
#         - Trend validation.
#
# # Volume-Volatility Correlation (volume_volatility_correlation)
#     - Definition: Rolling correlation between volume and volatility.
#     - High Impact Use:
#         - Validates trend sustainability and market efficiency.
#     - Key Insight:
#         - >0.7 → strong trend.
#         - 0–0.3 → choppy/inefficient market.
#         - <0 → exhaustion or potential manipulation.
#     - Best Markets: FX, Equities, Crypto.
#     - Applications:
#         - Trend confirmation.
#         - Regime classification.
#         - Risk assessment.
#
# # Volume Profile (volume_profile)
#     - Definition: Rolling standard deviation of volume.
#     - High Impact Use:
#         - Identifies concentration vs. dispersion of volume.
#     - Key Insight:
#         - High → accumulation/distribution.
#         - Low → consolidation/indecision.
#     - Best Markets: FX, Futures, Crypto.
#     - Applications:
#         - Breakout anticipation.
#         - Range trading.
#         - Position building/reducing.
#
# # Volume-Return Divergence (volume_return_divergence)
#     - Definition: Efficiency of returns per unit volume and volatility.
#     - High Impact Use:
#         - Detects whether price moves are supported by volume.
#     - Key Insight:
#         - Strong positive → efficient bullish movement.
#         - Strong negative → efficient bearish movement.
#         - Divergence → exhaustion risk.
#     - Best Markets: FX, Equities, Crypto.
#     - Applications:
#         - Reversal trading.
#         - Momentum confirmation.
#         - Risk management.
#
# # Abnormal Volume (abnormal_volume)
#     - Definition: Z-score of current volume relative to historical average.
#     - High Impact Use:
#         - Detects unusual market activity and news-driven spikes.
#     - Key Insight:
#         - |Z| > 2 → abnormal event.
#         - >3 → extreme volume.
#     - Best Markets: FX, Equities, Crypto, Futures.
#     - Applications:
#         - News/event trading.
#         - Reversal signals.
#         - Gap prediction.
#
# # Volume-Weighted Rate of Change (volume_weighted_rate_of_change)
#     - Definition: ROC adjusted by participation strength (volume).
#     - High Impact Use:
#         - Filters false breakouts lacking volume confirmation.
#     - Key Insight:
#         - High VWROC → strong momentum with conviction.
#     - Best Markets: Equities, Indices, Crypto.
#     - Applications:
#         - Trend following.
#         - Breakout confirmation.
#
# # VWAP-Adjusted ROC (vwap_adjusted_roc)
#     - Definition: Price ROC relative to VWAP shift.
#     - High Impact Use:
#         - Highlights institutional buying/selling zones.
#     - Key Insight:
#         - ROC > VWAP shift → strong trend.
#         - ROC < VWAP shift → trend weakening.
#     - Best Markets: Equities, FX.
#     - Applications:
#         - Institutional footprint detection.
#         - Trend quality assessment.
#
# # VWAP Deviation Momentum (vwap_dev_momentum)
#     - Definition: Rolling change of VWAP deviation.
#     - High Impact Use:
#         - Measures crowding or exit momentum near average cost.
#     - Key Insight:
#         - Rising deviation → strong momentum away from cost basis.
#         - Falling deviation → price returning to cost.
#     - Best Markets: FX, Equities, Crypto.
#     - Applications:
#         - Entry/exit timing.
#         - Flow momentum detection.
#
# # Volume Surge ROC (volume_surge_rate_of_change)
#     - Definition: ROC during volume surge periods only.
#     - High Impact Use:
#         - Detects momentum bursts confirmed by strong participation.
#     - Key Insight:
#         - Filters low-volume noise.
#         - Highlights conviction moves.
#     - Best Markets: FX, Crypto, Futures.
#     - Applications:
#         - Breakout/reversal detection.
#         - Intraday quant strategies.
#
# # Money Flow Index (mfi)
#     - Definition: MFI oscillator combining price and volume (TA-Lib).
#     - High Impact Use:
#         - Detects momentum exhaustion or confirmation under participation.
#     - Key Insight:
#         - Extreme MFI → potential reversal.
#         - Rising MFI → confirmed momentum.
#     - Best Markets: Equities, FX, Crypto.
#     - Applications:
#         - Trend confirmation.
#         - Reversal detection.
#
# # Binned MFI Delta (binned_mfi_delta)
#     - Definition: Change in MFI within each volume bin.
#     - High Impact Use:
#         - Measures shifts in flow intensity under specific liquidity regimes.
#     - Key Insight:
#         - Large ΔMFI in high-volume bin → strong momentum shift.
#     - Best Markets: FX, Crypto.
#     - Applications:
#         - Volume-conditioned momentum detection.
#         - ML feature for regime classification.
#
# # MFI Volume Bin Ratio (mfi_volume_bin_ratio)
#     - Definition: Ratio of current MFI to average MFI in volume bin.
#     - High Impact Use:
#         - Detects abnormal flow strength relative to liquidity regime.
#     - Key Insight:
#         - Ratio > 1 → stronger-than-average flow.
#         - Ratio < 1 → weaker flow.
#     - Best Markets: Equities, FX, Futures.
#     - Applications:
#         - Hidden institutional footprint detection.
#         - Liquidity-adjusted momentum signals.
#
# # Categorized MFI Trend (categorized_mfi_trend)
#     - Definition: Directional MFI trend (+1 up, -1 down) per volume bin.
#     - High Impact Use:
#         - Encodes flow direction relative to liquidity conditions.
#     - Key Insight:
#         - Positive → uptrend under given volume regime.
#         - Negative → downtrend.
#     - Best Markets: FX, Crypto intraday.
#     - Applications:
#         - Regime labeling.
#         - ML feature generation.
#
# # Volume Bin MFI Persistence (volume_bin_mfi_persistence)
#     - Definition: Autocorrelation of MFI changes per volume bin.
#     - High Impact Use:
#         - Detects stable flow regimes and accumulation/distribution zones.
#     - Key Insight:
#         - High autocorrelation → persistent flow.
#         - Low autocorrelation → random/consolidating flow.
#     - Best Markets: FX, Futures, Commodities.
#     - Applications:
#         - Stable flow detection.
#         - Regime classification.
#


def core_volume_features(df: pd.DataFrame, fast_window: int = 5, slow_window: int = 20, correlation_window: int = 20,
                         profile_window: int = 20, abnormal_volume_window: int = 50, abnormal_z_threshold: float = 2.0,
                         vwroc_period: int = 14, vwap_roc_period: int = 14, vwap_dev_period: int = 20, volume_surge_period: int = 14,
                         volume_surge_multiplier: float = 1.5, mfi_period: int = 14, mfi_bins: int = 5, mfi_persistence_lag: int = 5,
                         ad_window: int = 24, volatility_series: Optional[pd.Series] = None, returns_series: Optional[pd.Series] = None,
                         vol_col: str = "tick_volume") -> pd.DataFrame:
    """
    Generate comprehensive volume features from an OHLCV dataframe.

    This function computes all core volume analysis features including momentum, correlation, profile, divergence,
    abnormal activity detection, VWAP-based metrics, and MFI variations.
    These features capture volume dynamics, efficiency, and market participation patterns.

    Features Generated
    ------------------
    Core Volume Metrics:
    - volume_bins: Discretized volume bins (0 to bins-1)
    - volume_momentum: Fast/slow SMA ratio
    - volume_profile: Rolling standard deviation
    - abnormal_volume: Z-score relative to historical average

    Volume-Price Relationships:
    - volume_volatility_corr: Rolling correlation with volatility
    - volume_return_divergence: Returns per unit volume and volatility
    - volume_weighted_roc: Volume-weighted rate of change
    - volume_surge_roc: ROC during volume surges only

    Accumulation/Distribution Features:
    - ad_zscore: Z-score of A/D line (statistically normalized)
    - ad_pct_rank: Percentile rank of A/D line (0 to 1)

    VWAP-Based Features:
    - vwap_adjusted_roc: Price ROC relative to VWAP shift
    - vwap_dev_momentum: Rate of change in VWAP deviation

    Money Flow Index (MFI) Features:
    - mfi: Standard MFI oscillator
    - binned_mfi_delta: MFI change within volume bins
    - mfi_volume_bin_ratio: MFI relative to bin average
    - categorized_mfi_trend: Directional MFI trend per bin
    - volume_bin_mfi_persistence: MFI autocorrelation per bin

    Notes
    -----
    - All features handle missing data gracefully.
    - Discretize_volume returns bins and bin_map; only bins are included in output.
    - If volatility_series is None, uses (high - low) as proxy.
    - If returns_series is None, computes log returns from close prices.
    - Recommended for FX, Crypto, Equities, and Futures across all timeframes.
    """
    result = pd.DataFrame(index=df.index)

    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df[vol_col]

    if volatility_series is None:
        volatility_series = high - low
    if returns_series is None:
        returns_series = np.log(close / close.shift(1))

    volume_bins, _ = discretize_volume(volume.values, bins=4)
    result["volume_bins"] = volume_bins
    result["volume_momentum"] = volume_momentum(volume, fast_window=fast_window, slow_window=slow_window)
    result["volume_profile"] = volume_profile(volume, window=profile_window)
    result["abnormal_volume"] = abnormal_volume(volume, window=abnormal_volume_window, z_threshold=abnormal_z_threshold)
    result["volume_volatility_corr"] = volume_volatility_correlation(volume, volatility_series, window=correlation_window)
    result["volume_return_divergence"] = volume_return_divergence(returns_series, volume, volatility_series)
    result["volume_weighted_roc"] = volume_weighted_rate_of_change(close, volume, period=vwroc_period)
    result["volume_surge_roc"] = volume_surge_rate_of_change(close, volume, period=volume_surge_period, surge_mult=volume_surge_multiplier)

    _, ad_w_zscore, ad_w_pct_rank = ad(df, method="williams", vol_col=vol_col, window=ad_window)
    _, ad_m_zscore, ad_m_pct_rank = ad(df, method="mt", vol_col=vol_col, window=ad_window)
    result["ad_w_zscore"] = ad_w_zscore
    result["ad_w_pct_rank"] = ad_w_pct_rank
    result["ad_m_zscore"] = ad_m_zscore
    result["ad_m_pct_rank"] = ad_m_pct_rank

    result["vwap_adjusted_roc"] = vwap_adjusted_roc(high, low, close, volume, period=vwap_roc_period)
    result["vwap_dev_momentum"] = vwap_dev_momentum(high, low, close, volume, period=vwap_dev_period)
    result["mfi"] = mfi(high, low, close, volume, period=mfi_period)
    result["binned_mfi_delta"] = binned_mfi_delta(high, low, close, volume, bins=mfi_bins, period=mfi_period)
    result["mfi_volume_bin_ratio"] = mfi_volume_bin_ratio(high, low, close, volume, bins=mfi_bins, period=mfi_period)
    result["categorized_mfi_trend"] = categorized_mfi_trend(high, low, close, volume, bins=mfi_bins, period=mfi_period)
    result["volume_bin_mfi_persistence"] = volume_bin_mfi_persistence(high, low, close, volume, bins=mfi_bins,
                                                                      period=mfi_period, lag=mfi_persistence_lag)
    return result


def mfi_volume_features(df: pd.DataFrame, include_classic_mfi: bool = True, include_advanced_features: bool = True,
                        feature_type: str = "both", rolling_window: int = 60, use_expanding: bool = True,
                        short_window: Optional[int] = None, long_window: Optional[int] = None, bin_percentiles: list = None,
                        fixed_bin_edges: Optional[list] = None, mfi_bin_percentiles: list = None,
                        fixed_mfi_bin_edges: Optional[list] = None,
                        open_col: str = "open", high_col: str = "high", low_col: str = "low", close_col: str = "close",
                        volume_col: str = "tick_volume") -> pd.DataFrame:
    """
    Generate comprehensive Market Facilitation Index (MFI) volume features.

    This function combines Bill Williams' classic MFI interpretation with advanced directional and directionless
    volume-price efficiency features. It provides a complete toolkit for analyzing market microstructure, order flow,
    and efficiency.

    Features Generated
    ------------------
    CLASSIC MFI FEATURES (if include_classic_mfi=True):
    • mfi_classic: Raw MFI (high-low)/volume
    • mfi_classic_log: Log-transformed MFI
    • mfi_pattern_code: Pattern codes (mfi_up__vol_down, etc.)
        - GREEN: mfi_up__vol_up (genuine pressure)
        - FADE: mfi_up__vol_down (weak move)
        - FAKE: mfi_down__vol_up (false breakout)
        - SQUAT: mfi_down__vol_down (accumulation)

    DIRECTIONLESS FEATURES (if include_advanced_features=True and feature_type='directionless' or 'both'):
    • volume_ratio: Current volume / rolling mean
    • mfi_raw, mfi_log: Raw and log-space MFI
    • mfi_norm, mfi_z: Normalized MFI and Z-score
    • mfi_delta, mfi_logchg: MFI momentum
    • mfi_rangeadj, mfi_volumeadj: Context-adjusted MFI

    DIRECTIONAL FEATURES (if include_advanced_features=True and feature_type='directional' or 'both'):
    • Instantaneous: dmfi, dfp, bsdi, eom, dir_eff_ratio
    • Cumulative: cum_dfp, cum_dmfi, cum_bsdi
    • Statistical: *_mean, *_z scores for regime detection
    • Normalized: norm_cum_bsdi, norm_cum_dfp, dominance_ratio
    • Momentum: flow_momentum_dfp, flow_momentum_bsdi

    Notes
    -----
    - Classic MFI is ideal for pattern recognition and discretionary trading
    - Advanced features are optimized for machine learning models
    - Combine both for comprehensive market microstructure analysis
    - Use fixed bin edges in production to ensure consistency with training data

    See Also
    --------
    - market_facilitation_index: Classic Bill Williams MFI implementation
    - mfi_features: Comprehensive MFI feature engineering
    """
    result = df[["close"]]

    # Set default percentiles if not provided
    if bin_percentiles is None:
        bin_percentiles = list(np.arange(0.0, 1.0, 0.05))  # vigintile
        bin_percentiles[0] = 1e-9
    if mfi_bin_percentiles is None:
        mfi_bin_percentiles = list(np.arange(0.0, 1.0, 0.05))  # vigintile
        mfi_bin_percentiles[0] = 1e-9

    if include_classic_mfi:
        mfi_raw, mfi_log, mfi_codes, vol_edges, mfi_edges = market_facilitation_index(
            high_prices=df[high_col],
            low_prices=df[low_col],
            volumes=df[volume_col],
            bin_percentiles=bin_percentiles,
            fixed_bin_edges=fixed_bin_edges,
            mfi_bin_percentiles=mfi_bin_percentiles,
            fixed_mfi_bin_edges=fixed_mfi_bin_edges,
        )

        result["mfi_classic"] = mfi_raw
        result["mfi_classic_log"] = mfi_log
        result["mfi_pattern_code"] = mfi_codes

        # Store bin edges as metadata (useful for production deployment)
        result.attrs["volume_bin_edges"] = vol_edges
        result.attrs["mfi_bin_edges"] = mfi_edges

    if include_advanced_features:
        advanced_df = mfi_features(
            df=df,
            feature_type=feature_type,
            rolling_window=rolling_window,
            use_expanding=use_expanding,
            short_window=short_window,
            long_window=long_window,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            volume_col=volume_col,
        )
        result = pd.concat([result, advanced_df], axis=1)
    return result
