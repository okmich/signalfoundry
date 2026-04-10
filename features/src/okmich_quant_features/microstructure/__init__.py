"""
Market Microstructure Features

This package provides advanced market microstructure features derived from OHLCV + spread data.
Unlike traditional volume analysis, these features focus on:
  - Institutional footprints and informed trading detection
  - Liquidity dynamics and transaction cost estimation
  - Information asymmetry and adverse selection
  - Order flow toxicity and market fragility

Theoretical Foundation:
    - Kyle (1985): Market microstructure with informed traders
    - Easley & O'Hara (1987): Probability of informed trading (PIN)
    - Glosten & Milgrom (1985): Bid-ask spreads and adverse selection
    - Corwin & Schultz (2012): High-low spread estimators

Module Structure:
    - _primitives: Core building blocks (beta, buy/sell volume, etc.)
    - order_flow: Order flow imbalance, VPIN, Kyle's lambda, VWAP
    - liquidity: Spread estimators, Amihud illiquidity, liquidity indices
    - depth: Order book depth proxies from OHLC
    - information: PIN, adverse selection, smart money detection
    - regime: Market regime features (correlations, Hurst, etc.)
    - price_structure: Microstructure price features
    - composites: High-level meta-features combining primitives

Key Differences from volume.* package:
    - volume.*: Traditional volume confirmation of price moves
    - microstructure.*: Institutional behavior, liquidity crises, information flow

Usage:
    >>> from okmich_quant_features.microstructure import beta_clv, buy_sell_volume
    >>> from okmich_quant_features.microstructure import vpin, kyles_lambda
    >>>
    >>> # Split volume into buy/sell using close location
    >>> beta = beta_clv(high, low, close)
    >>> buy_vol, sell_vol = buy_sell_volume(high, low, close, volume)
    >>>
    >>> # Detect toxic flow
    >>> toxicity = vpin(buy_vol, sell_vol, window=20)
"""

from typing import Optional

import pandas as pd

from ._primitives import beta_clv, buy_sell_volume, typical_price, normalized_spread
from .depth import bar_absorption_ratio, wick_imbalance, range_volume_depth, absorption_weighted_depth_score, \
    multi_bar_depth_pressure, stealth_trading_indicator
from .price_structure import intrabar_efficiency_ratio, realized_skewness, realized_kurtosis, distance_to_extremes, \
    range_compression_ratio, price_path_fractal_dimension, close_open_gap_analysis, return_spread_cross_correlation
from .liquidity import corwin_schultz_spread, roll_spread, effective_tick_ratio, liquidity_score, cs_spread, \
    amihud_illiquidity, realized_liquidity_premium, spread_zscore, spread_expansion_momentum, spread_volume_ratio, \
    liquidity_drought_index, depth_imbalance_proxy, \
    liquidity_commonality, liquidity_resilience, spread_volatility_elasticity
from .order_flow import vir, cvd, vwcl, volume_concentration, volume_entropy, cvd_price_divergence, vpin, trade_intensity, \
    vwap_rolling, vwap_anchored, vwap_accumulation, peer_rel_vir, vir_zscore, delta_vpin, vwap_std_bands, kyles_lambda, \
    kyles_lambda_zscore, signed_volume_run_length, volume_clock_acceleration, net_order_flow_impulse, order_flow_persistence
from .information import pin_proxy, adverse_selection_component, smart_money_confidence_index
from .composites import liquidity_adjusted_momentum, volume_price_divergence, informed_liquidity_pressure, \
    institutional_footprint_score, regime_fragility_index, supply_demand_pressure_differential, \
    predictive_liquidity_transition_score
from .regime import volatility_volume_correlation, spread_volume_correlation, return_autocorrelation_decay, \
    volume_return_asymmetry


__all__ = [
    # Primitives
    "beta_clv",
    "buy_sell_volume",
    "typical_price",
    "normalized_spread",
    # Depth Proxies
    "bar_absorption_ratio",
    "wick_imbalance",
    "range_volume_depth",
    "absorption_weighted_depth_score",
    "multi_bar_depth_pressure",
    "stealth_trading_indicator",
    # Price Structure
    "intrabar_efficiency_ratio",
    "realized_skewness",
    "realized_kurtosis",
    "distance_to_extremes",
    "range_compression_ratio",
    "price_path_fractal_dimension",
    "close_open_gap_analysis",
    "return_spread_cross_correlation",
    # Liquidity
    "corwin_schultz_spread",
    "roll_spread",
    "effective_tick_ratio",
    "liquidity_score",
    "cs_spread",
    "amihud_illiquidity",
    "realized_liquidity_premium",
    "spread_zscore",
    "spread_expansion_momentum",
    "spread_volume_ratio",
    "liquidity_drought_index",
    "depth_imbalance_proxy",
    "liquidity_commonality",
    "liquidity_resilience",
    "spread_volatility_elasticity",
    # Order Flow
    "vir",
    "cvd",
    "vwcl",
    "volume_concentration",
    "volume_entropy",
    "cvd_price_divergence",
    "vpin",
    "trade_intensity",
    "vwap_rolling",
    "vwap_anchored",
    "vwap_accumulation",
    "peer_rel_vir",
    "vir_zscore",
    "delta_vpin",
    "vwap_std_bands",
    "kyles_lambda",
    "kyles_lambda_zscore",
    "signed_volume_run_length",
    "volume_clock_acceleration",
    "net_order_flow_impulse",
    "order_flow_persistence",
    # Information Asymmetry
    "pin_proxy",
    "adverse_selection_component",
    "smart_money_confidence_index",
    # Composites
    "liquidity_adjusted_momentum",
    "volume_price_divergence",
    "informed_liquidity_pressure",
    "institutional_footprint_score",
    "regime_fragility_index",
    "supply_demand_pressure_differential",
    "predictive_liquidity_transition_score",
    # Regime Detection
    "volatility_volume_correlation",
    "spread_volume_correlation",
    "return_autocorrelation_decay",
    "volume_return_asymmetry",
    # Aggregate
    "core_microstructure_features"]


def core_microstructure_features(df: pd.DataFrame, window: int = 20,
                                 # Depth parameters
                                 absorption_threshold: float = 1.5,
                                 # Price structure parameters
                                 rcr_short_span: int = 5,
                                 # Order flow parameters
                                 vpin_window: int = 50, vwap_window: int = 20, vwap_ad_window: int = 10,
                                 vwap_n_std: float = 2.0, cvd_lookback: int = 10, volume_entropy_bins: int = 10,
                                 kyles_zscore_window: int = 20, bar_duration_minutes: int = 5, delta_vpin_lookback: int = 5,
                                 # Regime detection parameters
                                 ac_max_lag: int = 5, ac_window: int = 72, vol_vol_window: int = 10, vra_window: int = 40,
                                 # Information asymmetry parameters
                                 smci_rvol_threshold: float = 2.0,
                                 # Liquidity parameters (spread-based, only used when spread_col is provided)
                                 cs_spread_window: int = 2, spread_ema_span: int = 5, liquidity_resilience_window: int = 40,
                                 spread_vol_elasticity_window: int = 40, xcorr_window: int = 20, xcorr_lags: Optional[list] = None,
                                 # Column names
                                 open_col: str = "open", high_col: str = "high", low_col: str = "low", close_col: str = "close",
                                 volume_col: str = "tick_volume", spread_col: Optional[str] = "spread") -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)

    # Extract OHLCV
    open_price = df[open_col]
    high_price = df[high_col]
    low_price = df[low_col]
    close_price = df[close_col]
    volume = df[volume_col]

    # Spread data (optional)
    has_spread = spread_col is not None and spread_col in df.columns
    if has_spread:
        spread = df[spread_col]
        mid_price = (open_price + close_price) / 2
        # Disable spread features if spread has no meaningful variation (e.g. synthetic instruments)
        if spread.std() < 1e-8 or spread.isna().all():
            import warnings
            warnings.warn(
                f"Spread column '{spread_col}' has no meaningful variation — "
                "skipping spread-based microstructure features.",
                stacklevel=2,
            )
            has_spread = False

    # ==================== Primitives ====================
    result["beta_clv"] = beta_clv(high_price, low_price, close_price)
    buy_vol, sell_vol = buy_sell_volume(high_price, low_price, close_price, volume)
    result["buy_volume"] = buy_vol
    result["sell_volume"] = sell_vol

    # ==================== Depth Proxies ====================
    result["bar_absorption_ratio"] = bar_absorption_ratio(open_price, close_price, volume)
    result["wick_imbalance"] = wick_imbalance(high_price, low_price, open_price, close_price)
    rvd_df = range_volume_depth(high_price, low_price, volume, window=window)
    result["rvd"] = rvd_df["rvd"]
    result["rvd_z"] = rvd_df["rvd_z"]
    result["absorption_depth_score"] = absorption_weighted_depth_score(open_price, close_price, volume, window=window)
    result["multi_bar_depth_pressure"] = multi_bar_depth_pressure(open_price, close_price, volume, window=window,
                                                                   threshold=absorption_threshold)
    stealth_df = stealth_trading_indicator(high_price, low_price, close_price, volume, window=window)
    result["stealth_trading"] = stealth_df["ST"]
    result["stealth_direction"] = stealth_df["ST_direction"]

    # ==================== Price Structure ====================
    result["intrabar_efficiency_ratio"] = intrabar_efficiency_ratio(open_price, high_price, low_price,
                                                                     close_price, window=window)
    result["realized_skewness"] = realized_skewness(close_price, window=window)
    result["realized_kurtosis"] = realized_kurtosis(close_price, window=window)
    result["distance_to_extremes"] = distance_to_extremes(high_price, low_price, close_price, window=window)
    result["range_compression_ratio"] = range_compression_ratio(high_price, low_price,
                                                                 short_span=rcr_short_span, long_span=window)
    fd_df = price_path_fractal_dimension(open_price, high_price, low_price, close_price, window=window)
    result["fractal_dimension"] = fd_df["FD"]
    result["fractal_dimension_ema"] = fd_df["FD_ema"]
    result["delta_fractal_dimension"] = fd_df["delta_FD"]
    gap_df = close_open_gap_analysis(open_price, close_price)
    result["gap"] = gap_df["gap"]
    result["gap_fill_ratio"] = gap_df["gap_fill_ratio"]

    # ==================== Order Flow ====================
    result["vir"] = vir(high_price, low_price, close_price, volume)
    result["vir_zscore"] = vir_zscore(high_price, low_price, close_price, volume, window=window)
    result["cvd"] = cvd(high_price, low_price, close_price, volume, window=window)
    result["vwcl"] = vwcl(high_price, low_price, close_price, volume, window=window)
    result["volume_concentration"] = volume_concentration(volume, window=window)
    result["volume_entropy"] = volume_entropy(volume, window=window, n_bins=volume_entropy_bins)
    result["cvd_price_divergence"] = cvd_price_divergence(high_price, low_price, close_price, volume,
                                                           cvd_window=window, lookback=cvd_lookback)
    result["vpin"] = vpin(high_price, low_price, close_price, volume, window=vpin_window)
    result["delta_vpin"] = delta_vpin(high_price, low_price, close_price, volume,
                                      vpin_window=vpin_window, lookback=delta_vpin_lookback)
    result["trade_intensity"] = trade_intensity(high_price, low_price, volume, window=window)
    result["vwap_rolling"] = vwap_rolling(high_price, low_price, close_price, volume, window=vwap_window)
    result["vwap_accumulation"] = vwap_accumulation(high_price, low_price, close_price, volume,
                                                     vwap_window=vwap_window, ad_window=vwap_ad_window)
    vwap_upper, vwap_mid, vwap_lower = vwap_std_bands(high_price, low_price, close_price, volume,
                                                       window=vwap_window, n_std=vwap_n_std)
    result["vwap_upper"] = vwap_upper
    result["vwap_lower"] = vwap_lower
    result["kyles_lambda"] = kyles_lambda(high_price, low_price, close_price, volume, window=window)
    result["kyles_lambda_zscore"] = kyles_lambda_zscore(high_price, low_price, close_price, volume,
                                                         window=window, zscore_window=kyles_zscore_window)
    result["signed_volume_run_length"] = signed_volume_run_length(high_price, low_price, close_price, volume)
    vca_df = volume_clock_acceleration(volume, bar_duration_minutes=bar_duration_minutes, window=window)
    result["volume_clock_acceleration"] = vca_df["VCA"]
    result["volume_clock_jerk"] = vca_df["VCA_jerk"]
    ofi_df = net_order_flow_impulse(high_price, low_price, close_price, volume, window=window)
    result["ofi"] = ofi_df["OFI"]
    result["ofi_z"] = ofi_df["OFI_z"]
    result["order_flow_persistence"] = order_flow_persistence(high_price, low_price, close_price, volume, window=window)

    # ==================== Liquidity (OHLCV-only) ====================
    result["amihud_illiquidity"] = amihud_illiquidity(close_price, volume, window=window)
    result["effective_tick_ratio"] = effective_tick_ratio(high_price, low_price, volume, window=window)
    result["liquidity_score"] = liquidity_score(high_price, low_price, close_price, window=window)

    # ==================== Liquidity (Spread-based) ====================
    if has_spread:
        result["corwin_schultz_spread"] = corwin_schultz_spread(high_price, low_price, close_price,
                                                                 window=cs_spread_window)
        result["roll_spread"] = roll_spread(close_price, window=window)
        result["realized_liquidity_premium"] = realized_liquidity_premium(close_price, spread, mid_price, window=window)
        result["spread_zscore"] = spread_zscore(spread, mid_price, window=window)
        result["spread_expansion_momentum"] = spread_expansion_momentum(spread, mid_price, ema_span=spread_ema_span)
        result["spread_volume_ratio"] = spread_volume_ratio(spread, mid_price, volume)
        result["liquidity_drought_index"] = liquidity_drought_index(close_price, volume, spread, mid_price,
                                                                     window=window)
        result["depth_imbalance_proxy"] = depth_imbalance_proxy(high_price, low_price, close_price, spread, mid_price,
                                                                  window=window)
        liq_res_df = liquidity_resilience(spread, mid_price, window=liquidity_resilience_window)
        result["liquidity_resilience_phi"] = liq_res_df["phi"]
        result["liquidity_resilience_halflife"] = liq_res_df["half_life"]
        result["spread_volatility_elasticity"] = spread_volatility_elasticity(spread, mid_price, high_price,
                                                                               low_price,
                                                                               window=spread_vol_elasticity_window)
        svc_df = spread_volume_correlation(spread, mid_price, volume, window=window)
        result["rho_spread_volume"] = svc_df["rho_SV"]
        result["delta_rho_spread_volume"] = svc_df["delta_rho_SV"]
        xcorr_df = return_spread_cross_correlation(close_price, spread, mid_price, window=xcorr_window,
                                                   lags=xcorr_lags or [1, 2, 3])
        for col in xcorr_df.columns:
            result[f"ret_spread_{col}"] = xcorr_df[col]

    # ==================== Information Asymmetry ====================
    result["adverse_selection"] = adverse_selection_component(high_price, low_price, close_price, volume, window=window)
    if has_spread:
        result["pin_proxy"] = pin_proxy(high_price, low_price, close_price, volume, spread, mid_price, window=window)
        result["smart_money_confidence"] = smart_money_confidence_index(high_price, low_price, close_price, volume,
                                                                         spread, mid_price, window=window,
                                                                         rvol_threshold=smci_rvol_threshold)

    # ==================== Regime Detection ====================
    result["volatility_volume_corr"] = volatility_volume_correlation(high_price, low_price, volume,
                                                                       vol_window=vol_vol_window, window=window)
    result["return_autocorr_decay"] = return_autocorrelation_decay(close_price, max_lag=ac_max_lag, window=ac_window)
    result["volume_return_asymmetry"] = volume_return_asymmetry(close_price, volume, window=vra_window)

    # ==================== Composites ====================
    result["volume_price_divergence"] = volume_price_divergence(close_price, volume, window=window)
    result["institutional_footprint"] = institutional_footprint_score(open_price, high_price, low_price,
                                                                       close_price, volume, window=window)
    if has_spread:
        result["liquidity_adj_momentum"] = liquidity_adjusted_momentum(close_price, spread, mid_price,
                                                                         vol_window=window, momentum_window=window)
        result["informed_liquidity_pressure"] = informed_liquidity_pressure(high_price, low_price, close_price,
                                                                             volume, spread, mid_price, window=window)
        result["regime_fragility_index"] = regime_fragility_index(high_price, low_price, close_price, volume,
                                                                    spread, mid_price, window=window)
        result["supply_demand_pressure"] = supply_demand_pressure_differential(high_price, low_price, open_price,
                                                                                close_price, volume, spread,
                                                                                mid_price, window=window)
        result["liquidity_transition_score"] = predictive_liquidity_transition_score(high_price, low_price,
                                                                                       open_price, close_price,
                                                                                       volume, spread, mid_price,
                                                                                       window=window)

    return result
