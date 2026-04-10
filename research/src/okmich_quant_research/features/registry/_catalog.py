"""
Feature Catalog
===============
Domain-knowledge metadata for every feature-computing function in okmich_quant_features. Covers ~272 functions across 12 modules.

Shorthand used in this file:
    Relevance  : C=CRITICAL, H=HIGH, M=MEDIUM, L=LOW, N=NONE
    Horizon    : I=intraday, S=short, ME=medium, LG=long, A=any
    Regime     : TR=trending, RA=ranging, VO=volatile, LV=low_vol, CR=crisis
"""
from ._schema import FeatureEntry, CRITICAL as C, HIGH as H, MEDIUM as M, LOW as L, NONE as N, H_INTRADAY as I, \
    H_SHORT as S, H_MEDIUM as ME, H_LONG as LG, H_ANY as A, R_TRENDING as TR, R_RANGING as RA, \
    R_VOLATILE as VO, R_LOW_VOL as LV, R_CRISIS as CR


def _fe(name, module, st, desc, *, rr=M, ret=M, dr=L, hor=A, wbi=None, dir_=False, causal=True, ot="series",
        spread=False, vol=False, bench=False, notes=""):
    """Compact FeatureEntry constructor."""
    return FeatureEntry(name=name, module=module, signal_type=st, description=desc,
        regime_relevance=rr, return_relevance=ret, direction_relevance=dr, horizon=hor, works_best_in=wbi or [],
        directional=dir_, causal=causal, output_type=ot,
        needs_spread=spread, needs_volume=vol, needs_benchmark=bench, notes=notes)


# ─────────────────────────────────────────────────────────────────────────────
# MICROSTRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
_MICROSTRUCTURE_PRIMITIVES = [
    _fe("beta_clv",          "microstructure._primitives", "price_structure",
        "Close location within the high-low range (CLV-weighted)", rr=M, ret=L, dr=M, hor=I, wbi=[TR, RA], dir_=True, vol=True),
    _fe("buy_sell_volume",   "microstructure._primitives", "order_flow",
        "Split bar volume into buy-side and sell-side components via CLV", rr=H, ret=M, dr=H, hor=I, dir_=True, vol=True,
        ot="dataframe", notes="Returns DataFrame with buy_vol and sell_vol columns"),
    _fe("typical_price",     "microstructure._primitives", "price_structure",
        "Simple (H+L+C)/3 price proxy", rr=L, ret=L, dr=L, hor=I),
    _fe("normalized_spread", "microstructure._primitives", "liquidity",
        "Bid-ask spread proxy normalised by mid-price", rr=H, ret=L, dr=N, hor=I, wbi=[RA, CR], spread=True),
]

_MICROSTRUCTURE_DEPTH = [
    _fe("bar_absorption_ratio",          "microstructure.depth", "order_flow",
        "Absorption of selling pressure within the bar range", rr=H, ret=M, dr=M, hor=I, wbi=[TR, VO], vol=True),
    _fe("wick_imbalance",                "microstructure.depth", "price_structure",
        "Upper vs lower wick ratio — directional rejection strength", rr=M, ret=M, dr=H, hor=I, dir_=True),
    _fe("range_volume_depth",            "microstructure.depth", "liquidity",
        "Volume relative to price range — depth proxy", rr=M, ret=L, dr=L, hor=I, vol=True),
    _fe("absorption_weighted_depth_score","microstructure.depth", "order_flow",
        "Composite bar depth weighted by absorption quality", rr=H, ret=M, dr=M, hor=I, wbi=[TR, VO], vol=True),
    _fe("multi_bar_depth_pressure",      "microstructure.depth", "order_flow",
        "Cumulative directional depth pressure over multiple bars", rr=M, ret=H, dr=H, hor=S, wbi=[TR], dir_=True, vol=True),
    _fe("stealth_trading_indicator",     "microstructure.depth", "toxicity",
        "Detects institutional stealth trading via low-impact high-volume patterns", rr=H, ret=H, dr=H, hor=S, wbi=[TR], dir_=True, vol=True),
]

_MICROSTRUCTURE_PRICE_STRUCTURE = [
    _fe("intrabar_efficiency_ratio",        "microstructure.price_structure", "price_structure",
        "How efficiently price moves within the bar (path efficiency)", rr=H, ret=M, dr=L, hor=I, wbi=[TR, RA]),
    _fe("realized_skewness",               "microstructure.price_structure", "price_structure",
        "Rolling skewness of intrabar log-returns", rr=H, ret=M, dr=M, hor=S, wbi=[VO, CR], dir_=True),
    _fe("realized_kurtosis",               "microstructure.price_structure", "price_structure",
        "Rolling kurtosis (tail weight) of intrabar log-returns", rr=H, ret=L, dr=N, hor=S, wbi=[VO, CR]),
    _fe("distance_to_extremes",            "microstructure.price_structure", "price_structure",
        "Close proximity to rolling high and low extremes",
        rr=M, ret=M, dr=M, hor=S, dir_=True, ot="dataframe", notes="Returns dist_to_high and dist_to_low columns"),
    _fe("range_compression_ratio", "microstructure.price_structure", "regime",
        "Ratio of current range to rolling average range — detects compression", rr=H, ret=M, dr=L, hor=S, wbi=[LV, RA]),
    _fe("price_path_fractal_dimension",    "microstructure.price_structure", "regime",
        "Fractal dimension of the price path — complexity measure", rr=H, ret=L, dr=N, hor=ME, wbi=[TR, RA]),
    _fe("close_open_gap_analysis",         "microstructure.price_structure", "price_structure",
        "Gap characterisation: gap size normalised by ATR", rr=M, ret=M, dr=H, hor=I, wbi=[VO], dir_=True),
    _fe("return_spread_cross_correlation", "microstructure.price_structure", "information",
        "Rolling cross-correlation between returns and spread changes", rr=H, ret=M, dr=L, hor=S, wbi=[CR, VO], spread=True),
]

_MICROSTRUCTURE_LIQUIDITY = [
    _fe("corwin_schultz_spread",    "microstructure.liquidity", "liquidity",
        "Corwin & Schultz (2012) bid-ask spread estimator from high-low", rr=H, ret=L, dr=N, hor=I),
    _fe("roll_spread",              "microstructure.liquidity", "liquidity",
        "Roll (1984) serial-covariance bid-ask spread estimator", rr=H, ret=L, dr=N, hor=I),
    _fe("effective_tick_ratio",     "microstructure.liquidity", "liquidity",
        "Effective tick size relative to price level", rr=M, ret=L, dr=N, hor=I),
    _fe("liquidity_score",          "microstructure.liquidity", "liquidity",
        "Composite liquidity score from spread + volume + range", rr=H, ret=M, dr=N, hor=S, wbi=[RA, LV], vol=True),
    _fe("cs_spread",                "microstructure.liquidity", "liquidity",
        "Alternative Corwin-Schultz spread for daily/non-overlapping windows", rr=H, ret=L, dr=N, hor=I),
    _fe("amihud_illiquidity",       "microstructure.liquidity", "liquidity",
        "Amihud (2002) illiquidity ratio |return| / dollar volume", rr=H, ret=M, dr=N, hor=ME, vol=True),
    _fe("realized_liquidity_premium","microstructure.liquidity", "liquidity",
        "Return premium earned per unit of spread — liquidity cost", rr=M, ret=M, dr=L, hor=ME, spread=True),
    _fe("spread_zscore",            "microstructure.liquidity", "liquidity",
        "Z-score of current spread vs rolling mean", rr=H, ret=L, dr=N, hor=I),
    _fe("spread_expansion_momentum","microstructure.liquidity", "liquidity",
        "Rate of spread widening — early stress signal", rr=H, ret=M, dr=N, hor=S, wbi=[VO, CR]),
    _fe("spread_volume_ratio",      "microstructure.liquidity", "liquidity",
        "Spread relative to volume — normalised transaction cost", rr=M, ret=L, dr=N, hor=S, vol=True),
    _fe("liquidity_drought_index",  "microstructure.liquidity", "regime",
        "Volume scarcity index — detects liquidity drought", rr=C, ret=M, dr=N, hor=S, wbi=[CR, LV], vol=True),
    _fe("depth_imbalance_proxy",    "microstructure.liquidity", "order_flow",
        "Buy-sell depth imbalance from range-volume decomposition", rr=M, ret=M, dr=H, hor=I, dir_=True, vol=True),
    _fe("liquidity_commonality",    "microstructure.liquidity", "liquidity",
        "Rolling correlation of asset spread change vs benchmark spread change", rr=H, ret=L, dr=N, hor=ME, wbi=[CR], spread=True, bench=True),
    _fe("liquidity_resilience",     "microstructure.liquidity", "regime",
        "AR(1) coefficient and half-life of spread reversion", rr=C, ret=M, dr=N, hor=ME, spread=True,
        ot="dataframe", notes="Returns phi and half_life columns"),
    _fe("spread_volatility_elasticity","microstructure.liquidity", "regime",
        "Sensitivity of spread changes to volatility changes (OLS beta)", rr=H, ret=M, dr=N, hor=ME, wbi=[VO], spread=True),
]

_MICROSTRUCTURE_ORDER_FLOW = [
    _fe("vir", "microstructure.order_flow", "order_flow",
        "Volume Intensity Ratio — volume relative to typical bar volume", rr=H, ret=H, dr=H, hor=I, dir_=True, vol=True),
    _fe("cvd", "microstructure.order_flow", "order_flow",
        "Cumulative Volume Delta — running buy-minus-sell volume", rr=H, ret=H, dr=C, hor=S, wbi=[TR], dir_=True, vol=True),
    _fe("vwcl", "microstructure.order_flow", "order_flow",
        "Volume-Weighted Close Location — directional volume pressure", rr=M, ret=M, dr=H, hor=I, wbi=[TR], dir_=True, vol=True),
    _fe("volume_concentration",   "microstructure.order_flow", "volume_structure",
        "Concentration of volume in tail bins — activity clustering", rr=H, ret=M, dr=L, hor=S, vol=True),
    _fe("volume_entropy",         "microstructure.order_flow", "information",
        "Shannon entropy of volume distribution across bins", rr=H, ret=M, dr=N, hor=S, vol=True),
    _fe("cvd_price_divergence",   "microstructure.order_flow", "information",
        "Divergence between CVD trend and price trend", rr=H, ret=H, dr=H, hor=ME, wbi=[TR, VO], dir_=True, vol=True),
    _fe("vpin", "microstructure.order_flow", "toxicity",
        "Volume-Synchronized Probability of Informed Trading (Easley et al.)", rr=C, ret=H, dr=N, hor=S, wbi=[VO, CR], vol=True),
    _fe("trade_intensity", "microstructure.order_flow", "volume_structure",
        "Relative volume intensity per bar vs rolling mean", rr=H, ret=M, dr=N, hor=S, vol=True),
    _fe("vwap_rolling", "microstructure.order_flow", "price_structure",
        "Rolling VWAP — volume-weighted average price over window", rr=L, ret=M, dr=H, hor=S, dir_=True, vol=True),
    _fe("vwap_anchored", "microstructure.order_flow", "price_structure", "VWAP anchored from a specific bar index",
        rr=L, ret=M, dr=H, hor=A, dir_=True, vol=True,
        notes="Requires anchor_bar parameter; not suitable for automated pipelines — use vwap_rolling"),
    _fe("vwap_accumulation", "microstructure.order_flow", "order_flow",
        "Cumulative signed deviation from anchored VWAP", rr=M, ret=H, dr=H, hor=S, wbi=[TR], dir_=True, vol=True),
    _fe("peer_rel_vir", "microstructure.order_flow", "order_flow",
        "VIR relative to peer/benchmark instrument VIR",
        rr=M, ret=H, dr=H, hor=I, dir_=True, vol=True, bench=True),
    _fe("vir_zscore", "microstructure.order_flow", "order_flow", "Z-score normalised VIR",
        rr=M, ret=H, dr=H, hor=I, dir_=True, vol=True),
    _fe("delta_vpin", "microstructure.order_flow", "toxicity", "Rate-of-change of VPIN — toxicity acceleration",
        rr=H, ret=H, dr=H, hor=S, wbi=[VO, CR], dir_=True, vol=True),
    _fe("vwap_std_bands", "microstructure.order_flow", "price_structure", "VWAP ± N standard deviation bands",
        rr=M, ret=M, dr=H, hor=S, dir_=True, vol=True, ot="dataframe", notes="Returns upper, vwap, lower columns"),
    _fe("kyles_lambda", "microstructure.order_flow", "toxicity",
        "Kyle's (1985) lambda — price impact per unit of signed volume", rr=H, ret=H, dr=N, hor=ME, vol=True),
    _fe("kyles_lambda_zscore", "microstructure.order_flow", "toxicity",
        "Z-score normalised Kyle's lambda", rr=H, ret=H, dr=N, hor=ME, vol=True),
    _fe("signed_volume_run_length","microstructure.order_flow", "order_flow",
        "Consecutive bars of same-direction volume — persistence measure", rr=H, ret=H, dr=H, hor=S, wbi=[TR], dir_=True, vol=True),
    _fe("volume_clock_acceleration","microstructure.order_flow", "volume_structure",
        "Rate of change in volume pace — activity acceleration", rr=H, ret=H, dr=N, hor=S, wbi=[VO, CR], vol=True),
    _fe("net_order_flow_impulse", "microstructure.order_flow", "order_flow",
        "Short-window net buy-minus-sell volume momentum", rr=M, ret=H, dr=H, hor=S, wbi=[TR], dir_=True, vol=True),
    _fe("order_flow_persistence", "microstructure.order_flow", "order_flow",
        "Autocorrelation of signed volume — directional flow continuity",
        rr=H, ret=H, dr=H, hor=S, wbi=[TR], dir_=True, vol=True),
]

_MICROSTRUCTURE_INFORMATION = [
    _fe("pin_proxy", "microstructure.information", "toxicity",
        "Probability of Informed Trading proxy (Easley & O'Hara)", rr=C, ret=H, dr=N, hor=ME, wbi=[CR, VO], vol=True),
    _fe("adverse_selection_component","microstructure.information", "toxicity",
        "Adverse selection cost component of the bid-ask spread", rr=H, ret=H, dr=N, hor=ME, wbi=[CR], vol=True),
    _fe("smart_money_confidence_index","microstructure.information", "toxicity",
        "Composite index of institutional / smart-money directional confidence", rr=H, ret=H, dr=H, hor=ME, dir_=True, vol=True),
]

_MICROSTRUCTURE_COMPOSITES = [
    _fe("liquidity_adjusted_momentum", "microstructure.composites", "composite",
        "Price momentum scaled by liquidity cost (spread-adjusted)", rr=M, ret=H, dr=H, hor=ME, dir_=True, vol=True),
    _fe("volume_price_divergence", "microstructure.composites", "composite",
        "Divergence between volume delta trend and price trend", rr=H, ret=H, dr=H, hor=ME, wbi=[TR, VO], dir_=True, vol=True),
    _fe("informed_liquidity_pressure", "microstructure.composites", "composite",
        "Composite pressure from informed trading on liquidity", rr=H, ret=H, dr=H, hor=ME, wbi=[VO, CR], dir_=True, vol=True),
    _fe("institutional_footprint_score", "microstructure.composites", "composite",
        "Multi-component institutional activity composite", rr=C, ret=H, dr=H, hor=ME, dir_=True, vol=True),
    _fe("regime_fragility_index", "microstructure.composites", "composite",
        "Market fragility composite from vol, liquidity, and flow stress", rr=C, ret=M, dr=N, hor=ME, wbi=[CR, VO], vol=True),
    _fe("supply_demand_pressure_differential","microstructure.composites", "composite",
        "Net supply-demand pressure from four micro components", rr=M, ret=H, dr=H, hor=S, dir_=True, vol=True),
    _fe("predictive_liquidity_transition_score","microstructure.composites", "composite",
        "Predictive score for upcoming liquidity regime transition", rr=H, ret=H, dr=N, hor=ME, wbi=[CR, VO], vol=True),
]

_MICROSTRUCTURE_REGIME = [
    _fe("volatility_volume_correlation", "microstructure.regime", "regime",
        "Rolling correlation between volatility and volume", rr=H, ret=M, dr=N, hor=ME, vol=True),
    _fe("spread_volume_correlation",     "microstructure.regime", "regime",
        "Rolling correlation between spread and volume", rr=H, ret=M, dr=N, hor=ME, wbi=[CR, VO], spread=True, vol=True),
    _fe("return_autocorrelation_decay",  "microstructure.regime", "regime",
        "Decay rate of return autocorrelation — momentum vs mean-reversion regime", rr=C, ret=H, dr=H, hor=ME, dir_=True),
    _fe("volume_return_asymmetry",       "microstructure.regime", "regime",
        "Asymmetry of volume between up-bars and down-bars", rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, vol=True),
]

# ─────────────────────────────────────────────────────────────────────────────
# TIMOTHY MASTERS INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
_TM_MOMENTUM = [
    _fe("rsi", "timothymasters.momentum", "momentum",
        "Relative Strength Index with Wilder EMA smoothing", rr=M, ret=M, dr=H, hor=S, wbi=[RA], dir_=True),
    _fe("detrended_rsi", "timothymasters.momentum", "momentum",
        "RSI residual after linear detrending", rr=M, ret=M, dr=H, hor=S, dir_=True),
    _fe("stochastic", "timothymasters.momentum", "momentum", "Stochastic K and D oscillator (Masters variant)",
        rr=M, ret=M, dr=H, hor=S, wbi=[RA], dir_=True, ot="dataframe", notes="Returns K and D columns"),
    _fe("stoch_rsi", "timothymasters.momentum", "momentum", "Stochastic applied to RSI values",
        rr=M, ret=M, dr=H, hor=S, wbi=[RA], dir_=True, ot="dataframe", notes="Returns K and D columns"),
    _fe("ma_difference", "timothymasters.momentum", "momentum",
        "Normalised short-minus-long MA gap", rr=L, ret=M, dr=H, hor=S, wbi=[TR], dir_=True),
    _fe("macd", "timothymasters.momentum", "trend", "MACD line, signal, and histogram (Masters ATR-based variant)",
        rr=M, ret=M, dr=H, hor=ME, wbi=[TR], dir_=True, ot="dataframe", notes="Returns macd, signal, histogram columns"),
    _fe("ppo", "timothymasters.momentum", "momentum",
        "Percentage Price Oscillator", rr=L, ret=M, dr=H, hor=ME, dir_=True),
    _fe("price_change_osc","timothymasters.momentum", "momentum",
        "Average log-return ratio oscillator", rr=L, ret=M, dr=H, hor=S, dir_=True),
    _fe("close_minus_ma","timothymasters.momentum", "momentum",
        "Current close minus rolling mean of prior bars", rr=L, ret=M, dr=H, hor=S, wbi=[TR], dir_=True),
    _fe("price_intensity","timothymasters.momentum", "order_flow",
        "(Close-Open) / ATR — intrabar directional intensity", rr=M, ret=M, dr=H, hor=I, dir_=True),
    _fe("reactivity",    "timothymasters.momentum", "price_structure",
        "Gietzen's aspect-ratio indicator — range relative to ATR", rr=M, ret=M, dr=L, hor=S),
]

_TM_TREND = [
    _fe("linear_trend", "timothymasters.trend", "trend",
        "First Legendre polynomial coefficient — linear trend slope", rr=H, ret=M, dr=H, hor=ME, wbi=[TR], dir_=True),
    _fe("quadratic_trend", "timothymasters.trend", "trend",
        "Second Legendre polynomial coefficient — curvature", rr=H, ret=M, dr=H, hor=ME, dir_=True),
    _fe("cubic_trend", "timothymasters.trend", "trend",
        "Third Legendre polynomial coefficient — inflection", rr=M, ret=L, dr=M, hor=LG, dir_=True),
    _fe("linear_deviation", "timothymasters.trend", "price_structure",
        "Residual from linear Legendre fit — mean reversion potential", rr=H, ret=M, dr=L, hor=ME, wbi=[RA]),
    _fe("quadratic_deviation","timothymasters.trend", "price_structure",
        "Residual from quadratic Legendre fit", rr=H, ret=L, dr=L, hor=ME),
    _fe("cubic_deviation", "timothymasters.trend", "price_structure",
        "Residual from cubic Legendre fit", rr=M, ret=L, dr=N, hor=LG),
    _fe("adx", "timothymasters.trend", "regime",
        "Average Directional Index (Wilder's 3-phase smoothing)", rr=C, ret=M, dr=M, hor=ME, wbi=[TR, RA]),
    _fe("aroon_up", "timothymasters.trend", "trend",
        "Bars since highest high, normalised — bullish structure", rr=H, ret=M, dr=M, hor=ME, wbi=[TR]),
    _fe("aroon_down", "timothymasters.trend", "trend",
        "Bars since lowest low, normalised — bearish structure", rr=H, ret=M, dr=M, hor=ME, wbi=[TR]),
    _fe("aroon_diff", "timothymasters.trend", "trend",
        "Aroon Up minus Aroon Down — directional trend strength", rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True),
]

_TM_VARIANCE = [
    _fe("price_variance_ratio", "timothymasters.variance", "regime",
        "Ratio of price variance at two scales (F-CDF compressed)", rr=H, ret=M, dr=N, hor=ME),
    _fe("change_variance_ratio", "timothymasters.variance", "regime",
        "Ratio of change variance at two scales (F-CDF compressed)", rr=H, ret=M, dr=N, hor=ME, wbi=[VO, CR]),
]

_TM_VOLUME = [
    _fe("intraday_intensity", "timothymasters.volume", "order_flow",
        "(2C-H-L)/(H-L) * volume, MA'd — intraday directional pressure", rr=M, ret=M, dr=H, hor=I, dir_=True, vol=True),
    _fe("money_flow", "timothymasters.volume", "order_flow",
        "Intraday intensity normalised by mean volume", rr=M, ret=M, dr=H, hor=S, dir_=True, vol=True),
    _fe("price_volume_fit", "timothymasters.volume", "information",
        "OLS slope of log(close) on log(volume) — price-volume coupling", rr=H, ret=H, dr=M, hor=ME, vol=True),
    _fe("vwma_ratio", "timothymasters.volume", "price_structure",
        "VWAP / arithmetic mean price ratio — volume skew measure", rr=M, ret=M, dr=H, hor=S, dir_=True, vol=True),
    _fe("normalized_obv", "timothymasters.volume", "order_flow",
        "Signed volume balance normalised by total volume", rr=M, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, vol=True),
    _fe("delta_obv", "timothymasters.volume", "order_flow",
        "OBV difference over lag — acceleration of volume balance", rr=M, ret=H, dr=H, hor=S, wbi=[TR], dir_=True, vol=True),
    _fe("normalized_pvi", "timothymasters.volume", "volume_structure",
        "Log returns on rising-volume bars (smart money tracking)", rr=H, ret=M, dr=M, hor=ME, wbi=[TR, VO], vol=True),
    _fe("normalized_nvi", "timothymasters.volume", "volume_structure",
        "Log returns on falling-volume bars (reveals underlying trend)", rr=H, ret=M, dr=M, hor=ME, wbi=[RA, LV], vol=True),
    _fe("volume_momentum", "timothymasters.volume", "volume_structure",
        "Log ratio of short-window to long-window volume SMA", rr=H, ret=M, dr=N, hor=S, vol=True),
]

_TM_INFORMATION = [
    _fe("entropy", "timothymasters.information", "information",
        "Shannon entropy of log-return distribution", rr=H, ret=M, dr=N, hor=ME),
    _fe("mutual_information","timothymasters.information", "information",
        "Mutual information between returns and volume", rr=H, ret=H, dr=N, hor=ME, vol=True),
]

_TM_FTI = [
    _fe("fti_lowpass","timothymasters.fti", "trend", "FTI lowpass filter output at best frequency",
        rr=M, ret=M, dr=M, hor=ME, dir_=True),
    _fe("fti_best_width",  "timothymasters.fti", "regime",
        "Optimal FTI filter width for current price series", rr=H, ret=L, dr=N, hor=ME),
    _fe("fti_best_period", "timothymasters.fti", "regime",
        "Dominant cycle period identified by FTI", rr=H, ret=L, dr=N, hor=ME),
    _fe("fti_best_fti",    "timothymasters.fti", "trend",
        "Best FTI value — denoised trend signal", rr=M, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True),
]

# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────
_VOLATILITY = [
    _fe("atr", "volatility._atr", "volatility",
        "Average True Range — standard volatility unit", rr=H, ret=M, dr=N, hor=S),
    _fe("ttr_ema_ratio",      "volatility._atr", "volatility",
        "True range to EMA ratio — relative volatility", rr=H, ret=M, dr=N, hor=S, wbi=[VO]),
    _fe("atr_ratio",          "volatility._atr", "regime",
        "Fast ATR to slow ATR ratio — volatility regime shift", rr=H, ret=M, dr=N, hor=S, wbi=[VO]),
    _fe("atr_sma_ratio",      "volatility._atr", "volatility",
        "ATR to SMA ratio — volatility as fraction of price level", rr=M, ret=L, dr=N, hor=S),
    _fe("volatility_ratio",   "volatility._volatility", "regime",
        "Parkinson to realised vol ratio — trend vs chop detector", rr=C, ret=M, dr=N, hor=ME, wbi=[TR, RA]),
    _fe("parkinson_volatility","volatility._volatility", "volatility",
        "High-low range volatility estimator (Parkinson 1980)", rr=H, ret=M, dr=N, hor=S),
    _fe("volatility_signature","volatility._volatility", "regime",
        "Volatility term structure across multiple lags", rr=C, ret=H, dr=N, hor=ME,
        ot="dataframe", notes="Returns vol at each lag as separate column"),
    _fe("garman_klass_volatility","volatility._volatility", "volatility",
        "Efficient OHLC volatility estimator (Garman-Klass 1980)", rr=H, ret=M, dr=N, hor=S),
    _fe("rolling_volatility", "volatility._volatility", "volatility",
        "Rolling standard deviation of log-returns", rr=H, ret=M, dr=N, hor=S),
    _fe("realized_volatility", "volatility._volatility", "volatility",
        "Sum of squared log-returns over window", rr=H, ret=M, dr=N, hor=S),
    _fe("realized_volatility_for_windows","volatility._volatility", "regime",
        "Realised volatility computed at multiple window lengths",
        rr=H, ret=H, dr=N, hor=ME, ot="dataframe", notes="One column per window length"),
    _fe("realized_volatility_with_bipower_jump_variations","volatility._volatility", "regime",
        "Realised vol decomposed into bipower variation and jump variation",
        rr=H, ret=H, dr=N, hor=ME, wbi=[CR, VO], ot="dataframe", notes="Returns rv, bpv, jv columns"),
    _fe("realized_volatility_window_with_bipower_jump_variations","volatility._volatility", "regime",
        "Multi-window BPV/JV decomposition", rr=H, ret=H, dr=N, hor=ME, ot="dataframe"),
    _fe("rogers_satchell_volatility","volatility._volatility", "volatility",
        "Drift-neutral OHLC volatility estimator (best for trending markets)", rr=H, ret=M, dr=N, hor=S, wbi=[TR]),
    _fe("yang_zhang_volatility","volatility._volatility", "volatility",
        "Full OHLC estimator with overnight gap correction", rr=H, ret=M, dr=N, hor=S),
    _fe("volume_weighted_volatility","volatility._volatility", "volatility",
        "Volatility weighted by volume — emphasises active-session risk", rr=M, ret=M, dr=N, hor=S, vol=True),
    _fe("volatility_of_volatility","volatility._volatility", "regime",
        "Volatility of volatility — vol clustering and crisis signal", rr=C, ret=H, dr=N, hor=ME, wbi=[CR, VO]),
    _fe("vov_normalized",    "volatility._volatility", "regime",
        "Normalised volatility-of-volatility", rr=H, ret=H, dr=N, hor=ME, wbi=[VO]),
    _fe("volatility_term_structure","volatility._volatility", "regime",
        "Volatility across multiple timeframes — risk term structure",
        rr=C, ret=H, dr=N, hor=ME, ot="dataframe", notes="One column per timeframe"),
    _fe("gap_risk_ratio",    "volatility", "regime",
        "Yang-Zhang to Parkinson ratio — overnight gap risk component", rr=H, ret=M, dr=N, hor=ME, wbi=[VO, CR]),
    _fe("trend_quality_ratio","volatility", "regime",
        "Rogers-Satchell to Parkinson ratio — trend quality signal",
        rr=H, ret=M, dr=N, hor=ME, wbi=[TR, RA]),
    _fe("overnight_ratio",   "volatility", "regime",
        "Rolling vol to Parkinson ratio — overnight vs intraday risk", rr=M, ret=L, dr=N, hor=ME),
    _fe("jump_detection",    "volatility", "regime",
        "Jump indicator derived from bipower variation vs realised vol",
        rr=H, ret=H, dr=N, hor=I, wbi=[CR, VO]),
    _fe("quantile_based_volatility_labeling","volatility.quantile_based_volatility", "regime",
        "Discrete vol-regime labels (LOW/MEDIUM/HIGH) via quantile thresholds",
        rr=C, ret=L, dr=N, hor=ME, ot="array", notes="Returns categorical array; useful as regime target or input"),
]

# ─────────────────────────────────────────────────────────────────────────────
# PATH STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
_PATH_STRUCTURE = [
    _fe("auto_corr",          "path_structure", "regime",
        "Rolling autocorrelation of returns at a specified lag",
        rr=H, ret=H, dr=H, hor=ME, wbi=[RA], dir_=True),
    _fe("hurst_exponent",     "path_structure", "regime",
        "Hurst exponent: H<0.5=mean-reverting, H>0.5=trending, H≈0.5=random",
        rr=C, ret=H, dr=N, hor=LG, wbi=[TR, RA]),
    _fe("trend_strength",     "path_structure", "regime",
        "Regression-based trend strength — efficiency of price path",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True),
    _fe("detrended_trend_strength","path_structure", "regime",
        "Trend strength after removing the linear drift component",
        rr=H, ret=H, dr=H, hor=ME, dir_=True),
    _fe("variance_ratio",     "path_structure", "regime",
        "Variance ratio test statistic — random walk vs mean-reversion",
        rr=H, ret=H, dr=N, hor=ME, wbi=[TR, RA]),
    _fe("zigzag_density",     "path_structure", "regime",
        "Reversal density in a window — choppiness indicator",
        rr=H, ret=M, dr=N, hor=ME, wbi=[RA, VO]),
    _fe("choppiness_index",   "path_structure", "regime",
        "Choppiness Index: ~100=range-bound, ~0=strongly trending",
        rr=C, ret=M, dr=N, hor=ME, wbi=[TR, RA]),
    _fe("efficiency_ratio",   "path_structure", "regime",
        "Kaufman efficiency ratio — net displacement / total path length",
        rr=H, ret=M, dr=N, hor=ME, wbi=[TR, RA]),
    _fe("kendall_tau",        "path_structure", "regime",
        "Rolling Kendall rank correlation — monotonic trend test",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True),
    _fe("runs_test",          "path_structure", "regime",
        "Wald-Wolfowitz runs test statistic — randomness of return signs",
        rr=H, ret=M, dr=N, hor=ME),
    _fe("shannon_entropy",    "path_structure", "information",
        "Shannon entropy of binned return distribution",
        rr=H, ret=M, dr=N, hor=ME),
    _fe("ljung_box_test",     "path_structure", "regime",
        "Ljung-Box autocorrelation test — stat and p-value",
        rr=H, ret=H, dr=N, hor=ME,
        ot="dataframe", notes="Returns stat and pvalue columns"),
    _fe("bds_test", "path_structure", "regime", "BDS nonlinearity test — detects non-random structure",
        rr=H, ret=H, dr=N, hor=ME, ot="dataframe", notes="Returns stat and pvalue per dimension"),
]

# ─────────────────────────────────────────────────────────────────────────────
# VOLUME MODULE
# ─────────────────────────────────────────────────────────────────────────────
_VOLUME = [
    _fe("abnormal_volume",         "volume", "volume_structure",
        "Z-score of volume vs rolling mean — anomalous activity flag", rr=H, ret=M, dr=N, hor=I, wbi=[VO, CR], vol=True),
    _fe("volume_profile",          "volume", "volume_structure",
        "Rolling standard deviation of volume — activity dispersion", rr=M, ret=L, dr=N, hor=ME, vol=True),
    _fe("volume_momentum", "volume", "volume_structure",
        "Fast-to-slow SMA volume ratio — volume acceleration", rr=H, ret=M, dr=N, hor=S, vol=True),
    _fe("volume_return_divergence","volume", "information",
        "Return per unit volume — price efficiency signal", rr=H, ret=H, dr=H, hor=ME, wbi=[TR, VO], dir_=True, vol=True),
    _fe("volume_volatility_correlation","volume", "regime","Rolling correlation between volume and volatility",
        rr=H, ret=M, dr=N, hor=ME, wbi=[VO], vol=True),
    _fe("volume_bin_mfi_persistence","volume", "volume_structure",
        "MFI autocorrelation within each volume bin", rr=M, ret=M, dr=M, hor=ME, vol=True),
    _fe("mfi_volume_bin_ratio", "volume", "volume_structure",
        "MFI relative to bin-average MFI", rr=M, ret=M, dr=M, hor=S, vol=True),
    _fe("binned_mfi_delta", "volume", "volume_structure",
        "MFI change within volume bins — flow shift detector", rr=M, ret=H, dr=H, hor=S, dir_=True, vol=True),
    _fe("mfi", "volume", "volume_structure", "Money Flow Index oscillator (0-100)",
        rr=M, ret=M, dr=H, hor=S, wbi=[RA], dir_=True, vol=True),
    _fe("categorized_mfi_trend",   "volume", "volume_structure",
        "Directional MFI trend per volume bin category", rr=M, ret=M, dr=H, hor=S, dir_=True, vol=True),
    _fe("volume_weighted_rate_of_change","volume", "momentum",
        "Rate of change weighted by volume intensity", rr=M, ret=H, dr=H, hor=S, wbi=[TR], dir_=True, vol=True),
    _fe("volume_surge_rate_of_change","volume", "momentum",
        "ROC computed only on volume-surge bars", rr=M, ret=H, dr=H, hor=S, wbi=[VO], dir_=True, vol=True),
    _fe("vwap_dev_momentum",       "volume", "momentum", "Rate of change of deviation from VWAP",
        rr=M, ret=H, dr=H, hor=S, dir_=True, vol=True),
    _fe("vwap_adjusted_roc",       "volume", "momentum", "ROC adjusted for VWAP drift",
        rr=M, ret=H, dr=H, hor=S, wbi=[TR], dir_=True, vol=True),
    _fe("ad","volume", "order_flow", "Accumulation/Distribution (Williams or MT method)",
        rr=M, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, vol=True),
    _fe("market_facilitation_index","volume", "volume_structure",
        "BW MFI + pattern codes (Green/Fade/Fake/Squat)", rr=H, ret=M, dr=N, hor=I, vol=True,
        ot="dataframe", notes="Returns mfi value and pattern code columns"),
    _fe("mfi_features",            "volume", "volume_structure",
        "Advanced MFI feature engineering wrapper — multiple derived columns",
        rr=H, ret=M, dr=M, hor=ME, vol=True, ot="dataframe"),
]

# ─────────────────────────────────────────────────────────────────────────────
# MOMENTUM MODULE
# ─────────────────────────────────────────────────────────────────────────────
_MOMENTUM = [
    # Rate of Change
    _fe("log_returns","momentum", "momentum", "Log returns: ln(close[t] / close[t-period])",
        rr=L, ret=H, dr=H, hor=I, dir_=True),
    _fe("roc","momentum", "momentum", "Rate of change: (close[t] - close[t-n]) / close[t-n]",
        rr=L, ret=H, dr=H, hor=S, dir_=True),
    _fe("roc_smoothed", "momentum", "momentum", "ROC smoothed by an EMA signal line",
        rr=L, ret=H, dr=H, hor=S, dir_=True),
    _fe("momentum", "momentum", "momentum", "Classic price displacement: close[t] - close[t-period]",
        rr=L, ret=H, dr=H, hor=S, dir_=True),
    _fe("momentum_acceleration","momentum", "momentum", "Short-window minus long-window momentum",
        rr=M, ret=H, dr=H, hor=S, wbi=[TR], dir_=True),
    _fe("roc_velocity", "momentum", "momentum", "Acceleration of rate-of-change",
        rr=M, ret=H, dr=H, hor=S, wbi=[TR], dir_=True),
    _fe("jerk", "momentum", "momentum", "3rd derivative of price — rate of change of acceleration",
        rr=L, ret=M, dr=H, hor=S, dir_=True),
    _fe("rolling_slope", "momentum", "trend", "OLS regression slope of close over window",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True),
    # Oscillators
    _fe("rsi", "momentum", "momentum", "Relative Strength Index (Wilder EMA, momentum.py variant)",
        rr=M, ret=M, dr=H, hor=S, wbi=[RA], dir_=True),
    _fe("stochastic", "momentum", "momentum", "Stochastic K and D (momentum.py variant)",
        rr=M, ret=M, dr=H, hor=S, wbi=[RA], dir_=True, ot="dataframe"),
    _fe("williams_r","momentum", "momentum",
        "Williams %R oscillator (-100 to 0)", rr=M, ret=M, dr=H, hor=S, wbi=[RA], dir_=True),
    _fe("mean_adjusted_ratio", "momentum", "momentum", "Close / rolling MA ratio",
        rr=L, ret=M, dr=H, hor=S, dir_=True),
    _fe("ema_residual","momentum", "momentum",
        "Close minus EMA — deviation from trend", rr=L, ret=M, dr=H, hor=S, dir_=True),
    _fe("detrended_price",      "momentum", "momentum",
        "Detrended price oscillator", rr=M, ret=M, dr=H, hor=S, wbi=[RA], dir_=True),
    _fe("macd", "momentum", "trend", "MACD line and signal line (momentum.py variant)",
        rr=M, ret=M, dr=H, hor=ME, wbi=[TR], dir_=True, ot="dataframe"),
    # Volatility-adjusted
    _fe("momentum_volatility_ratio","momentum", "momentum",
        "Momentum divided by ATR — volatility-adjusted momentum", rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True),
    _fe("trend_quality", "momentum", "regime",
        "OLS slope divided by ATR — trend quality ratio", rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True),
    _fe("vol_adj_mom_osc","momentum", "momentum", "ROC normalised by ATR", rr=M, ret=H, dr=H, hor=ME, dir_=True),
    # Efficiency & directional strength
    _fe("efficiency_ratio","momentum", "regime",
        "Kaufman efficiency ratio (momentum.py variant)", rr=H, ret=M, dr=N, hor=ME, wbi=[TR, RA]),
    _fe("adx", "momentum", "regime", "Average Directional Index with +DI/-DI (momentum.py variant)",
        rr=C, ret=M, dr=M, hor=ME, wbi=[TR, RA], ot="dataframe", notes="Returns adx, plus_di, minus_di columns"),
    # William Blau
    _fe("true_strength_index", "momentum", "momentum", "Blau True Strength Index with signal line",
        rr=M, ret=H, dr=H, hor=ME, dir_=True, ot="dataframe", notes="Returns tsi and signal columns"),
    _fe("stochastic_momentum_index", "momentum", "momentum",
        "Blau Stochastic Momentum Index with signal", rr=M, ret=H, dr=H, hor=S, dir_=True, ot="dataframe"),
    _fe("directional_trend_index_blau",  "momentum", "trend",
        "Blau Directional Trend Index — smoothed directional bias", rr=H, ret=H, dr=H, hor=ME, dir_=True, ot="dataframe"),
    _fe("directional_efficiency_index",  "momentum", "regime",
        "Blau Directional Efficiency Index", rr=H, ret=H, dr=H, hor=ME, dir_=True, ot="dataframe"),
    _fe("slope_divergence_tsi", "momentum", "momentum",
        "TSI slope divergence — momentum divergence detector", rr=M, ret=H, dr=H, hor=ME, dir_=True),
    # Proximity & range
    _fe("high_proximity", "momentum", "momentum","Normalised distance of close below rolling high",
        rr=M, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True),
    _fe("rolling_high_proximity","momentum", "momentum",
        "Short-window proximity to rolling high", rr=M, ret=H, dr=M, hor=S, wbi=[TR], dir_=True),
    _fe("session_high_low_pct",  "momentum", "price_structure",
        "Close position within session high-low range", rr=M, ret=M, dr=H, hor=I, dir_=True),
    # Lagged / skip
    _fe("lagged_return_skip",    "momentum", "momentum",
        "Lagged return with skip (momentum factor style)", rr=L, ret=H, dr=H, hor=ME, dir_=True),
    _fe("pr_skip","momentum", "momentum",
        "Price ratio with skip period", rr=L, ret=H, dr=H, hor=ME, dir_=True),
    _fe("lagged_delta_returns",  "momentum", "momentum",
        "Change in returns over a lag", rr=L, ret=H, dr=H, hor=S, dir_=True),
    # Velocity
    _fe("sustained_velocity",    "momentum", "momentum",
        "Average velocity (dP/dt) over lookback window", rr=M, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True),
    _fe("exponential_velocity", "momentum", "momentum",
        "EWMA velocity — exponentially weighted dP/dt", rr=M, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True),
    _fe("velocity_magnitude",    "momentum", "momentum",
        "Absolute velocity — speed regardless of direction", rr=M, ret=M, dr=N, hor=ME),
    _fe("velocity_consistency",  "momentum", "regime",
        "Coefficient of variation of velocity — trend smoothness", rr=H, ret=M, dr=N, hor=ME, wbi=[TR, RA]),
    # Aggregate & cross-sectional
    _fe("aggregate_m","momentum", "composite",
        "David Varadi's Aggregate M++ multi-timeframe momentum", rr=L, ret=H, dr=H, hor=ME, dir_=True),
    _fe("rolling_sharpe","momentum", "momentum",
        "Rolling Sharpe ratio of returns", rr=M, ret=M, dr=H, hor=ME, dir_=True),
    _fe("imom","momentum", "momentum",
        "Intra-sectional momentum", rr=L, ret=H, dr=H, hor=ME, dir_=True),
    _fe("csz_mom", "momentum", "momentum",
        "Cross-sectional z-score momentum", rr=L, ret=H, dr=H, hor=ME, dir_=True, bench=True),
    _fe("peer_rel_z_mom", "momentum", "momentum",
        "Peer-relative z-score momentum", rr=L, ret=H, dr=H, hor=ME, dir_=True, bench=True),
    _fe("idiomatic_intraday_mom","momentum", "momentum",
        "Idiomatic intraday momentum signal", rr=L, ret=H, dr=H, hor=I, dir_=True),
]

# ─────────────────────────────────────────────────────────────────────────────
# TREND MODULE
# ─────────────────────────────────────────────────────────────────────────────
_TREND = [
    # Causal labeling functions (produce 1/-1/0 signals)
    _fe("continuous_trend_labeling", "trend", "trend",
        "State-machine trend label: +1 up, -1 down, 0 neutral", rr=H, ret=M, dr=H, hor=S, dir_=True),
    _fe("continuous_ma_trend_labeling", "trend", "trend",
        "MA-deviation trend label: +1/-1/0/NaN", rr=H, ret=M, dr=H, hor=S, wbi=[TR], dir_=True),
    _fe("zscore_trend_labeling","trend", "trend","Z-score threshold trend label: +1/-1/0", rr=H, ret=M, dr=H, hor=S, dir_=True),
    _fe("trend_persistence_labeling",   "trend", "trend",
        "Volatility-adjusted directional drift label", rr=H, ret=H, dr=H, hor=ME, dir_=True),
    # Supplementary indicators
    _fe("bollinger_band", "trend", "price_structure", "Bollinger Bands: upper/mid/lower/%B/width",
        rr=H, ret=M, dr=M, hor=S, wbi=[RA], ot="dataframe", notes="Returns 5 columns: upper, mid, lower, pct_b, width"),
    _fe("cci","trend", "momentum", "Commodity Channel Index oscillator", rr=M, ret=M, dr=H, hor=S, dir_=True),
    _fe("fibonacci_range", "trend", "price_structure",
        "Fibonacci support/resistance levels from OHLC range", rr=M, ret=M, dr=M, hor=ME, wbi=[RA], ot="dataframe"),
    _fe("heiken_ashi", "trend", "price_structure", "Heikin Ashi candle values (smoothed OHLC)",
        rr=M, ret=M, dr=H, hor=I, dir_=True, ot="dataframe", notes="Returns HA open, high, low, close, flag columns"),
    _fe("heiken_ashi_momentum", "trend", "momentum",
        "Momentum of Heikin Ashi close vs open", rr=M, ret=H, dr=H, hor=S, wbi=[TR], dir_=True),
    _fe("heiken_ashi_momentum_advanced","trend", "momentum",
        "Advanced HA momentum with smoothing", rr=M, ret=H, dr=H, hor=S, wbi=[TR], dir_=True, ot="dataframe"),
    _fe("precision_trend", "trend", "trend",
        "Precision Trend indicator", rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True),
    _fe("trading_the_trend", "trend", "trend",
        "Trading The Trend (TTT) indicator", rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True),
    _fe("ttm_trend", "trend", "trend", "TTM Trend indicator", rr=M, ret=M, dr=H, hor=S, dir_=True),
]

# ─────────────────────────────────────────────────────────────────────────────
# TIMOTHY MASTERS — CROSS-MARKET (PAIRED) INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
_TM_CROSS_MARKET = [
    _fe("correlation", "timothymasters.cross_market", "information",
        "Rolling Spearman rank correlation × 50 between two aligned markets",
        rr=H, ret=M, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True,
        notes="Inputs must be date-aligned. Range ~[-50, 50]."),
    _fe("delta_correlation", "timothymasters.cross_market", "information",
        "Change in rolling Spearman rank correlation over a lag period",
        rr=H, ret=M, dr=H, hor=S, wbi=[TR, VO], dir_=True, bench=True,
        notes="Range ~[-100, 100]. Detects correlation regime shifts."),
    _fe("deviation", "timothymasters.cross_market", "price_structure",
        "Log-space OLS spread deviation: normal CDF-compressed residual from regression of market1 on market2",
        rr=H, ret=H, dr=H, hor=ME, wbi=[RA], dir_=True, bench=True,
        notes="Range ~[-50, 50]. Optional EMA smoothing available."),
    _fe("purify", "timothymasters.cross_market", "price_structure",
        "SVD-based spread purification (raw prices): market1 residual after removing trend/acceleration/vol of market2",
        rr=H, ret=H, dr=H, hor=ME, wbi=[RA], dir_=True, bench=True,
        notes="Range ~[-50, 50]. Removes confounding macro factors from raw-price spread."),
    _fe("log_purify", "timothymasters.cross_market", "price_structure",
        "SVD-based spread purification in log-price space: log-space variant of purify",
        rr=H, ret=H, dr=H, hor=ME, wbi=[RA], dir_=True, bench=True,
        notes="Range ~[-50, 50]. Preferred for instruments with large price-level differences."),
    _fe("trend_diff", "timothymasters.cross_market", "trend",
        "Difference of linear Legendre trend between two markets (market1 - market2)",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True,
        notes="Range ~[-100, 100]. Identifies relative trend divergence between paired instruments."),
    _fe("cmma_diff", "timothymasters.cross_market", "trend",
        "Difference of Close-Minus-MA between two markets (market1 - market2)",
        rr=M, ret=H, dr=H, hor=S, wbi=[TR, RA], dir_=True, bench=True,
        notes="Range ~[-100, 100]. Short-term relative momentum divergence."),
]

# ─────────────────────────────────────────────────────────────────────────────
# TIMOTHY MASTERS — MULTI-MARKET PORTFOLIO STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
_TM_PORTFOLIO_STATS = [
    _fe("trend_rank", "timothymasters.multi_market", "composite",
        "Fractile rank of target market linear trend among all universe markets",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True,
        notes="closes[0]/highs[0]/lows[0] = target market. Range [0, 1]."),
    _fe("trend_median", "timothymasters.multi_market", "composite",
        "Median linear trend across all markets in the universe",
        rr=H, ret=M, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True,
        notes="Measures universe-wide trend environment strength."),
    _fe("trend_range", "timothymasters.multi_market", "composite",
        "Max minus min of linear trend across all markets — universe trend dispersion",
        rr=H, ret=M, dr=N, hor=ME, wbi=[TR, VO], bench=True,
        notes="High range signals divergent trend environment across markets."),
    _fe("trend_iqr", "timothymasters.multi_market", "composite",
        "IQR (P75-P25) of linear trend across all markets — robust trend dispersion",
        rr=H, ret=M, dr=N, hor=ME, wbi=[TR, VO], bench=True,
        notes="Outlier-robust version of trend_range."),
    _fe("trend_clump", "timothymasters.multi_market", "composite",
        "40th/60th fractile clump of linear trend — fraction of markets near the center",
        rr=M, ret=M, dr=N, hor=ME, bench=True,
        notes="High clump = most markets share similar trend levels (consensus regime)."),
    _fe("cmma_rank", "timothymasters.multi_market", "composite",
        "Fractile rank of target market close_minus_ma among all universe markets",
        rr=H, ret=H, dr=H, hor=S, wbi=[TR, RA], dir_=True, bench=True,
        notes="closes[0] is the target market. Range [0, 1]."),
    _fe("cmma_median", "timothymasters.multi_market", "composite",
        "Median close_minus_ma across all markets in the universe",
        rr=H, ret=M, dr=H, hor=S, wbi=[TR], dir_=True, bench=True,
        notes="Universe-wide short-term momentum consensus."),
    _fe("cmma_range", "timothymasters.multi_market", "composite",
        "Max minus min close_minus_ma across all markets — short-term spread dispersion",
        rr=M, ret=M, dr=N, hor=S, wbi=[VO], bench=True),
    _fe("cmma_iqr", "timothymasters.multi_market", "composite",
        "IQR of close_minus_ma across all markets — robust short-term momentum dispersion",
        rr=M, ret=M, dr=N, hor=S, bench=True),
    _fe("cmma_clump", "timothymasters.multi_market", "composite",
        "40th/60th fractile clump of close_minus_ma — short-term momentum clustering",
        rr=M, ret=M, dr=N, hor=S, bench=True,
        notes="High clump = markets clustered in similar short-term momentum range."),
]

# ─────────────────────────────────────────────────────────────────────────────
# TIMOTHY MASTERS — MULTI-MARKET RISK INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
_TM_RISK = [
    _fe("mahal", "timothymasters.multi_market", "regime",
        "Mahalanobis distance (log-odds of F-statistic) — joint market anomaly / stress detector",
        rr=C, ret=H, dr=N, hor=ME, wbi=[CR, VO], bench=True,
        notes="High values signal unusual joint market behaviour. Window excludes current bar."),
    _fe("abs_ratio", "timothymasters.multi_market", "regime",
        "Fraction of total variance in top-k eigenvalues — dimensionality of market comovement",
        rr=H, ret=M, dr=N, hor=ME, wbi=[TR, CR], bench=True,
        notes="High abs_ratio = highly correlated market universe (herding / synchronisation)."),
    _fe("abs_shift", "timothymasters.multi_market", "regime",
        "Z-score shift in abs_ratio between short and long windows — comovement acceleration",
        rr=H, ret=M, dr=N, hor=ME, wbi=[CR, VO], bench=True,
        notes="Rising abs_shift signals increasing inter-market synchronisation. Requires long_lookback > short_lookback."),
    _fe("coherence", "timothymasters.multi_market", "regime",
        "Eigenvalue-weighted correlation coherence across the market universe",
        rr=C, ret=H, dr=N, hor=ME, wbi=[TR, CR], bench=True,
        notes="High coherence = markets moving in lockstep. Window includes current bar."),
    _fe("delta_coherence", "timothymasters.multi_market", "regime",
        "Change in coherence over delta_length bars — coherence acceleration",
        rr=H, ret=M, dr=N, hor=ME, wbi=[CR, VO], bench=True,
        notes="Rising delta_coherence signals tightening inter-market correlation."),
]

# ─────────────────────────────────────────────────────────────────────────────
# TIMOTHY MASTERS — JANUS RELATIVE STRENGTH SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
_TM_JANUS = [
    _fe("janus_market_index", "timothymasters.multi_market", "composite",
        "JANUS market index: OOS equity of a naive long-only strategy across the universe",
        rr=M, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True,
        notes="Baseline reference for all JANUS relative strength comparisons."),
    _fe("janus_rs", "timothymasters.multi_market", "composite",
        "JANUS offensive/defensive relative strength: target market OOS equity vs universe, clipped [-200, 200]",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True),
    _fe("janus_rs_fractile", "timothymasters.multi_market", "composite",
        "JANUS RS fractile rank of target market within the rolling lookback window [0, 1]",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True),
    _fe("janus_delta_rs_fractile", "timothymasters.multi_market", "composite",
        "Change in JANUS RS fractile over delta_length bars — RS momentum",
        rr=H, ret=H, dr=H, hor=S, wbi=[TR, VO], dir_=True, bench=True),
    _fe("janus_rss", "timothymasters.multi_market", "composite",
        "JANUS RSS: cumulative relative strength signal aggregate (sum of RS signals across universe)",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True),
    _fe("janus_delta_rss", "timothymasters.multi_market", "composite",
        "Change in JANUS RSS over delta_length bars — RSS acceleration",
        rr=M, ret=H, dr=H, hor=S, wbi=[TR], dir_=True, bench=True),
    _fe("janus_dom", "timothymasters.multi_market", "composite",
        "JANUS DOM: Direction of Market — RS gains accumulated during RSS expansion phases",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True,
        notes="Increases when RSS is expanding; measures offensive positioning signal."),
    _fe("janus_doe", "timothymasters.multi_market", "composite",
        "JANUS DOE: Direction of Equity — RS changes accumulated during RSS contraction phases",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR, RA], dir_=True, bench=True,
        notes="Active during RSS contraction; measures defensive positioning signal."),
    _fe("janus_dom_index", "timothymasters.multi_market", "composite",
        "JANUS DOM index: DOM normalised and CDF-compressed to standard scale",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True),
    _fe("janus_rm", "timothymasters.multi_market", "composite",
        "JANUS RM: momentum-based relative strength via DOM transitions, clipped [-300, 300]",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True,
        notes="Captures momentum-driven relative strength shifts from DOM/DOE transitions."),
    _fe("janus_rm_fractile", "timothymasters.multi_market", "composite",
        "JANUS RM fractile rank of target market within the rolling lookback window [0, 1]",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True),
    _fe("janus_delta_rm_fractile", "timothymasters.multi_market", "composite",
        "Change in JANUS RM fractile over delta_length bars — RM momentum",
        rr=H, ret=H, dr=H, hor=S, wbi=[TR, VO], dir_=True, bench=True),
    _fe("janus_rs_leader_equity", "timothymasters.multi_market", "composite",
        "OOS equity of the RS leader (best market by RS) at each bar",
        rr=M, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True),
    _fe("janus_rs_laggard_equity", "timothymasters.multi_market", "composite",
        "OOS equity of the RS laggard (worst market by RS) at each bar",
        rr=M, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True),
    _fe("janus_rs_ps", "timothymasters.multi_market", "composite",
        "JANUS RS portfolio selectivity: RS leader minus RS laggard equity gap",
        rr=H, ret=H, dr=N, hor=ME, wbi=[TR], bench=True,
        notes="High ps = large spread between top and bottom RS performers in universe."),
    _fe("janus_rs_leader_advantage", "timothymasters.multi_market", "composite",
        "RS leader OOS equity advantage over the market index",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True),
    _fe("janus_rs_laggard_advantage", "timothymasters.multi_market", "composite",
        "RS laggard OOS equity advantage over the market index (negative = underperformance)",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True),
    _fe("janus_rm_leader_equity", "timothymasters.multi_market", "composite",
        "OOS equity of the RM leader (best market by RM) at each bar",
        rr=M, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True),
    _fe("janus_rm_laggard_equity", "timothymasters.multi_market", "composite",
        "OOS equity of the RM laggard (worst market by RM) at each bar",
        rr=M, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True),
    _fe("janus_rm_ps", "timothymasters.multi_market", "composite",
        "JANUS RM portfolio selectivity: RM leader minus RM laggard equity gap",
        rr=H, ret=H, dr=N, hor=ME, wbi=[TR], bench=True),
    _fe("janus_rm_leader_advantage", "timothymasters.multi_market", "composite",
        "RM leader OOS equity advantage over the market index",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True),
    _fe("janus_rm_laggard_advantage", "timothymasters.multi_market", "composite",
        "RM laggard OOS equity advantage over the market index",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True),
    _fe("janus_oos_avg", "timothymasters.multi_market", "composite",
        "JANUS OOS average equity across all markets — universe-wide performance baseline",
        rr=M, ret=M, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True),
    _fe("janus_cma_oos", "timothymasters.multi_market", "composite",
        "JANUS CMA out-of-sample equity with adaptive lookback minimising variance",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True,
        notes="CMA searches optimal lookback in [min_cma, max_cma]. Adaptive to changing market regimes."),
    _fe("janus_leader_cma_oos", "timothymasters.multi_market", "composite",
        "JANUS leader CMA OOS equity: CMA adaptive lookback applied to the RS leader market",
        rr=H, ret=H, dr=H, hor=ME, wbi=[TR], dir_=True, bench=True),
]

# ─────────────────────────────────────────────────────────────────────────────
# CANDLE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
_CANDLE = [
    _fe("candle_features", "candle", "price_structure",
        "Candlestick anatomy features: body_frac, upper_tail, lower_tail, range, frac_high, frac_low",
        rr=M, ret=M, dr=M, hor=I, ot="dataframe",
        notes="6 normalised scale-invariant anatomy features. Useful for pattern recognition and regime classification."),
]

# ─────────────────────────────────────────────────────────────────────────────
# FRACTIONAL DIFFERENTIATION
# ─────────────────────────────────────────────────────────────────────────────
_FRACTIONAL_DIFF = [
    _fe("fractional_differentiate_series", "fractional_diff", "information",
        "Apply fractional differentiation (order d) — preserves memory while inducing stationarity",
        rr=M, ret=H, dr=N, hor=LG,
        notes="Returns (result_array, window_size). Window size auto-determined by weight convergence threshold."),
    _fe("get_optimal_fractional_differentiation_order", "fractional_diff", "information",
        "Find minimum d that achieves ADF-stationarity while preserving maximum series memory",
        rr=M, ret=H, dr=N, hor=LG,
        notes="Returns (optimal_d, window_size, diff_series, adf_results). Scans d from min_d to max_d by step."),
]

# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL / CALENDAR FEATURES
# ─────────────────────────────────────────────────────────────────────────────
_TEMPORAL = [
    _fe("hour_of_day", "temporal", "temporal",
        "Hour of day (0–23) from DatetimeIndex — captures intraday session patterns",
        rr=M, ret=M, dr=N, hor=I,
        notes="Part of TemporalFeature class. Requires DatetimeIndex."),
    _fe("day_of_week", "temporal", "temporal",
        "Day of week (0=Monday, 6=Sunday) from DatetimeIndex — captures weekly seasonality",
        rr=M, ret=M, dr=N, hor=S,
        notes="Part of TemporalFeature class."),
    _fe("hour_of_day_cyclic", "temporal", "temporal",
        "Cyclic sin/cos encoding of hour-of-day — preserves circular periodicity for ML",
        rr=M, ret=M, dr=N, hor=I, ot="dataframe",
        notes="Returns (hour_sin, hour_cos) tuple. Preferred over raw hour for gradient-based models."),
    _fe("day_of_week_cyclic", "temporal", "temporal",
        "Cyclic sin/cos encoding of day-of-week — preserves circular periodicity for ML",
        rr=M, ret=M, dr=N, hor=S, ot="dataframe",
        notes="Returns (dow_sin, dow_cos) tuple."),
    _fe("market_session", "temporal", "temporal",
        "Market session label: closed / pre_market / open / after_hours",
        rr=M, ret=M, dr=N, hor=I,
        notes="Supports US, EU, ASIA markets. Timezone-aware. Part of TemporalFeature class."),
]

# ─────────────────────────────────────────────────────────────────────────────
# MOVING AVERAGE / CHANNEL FEATURES (ma.py)
# ─────────────────────────────────────────────────────────────────────────────
_MA_FEATURES = [
    _fe("keltner_channels", "ma", "price_structure",
        "Keltner Channels: EMA ± ATR-multiple bands — volatility-based price channel",
        rr=M, ret=M, dr=M, hor=S, wbi=[TR, RA],
        ot="dataframe", notes="Returns (upper, mid, lower) tuple of Series/arrays"),
]

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL SMOOTHING / FILTERING (filters.py)
# ─────────────────────────────────────────────────────────────────────────────
_FILTERS = [
    _fe("smooth_kalman", "filters", "price_structure",
        "Causal 1-D Kalman filter smoother — optimal linear state estimation",
        rr=M, ret=M, dr=N, hor=A, causal=True,
        notes="Causal (real-time safe). Tune process_noise / measurement_noise to balance lag vs smoothness."),
    _fe("smooth_gaussian", "filters", "price_structure",
        "Causal Gaussian kernel smoother — asymmetric window using only past bars",
        rr=L, ret=M, dr=N, hor=A, causal=True,
        notes="causal=True (default) is real-time safe. causal=False uses future data — research only."),
    _fe("smooth_savitzky_golay", "filters", "price_structure",
        "Causal Savitzky-Golay polynomial smoother — fits polynomial to past window",
        rr=L, ret=M, dr=N, hor=A, causal=True,
        notes="causal=True (default) is real-time safe. Window must be odd; polyorder < window."),
    _fe("smooth_median", "filters", "price_structure",
        "Causal rolling median filter — robust to outliers and price spikes",
        rr=L, ret=M, dr=N, hor=A, causal=True,
        notes="causal=True (default) uses backward-looking window. Robust alternative to EMA."),
    _fe("smooth_wavelet", "filters", "price_structure",
        "Wavelet denoising smoother — soft-threshold multi-resolution decomposition",
        rr=L, ret=M, dr=N, hor=A, causal=False,
        notes="NON-CAUSAL: uses future data. Research / offline analysis only. Not suitable for live trading."),
    _fe("smooth_loess", "filters", "price_structure",
        "LOESS locally-weighted scatterplot smoother",
        rr=L, ret=M, dr=N, hor=A, causal=False,
        notes="NON-CAUSAL: uses future data. Research / offline analysis only. Not suitable for live trading."),
]

# ─────────────────────────────────────────────────────────────────────────────
# FULL CATALOG
# ─────────────────────────────────────────────────────────────────────────────
CATALOG: list[FeatureEntry] = (
    _MICROSTRUCTURE_PRIMITIVES
    + _MICROSTRUCTURE_DEPTH
    + _MICROSTRUCTURE_PRICE_STRUCTURE
    + _MICROSTRUCTURE_LIQUIDITY
    + _MICROSTRUCTURE_ORDER_FLOW
    + _MICROSTRUCTURE_INFORMATION
    + _MICROSTRUCTURE_COMPOSITES
    + _MICROSTRUCTURE_REGIME
    + _TM_MOMENTUM
    + _TM_TREND
    + _TM_VARIANCE
    + _TM_VOLUME
    + _TM_INFORMATION
    + _TM_FTI
    + _TM_CROSS_MARKET
    + _TM_PORTFOLIO_STATS
    + _TM_RISK
    + _TM_JANUS
    + _VOLATILITY
    + _PATH_STRUCTURE
    + _VOLUME
    + _MOMENTUM
    + _TREND
    + _CANDLE
    + _FRACTIONAL_DIFF
    + _TEMPORAL
    + _MA_FEATURES
    + _FILTERS
)