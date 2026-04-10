import pandas as pd


def candle_features(df: pd.DataFrame, open_col: str = "open", high_col: str = "high", low_col: str = "low",
                    close_col: str = "close", volume_col: str = "tick_volume") -> pd.DataFrame:
    """
    Extract candlestick shape and structure features. It computes normalized features that describe the anatomy of each candle:
    body size, wick proportions, range metrics, and directional characteristics.

    These features are useful for:
    - Candlestick pattern recognition
    - Market regime classification
    - Price action analysis
    - ML feature engineering

    Returns:
    --------
    pd.DataFrame with columns:
        • range: (high - low) / open - Normalized candle range
            └─ Measures volatility relative to price level
        • upper_tail: (high - max(open, close)) / range - Upper wick ratio
            └─ High values suggest rejection of higher prices (bearish)
        • lower_tail: (min(open, close) - low) / range - Lower wick ratio
            └─ High values suggest rejection of lower prices (bullish)
        • body_frac: (close - open) / (high - low) - Body proportion
            └─ Positive = bullish body, Negative = bearish body
            └─ Magnitude: body dominance (1 = marubozu, 0 = doji)
        • frac_high: (high - open) / open - Fractional move to high
            └─ Measures upward exploration from open
        • frac_low: (open - low) / low - Fractional move to low
            └─ Measures downward exploration from open

    Feature Interpretation:
    -----------------------
    BODY CHARACTERISTICS:
    • body_frac close to ±1: Strong directional candle (marubozu-like)
    • body_frac close to 0: Indecision candle (doji-like)
    • |body_frac| > 0.7: Strong conviction move
    • |body_frac| < 0.3: Weak/indecisive move

    WICK CHARACTERISTICS:
    • upper_tail > 0.4: Strong rejection at highs (potential resistance)
    • lower_tail > 0.4: Strong rejection at lows (potential support)
    • Both tails > 0.3: Spinning top (high uncertainty)
    • Both tails < 0.1: Strong body dominance (conviction)

    RANGE CHARACTERISTICS:
    • range > 2%: High volatility candle
    • range < 0.5%: Low volatility consolidation
    • Expanding range: Increasing volatility
    • Contracting range: Decreasing volatility

    Common Patterns (can be detected with these features):
    ------------------------------------------------------
    • DOJI: |body_frac| < 0.1, both tails > 0.3
    • HAMMER: lower_tail > 0.5, upper_tail < 0.2, small body
    • SHOOTING STAR: upper_tail > 0.5, lower_tail < 0.2, small body
    • MARUBOZU: |body_frac| > 0.9, both tails < 0.05
    • SPINNING TOP: |body_frac| < 0.3, both tails > 0.3

    Use Cases by Strategy:
    ----------------------
    REVERSAL TRADING:
    • Look for: High upper_tail at resistance (shooting star)
    • Look for: High lower_tail at support (hammer)
    • Signal: Extreme wick ratios at key levels

    BREAKOUT TRADING:
    • Look for: High |body_frac| (>0.7) with increasing range
    • Look for: Low wick ratios (<0.2) suggesting conviction
    • Avoid: High tail ratios during breakout (false break)

    RANGE TRADING:
    • Look for: Low range + high tail ratios (consolidation)
    • Look for: |body_frac| < 0.3 (indecision)
    • Signal: Doji-like candles at range extremes

    ML FEATURE ENGINEERING:
    • All features are normalized (scale-invariant)
    • Combine with volume features for confirmation
    • Use rolling statistics (mean, std) for regime detection
    • Calculate consecutive pattern sequences

    Important Notes:
    ----------------
    • All features are normalized (dimensionless ratios)
    • Epsilon (1e-9) prevents division by zero
    • Features work across all timeframes and instruments
    • No look-ahead bias (only uses current bar data)
    • Suitable for both rule-based and ML strategies

    Example Usage:
    --------------
    >>> # Basic usage
    >>> candle_feats = candle_features(df)
    >>>
    >>> # Detect hammer pattern
    >>> hammers = candle_feats[
    ...     (candle_feats['lower_tail'] > 0.5) &
    ...     (candle_feats['upper_tail'] < 0.2) &
    ...     (candle_feats['body_frac'].abs() < 0.3)
    ... ]
    >>>
    >>> # Detect strong bullish candles
    >>> strong_bull = candle_feats[
    ...     (candle_feats['body_frac'] > 0.7) &
    ...     (candle_feats['upper_tail'] < 0.2)
    ... ]
    >>>
    >>> # Combine with volume for confirmation
    >>> from okmich_quant_features.volume import mfi_features
    >>> vol_feats = mfi_features(df, feature_type='directionless')
    >>> confirmed_signals = strong_bull[vol_feats['volume_ratio'] > 1.5]

    See Also:
    ---------
    - mfi_features(): For volume-price efficiency analysis
    - Volume features: For confirmation of candlestick patterns
    """
    # Input validation
    required_cols = [open_col, high_col, low_col, close_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    data = pd.DataFrame(index=df.index)
    eps = 1e-9

    _open = df[open_col]
    _high = df[high_col]
    _low = df[low_col]
    _close = df[close_col]

    # Basic range metric
    data["range"] = (_high - _low) / (_open + eps)

    # Wick ratios (tails)
    high_of_body = pd.concat([_open, _close], axis=1).max(axis=1)
    low_of_body = pd.concat([_open, _close], axis=1).min(axis=1)

    data["upper_tail"] = (_high - high_of_body) / ((_high - _low) + eps)
    data["lower_tail"] = (low_of_body - _low) / ((_high - _low) + eps)

    # Body characteristics
    data["body_frac"] = (_close - _open) / ((_high - _low) + eps)

    # Fractional price moves
    data["frac_high"] = (_high - _open) / (_open + eps)
    data["frac_low"] = (_open - _low) / (_low + eps)

    return data
