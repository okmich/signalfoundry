import numpy as np
import pandas as pd


def example_momentum_features(
    df: pd.DataFrame, lookback: int = 20, include_volume: bool = True
) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    # Log returns
    features["log_return"] = np.log(df["close"] / df["close"].shift(1))
    features["log_return_5"] = features["log_return"].rolling(5).mean()
    features["log_return_10"] = features["log_return"].rolling(10).mean()

    # Price momentum
    features["momentum_5"] = df["close"] / df["close"].shift(5) - 1
    features["momentum_10"] = df["close"] / df["close"].shift(10) - 1
    features["momentum_20"] = df["close"] / df["close"].shift(lookback) - 1

    # RSI-like (simplified)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features["rsi_14"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    features["atr_14"] = true_range.rolling(14).mean()
    features["atr_pct_14"] = features["atr_14"] / df["close"]

    # Volume features
    if include_volume and "volume" in df.columns:
        features["volume_ma_20"] = df["volume"].rolling(lookback).mean()
        features["volume_ratio"] = df["volume"] / features["volume_ma_20"]

    return features


def example_volatility_features(
    df: pd.DataFrame, short_window: int = 10, long_window: int = 30
) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    # Log returns
    log_returns = np.log(df["close"] / df["close"].shift(1))

    # Realized volatility (multiple windows)
    features[f"realized_vol_{short_window}"] = log_returns.rolling(short_window).std()
    features[f"realized_vol_{long_window}"] = log_returns.rolling(long_window).std()
    features["vol_ratio"] = (
        features[f"realized_vol_{short_window}"]
        / features[f"realized_vol_{long_window}"]
    )

    # Parkinson volatility (high-low based)
    log_hl = np.log(df["high"] / df["low"])
    features[f"parkinson_vol_{short_window}"] = np.sqrt(
        (1 / (4 * short_window * np.log(2))) * (log_hl**2).rolling(short_window).sum()
    )

    # Range-based features
    features["hl_range"] = (df["high"] - df["low"]) / df["close"]
    features["hl_range_ma"] = features["hl_range"].rolling(short_window).mean()

    # Volatility of volatility
    features["vol_of_vol"] = (
        features[f"realized_vol_{short_window}"].rolling(short_window).std()
    )

    return features


def example_path_structure_features(
    df: pd.DataFrame, hurst_window: int = 12, efficiency_window: int = 20
) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    # Log returns
    log_returns = np.log(df["close"] / df["close"].shift(1))

    # Efficiency ratio (Kaufman)
    def efficiency_ratio(returns, window):
        net_change = returns.rolling(window).sum().abs()
        total_change = returns.abs().rolling(window).sum()
        return net_change / total_change

    features[f"efficiency_{efficiency_window}"] = efficiency_ratio(
        log_returns, efficiency_window
    )

    # Autocorrelation
    features["autocorr_lag1"] = log_returns.rolling(20).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
    )

    # Trend strength (R-squared of linear regression)
    def trend_strength(prices, window):
        def calc_r2(series):
            if len(series) < 3:
                return np.nan
            x = np.arange(len(series))
            y = series.values
            corr = np.corrcoef(x, y)[0, 1]
            return corr**2

        return prices.rolling(window).apply(calc_r2, raw=False)

    features[f"trend_strength_{efficiency_window}"] = trend_strength(
        df["close"], efficiency_window
    )
    features["fractal_dim"] = 1 + (
        log_returns.abs().rolling(hurst_window).sum()
        / log_returns.rolling(hurst_window).std()
    )

    return features


def combined_features_v1(
    df: pd.DataFrame, lookback: int = 20, include_volume: bool = True
) -> pd.DataFrame:
    # Generate all feature sets
    momentum_feats = example_momentum_features(
        df, lookback=lookback, include_volume=include_volume
    )
    volatility_feats = example_volatility_features(df, short_window=10, long_window=30)
    path_feats = example_path_structure_features(
        df, hurst_window=12, efficiency_window=lookback
    )

    # Combine
    combined = pd.concat([momentum_feats, volatility_feats, path_feats], axis=1)

    return combined
