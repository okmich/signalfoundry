import numpy as np
import pandas as pd
from typing import Optional


class FixedForwardReturnLabeler:
    """
    Generate fixed-horizon forward return labels.

    Labels bar i with the return from close[i] to close[i+H].
    Positive = price rises, Negative = price falls, Near-zero = sideways.

    This is the most fundamental prediction target in quantitative finance.
    The look-ahead is intentional (leaks_future=True) - we're training a model to predict what will happen H bars ahead.

    Parameters
    ----------
    horizon : int
        Number of bars ahead to predict (e.g., 12 for 1 hour on 5-min data)
    normalize : bool, default=True
        Whether to normalize by rolling volatility (makes it scale-invariant)
    normalize_window : int, default=20
        Window for volatility calculation if normalize=True
    clip_percentile : float, optional
        If provided, clip to [100-p, p] percentile bounds (e.g., 99)
        Helps remove extreme outliers that may hurt model training
    use_log_returns : bool, default=True
        Use log returns (True) or simple returns (False)
        Log returns are additive and more stable for ML

    Attributes
    ----------
    leaks_future : bool
        Always True - this labeler explicitly uses future data

    Notes
    -----
    Expected Performance:
    - Random walk: ~50% directional accuracy
    - With edge: 52-58% directional accuracy is realistic and profitable
    - Higher accuracy claims are often overfitting or data leakage

    Normalization:
    - Without normalization: returns have time-varying variance (volatility clustering)
    - With normalization: targets are more stationary, easier for models to learn
    - Vol normalization = return / rolling_volatility

    Common Horizons:
    - Intraday (5-min bars): 12 bars (1 hour), 60 bars (5 hours)
    - Daily bars: 5 bars (1 week), 20 bars (1 month)
    - Choose based on your trading frequency and holding period
    """

    def __init__(self, horizon: int, normalize: bool = True, normalize_window: int = 20, clip_percentile: float = 90.0,
                 use_log_returns: bool = True):
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if normalize and normalize_window <= 0:
            raise ValueError(f"normalize_window must be positive, got {normalize_window}")
        if clip_percentile is not None and not (0 < clip_percentile < 100):
            raise ValueError(f"clip_percentile must be in (0, 100), got {clip_percentile}")

        self.horizon = horizon
        self.normalize = normalize
        self.normalize_window = normalize_window
        self.clip_percentile = clip_percentile
        self.use_log_returns = use_log_returns
        self.leaks_future = True  # Explicit forward-looking

    def label(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """
        Generate forward return labels.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with price data
        price_col : str, default='close'
            Column name for price

        Returns
        -------
        pd.Series
            Forward return targets, same index as df.
            Last `horizon` bars will be NaN (no future available).

        Raises
        ------
        ValueError
            If price_col not in df, or if df has fewer than horizon+1 rows
        KeyError
            If price_col not found in DataFrame
        """
        if price_col not in df.columns:
            raise KeyError(f"Column '{price_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")

        if len(df) < self.horizon + 1:
            raise ValueError(
                f"DataFrame has {len(df)} rows but horizon={self.horizon} "
                f"requires at least {self.horizon + 1} rows"
            )

        prices = df[price_col]
        future_prices = prices.shift(-self.horizon)

        # Compute returns
        raw_returns = np.log(future_prices / prices) if self.use_log_returns else (future_prices - prices) / prices

        # Normalize by volatility if requested
        if self.normalize:
            # Compute rolling volatility of log returns
            log_rets = np.log(prices / prices.shift(1))
            rolling_vol = log_rets.rolling(window=self.normalize_window, min_periods=self.normalize_window).std()

            # Avoid division by zero
            rolling_vol = rolling_vol.replace(0, np.nan)

            # Normalize
            targets = raw_returns / rolling_vol
        else:
            targets = raw_returns

        # Clip outliers if requested
        if self.clip_percentile is not None:
            lower_percentile = 100 - self.clip_percentile
            upper_percentile = self.clip_percentile

            # Compute percentiles only on valid (non-NaN) values
            valid_targets = targets.dropna()
            if len(valid_targets) > 0:
                q_low = valid_targets.quantile(lower_percentile / 100)
                q_high = valid_targets.quantile(upper_percentile / 100)
                targets = targets.clip(lower=q_low, upper=q_high)

        # Ensure we return a Series with the same index
        targets.name = f'fwd_return_h{self.horizon}'

        return targets

    def __repr__(self):
        return (
            f"FixedForwardReturnLabeler("
            f"horizon={self.horizon}, "
            f"normalize={self.normalize}, "
            f"normalize_window={self.normalize_window}, "
            f"clip_percentile={self.clip_percentile}, "
            f"use_log_returns={self.use_log_returns})"
        )
