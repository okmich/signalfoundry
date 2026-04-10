"""
Core ruptures-based labeling functions.

Uses the ruptures library for offline change point detection to label
price series by trend direction, volatility regime, or multivariate regimes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.stats import linregress


class CostModel(Enum):
    """Cost function (model) for change point detection."""
    L1 = "l1"           # Mean shifts, robust to outliers
    L2 = "l2"           # Mean shifts, sensitive
    RBF = "rbf"         # Distribution changes (general regime)
    LINEAR = "linear"   # Trend/slope changes
    NORMAL = "normal"   # Mean + variance changes
    AR = "ar"           # AR coefficient changes
    RANK = "rank"       # Non-parametric, robust


class Algorithm(Enum):
    """Change point detection algorithm."""
    PELT = "pelt"           # Fast, exact, unknown # breakpoints (O(n))
    BINSEG = "binseg"       # Fast approximate (O(n log n))
    DYNP = "dynp"           # Exact, known # breakpoints (O(Kn²))
    WINDOW = "window"       # Sliding window (O(n))
    BOTTOMUP = "bottomup"   # Hierarchical segmentation (O(n log n))


class LabelMethod(Enum):
    """Method for assigning labels to segments."""
    SLOPE = "slope"             # Continuous slope value
    DIRECTION = "direction"     # Discretized to {-1, 0, 1}
    MAGNITUDE = "magnitude"     # Discretized slope magnitude


@dataclass
class RupturesConfig:
    """Configuration for ruptures-based labeling."""
    model: CostModel = CostModel.RBF
    algo: Algorithm = Algorithm.PELT
    penalty: Optional[float] = 10.0
    n_bkps: Optional[int] = None  # Alternative to penalty (for DYNP)
    min_size: int = 5
    jump: int = 1

    def __post_init__(self):
        if self.algo == Algorithm.DYNP and self.n_bkps is None:
            raise ValueError("Algorithm.DYNP requires n_bkps to be specified")


def _get_algo_instance(config: RupturesConfig, signal: np.ndarray) -> Any:
    """Create and fit the appropriate ruptures algorithm instance."""
    model_str = config.model.value

    if config.algo == Algorithm.PELT:
        algo = rpt.Pelt(model=model_str, min_size=config.min_size, jump=config.jump)
    elif config.algo == Algorithm.BINSEG:
        algo = rpt.Binseg(model=model_str, min_size=config.min_size, jump=config.jump)
    elif config.algo == Algorithm.DYNP:
        algo = rpt.Dynp(model=model_str, min_size=config.min_size, jump=config.jump)
    elif config.algo == Algorithm.WINDOW:
        algo = rpt.Window(model=model_str, min_size=config.min_size, jump=config.jump)
    elif config.algo == Algorithm.BOTTOMUP:
        algo = rpt.BottomUp(model=model_str, min_size=config.min_size, jump=config.jump)
    else:
        raise ValueError(f"Unknown algorithm: {config.algo}")

    algo.fit(signal)
    return algo


def _predict_breakpoints(algo: Any, config: RupturesConfig, signal: np.ndarray = None) -> List[int]:
    """Get breakpoints from fitted algorithm."""
    if config.algo == Algorithm.DYNP:
        return algo.predict(n_bkps=config.n_bkps)
    else:
        penalty = config.penalty

        # Auto-scale penalty based on signal variance if signal provided
        if signal is not None and penalty is not None:
            signal_var = np.var(signal)
            if signal_var > 0:
                # Scale penalty: user penalty is a multiplier (1.0 = moderate segmentation)
                # Typical range: 0.1 (many segments) to 10 (few segments)
                penalty = config.penalty * signal_var * len(signal) * 0.01

        return algo.predict(pen=penalty)


def ruptures_segment(
    signal: pd.Series,
    config: RupturesConfig = None,
) -> List[int]:
    """
    Detect change points in a signal.

    Parameters
    ----------
    signal : pd.Series
        Input time series (prices, returns, or any signal)
    config : RupturesConfig, optional
        Algorithm configuration. Defaults to PELT with RBF model.

    Returns
    -------
    List[int]
        Indices of detected change points (end of each segment).
        The last index is always len(signal).

    Notes
    -----
    The penalty parameter is automatically scaled based on signal variance.
    A penalty of 1.0 produces moderate segmentation. Use lower values (0.1-0.5)
    for more segments, higher values (2-10) for fewer segments.
    """
    if config is None:
        config = RupturesConfig()

    # Prepare signal for ruptures (needs 2D array)
    arr = signal.values.reshape(-1, 1).astype(float)

    # Handle NaN values
    if np.isnan(arr).any():
        # Forward fill then backward fill NaNs
        arr = pd.DataFrame(arr).ffill().bfill().values

    algo = _get_algo_instance(config, arr)
    breakpoints = _predict_breakpoints(algo, config, arr)

    return breakpoints


def _segments_from_breakpoints(breakpoints: List[int], n: int) -> List[Tuple[int, int]]:
    """Convert breakpoints to (start, end) segment tuples."""
    segments = []
    prev = 0
    for bp in breakpoints:
        segments.append((prev, bp))
        prev = bp
    return segments


def _compute_segment_slope(prices: np.ndarray) -> float:
    """Compute normalized slope of a price segment."""
    if len(prices) < 2:
        return 0.0

    x = np.arange(len(prices))
    try:
        slope, _, _, _, _ = linregress(x, prices)
        # Normalize by mean price to get relative slope
        mean_price = np.mean(prices)
        if mean_price != 0:
            return slope / mean_price
        return slope
    except Exception:
        return 0.0


def _assign_trend_direction(slope: float, threshold: float = 0.0) -> int:
    """Assign trend direction based on slope."""
    if slope > threshold:
        return 1
    elif slope < -threshold:
        return -1
    return 0


def ruptures_trend_labels(
    prices: pd.Series,
    config: RupturesConfig = None,
    label_method: LabelMethod = LabelMethod.DIRECTION,
    neutral_threshold: float = 0.0,
    use_returns: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    Label price series by trend regime using ruptures segmentation.

    Parameters
    ----------
    prices : pd.Series
        Price series to label
    config : RupturesConfig, optional
        Ruptures algorithm configuration. Defaults to PELT with L2 model on returns.
    label_method : LabelMethod
        How to assign labels to segments:
        - SLOPE: continuous normalized slope value
        - DIRECTION: {-1, 0, 1} based on slope sign
        - MAGNITUDE: discretized slope into quintiles
    neutral_threshold : float
        For DIRECTION method, slopes within [-threshold, +threshold] are labeled 0.
    use_returns : bool
        If True (default), detect change points on log returns rather than prices.
        This typically produces better segmentation for financial data.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (labels, segment_ids) where:
        - labels: trend labels for each timestamp
        - segment_ids: segment identifier for each timestamp
    """
    if config is None:
        config = RupturesConfig(model=CostModel.L2)

    # Decide what signal to use for change point detection
    if use_returns:
        # Detect on log returns - better for financial data
        signal = np.log(prices / prices.shift(1)).fillna(0)
    else:
        # Detect on raw prices
        signal = prices

    # Detect change points
    breakpoints = ruptures_segment(signal, config)
    segments = _segments_from_breakpoints(breakpoints, len(prices))

    # Initialize output arrays
    labels = np.zeros(len(prices), dtype=float)
    segment_ids = np.zeros(len(prices), dtype=int)

    price_values = prices.values.astype(float)

    for seg_id, (start, end) in enumerate(segments):
        segment_prices = price_values[start:end]
        slope = _compute_segment_slope(segment_prices)

        if label_method == LabelMethod.SLOPE:
            label_value = slope
        elif label_method == LabelMethod.DIRECTION:
            label_value = _assign_trend_direction(slope, neutral_threshold)
        elif label_method == LabelMethod.MAGNITUDE:
            # Will be post-processed below
            label_value = slope
        else:
            raise ValueError(f"Unknown label method: {label_method}")

        labels[start:end] = label_value
        segment_ids[start:end] = seg_id

    # Post-process MAGNITUDE method
    if label_method == LabelMethod.MAGNITUDE:
        # Discretize slopes into quintiles (-2, -1, 0, 1, 2)
        unique_slopes = []
        for start, end in segments:
            unique_slopes.append(labels[start])

        if len(unique_slopes) > 1:
            percentiles = np.percentile(unique_slopes, [20, 40, 60, 80])
            for start, end in segments:
                slope = labels[start]
                if slope <= percentiles[0]:
                    mag = -2
                elif slope <= percentiles[1]:
                    mag = -1
                elif slope <= percentiles[2]:
                    mag = 0
                elif slope <= percentiles[3]:
                    mag = 1
                else:
                    mag = 2
                labels[start:end] = mag

    labels_series = pd.Series(labels, index=prices.index, name="trend_label")
    segment_series = pd.Series(segment_ids, index=prices.index, name="segment_id")

    # Cast to int for DIRECTION and MAGNITUDE
    if label_method in (LabelMethod.DIRECTION, LabelMethod.MAGNITUDE):
        labels_series = labels_series.astype(int)

    return labels_series, segment_series


def ruptures_volatility_labels(
    prices: pd.Series,
    vol_window: int = 20,
    n_regimes: int = 3,
    config: RupturesConfig = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Label volatility regimes using ruptures on rolling volatility.

    Parameters
    ----------
    prices : pd.Series
        Price series
    vol_window : int
        Window size for rolling volatility calculation
    n_regimes : int
        Number of volatility regimes to detect (for labeling, not breakpoints)
    config : RupturesConfig, optional
        Ruptures configuration. Defaults to PELT with L2 model.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (regime_labels, segment_ids) where:
        - regime_labels: 0=low, 1=medium, 2=high (for n_regimes=3)
        - segment_ids: segment identifier for each timestamp
    """
    if config is None:
        config = RupturesConfig(model=CostModel.L2)

    # Compute rolling volatility
    returns = np.log(prices / prices.shift(1))
    volatility = returns.rolling(window=vol_window).std()

    # Handle initial NaNs
    volatility = volatility.ffill().bfill()

    # Detect change points on volatility
    breakpoints = ruptures_segment(volatility, config)
    segments = _segments_from_breakpoints(breakpoints, len(volatility))

    # Compute mean volatility per segment
    vol_values = volatility.values.astype(float)
    segment_vols = []
    for start, end in segments:
        segment_vols.append(np.mean(vol_values[start:end]))

    # Assign regime labels based on volatility ranking
    if n_regimes <= 1:
        regime_map = {i: 0 for i in range(len(segments))}
    else:
        # Create percentile thresholds
        percentiles = np.linspace(0, 100, n_regimes + 1)[1:-1]
        thresholds = np.percentile(segment_vols, percentiles)

        regime_map = {}
        for seg_idx, vol in enumerate(segment_vols):
            regime = 0
            for thresh in thresholds:
                if vol > thresh:
                    regime += 1
            regime_map[seg_idx] = regime

    # Build output arrays
    labels = np.zeros(len(prices), dtype=int)
    segment_ids = np.zeros(len(prices), dtype=int)

    for seg_id, (start, end) in enumerate(segments):
        labels[start:end] = regime_map[seg_id]
        segment_ids[start:end] = seg_id

    labels_series = pd.Series(labels, index=prices.index, name="vol_regime")
    segment_series = pd.Series(segment_ids, index=prices.index, name="segment_id")

    return labels_series, segment_series


def ruptures_multivariate_labels(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_regimes: int = 5,
    config: RupturesConfig = None,
) -> Tuple[pd.Series, pd.Series, Dict[int, Dict[str, float]]]:
    """
    Detect regime changes using multiple features simultaneously.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the feature columns
    feature_cols : List[str]
        Column names to use as features for regime detection
    n_regimes : int
        Target number of regimes (used for labeling segments)
    config : RupturesConfig, optional
        Ruptures configuration. Defaults to PELT with RBF model.

    Returns
    -------
    Tuple[pd.Series, pd.Series, Dict]
        (regime_labels, segment_ids, regime_stats) where:
        - regime_labels: regime identifier (0 to n_regimes-1)
        - segment_ids: segment identifier for each timestamp
        - regime_stats: statistics for each regime {regime_id: {feature: mean, ...}}
    """
    if config is None:
        config = RupturesConfig(model=CostModel.RBF)

    # Validate feature columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Extract and normalize features
    features = df[feature_cols].copy()

    # Handle NaNs
    features = features.ffill().bfill()

    # Standardize features for ruptures
    feature_means = features.mean()
    feature_stds = features.std().replace(0, 1)
    features_normalized = (features - feature_means) / feature_stds

    # Prepare signal (2D array for multivariate)
    signal = features_normalized.values.astype(float)

    # Detect change points
    algo = _get_algo_instance(config, signal)
    breakpoints = _predict_breakpoints(algo, config, signal)
    segments = _segments_from_breakpoints(breakpoints, len(df))

    # Compute segment statistics for clustering
    segment_stats = []
    for start, end in segments:
        seg_features = features.iloc[start:end]
        stats = {col: seg_features[col].mean() for col in feature_cols}
        segment_stats.append(stats)

    # Cluster segments into regimes based on feature similarity
    # Simple approach: use first principal component of means
    if len(segments) <= n_regimes:
        # Fewer segments than regimes - each segment is its own regime
        regime_map = {i: i for i in range(len(segments))}
    else:
        # Compute "score" for each segment (sum of normalized means)
        scores = []
        for stats in segment_stats:
            score = sum(
                (stats[col] - feature_means[col]) / feature_stds[col]
                for col in feature_cols
            )
            scores.append(score)

        # Assign regimes based on score percentiles
        percentiles = np.linspace(0, 100, n_regimes + 1)[1:-1]
        thresholds = np.percentile(scores, percentiles)

        regime_map = {}
        for seg_idx, score in enumerate(scores):
            regime = 0
            for thresh in thresholds:
                if score > thresh:
                    regime += 1
            regime_map[seg_idx] = regime

    # Build output arrays
    labels = np.zeros(len(df), dtype=int)
    segment_ids = np.zeros(len(df), dtype=int)

    for seg_id, (start, end) in enumerate(segments):
        labels[start:end] = regime_map[seg_id]
        segment_ids[start:end] = seg_id

    # Compute regime statistics
    regime_stats = {}
    for regime_id in range(n_regimes):
        regime_mask = labels == regime_id
        if regime_mask.sum() > 0:
            regime_data = df.loc[df.index[regime_mask], feature_cols]
            regime_stats[regime_id] = {
                col: float(regime_data[col].mean()) for col in feature_cols
            }
            regime_stats[regime_id]["count"] = int(regime_mask.sum())

    labels_series = pd.Series(labels, index=df.index, name="regime_label")
    segment_series = pd.Series(segment_ids, index=df.index, name="segment_id")

    return labels_series, segment_series, regime_stats
