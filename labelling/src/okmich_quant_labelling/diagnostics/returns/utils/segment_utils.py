import numpy as np
import pandas as pd


def apply_target_to_segment(index, start_idx, end_idx, value):
    """
    Assign a target value to all bars in a segment.

    Parameters
    ----------
    index : pd.Index
        Full index of the time series
    start_idx : int
        Segment start position
    end_idx : int
        Segment end position
    value : float
        Target value to assign

    Returns
    -------
    pd.Series
        Series with target value assigned to segment, NaN elsewhere
    """
    targets = pd.Series(index=index, dtype=float)
    targets.iloc[start_idx:end_idx + 1] = value
    return targets


def merge_segment_targets(segment_list, index):
    """
    Combine targets from multiple segments into a single series.

    Parameters
    ----------
    segment_list : list of dict
        List of segments, each dict containing:
        - 'start_idx': segment start position
        - 'end_idx': segment end position
        - 'value': target value for segment
    index : pd.Index
        Full index of the time series

    Returns
    -------
    pd.Series
        Merged target series with NaN for bars not in any segment
    """
    # Initialize with NaN
    merged_targets = pd.Series(np.nan, index=index, dtype=float)

    # Fill in each segment
    for segment in segment_list:
        start_idx = segment['start_idx']
        end_idx = segment['end_idx']
        value = segment['value']

        merged_targets.iloc[start_idx:end_idx + 1] = value

    return merged_targets


def smooth_segment_boundaries(targets, window=3):
    """
    Apply smoothing at segment transitions using rolling average.

    Helps reduce abrupt changes when segments switch.

    Parameters
    ----------
    targets : pd.Series
        Target series with potential discontinuities
    window : int, default=3
        Rolling window size for smoothing

    Returns
    -------
    pd.Series
        Smoothed targets
    """
    # Apply rolling mean
    smoothed = targets.rolling(window=window, center=True, min_periods=1).mean()

    return smoothed


def fill_neutral_segments(targets, neutral_value=0.0):
    """
    Fill NaN values (segments with no trend) with a neutral value.

    Parameters
    ----------
    targets : pd.Series
        Target series with NaN for neutral segments
    neutral_value : float, default=0.0
        Value to use for neutral segments

    Returns
    -------
    pd.Series
        Targets with NaN filled
    """
    return targets.fillna(neutral_value)


def detect_segment_changes(targets, threshold=0.5):
    """
    Identify indices where segment changes occur (significant value change).

    Parameters
    ----------
    targets : pd.Series
        Target series
    threshold : float, default=0.5
        Minimum absolute difference to consider a segment change

    Returns
    -------
    list of int
        Indices where segment changes occur
    """
    # Calculate absolute differences
    diffs = targets.diff().abs()

    # Find changes above threshold
    change_indices = diffs[diffs > threshold].index.tolist()

    return change_indices


def segment_statistics(targets, segment_info):
    """
    Calculate statistics for each segment.

    Parameters
    ----------
    targets : pd.Series
        Target series
    segment_info : pd.DataFrame
        DataFrame with columns: ['start_idx', 'end_idx']

    Returns
    -------
    pd.DataFrame
        DataFrame with segment statistics:
        - mean: average target in segment
        - std: standard deviation in segment
        - min: minimum value in segment
        - max: maximum value in segment
        - duration: number of bars in segment
    """
    stats_list = []

    for idx, row in segment_info.iterrows():
        start_idx = row['start_idx']
        end_idx = row['end_idx']

        segment_targets = targets.iloc[start_idx:end_idx + 1]

        stats = {
            'segment_id': idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'mean': segment_targets.mean(),
            'std': segment_targets.std(),
            'min': segment_targets.min(),
            'max': segment_targets.max(),
            'duration': len(segment_targets),
        }

        stats_list.append(stats)

    return pd.DataFrame(stats_list)


def interpolate_segment_targets(targets, method='linear'):
    """
    Interpolate targets within segments to create smooth transitions.

    Useful for creating time-varying targets within a segment.

    Parameters
    ----------
    targets : pd.Series
        Target series with potentially constant values per segment
    method : str, default='linear'
        Interpolation method: 'linear', 'quadratic', 'cubic'

    Returns
    -------
    pd.Series
        Interpolated targets
    """
    # Identify non-NaN segments
    non_nan_mask = ~targets.isna()

    if method == 'linear':
        interpolated = targets.interpolate(method='linear')
    elif method == 'quadratic':
        interpolated = targets.interpolate(method='quadratic')
    elif method == 'cubic':
        interpolated = targets.interpolate(method='cubic')
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    # Only interpolate within segments (don't fill leading/trailing NaNs)
    interpolated[~non_nan_mask] = np.nan

    return interpolated
