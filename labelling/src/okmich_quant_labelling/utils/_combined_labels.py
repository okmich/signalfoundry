"""
Combined Multi-Dimensional Labeling Utilities

This module provides functionality for combining multiple labeling dimensions
(path structure, volatility, momentum) into a single unified label space.

The canonical use case is 18-state combined labeling:
- 3 Path Structure states (High/Medium/Low Choppiness or Efficiency)
- 3 Volatility states (High/Medium/Low Volatility)
- 2 Momentum states (Positive/Negative Direction)
- Total: 3 × 3 × 2 = 18 combined states

This creates a rich, multi-dimensional regime characterization that captures:
- How choppy/smooth the price path is
- How volatile the price movement is
- What direction the momentum is in

Example Combined Regimes:
------------------------
- State 0 (0-0-0): High Chop + Low Vol + Negative Momentum → Weak downtrend consolidation
- State 8 (1-1-1): Med Chop + Med Vol + Positive Momentum → Moderate uptrend
- State 17 (2-2-1): Low Chop + High Vol + Positive Momentum → Strong explosive uptrend

The encoding scheme uses base-N notation:
    combined_label = label_1 * (N2 * N3) + label_2 * N3 + label_3

where N2, N3 are the number of states in dimensions 2 and 3.
"""

from typing import List, Dict, Union

import numpy as np
import pandas as pd


def combine_labels(
        labels: List[pd.Series], names: List[str] = None, label_order: List[int] = None
) -> pd.Series:
    """
    Combine multiple label series into a single multi-dimensional label.

    Uses base-N encoding to create unique integer labels for each combination.

    Parameters
    ----------
    labels : List[pd.Series]
        List of label series to combine. Each series should contain integer labels
        starting from 0. The first series is the most significant dimension.

    names : List[str], optional
        Names for each dimension (for metadata). If None, uses generic names.

    label_order : List[int], optional
        Expected number of states for each dimension. If None, infers from data.
        Useful for validation and ensuring consistent encoding.

    Returns
    -------
    pd.Series
        Combined labels with unique integer for each combination.
        NaN where any input label is NaN.

    Examples
    --------
    >>> path_labels = pd.Series([0, 1, 2, 0, 1])  # 3 states
    >>> vol_labels = pd.Series([0, 0, 1, 2, 2])   # 3 states
    >>> mom_labels = pd.Series([0, 1, 1, 0, 1])   # 2 states
    >>> combined = combine_labels([path_labels, vol_labels, mom_labels])
    >>> # Result: [0, 7, 17, 6, 11]
    >>> # State 0 = (0,0,0), State 7 = (1,0,1), State 17 = (2,1,1), etc.

    Notes
    -----
    The encoding uses least-significant dimension last:
        combined = labels[0] * prod(n_states[1:]) + labels[1] * prod(n_states[2:]) + ... + labels[-1]

    This ensures that the first dimension varies slowest (most significant).
    """
    if len(labels) == 0:
        raise ValueError("Must provide at least one label series")

    # Validate all series have same length
    lengths = [len(s) for s in labels]
    if len(set(lengths)) > 1:
        raise ValueError(f"All label series must have same length. Got: {lengths}")

    n_dims = len(labels)
    n_obs = lengths[0]

    # Infer or validate number of states per dimension
    if label_order is None:
        label_order = [int(s.max()) + 1 for s in labels]
    else:
        if len(label_order) != n_dims:
            raise ValueError(
                f"label_order length ({len(label_order)}) must match number of dimensions ({n_dims})"
            )

        # Validate data doesn't exceed expected states
        for i, (s, expected) in enumerate(zip(labels, label_order)):
            actual_max = int(s.max())
            if actual_max >= expected:
                raise ValueError(
                    f"Dimension {i} has label {actual_max} but expected max is {expected - 1}"
                )

    # Use provided names or defaults
    if names is None:
        names = [f"dim_{i}" for i in range(n_dims)]
    elif len(names) != n_dims:
        raise ValueError(
            f"names length ({len(names)}) must match number of dimensions ({n_dims})"
        )

    # Calculate combined labels using base-N encoding
    combined = pd.Series(0, index=labels[0].index, dtype="Int64")

    # Track NaN mask - combined label is NaN if ANY input is NaN
    nan_mask = pd.Series(False, index=labels[0].index)
    for s in labels:
        nan_mask |= s.isna()

    # Build combined label (most significant dimension first)
    multiplier = 1
    for i in range(n_dims - 1, -1, -1):
        combined += labels[i].fillna(0).astype(int) * multiplier
        multiplier *= label_order[i]

    # Apply NaN mask
    combined = combined.astype("float64")  # Convert to float to support NaN
    combined.loc[nan_mask] = np.nan

    # Add metadata
    combined.name = "combined_label"

    # Store encoding scheme as attributes (for decoding later)
    combined.attrs["n_dimensions"] = n_dims
    combined.attrs["label_order"] = label_order
    combined.attrs["dimension_names"] = names
    combined.attrs["n_states"] = int(np.prod(label_order))

    return combined


def decode_combined_labels(combined: pd.Series) -> pd.DataFrame:
    """
    Decode combined labels back into individual dimension labels.

    Parameters
    ----------
    combined : pd.Series
        Combined labels from combine_labels(). Must have 'label_order' attribute.

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per dimension containing decoded labels.

    Examples
    --------
    >>> combined = combine_labels([path_labels, vol_labels, mom_labels])
    >>> decoded = decode_combined_labels(combined)
    >>> # Returns DataFrame with columns: dim_0, dim_1, dim_2

    Notes
    -----
    This is the inverse operation of combine_labels().
    """
    if "label_order" not in combined.attrs:
        raise ValueError(
            "Combined labels must have 'label_order' attribute. "
            "Was this created with combine_labels()?"
        )

    label_order = combined.attrs["label_order"]
    names = combined.attrs.get(
        "dimension_names", [f"dim_{i}" for i in range(len(label_order))]
    )
    n_dims = len(label_order)

    # Decode using base-N notation
    decoded = {}
    remainder = combined.fillna(-1).astype(int)  # Use -1 for NaN temporarily

    for i in range(n_dims - 1, -1, -1):
        decoded[names[i]] = remainder % label_order[i]
        remainder = remainder // label_order[i]

    df = pd.DataFrame(decoded, index=combined.index)

    # Restore NaN where original was NaN
    nan_mask = combined.isna()
    for col in df.columns:
        df.loc[nan_mask, col] = np.nan
        df[col] = df[col].astype("Int64")

    return df[names]  # Reorder columns to match input order


def get_combined_label_description(
        combined_label: int,
        label_order: List[int],
        dimension_names: List[str],
        state_names: List[List[str]] = None,
) -> Dict[str, Union[int, str]]:
    """
    Get human-readable description of a combined label state.

    Parameters
    ----------
    combined_label : int
        The combined label to describe (0 to n_states-1)

    label_order : List[int]
        Number of states in each dimension

    dimension_names : List[str]
        Names of each dimension

    state_names : List[List[str]], optional
        Human-readable names for states in each dimension.
        If None, uses generic "State 0", "State 1", etc.

    Returns
    -------
    dict
        Dictionary with dimension names as keys and state info as values.

    Examples
    --------
    >>> label_order = [3, 3, 2]
    >>> dim_names = ['path_structure', 'volatility', 'momentum']
    >>> state_names = [
    ...     ['High Chop', 'Med Chop', 'Low Chop'],
    ...     ['Low Vol', 'Med Vol', 'High Vol'],
    ...     ['Negative', 'Positive']
    ... ]
    >>> desc = get_combined_label_description(17, label_order, dim_names, state_names)
    >>> # Returns: {'path_structure': 'Low Chop', 'volatility': 'High Vol', 'momentum': 'Positive'}
    """
    if combined_label < 0 or combined_label >= np.prod(label_order):
        raise ValueError(
            f"combined_label {combined_label} out of range [0, {np.prod(label_order) - 1}]"
        )

    n_dims = len(label_order)

    # Decode the label
    indices = []
    remainder = combined_label
    for i in range(n_dims - 1, -1, -1):
        indices.append(remainder % label_order[i])
        remainder = remainder // label_order[i]
    indices = indices[::-1]  # Reverse to get original order

    # Build description
    description = {"combined_label": combined_label}

    for i, (dim_name, state_idx) in enumerate(zip(dimension_names, indices)):
        if state_names is not None and i < len(state_names):
            state_name = state_names[i][state_idx]
        else:
            state_name = f"State {state_idx}"

        description[dim_name] = state_name
        description[f"{dim_name}_idx"] = state_idx

    return description


def create_18state_labels(
        path_structure_labels: pd.Series,
        volatility_labels: pd.Series,
        momentum_labels: pd.Series,
        path_order: str = "ascending",
        vol_order: str = "ascending",
        mom_order: str = "ascending",
) -> pd.Series:
    """
    Create canonical 18-state combined labels from path structure, volatility, and momentum.

    This is a convenience wrapper around combine_labels() with predefined ordering
    and interpretation for the standard 3×3×2 regime space.

    Parameters
    ----------
    path_structure_labels : pd.Series
        Path structure labels (0, 1, 2).
        Typically from directional_efficiency labeling.

    volatility_labels : pd.Series
        Volatility labels (0, 1, 2).
        Typically from forward_realized_volatility labeling.

    momentum_labels : pd.Series
        Momentum labels (0, 1).
        Should be binary: 0 = Negative, 1 = Positive.
        If 3-state momentum labels provided, automatically maps to binary:
        - Original 0 (negative) → 0
        - Original 1 (neutral) → (dropped or mapped based on velocity sign)
        - Original 2 (positive) → 1

    path_order : str, default='ascending'
        How path structure labels map to interpretation:
        - 'ascending': 0=High Chop, 1=Med Chop, 2=Low Chop (smooth)
        - 'descending': 0=Low Chop (smooth), 1=Med Chop, 2=High Chop

    vol_order : str, default='ascending'
        How volatility labels map to interpretation:
        - 'ascending': 0=Low Vol, 1=Med Vol, 2=High Vol
        - 'descending': 0=High Vol, 1=Med Vol, 2=Low Vol

    mom_order : str, default='ascending'
        How momentum labels map to interpretation:
        - 'ascending': 0=Negative, 1=Positive
        - 'descending': 0=Positive, 1=Negative

    Returns
    -------
    pd.Series
        Combined 18-state labels (0-17) with metadata attributes.

    Examples
    --------
    >>> from okmich_quant_features.path_structure.labelling import (
    ...     directional_efficiency, percentile_labels
    ... )
    >>> from okmich_quant_features.volatility.labelling import (
    ...     forward_realized_volatility
    ... )
    >>> from okmich_quant_features.momentum.labelling import (
    ...     sustained_forward_velocity, percentile_labels_symmetric
    ... )
    >>>
    >>> # Calculate forward features
    >>> efficiency = directional_efficiency(prices, lookahead=24)
    >>> volatility = forward_realized_volatility(prices, lookahead=24)
    >>> velocity = sustained_forward_velocity(prices, lookahead=24)
    >>>
    >>> # Create labels
    >>> path_labels = percentile_labels(efficiency, n_states=3)
    >>> vol_labels = percentile_labels(volatility, n_states=3)
    >>> mom_labels_3state = percentile_labels_symmetric(velocity, n_states=3)
    >>>
    >>> # Convert to 18-state
    >>> combined = create_18state_labels(path_labels, vol_labels, mom_labels_3state)
    >>> print(f"Total states: {combined.attrs['n_states']}")  # 18

    Notes
    -----
    Standard interpretation (with ascending order):

    Path Structure (Directional Efficiency):
    - 0: High Choppiness (low efficiency) - ranging/choppy
    - 1: Medium Choppiness - transitional
    - 2: Low Choppiness (high efficiency) - smooth trending

    Volatility (Realized Volatility):
    - 0: Low Volatility - quiet/compressed
    - 1: Medium Volatility - normal
    - 2: High Volatility - expanded/volatile

    Momentum (Sustained Velocity):
    - 0: Negative - downward momentum
    - 1: Positive - upward momentum

    Example Combined States:
    - State 0 (0,0,0): High Chop + Low Vol + Negative = Weak bear consolidation
    - State 8 (1,1,0): Med Chop + Med Vol + Negative = Moderate bear trend
    - State 9 (1,1,1): Med Chop + Med Vol + Positive = Moderate bull trend
    - State 17 (2,2,1): Low Chop + High Vol + Positive = Strong bull breakout
    """
    # Validate inputs
    for labels, name in [
        (path_structure_labels, "path_structure"),
        (volatility_labels, "volatility"),
        (momentum_labels, "momentum"),
    ]:
        if not isinstance(labels, pd.Series):
            raise TypeError(f"{name}_labels must be a pandas Series")

    # Handle 3-state momentum conversion to binary
    mom_processed = momentum_labels.copy()
    n_mom_states = int(momentum_labels.max()) + 1

    if n_mom_states == 3:
        # Map 3-state to binary: 0→0, 1→based on context, 2→1
        # For neutral (1), we'll drop them or map based on sign (requires velocity)
        # Simplest: 0→0, 1,2→1 (conservative mapping)
        mom_processed = momentum_labels.copy()
        mom_processed = mom_processed.replace({0: 0, 1: np.nan, 2: 1})  # Drop neutral
        print(
            "Warning: 3-state momentum converted to binary. Neutral labels (1) set to NaN."
        )
        print(
            "For better results, use binary momentum labels (negative/positive only)."
        )
    elif n_mom_states != 2:
        raise ValueError(f"momentum_labels must have 2 or 3 states, got {n_mom_states}")

    # Reorder labels if needed based on interpretation
    path_proc = _reorder_labels(path_structure_labels, path_order)
    vol_proc = _reorder_labels(volatility_labels, vol_order)
    mom_proc = _reorder_labels(mom_processed, mom_order)

    # Combine
    combined = combine_labels(
        [path_proc, vol_proc, mom_proc],
        names=["path_structure", "volatility", "momentum"],
        label_order=[3, 3, 2],
    )

    # Add interpretation metadata
    combined.attrs["path_interpretation"] = {
        0: "High Chop" if path_order == "ascending" else "Low Chop",
        1: "Med Chop",
        2: "Low Chop" if path_order == "ascending" else "High Chop",
    }
    combined.attrs["vol_interpretation"] = {
        0: "Low Vol" if vol_order == "ascending" else "High Vol",
        1: "Med Vol",
        2: "High Vol" if vol_order == "ascending" else "Low Vol",
    }
    combined.attrs["mom_interpretation"] = {
        0: "Negative" if mom_order == "ascending" else "Positive",
        1: "Positive" if mom_order == "ascending" else "Negative",
    }

    return combined


def _reorder_labels(labels: pd.Series, order: str) -> pd.Series:
    """Helper to reorder labels based on interpretation."""
    if order == "ascending":
        return labels.copy()
    elif order == "descending":
        n_states = int(labels.max()) + 1
        return (n_states - 1 - labels).astype("Int64")
    else:
        raise ValueError(f"order must be 'ascending' or 'descending', got {order}")


def describe_18state_regime(state: int) -> str:
    """
    Get human-readable description of an 18-state regime.

    Parameters
    ----------
    state : int
        Combined state (0-17)

    Returns
    -------
    str
        Human-readable description

    Examples
    --------
    >>> describe_18state_regime(0)
    'State 0: High Chop + Low Vol + Negative → Weak bear consolidation'
    >>> describe_18state_regime(17)
    'State 17: Low Chop + High Vol + Positive → Strong bull breakout'
    """
    if state < 0 or state > 17:
        raise ValueError(f"state must be 0-17, got {state}")

    # Decode
    path_idx = state // 6
    vol_idx = (state % 6) // 2
    mom_idx = state % 2

    path_names = ["High Chop", "Med Chop", "Low Chop"]
    vol_names = ["Low Vol", "Med Vol", "High Vol"]
    mom_names = ["Negative", "Positive"]

    # Interpretation
    interpretations = {
        0: "Weak bear consolidation",
        1: "Bear consolidation",
        2: "Moderate bear consolidation",
        3: "Bull consolidation",
        4: "Moderate bull consolidation",
        5: "Strong bull consolidation",
        6: "Choppy bear range",
        7: "Choppy bull range",
        8: "Moderate bear trend",
        9: "Moderate bull trend",
        10: "Volatile bear range",
        11: "Volatile bull range",
        12: "Smooth bear trend",
        13: "Smooth bull trend",
        14: "Strong bear trend",
        15: "Strong bull trend",
        16: "Strong bear breakout",
        17: "Strong bull breakout",
    }

    components = f"{path_names[path_idx]} + {vol_names[vol_idx]} + {mom_names[mom_idx]}"
    interpretation = interpretations.get(state, "Unknown regime")

    return f"State {state}: {components} → {interpretation}"


def get_combined_state_statistics(
        df: pd.DataFrame,
        combined_label_col: str,
        price_col: str = "close",
        returns_col: str = "returns",
) -> pd.DataFrame:
    """
    Calculate statistics for each combined state.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing combined labels, prices, and returns

    combined_label_col : str
        Column name containing combined labels

    price_col : str, default='close'
        Column name for prices

    returns_col : str, default='returns'
        Column name for returns

    Returns
    -------
    pd.DataFrame
        Statistics for each state including:
        - count, frequency, mean_return, volatility, sharpe, max_drawdown
    """
    combined = df[combined_label_col]

    stats = []
    for state in range(int(combined.min()), int(combined.max()) + 1):
        mask = combined == state
        state_data = df[mask]

        if len(state_data) == 0:
            continue

        stat = {
            "state": state,
            "description": describe_18state_regime(state),
            "count": len(state_data),
            "frequency_pct": len(state_data) / len(df) * 100,
            "mean_return": (
                state_data[returns_col].mean() if returns_col in df.columns else np.nan
            ),
            "volatility": (
                state_data[returns_col].std() if returns_col in df.columns else np.nan
            ),
        }

        # Sharpe ratio (annualized for 5min bars: sqrt(252*24*12))
        if returns_col in df.columns and state_data[returns_col].std() > 0:
            stat["sharpe"] = (
                                     state_data[returns_col].mean() / state_data[returns_col].std()
                             ) * np.sqrt(252 * 24 * 12)
        else:
            stat["sharpe"] = np.nan

        stats.append(stat)

    return pd.DataFrame(stats)
