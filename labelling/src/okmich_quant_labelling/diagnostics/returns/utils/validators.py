import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compare_regression_to_classification(reg_targets, class_labels):
    """
    Verify that regression target signs match classification labels.

    Parameters
    ----------
    reg_targets : pd.Series or np.ndarray
        Continuous regression targets
    class_labels : pd.Series or np.ndarray
        Discrete classification labels (-1, 0, +1)

    Returns
    -------
    dict
        Dictionary containing:
        - agreement_rate: fraction where sign(target) == label
        - mismatches: number of mismatches
        - total: total number of samples
        - agreement_by_class: agreement rate per class
    """
    # Convert to numpy for easier processing
    if isinstance(reg_targets, pd.Series):
        reg_targets = reg_targets.values
    if isinstance(class_labels, pd.Series):
        class_labels = class_labels.values

    # Get signs of regression targets
    reg_signs = np.sign(reg_targets)

    # Calculate agreement
    matches = (reg_signs == class_labels)
    agreement_rate = np.mean(matches)
    mismatches = np.sum(~matches)
    total = len(matches)

    # Agreement by class
    agreement_by_class = {}
    for label in [-1, 0, 1]:
        mask = (class_labels == label)
        if np.sum(mask) > 0:
            class_agreement = np.mean(matches[mask])
            agreement_by_class[label] = class_agreement

    return {
        "agreement_rate": agreement_rate,
        "mismatches": mismatches,
        "total": total,
        "agreement_by_class": agreement_by_class,
    }


def calculate_regression_metrics(targets, forward_returns, periods=5):
    """
    Evaluate regression target quality against forward returns.

    Parameters
    ----------
    targets : pd.Series or np.ndarray
        Regression targets
    forward_returns : pd.Series or np.ndarray
        Actual forward returns
    periods : int, default=5
        Number of periods for forward returns

    Returns
    -------
    dict
        Dictionary containing:
        - r2_score: R² (coefficient of determination)
        - correlation: Pearson correlation
        - mse: Mean squared error
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - directional_accuracy: How often sign(target) == sign(return)
    """
    # Handle NaN values
    if isinstance(targets, pd.Series):
        mask = ~(targets.isna() | forward_returns.isna())
        targets_clean = targets[mask].values
        returns_clean = forward_returns[mask].values
    else:
        mask = ~(np.isnan(targets) | np.isnan(forward_returns))
        targets_clean = targets[mask]
        returns_clean = forward_returns[mask]

    if len(targets_clean) == 0:
        return {
            "r2_score": np.nan,
            "correlation": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "directional_accuracy": np.nan,
        }

    # Calculate metrics
    r2 = r2_score(returns_clean, targets_clean)
    correlation = np.corrcoef(targets_clean, returns_clean)[0, 1]
    mse = mean_squared_error(returns_clean, targets_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(returns_clean, targets_clean)

    # Directional accuracy
    target_signs = np.sign(targets_clean)
    return_signs = np.sign(returns_clean)
    directional_accuracy = np.mean(target_signs == return_signs)

    return {
        "r2_score": r2,
        "correlation": correlation,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": directional_accuracy,
    }


def detect_lookahead_bias(df, target_col, price_col="close", max_lag=10):
    """
    Detect potential lookahead bias by checking if targets correlate with future prices.

    A causal target should NOT correlate strongly with future price changes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with targets and prices
    target_col : str
        Column name for regression targets
    price_col : str, default="close"
        Column name for prices
    max_lag : int, default=10
        Maximum number of forward lags to check

    Returns
    -------
    dict
        Dictionary containing:
        - max_future_correlation: highest correlation with future prices
        - max_lag: lag with highest correlation
        - all_correlations: dict of {lag: correlation}
        - has_lookahead: bool (True if max_correlation > 0.3)
    """
    # Calculate future price changes
    future_correlations = {}

    for lag in range(1, max_lag + 1):
        future_returns = df[price_col].pct_change(lag).shift(-lag)

        # Calculate correlation
        mask = ~(df[target_col].isna() | future_returns.isna())
        if mask.sum() > 0:
            corr = df.loc[mask, target_col].corr(future_returns[mask])
            future_correlations[lag] = corr

    if len(future_correlations) == 0:
        return {
            "max_future_correlation": np.nan,
            "max_lag": None,
            "all_correlations": {},
            "has_lookahead": False,
        }

    # Find maximum correlation
    max_lag_key = max(future_correlations, key=lambda k: abs(future_correlations[k]))
    max_correlation = future_correlations[max_lag_key]

    return {
        "max_future_correlation": max_correlation,
        "max_lag": max_lag_key,
        "all_correlations": future_correlations,
        "has_lookahead": abs(max_correlation) > 0.3,  # Threshold for concern
    }


def validate_causality(targets, segment_info):
    """
    Ensure targets only use past data by checking segment structure.

    Parameters
    ----------
    targets : pd.Series
        Regression targets
    segment_info : pd.DataFrame
        DataFrame with columns: ['start_idx', 'end_idx', 'direction']

    Returns
    -------
    dict
        Dictionary containing:
        - is_causal: bool (True if all targets use only past data)
        - violations: list of segment indices with violations
        - total_segments: total number of segments checked
    """
    violations = []

    for idx, row in segment_info.iterrows():
        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])

        if end_idx <= start_idx:
            continue

        segment_targets = targets.iloc[start_idx : end_idx + 1].dropna()
        if len(segment_targets) < 2:
            continue

        t_start = float(segment_targets.iloc[0])
        t_end = float(segment_targets.iloc[-1])

        # Pattern 1: FORWARD_RETURN / RETURN_TO_EXTREME
        #   Target at segment end converges to ~0 while start is non-zero.
        #   Remaining-return targets shrink to zero as bar approaches end_idx.
        if abs(t_start) > 1e-8 and abs(t_end) < abs(t_start) * 0.05:
            violations.append(
                {
                    "segment_idx": idx,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "reason": "forward_return_pattern",
                    "t_start": t_start,
                    "t_end": t_end,
                }
            )
            continue

        # Pattern 2: AMPLITUDE_PER_BAR
        #   All bars in segment carry the exact same constant value (segment-level
        #   aggregate assigned uniformly — computed from the full segment endpoint).
        if len(segment_targets) >= 3:
            seg_std = float(segment_targets.std())
            seg_mean = float(segment_targets.mean())
            if seg_std < 1e-10 * max(abs(seg_mean), 1.0):
                violations.append(
                    {
                        "segment_idx": idx,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "reason": "constant_segment_value",
                        "value": t_start,
                    }
                )

    is_causal = len(violations) == 0

    return {
        "is_causal": is_causal,
        "violations": violations,
        "total_segments": len(segment_info),
    }
