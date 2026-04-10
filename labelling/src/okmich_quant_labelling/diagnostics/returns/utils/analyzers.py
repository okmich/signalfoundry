import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


def analyze_target_distribution(targets, target_name="target"):
    """
    Generate statistical summary of target value distribution.

    Parameters
    ----------
    targets : pd.Series or np.ndarray
        Regression targets
    target_name : str, default="target"
        Name for display purposes

    Returns
    -------
    dict
        Dictionary containing statistical measures:
        - count: number of non-NaN values
        - mean: mean value
        - std: standard deviation
        - min/max: minimum and maximum values
        - quartiles: 25th, 50th, 75th percentiles
        - skewness: distribution skewness
        - kurtosis: distribution kurtosis
        - zero_fraction: fraction of zero values
    """
    if isinstance(targets, pd.Series):
        clean_targets = targets.dropna()
    else:
        clean_targets = targets[~np.isnan(targets)]

    if len(clean_targets) == 0:
        return {"error": "No valid target values"}

    zero_fraction = np.sum(clean_targets == 0) / len(clean_targets)

    return {
        "target_name": target_name,
        "count": len(clean_targets),
        "mean": float(np.mean(clean_targets)),
        "std": float(np.std(clean_targets)),
        "min": float(np.min(clean_targets)),
        "max": float(np.max(clean_targets)),
        "q25": float(np.percentile(clean_targets, 25)),
        "q50": float(np.percentile(clean_targets, 50)),
        "q75": float(np.percentile(clean_targets, 75)),
        "skewness": float(stats.skew(clean_targets)),
        "kurtosis": float(stats.kurtosis(clean_targets)),
        "zero_fraction": float(zero_fraction),
    }


def calculate_directional_accuracy(targets, forward_returns):
    """
    Calculate how often sign(target) matches sign(forward_return).

    Parameters
    ----------
    targets : pd.Series or np.ndarray
        Regression targets
    forward_returns : pd.Series or np.ndarray
        Actual forward returns

    Returns
    -------
    float
        Directional accuracy [0, 1]
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
        return np.nan

    target_signs = np.sign(targets_clean)
    return_signs = np.sign(returns_clean)

    accuracy = np.mean(target_signs == return_signs)

    return accuracy


def calculate_target_correlation(targets, forward_returns, method="pearson"):
    """
    Calculate correlation between targets and forward returns.

    Parameters
    ----------
    targets : pd.Series or np.ndarray
        Regression targets
    forward_returns : pd.Series or np.ndarray
        Actual forward returns
    method : str, default="pearson"
        Correlation method: 'pearson', 'spearman', or 'kendall'

    Returns
    -------
    float
        Correlation coefficient
    """
    # Handle NaN values
    if isinstance(targets, pd.Series) and isinstance(forward_returns, pd.Series):
        mask = ~(targets.isna() | forward_returns.isna())
        targets_clean = targets[mask]
        returns_clean = forward_returns[mask]
    else:
        if isinstance(targets, pd.Series):
            targets = targets.values
        if isinstance(forward_returns, pd.Series):
            forward_returns = forward_returns.values

        mask = ~(np.isnan(targets) | np.isnan(forward_returns))
        targets_clean = pd.Series(targets[mask])
        returns_clean = pd.Series(forward_returns[mask])

    if len(targets_clean) < 2:
        return np.nan

    correlation = targets_clean.corr(returns_clean, method=method)

    return correlation


def plot_target_distribution(targets, target_name="Target", title=None, bins=50):
    """
    Visualize target value distribution using plotly.

    Parameters
    ----------
    targets : pd.Series or np.ndarray
        Regression targets
    target_name : str, default="Target"
        Name for axis labels
    title : str, optional
        Plot title
    bins : int, default=50
        Number of histogram bins

    Returns
    -------
    None
        Shows interactive plotly figure
    """
    if isinstance(targets, pd.Series):
        clean_targets = targets.dropna().values
    else:
        clean_targets = targets[~np.isnan(targets)]

    if len(clean_targets) == 0:
        print("No valid target values to plot")
        return

    # Create figure with subplots
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Histogram",
            "Box Plot",
            "Time Series" if isinstance(targets, pd.Series) else "Q-Q Plot",
            "Statistics",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "box"}],
            [{"type": "scatter"}, {"type": "table"}],
        ],
    )

    # Histogram
    fig.add_trace(
        go.Histogram(x=clean_targets, nbinsx=bins, name="Distribution", marker_color="cyan"),
        row=1,
        col=1,
    )

    # Box plot
    fig.add_trace(go.Box(y=clean_targets, name=target_name, marker_color="lightblue"), row=1, col=2)

    # Time series or Q-Q plot
    if isinstance(targets, pd.Series):
        fig.add_trace(
            go.Scatter(x=targets.index, y=targets.values, mode="lines", name="Time Series", line=dict(color="orange")),
            row=2,
            col=1,
        )
    else:
        # Q-Q plot
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(clean_targets)))
        sample_quantiles = np.sort(clean_targets)
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles, y=sample_quantiles, mode="markers", name="Q-Q Plot", marker=dict(color="green")
            ),
            row=2,
            col=1,
        )

    # Statistics table
    stats_dict = analyze_target_distribution(targets, target_name)
    table_data = [[k, f"{v:.4f}" if isinstance(v, float) else v] for k, v in stats_dict.items()]

    fig.add_trace(
        go.Table(
            header=dict(values=["Statistic", "Value"], fill_color="paleturquoise", align="left"),
            cells=dict(
                values=list(zip(*table_data)), fill_color="lavender", align="left"
            ),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=title or f"{target_name} Distribution Analysis",
        template="plotly_dark",
        height=800,
        showlegend=False,
    )

    fig.show()
