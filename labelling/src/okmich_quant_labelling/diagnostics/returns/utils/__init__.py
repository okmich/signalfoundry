from .analyzers import (
    analyze_target_distribution,
    calculate_directional_accuracy,
    calculate_target_correlation,
    plot_target_distribution,
)
from .binning import (
    bin_targets_by_quantile,
    bin_targets_by_threshold,
    create_balanced_bins,
)
from .normalizers import (
    normalize_by_volatility,
    calculate_log_returns,
    calculate_volatility,
    clip_by_percentile,
    clip_by_std,
)
from .segment_utils import (
    apply_target_to_segment,
    merge_segment_targets,
    smooth_segment_boundaries,
)
from .target_calculators import (
    calculate_slope,
    calculate_momentum,
    calculate_cumulative_return,
    calculate_amplitude_per_bar,
    calculate_return_to_extreme,
    calculate_forward_return,
    calculate_percentage_from_extreme,
)
from .validators import (
    compare_regression_to_classification,
    calculate_regression_metrics,
    detect_lookahead_bias,
    validate_causality,
)

__all__ = [
    # Target calculators
    "calculate_slope",
    "calculate_momentum",
    "calculate_cumulative_return",
    "calculate_amplitude_per_bar",
    "calculate_return_to_extreme",
    "calculate_forward_return",
    "calculate_percentage_from_extreme",
    # Normalizers
    "normalize_by_volatility",
    "calculate_log_returns",
    "calculate_volatility",
    "clip_by_percentile",
    "clip_by_std",
    # Validators
    "compare_regression_to_classification",
    "calculate_regression_metrics",
    "detect_lookahead_bias",
    "validate_causality",
    # Analyzers
    "analyze_target_distribution",
    "calculate_directional_accuracy",
    "calculate_target_correlation",
    "plot_target_distribution",
    # Binning
    "bin_targets_by_quantile",
    "bin_targets_by_threshold",
    "create_balanced_bins",
    # Segment utils
    "apply_target_to_segment",
    "merge_segment_targets",
    "smooth_segment_boundaries",
]