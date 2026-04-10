from enum import Enum


class RegressionTargetType(Enum):
    """
    Regression target types for continuous trend labeling.

    Causal — only use past data, safe for live predictions:
        PERCENTAGE_FROM_EXTREME  distance from segment start as % of start price
        CUMULATIVE_RETURN        total log (or simple) return from segment start
        MOMENTUM                 cumulative return normalized by elapsed bars
        SLOPE                    OLS beta of log-price fitted from segment start to current bar

    Forward-looking — use future segment information, valid only as training labels:
        AMPLITUDE_PER_BAR        total amplitude of the full segment / segment duration
        FORWARD_RETURN           total return from current bar to segment end
        FORWARD_RETURN_PER_BAR   forward return normalized by remaining bars
        RETURN_TO_EXTREME        remaining return from current bar to segment peak / trough

    Applicability by labeler:
        AutoLabelRegression              — all except AMPLITUDE_PER_BAR, FORWARD_RETURN,
                                           FORWARD_RETURN_PER_BAR
        AmplitudeBasedRegressionLabeler  — all except PERCENTAGE_FROM_EXTREME
    """
    # ------------------------------------------------------------------ causal
    PERCENTAGE_FROM_EXTREME = "percentage_from_extreme"
    CUMULATIVE_RETURN = "cumulative_return"
    MOMENTUM = "momentum"
    SLOPE = "slope"

    # --------------------------------------------------------- forward-looking
    AMPLITUDE_PER_BAR = "amplitude_per_bar"
    FORWARD_RETURN = "forward_return"
    FORWARD_RETURN_PER_BAR = "forward_return_per_bar"
    RETURN_TO_EXTREME = "return_to_extreme"


# ---------------------------------------------------------------------------
# Integer IDs for Numba dispatch — must stay in sync with _numba_kernels.py
# ---------------------------------------------------------------------------
TT_PERCENTAGE_FROM_EXTREME = 0
TT_CUMULATIVE_RETURN = 1
TT_MOMENTUM = 2
TT_SLOPE = 3
TT_AMPLITUDE_PER_BAR = 4
TT_FORWARD_RETURN = 5
TT_FORWARD_RETURN_PER_BAR = 6
TT_RETURN_TO_EXTREME = 7

REGRESSION_TARGET_TYPE_IDS: dict = {
    RegressionTargetType.PERCENTAGE_FROM_EXTREME: TT_PERCENTAGE_FROM_EXTREME,
    RegressionTargetType.CUMULATIVE_RETURN: TT_CUMULATIVE_RETURN,
    RegressionTargetType.MOMENTUM: TT_MOMENTUM,
    RegressionTargetType.SLOPE: TT_SLOPE,
    RegressionTargetType.AMPLITUDE_PER_BAR: TT_AMPLITUDE_PER_BAR,
    RegressionTargetType.FORWARD_RETURN: TT_FORWARD_RETURN,
    RegressionTargetType.FORWARD_RETURN_PER_BAR: TT_FORWARD_RETURN_PER_BAR,
    RegressionTargetType.RETURN_TO_EXTREME: TT_RETURN_TO_EXTREME,
}
