from enum import Enum, StrEnum, auto


# ----------------------------------------------------------
# Enum for supported distributions
# ----------------------------------------------------------
class DistType(Enum):
    NORMAL = auto()
    GAMMA = auto()
    LAMDA = auto()
    LOGNORMAL = auto()
    CATEGORICAL = auto()
    POISSON = auto()
    EXPONENTIAL = auto()
    STUDENTT = auto()
    BERNOULLI = auto()


class DurationType(StrEnum):
    """Duration distribution family for HSMM explicit-duration modelling."""
    POISSON = "poisson"
    NEGBIN = "negbin"
    GAMMA = "gamma"
    LOGNORMAL = "lognormal"
    NONPARAMETRIC = "nonparametric"


class InferenceMode(StrEnum):
    """
    Inference mode for HMM predictions.

    - FILTERING: Causal inference using only observations up to time t (Forward algorithm).
                 Best for backtesting and live trading to avoid temporal leakage.
    - SMOOTHING: Non-causal inference using all observations (Forward-Backward algorithm).
                 Best for offline labeling when you have the full dataset.
    - VITERBI: Most likely state sequence (Viterbi algorithm).
               Only applicable to predict(), not predict_proba().
    """

    FILTERING = "filtering"  # Causal - default
    SMOOTHING = "smoothing"  # Non-causal
    VITERBI = "viterbi"  # Most likely path
