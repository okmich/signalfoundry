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


class InferenceMode(StrEnum):
    """
    Inference mode for HMM predictions.

    - FILTERING: Causal inference using only observations up to time t (Forward algorithm).
                 Best for backtesting and live trading to avoid temporal leakage.
    - SMOOTHING: Non-causal inference using all observations (Forward-Backward algorithm).
                 Best for offline labeling when you have the full dataset.
    - VITERBI: Most likely state sequence (Viterbi algorithm).
               Only applicable to predict(), not predict_proba().
    - CAUSAL_VITERBI: Terminal state of the best path over observations up to time t
               (max-product forward recursion, no traceback). Causal, so it is the
               Viterbi-family member that is safe for backtesting and live trading.
               Only applicable to predict(), not predict_proba().

    On the two Viterbi members: standard VITERBI is non-causal not because it uses the
    transition matrix, but because of its final two steps - it selects the winner from the
    frontier at ``T`` and then traces backwards, rewriting every earlier label to agree with
    that choice. A label already emitted can therefore change when new bars arrive.
    CAUSAL_VITERBI keeps the identical forward recursion and simply drops the traceback,
    reading ``argmax_k delta_t(k)`` at every bar instead of only at ``T``.
    """

    FILTERING = "filtering"  # Causal - default
    SMOOTHING = "smoothing"  # Non-causal
    VITERBI = "viterbi"  # Most likely path - NON-CAUSAL (backward traceback)
    CAUSAL_VITERBI = "causal_viterbi"  # Most likely path SO FAR - causal, no traceback
