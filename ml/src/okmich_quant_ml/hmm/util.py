from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import Optional

import numpy as np


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
    - VITERBI: Most likely state SEQUENCE (true Viterbi: max-product forward + backward traceback).
               Differs from SMOOTHING, which is the per-bar argmax of the marginals. Only applicable
               to predict(), not predict_proba(). Before 2026-09-27 this mode returned pomegranate's
               DenseHMM.predict, i.e. the SMOOTHING labels; see BasePomegranateHMM.predict_viterbi.
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


class SelectionCriterion(StrEnum):
    """
    Information criterion ``train()`` uses to choose among fitted candidates. Lower is better for all three.

    - AIC: ``-2 log L + 2 nu``.
    - BIC: ``-2 log L + nu log n``.
    - ICL: ``BIC + 2 H(z | x)``, the integrated completed likelihood in its BIC form (Biernacki, Celeux & Govaert,
           2000). ``H(z | x)`` is the entropy of the posterior over the whole state path, so ICL also penalises
           states the posterior cannot tell apart. It looks for well-separated states, not for the number of
           density components, which BIC estimates consistently and ICL does not.

    All three compare models fitted to the same X only. None of them can rank feature sets.
    """

    AIC = "aic"
    BIC = "bic"
    ICL = "icl"


def as_selection_criterion(value: "SelectionCriterion | str") -> SelectionCriterion:
    """Coerce a criterion given as an enum member or its string value; an unknown value raises, naming the allowed."""
    try:
        return SelectionCriterion(value)
    except ValueError:
        allowed = [c.value for c in SelectionCriterion]
        raise ValueError(f"Unknown criterion '{value}'. Allowed values: {allowed}") from None


@dataclass(frozen=True)
class SelectionScores:
    """
    One fitted model's selection scores on one dataset.

    ``path_entropy`` is the exact ``H(z_1:T | x)`` in nats (``BasePomegranateHMM.path_entropy``). It is ``None`` when
    it was not computed, and ``icl`` then raises.
    """

    log_likelihood: float
    n_params: int
    n_obs: int
    path_entropy: Optional[float] = None

    @property
    def aic(self) -> float:
        return float(-2 * self.log_likelihood + 2 * self.n_params)

    @property
    def bic(self) -> float:
        return float(-2 * self.log_likelihood + self.n_params * np.log(self.n_obs))

    @property
    def icl(self) -> float:
        if self.path_entropy is None:
            raise ValueError("ICL needs the path entropy: score with with_entropy=True.")
        return self.bic + 2 * self.path_entropy

    def score(self, criterion: "SelectionCriterion | str") -> float:
        """The value of ``criterion`` (lower is better)."""
        return getattr(self, as_selection_criterion(criterion).value)
