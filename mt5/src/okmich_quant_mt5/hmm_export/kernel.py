"""NumPy mirror of ``HmmCore.mqh`` - the reference the MQL5 kernel is tested against.

Two things are being pinned here.

1. **Streaming equals batch.** The MQL5 kernel renormalises the frontier every
   bar (subtracting ``logsumexp`` under FILTERING, the row max under
   CAUSAL_VITERBI) so the scores stay bounded over a long history. Both shifts
   are exactly neutral - the filtered posterior is scale-free and ``argmax`` is
   shift-invariant - but "exactly neutral" is a claim, so
   ``tests/hmm_export/test_kernel.py`` checks this class against
   ``BasePomegranateHMM._forward_pass`` / ``_max_product_forward_pass``.

2. **Causality.** ``step`` consumes one observation and can never see a later
   one. There is no traceback, so a label once emitted is final. That is the
   whole reason CAUSAL_VITERBI is tradable where VITERBI is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .features import FeatureSpec

#: Mirrors the floor applied in ``BasePomegranateHMM._extract_hmm_parameters``.
#: Keeping it means the exported log-parameters are finite, so neither side
#: ever performs -inf arithmetic and neither can produce a NaN.
LOG_ZERO = float(np.log(1e-300))

#: Label written where no state is published (warm-up, burn-in, forming bar).
LABEL_NONE = -1


class Mql5InferenceMode(StrEnum):
    """The causal subset of ``okmich_quant_ml.hmm.util.InferenceMode``.

    Values match that enum's, so the two interoperate directly. SMOOTHING and
    VITERBI are absent on purpose: both rewrite already-emitted labels when new
    bars arrive, which repaints an indicator.
    """

    FILTERING = "filtering"
    CAUSAL_VITERBI = "causal_viterbi"


@dataclass(frozen=True)
class Mql5HmmParams:
    """Everything the MQL5 kernel needs, in the form it needs it.

    ``sigma_inv`` and ``log_norm`` are precomputed in float64 here so MQL5 never
    inverts a matrix or evaluates a log-determinant.
    """

    model_id: str
    spec: FeatureSpec
    log_pi: np.ndarray  # (K,)
    log_a: np.ndarray  # (K, K)
    mu: np.ndarray  # (K, D)
    sigma_inv: np.ndarray  # (K, D, D)
    log_norm: np.ndarray  # (K,)  = -0.5 * (D*log(2pi) + logdet(Sigma_k))
    feat_mean: np.ndarray  # (D,)
    feat_sd: np.ndarray  # (D,)

    def __post_init__(self) -> None:
        k, d = self.n_states, self.n_features
        expected = {
            "log_pi": (k,),
            "log_a": (k, k),
            "mu": (k, d),
            "sigma_inv": (k, d, d),
            "log_norm": (k,),
            "feat_mean": (d,),
            "feat_sd": (d,),
        }
        for name, shape in expected.items():
            got = np.asarray(getattr(self, name)).shape
            if got != shape:
                raise ValueError(f"Mql5HmmParams.{name}: expected shape {shape}, got {got}")
        if d != self.spec.n_features:
            raise ValueError(f"mu has {d} columns but spec declares {self.spec.n_features} features")
        if not np.all(self.feat_sd > 0.0):
            raise ValueError("feat_sd must be strictly positive (use 1.0 for no scaling)")

    @property
    def n_states(self) -> int:
        return int(np.asarray(self.log_pi).shape[0])

    @property
    def n_features(self) -> int:
        return int(np.asarray(self.mu).shape[1])

    def standardise(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.feat_mean) / self.feat_sd

    def log_emissions(self, x: np.ndarray) -> np.ndarray:
        """``log N(x; mu_k, Sigma_k)`` for every state. Mirrors ``HmmLogEmissions``."""
        delta = np.asarray(x, dtype=np.float64)[None, :] - self.mu  # (K, D)
        quad = np.einsum("kd,kde,ke->k", delta, self.sigma_inv, delta)
        return self.log_norm - 0.5 * quad


class Mql5HmmKernel:
    """Streaming causal recursion. Mirrors ``CHmmRecursion`` in ``HmmCore.mqh``."""

    def __init__(self, params: Mql5HmmParams, mode: Mql5InferenceMode) -> None:
        self.params = params
        self.mode = Mql5InferenceMode(mode)
        self._score = np.zeros(params.n_states, dtype=np.float64)
        self._started = False
        self._last_loglik = 0.0
        self._steps = 0

    def reset(self) -> None:
        self._score = np.zeros(self.params.n_states, dtype=np.float64)
        self._started = False
        self._last_loglik = 0.0
        self._steps = 0

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def score(self) -> np.ndarray:
        return self._score.copy()

    @property
    def last_predictive_loglik(self) -> float:
        """``log P(o_t | o_0..o_{t-1})`` - the constant removed by renormalising. FILTERING only."""
        return self._last_loglik

    def step(self, x: np.ndarray) -> bool:
        """Advance one bar. Returns False if no feasible path survives."""
        log_b = self.params.log_emissions(self.params.standardise(x))
        if not np.all(np.isfinite(log_b)):
            # An observation far enough from every mean overflows the quadratic
            # form. Report infeasibility rather than let a NaN reach argmax,
            # which would silently tie on state 0.
            return False

        if not self._started:
            self._score = self.params.log_pi + log_b
            self._started = True
        else:
            joint = self._score[:, None] + self.params.log_a  # (K_from, K_to)
            if self.mode == Mql5InferenceMode.FILTERING:
                # np.logaddexp.reduce, matching BasePomegranateHMM._forward_pass
                self._score = np.logaddexp.reduce(joint, axis=0) + log_b
            else:
                # np.max - the single operator that separates the two modes
                self._score = np.max(joint, axis=0) + log_b

        if self.mode == Mql5InferenceMode.FILTERING:
            shift = float(np.logaddexp.reduce(self._score))
            self._last_loglik = shift
        else:
            shift = float(np.max(self._score))

        if not np.isfinite(shift):
            return False

        self._score = self._score - shift
        self._steps += 1
        return True

    def argmax(self) -> int:
        return int(np.argmax(self._score))

    def confidence(self) -> float:
        """FILTERING: max filtered posterior. CAUSAL_VITERBI: path margin in nats.

        The max-product frontier is a vector of path log-probabilities, not a
        distribution; normalising it would produce something that sums to 1 and
        is not a posterior. The best-minus-second-best margin is the honest
        confidence for that decoder.
        """
        if self.params.n_states < 2:
            return 1.0

        ordered = np.sort(self._score)[::-1]
        if self.mode == Mql5InferenceMode.FILTERING:
            return float(np.exp(ordered[0]))
        return float(ordered[0] - ordered[1])

    def posterior(self) -> np.ndarray:
        if self.mode != Mql5InferenceMode.FILTERING:
            raise ValueError("posterior() is only defined for FILTERING; CAUSAL_VITERBI scores paths, not states")
        return np.exp(self._score)

    def run(self, features: np.ndarray, burn_in: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Replay a feature matrix exactly as ``CHmmIndicator::Calculate`` would.

        Non-finite rows are skipped without advancing the recursion, matching
        the MQL5 side, which refuses to feed a fabricated observation into the
        chain. Returns ``(labels, confidence)`` with ``LABEL_NONE`` / NaN where
        no state is published.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.params.n_features:
            raise ValueError(f"features must be (T, {self.params.n_features}), got {features.shape}")

        n = features.shape[0]
        labels = np.full(n, LABEL_NONE, dtype=np.int64)
        conf = np.full(n, np.nan, dtype=np.float64)

        self.reset()
        for t in range(n):
            row = features[t]
            if not np.all(np.isfinite(row)):
                continue
            if not self.step(row):
                raise ValueError(
                    f"no feasible state path at bar {t}: every candidate path is forced through a "
                    "zero-probability transition, or an emission was non-finite. Inspect the "
                    "exported transition matrix for over-restrictive zeros."
                )
            if self._steps <= burn_in:
                continue
            labels[t] = self.argmax()
            conf[t] = self.confidence()

        return labels, conf
