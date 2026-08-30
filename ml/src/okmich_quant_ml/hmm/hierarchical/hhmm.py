"""
HierarchicalHMM — native-pomegranate macro-regime (Run / Reversal) HHMM.

The 3-level Tayal/Wisebourt topology is reduced (Murphy 2002 DBN flattening) to a **flat
4-state ``DenseHMM`` with a masked transition matrix**:

    state 0 = block-A P+ (up)     state 1 = block-A P- (down)
    state 2 = block-B P+ (up)     state 3 = block-B P- (down)

- The two macro *blocks* (A, B) are the hidden Run/Reversal regime; the label assignment
  (which block is Run) is done post-hoc and kept stable across refits via a 2x2 match.
- Direction (P+/P-) is *observed*, not hidden — every transition flips direction, matching the
  strictly alternating zigzag stream. Termination is folded into the cross-block transition.
- Emission is pluggable (``DistType``, default CATEGORICAL). For the canonical categorical model,
  P+ states are pinned to the 9 up-symbols and P- states to the 9 down-symbols; EM preserves those
  structural zeros, locking direction identity so only the macro label can differ across refits.

Because the model is a real ``DenseHMM`` wrapped by ``BasePomegranateHMM``, fixed-lag smoothing,
forward-backward, per-bar predictive log-lik and serialisation are all reused, and the causal-first
posterior modes (FILTER live; FIXED_LAG/SMOOTHER teacher/diagnostic) come for free.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch
from pomegranate.distributions import (
    Categorical,
    Exponential,
    Gamma,
    LogNormal,
    Normal,
    StudentT,
)
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM
from scipy.optimize import linear_sum_assignment

from ..base_pomegranate import BasePomegranateHMM
from ..util import DistType, InferenceMode
from .config import (
    ALPHABET_SIZE,
    N_FLOW_BINS,
    N_MACRO_STATES,
    N_STATES,
    N_STRENGTH_BINS,
    SUB_ALPHABET_SIZE,
    HHMMLevel,
    MacroRegime,
    PosteriorMode,
    SessionPolicy,
    TOPOLOGY_NAME,
    ZigzagDirection,
    symbols_for_direction,
)
from .observations import ZigzagObservations

# Flat state layout (fixed).
_UP_STATES: tuple[int, ...] = (0, 2)
_DOWN_STATES: tuple[int, ...] = (1, 3)
_BLOCK_STATES: tuple[tuple[int, ...], ...] = ((0, 1), (2, 3))  # block A, block B


def state_block(state: int) -> int:
    """Macro block (0 or 1) a flat state belongs to."""
    return state // 2


def state_direction(state: int) -> ZigzagDirection:
    """Zigzag direction a flat state emits (P+ = up on even index, P- = down on odd)."""
    return ZigzagDirection.UP if state % 2 == 0 else ZigzagDirection.DOWN


def build_transition_mask() -> np.ndarray:
    """
    Hard structural mask (1 = allowed, 0 = forbidden) for the flat topology.

    A transition is allowed iff it *flips direction* (``s % 2 != s' % 2``): this simultaneously
    forbids self-loops on production states and enforces P+/P- alternation. Staying in a block is a
    macro hold; flipping block is the termination-driven macro switch.
    """
    mask = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    for s in range(N_STATES):
        for t in range(N_STATES):
            if s % 2 != t % 2:
                mask[s, t] = 1.0
    return mask


class HierarchicalHMM(BasePomegranateHMM):
    """
    Posterior-first hierarchical HMM of Run/Reversal macro regimes over a zigzag stream.

    Parameters
    ----------
    distribution_type
        Emission family. Default ``DistType.CATEGORICAL`` — the canonical 18-symbol Tayal/Wisebourt
        model this module exists to implement (and the only family the param-dict persistence
        supports). Continuous families (NORMAL/STUDENTT/LOGNORMAL/GAMMA/EXPONENTIAL) and mixtures
        serve the generalised variant over continuous per-zigzag features.
    n_components
        Mixture components per state (``>= 2`` enables ``GeneralMixtureModel`` emissions).
        Ignored for CATEGORICAL.
    random_state, max_iter, inference_mode
        As in :class:`BasePomegranateHMM`. ``inference_mode`` sets the default recursion used by the
        inherited ``predict``/``predict_proba``; :meth:`predict_proba` takes an explicit
        :class:`PosteriorMode` and does not depend on it.
    """

    def __init__(self, distribution_type: DistType = DistType.CATEGORICAL, *, n_components: int = 1,
                 n_init: int = 2, tol: float = 1e-4, random_state: int = 100, max_iter: int = 100,
                 inference_mode: Optional[InferenceMode] = None, **dist_kwargs):
        super().__init__(distribution_type, n_states=N_STATES, random_state=random_state,
                         max_iter=max_iter, inference_mode=inference_mode, **dist_kwargs)
        self.n_components = n_components
        self.n_init = max(1, int(n_init))
        self.tol = float(tol)
        if self.is_categorical and self.is_mixture:
            raise ValueError(
                "Mixture emissions (n_components >= 2) are not supported for CATEGORICAL: a mixture of "
                "categoricals over the same finite alphabet is itself a categorical (non-identifiable), "
                "so EM degenerates. Use n_components=1 for the categorical HHMM, or a continuous "
                "distribution (e.g. NORMAL/STUDENTT) for mixture emissions."
            )
        self._mask = build_transition_mask()
        # Block -> MacroRegime assignment, resolved at fit time. Default identity until fitted.
        self.macro_labels_: dict[int, MacroRegime] = {0: MacroRegime.RUN, 1: MacroRegime.REVERSAL}
        self._fitted = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def is_categorical(self) -> bool:
        return self.distribution_type == DistType.CATEGORICAL

    @property
    def is_mixture(self) -> bool:
        return self.n_components is not None and self.n_components >= 2

    def macro_block(self, regime: MacroRegime) -> int:
        """Flat-state block index currently labelled as ``regime``."""
        regime = MacroRegime(regime)
        for block, lab in self.macro_labels_.items():
            if lab is regime:
                return block
        raise KeyError(f"No block is labelled {regime}")

    def production_state_labels(self) -> list[str]:
        """Human-readable label per flat state column, e.g. ``['run:P+', 'run:P-', ...]``."""
        labels = []
        for s in range(N_STATES):
            regime = self.macro_labels_[state_block(s)]
            sign = "P+" if state_direction(s) is ZigzagDirection.UP else "P-"
            labels.append(f"{regime.value}:{sign}")
        return labels

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _build_model(self) -> DenseHMM:
        distributions = [self._build_state_distribution(s) for s in range(N_STATES)]
        edges = self._init_edges()
        starts = np.full(N_STATES, 1.0 / N_STATES, dtype=np.float64)
        return DenseHMM(distributions=distributions, edges=edges, starts=starts,
                        max_iter=self.max_iter, tol=self.tol, random_state=self.random_state, dtype=torch.float64)

    def _init_edges(self) -> np.ndarray:
        """Mask-consistent random initial transition matrix (rows renormalised)."""
        rng = np.random.default_rng(self.random_state)
        edges = self._mask * rng.uniform(0.5, 1.0, size=(N_STATES, N_STATES))
        row_sums = edges.sum(axis=1, keepdims=True)
        return edges / np.maximum(row_sums, 1e-300)

    def _build_state_distribution(self, state: int):
        # Categorical mixtures are rejected at construction (non-identifiable), so the categorical
        # path is always a single Categorical; mixtures apply only to continuous emissions.
        if self.is_categorical:
            return self._build_categorical(state)
        if self.is_mixture:
            rng = np.random.default_rng(self.random_state + 101 * state)
            return GeneralMixtureModel(distributions=[self._build_continuous(state, rng) for _ in range(self.n_components)])
        return self._build_continuous(state, None)

    def _build_categorical(self, state: int) -> Categorical:
        """
        Direction-pinned categorical init with strength-biased symmetry breaking.

        Mass sits only on the state's 9-symbol sub-alphabet (direction pinning). Additionally, block 0
        states are biased toward *high*-strength symbols and block 1 toward *low*-strength symbols so
        the two macro blocks start emission-distinguishable. Without this, random-symmetric inits let
        both blocks collapse to similar emissions and EM needs many (slow) iterations — and often
        fails — to separate Run from Reversal. The bias is a symmetry break, not an assumption: the
        Run/Reversal label is still assigned post-hoc by zigzag magnitude.
        """
        rng = np.random.default_rng(self.random_state + 17 * state)
        probs = np.full(ALPHABET_SIZE, 1e-12, dtype=np.float64)
        sub = list(symbols_for_direction(state_direction(state)))
        favored_strength = N_STRENGTH_BINS - 1 if state_block(state) == 0 else 0
        alpha = np.ones(SUB_ALPHABET_SIZE, dtype=np.float64)
        for local in range(SUB_ALPHABET_SIZE):
            if local // N_FLOW_BINS == favored_strength:
                alpha[local] += 4.0  # concentrate initial mass on the block's favoured strength band
        probs[sub] = rng.dirichlet(alpha)
        probs = probs / probs.sum()
        return Categorical(probs=probs.reshape(1, -1), dtype=torch.float64)

    def _build_continuous(self, state: int, rng: Optional[np.random.Generator]):
        """
        Build a continuous emission for a state, seeded from the k-means init when available.

        Pre-seeding location/scale is essential under the masked topology: pomegranate's own
        mixture initialisation can leave a masked state's component with no responsibility, which
        yields NaN covariances. ``rng`` (non-None for mixture components) jitters the seed so a
        state's components start distinct. Gaussian-family means/covs come from the k-means stats
        the base ``fit`` computes; positive-support families (LogNormal/Gamma/Exponential) are left
        for pomegranate to initialise from data.
        """
        cov_type = str(self.dist_kwargs.get("covariance_type", "diag")).lower()
        min_cov = self.dist_kwargs.get("min_cov")
        match self.distribution_type:
            case DistType.NORMAL:
                means, covs = self._gaussian_init(state, cov_type, min_cov, rng)
                return Normal(means=means, covs=covs, covariance_type=cov_type, min_cov=min_cov, dtype=torch.float64)
            case DistType.STUDENTT:
                ct = "diag" if cov_type == "full" else cov_type
                means, covs = self._gaussian_init(state, ct, min_cov, rng)
                return StudentT(dofs=self.dist_kwargs.get("dofs", 3), means=means, covs=covs, covariance_type=ct,
                                min_cov=min_cov, dtype=torch.float64)
            case DistType.LOGNORMAL:
                return LogNormal(covariance_type=cov_type, min_cov=min_cov, dtype=torch.float64)
            case DistType.GAMMA:
                return Gamma(dtype=torch.float64)
            case DistType.EXPONENTIAL:
                return Exponential(dtype=torch.float64)
            case _:
                raise ValueError(f"Unsupported distribution_type for HHMM continuous emissions: {self.distribution_type}")

    def _gaussian_init(self, state: int, cov_type: str, min_cov: Optional[float],
                       rng: Optional[np.random.Generator]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (means, covs) seeded from k-means stats for the state, or (None, None) if unavailable."""
        stats = getattr(self, "_kmeans_stats", None)
        if not isinstance(stats, dict):
            return None, None
        floor = 1e-6 if min_cov is None else max(float(min_cov), 1e-12)
        means = np.asarray(stats["centroids"][state], dtype=np.float64).copy()
        covs_diag = np.maximum(np.asarray(stats["covs_diag"][state], dtype=np.float64).copy(), floor)
        if rng is not None:  # jitter mixture components apart
            means = means + rng.normal(0.0, np.sqrt(covs_diag) * 0.1, size=means.shape)
        if cov_type == "sphere":
            return means, np.array([float(np.maximum(covs_diag.mean(), floor))], dtype=np.float64)
        if cov_type == "full":
            covs_full = np.asarray(stats["covs_full"][state], dtype=np.float64).copy()
            covs_full[np.diag_indices_from(covs_full)] = np.maximum(np.diag(covs_full), floor)
            return means, covs_full
        return means, covs_diag  # diag

    # ------------------------------------------------------------------
    # Categorical-aware emission log-probability (used by fixed-lag / per-bar loglik)
    # ------------------------------------------------------------------
    def _compute_log_emissions(self, X: np.ndarray) -> np.ndarray:
        T = X.shape[0]
        if self.is_categorical:
            X_tensor = torch.tensor(np.asarray(X).astype(np.int64), dtype=torch.int64)
        else:
            X_tensor = torch.tensor(np.asarray(X, dtype=np.float64), dtype=torch.float64)
        log_emit = np.empty((T, N_STATES), dtype=np.float64)
        for j, dist in enumerate(self._model.distributions):
            lp = dist.log_probability(X_tensor)
            if hasattr(lp, "detach"):
                lp = lp.detach().cpu().numpy()
            log_emit[:, j] = np.asarray(lp, dtype=np.float64)
        return log_emit

    # ------------------------------------------------------------------
    # Input resolution
    # ------------------------------------------------------------------
    def _resolve_input(self, observations: Union[ZigzagObservations, np.ndarray],
                       magnitudes: Optional[np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if isinstance(observations, ZigzagObservations):
            if not self.is_categorical:
                raise TypeError(
                    "Continuous-emission HHMM expects a raw (T, D) feature matrix, not ZigzagObservations. "
                    "Pass an ndarray plus optional magnitudes."
                )
            X = observations.to_model_input()
            if X.size == 0:
                raise ValueError("observation sequence is empty; nothing to fit or infer.")
            return X, observations.magnitudes
        X = np.asarray(observations)
        if X.size == 0:
            raise ValueError("observation sequence is empty; nothing to fit or infer.")
        if self.is_categorical:
            X = X.reshape(-1, 1).astype(np.int64)
            if X.min() < 0 or X.max() >= ALPHABET_SIZE:
                raise ValueError(f"categorical symbols must be in [0, {ALPHABET_SIZE}); got [{X.min()}, {X.max()}]")
        else:
            X = X.astype(np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if not np.all(np.isfinite(X)):
                raise ValueError("Continuous feature matrix contains NaN/Inf.")
        return X, magnitudes

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, observations: Union[ZigzagObservations, np.ndarray], *,
            magnitudes: Optional[np.ndarray] = None, prev: Optional["HierarchicalHMM"] = None,
            lengths: Optional[Sequence[int]] = None) -> "HierarchicalHMM":
        """
        Fit the HHMM and assign macro (Run/Reversal) labels.

        Parameters
        ----------
        observations
            A :class:`ZigzagObservations` (categorical default) or a raw model-input array
            (int symbols for categorical; ``(T, D)`` float features for continuous emissions).
        magnitudes
            Per-zigzag magnitudes used to label blocks (Run = higher mean magnitude). Taken from
            ``observations`` automatically when a :class:`ZigzagObservations` is passed. When absent
            the labelling falls back to an emission-based strength proxy.
        prev
            A previously fitted HHMM; blocks are matched to it (2x2 assignment) so the Run/Reversal
            identity stays stable across refits.
        lengths
            Optional per-sequence lengths for a concatenated multi-sequence ``observations``.
        """
        X, mags = self._resolve_input(observations, magnitudes)
        if self.is_categorical:
            self._fit_categorical_best(X, lengths)
        else:
            # Continuous emissions benefit from the base kmeans-init + covariance-retry machinery.
            super().fit(X, lengths)
        self._fitted = True
        self._assign_macro_labels(X, mags, prev)
        return self

    def _fit_categorical_best(self, X: np.ndarray, lengths: Optional[Sequence[int]]) -> None:
        """
        Fit the categorical HHMM ``n_init`` times from different random inits and keep the highest
        log-likelihood solution.

        Categorical EM under the masked topology can converge to a degenerate optimum where the two
        macro blocks collapse to similar emissions (no Run/Reversal separation). Best-of-N restarts
        make the fit robust: the well-separated solution fits the strength/flow bimodality better and
        therefore has higher likelihood, so it is the one retained.
        """
        X_list = self._split_sequences(X, lengths)
        base_seed = self.random_state
        best_ll, best_model = -np.inf, None
        for i in range(self.n_init):
            self.random_state = base_seed + 1000 * i
            candidate = self._build_model()
            candidate.fit(X_list)
            ll = candidate.log_probability([X])
            ll = float(ll.sum().item()) if hasattr(ll, "sum") else float(ll)
            if np.isfinite(ll) and ll > best_ll:
                best_ll, best_model = ll, candidate
        self.random_state = base_seed
        if best_model is None:
            raise RuntimeError("Categorical HHMM fit failed to produce a finite-likelihood model across all restarts.")
        self._model = best_model

    def _assign_macro_labels(self, X: np.ndarray, magnitudes: Optional[np.ndarray],
                             prev: Optional["HierarchicalHMM"]) -> None:
        """Label the two blocks Run/Reversal; keep identity stable vs ``prev`` when supplied."""
        if magnitudes is not None and len(magnitudes) == len(X):
            block_score = self._block_scores_from_magnitude(X, np.asarray(magnitudes, dtype=np.float64))
        else:
            block_score = self._block_scores_from_emission()

        run_block = int(np.argmax(block_score))  # higher score (magnitude/strength) => Run
        labels = {run_block: MacroRegime.RUN, 1 - run_block: MacroRegime.REVERSAL}

        if prev is not None and prev._fitted:
            labels = self._match_to_previous(prev, labels)
        self.macro_labels_ = labels

    def _block_scores_from_magnitude(self, X: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
        """Mean zigzag magnitude of the observations MAP-assigned to each block."""
        smoothed = self._model.predict_proba([X])
        if hasattr(smoothed, "detach"):
            smoothed = smoothed.detach().cpu().numpy()
        smoothed = np.asarray(smoothed)
        if smoothed.ndim == 3:
            smoothed = smoothed[0]
        states = np.argmax(smoothed, axis=1)
        scores = np.zeros(N_MACRO_STATES, dtype=np.float64)
        for block in range(N_MACRO_STATES):
            mask = np.isin(states, _BLOCK_STATES[block])
            scores[block] = magnitudes[mask].mean() if mask.any() else 0.0
        return scores

    def _block_scores_from_emission(self) -> np.ndarray:
        """Fallback: per-block expected trend-strength from the categorical emissions."""
        if not self.is_categorical:
            # Continuous fallback: use the L2 norm of the emission mean per block as a trend proxy.
            scores = np.zeros(N_MACRO_STATES, dtype=np.float64)
            for block in range(N_MACRO_STATES):
                vals = [self._mean_norm(self._model.distributions[s]) for s in _BLOCK_STATES[block]]
                vals = [v for v in vals if v is not None]
                scores[block] = float(np.mean(vals)) if vals else 0.0
            return scores
        scores = np.zeros(N_MACRO_STATES, dtype=np.float64)
        for block in range(N_MACRO_STATES):
            expected = []
            for s in _BLOCK_STATES[block]:
                probs = self._emission_probs(s)
                sub = list(symbols_for_direction(state_direction(s)))
                p = probs[sub]
                p = p / p.sum() if p.sum() > 0 else p
                strength = (np.arange(SUB_ALPHABET_SIZE) // 3)  # strength bin per sub-symbol
                expected.append(float((p * strength).sum()))
            scores[block] = float(np.mean(expected))
        return scores

    @staticmethod
    def _mean_norm(dist) -> Optional[float]:
        """L2 norm of a continuous emission's mean, averaged over mixture components if present."""
        if hasattr(dist, "distributions"):  # GeneralMixtureModel
            norms = [float(np.linalg.norm(c.means.detach().cpu().numpy()))
                     for c in dist.distributions if getattr(c, "means", None) is not None]
            return float(np.mean(norms)) if norms else None
        mean = getattr(dist, "means", None)
        return float(np.linalg.norm(mean.detach().cpu().numpy())) if mean is not None else None

    def _emission_probs(self, state: int) -> np.ndarray:
        """Flat categorical emission vector (length 18) for a state (mixture-weighted if needed)."""
        dist = self._model.distributions[state]
        if hasattr(dist, "distributions"):  # GeneralMixtureModel
            priors = dist.priors.detach().cpu().numpy() if hasattr(dist, "priors") else None
            comps = [d.probs[0].detach().cpu().numpy() for d in dist.distributions]
            if priors is not None:
                return np.average(np.stack(comps), axis=0, weights=priors)
            return np.mean(np.stack(comps), axis=0)
        return dist.probs[0].detach().cpu().numpy()

    def _match_to_previous(self, prev: "HierarchicalHMM", labels: dict[int, MacroRegime]) -> dict[int, MacroRegime]:
        """2x2 assignment: relabel current blocks to minimise emission distance to ``prev``."""
        cost = np.zeros((N_MACRO_STATES, N_MACRO_STATES), dtype=np.float64)
        for cur in range(N_MACRO_STATES):
            for old in range(N_MACRO_STATES):
                cost[cur, old] = self._block_emission_distance(cur, prev, old)
        row, col = linear_sum_assignment(cost)
        matched: dict[int, MacroRegime] = {}
        for cur, old in zip(row, col):
            matched[int(cur)] = prev.macro_labels_[int(old)]
        return matched

    def _dist_mean_vector(self, state: int) -> np.ndarray:
        """Emission mean vector for a continuous state (prior-weighted across mixture components)."""
        dist = self._model.distributions[state]
        if hasattr(dist, "distributions"):  # GeneralMixtureModel
            priors = dist.priors.detach().cpu().numpy() if hasattr(dist, "priors") else None
            means = [c.means.detach().cpu().numpy() for c in dist.distributions if getattr(c, "means", None) is not None]
            if not means:
                return np.zeros(1, dtype=np.float64)
            return np.average(np.stack(means), axis=0, weights=priors) if priors is not None else np.mean(np.stack(means), axis=0)
        mean = getattr(dist, "means", None)
        return mean.detach().cpu().numpy() if mean is not None else np.zeros(1, dtype=np.float64)

    def _block_signature(self, block: int) -> np.ndarray:
        """Family-agnostic feature vector for a block used to match blocks across refits."""
        states = _BLOCK_STATES[block]
        if self.is_categorical:
            return np.concatenate([self._emission_probs(s) for s in states])
        return np.concatenate([self._dist_mean_vector(s) for s in states])

    def _block_emission_distance(self, cur_block: int, prev: "HierarchicalHMM", old_block: int) -> float:
        """
        L1 distance between two blocks' emission signatures (self vs prev).

        Uses categorical emission probs for the discrete model and emission means for continuous
        families, so Hungarian refit-matching works for both (probs exist only on Categorical).
        """
        return float(np.abs(self._block_signature(cur_block) - prev._block_signature(old_block)).sum())

    # ------------------------------------------------------------------
    # Posterior inference
    # ------------------------------------------------------------------
    def predict_proba(self, observations: Union[ZigzagObservations, np.ndarray], level: HHMMLevel = HHMMLevel.MACRO,
                      mode: PosteriorMode = PosteriorMode.FILTER, *, lag: int = 0,
                      apply_session_policy: bool = True,
                      session_breaks: Optional[Sequence[int]] = None) -> np.ndarray:
        """
        Posterior probabilities at the requested level and recursion.

        Parameters
        ----------
        observations
            Zigzag observations (or raw model-input array) to score. Must match the fitted emission
            family.
        level
            ``HHMMLevel.MACRO`` -> ``(T, 2)`` with columns ``[Run, Reversal]``;
            ``HHMMLevel.PRODUCTION`` -> ``(T, 4)`` over the flat production states (see
            :meth:`production_state_labels`).
        mode
            ``PosteriorMode.FILTER`` (causal, live), ``FIXED_LAG`` (needs ``lag``), or ``SMOOTHER``.
            FIXED_LAG/SMOOTHER are look-ahead — teacher/diagnostic only.
        lag
            Number of future zigzags to condition on when ``mode == FIXED_LAG``.
        apply_session_policy
            If True, apply the group's :class:`SessionPolicy` gate (HARD segmentation / SOFT
            confidence downweight) using the observations' event times.
        session_breaks
            Explicit segment boundaries (positions) overriding the auto-derived HARD breaks.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() before predict_proba().")
        level, mode = HHMMLevel(level), PosteriorMode(mode)
        X, _ = self._resolve_input(observations, None)
        event_times = observations.event_times if isinstance(observations, ZigzagObservations) else None

        segments = self._derive_segments(len(X), event_times, session_breaks, apply_session_policy)
        flat = self._flat_posterior(X, mode, lag, segments)

        if apply_session_policy and event_times is not None:
            flat = self._apply_soft_policy(flat, event_times)

        if level is HHMMLevel.PRODUCTION:
            return flat
        return self._macro_from_flat(flat)

    def _flat_posterior(self, X: np.ndarray, mode: PosteriorMode, lag: int,
                        segments: list[tuple[int, int]]) -> np.ndarray:
        out = np.empty((len(X), N_STATES), dtype=np.float64)
        for start, end in segments:
            out[start:end] = self._segment_posterior(X[start:end], mode, lag)
        return out

    def _segment_posterior(self, X_seg: np.ndarray, mode: PosteriorMode, lag: int) -> np.ndarray:
        X_seg = self._preprocess_input(X_seg)
        if mode is PosteriorMode.FILTER:
            return self._predict_proba_filtered(X_seg)
        if mode is PosteriorMode.FIXED_LAG:
            return self.predict_proba_fixed_lag(X_seg, lag)
        # SMOOTHER
        proba = self._model.predict_proba([X_seg])
        if hasattr(proba, "detach"):
            proba = proba.detach().cpu().numpy()
        proba = np.asarray(proba)
        if proba.ndim == 3:
            proba = proba[0]
        return proba

    def _macro_from_flat(self, flat: np.ndarray) -> np.ndarray:
        """Block-sum the flat posterior into ``(T, 2)`` columns ordered ``[Run, Reversal]``."""
        run_block = self.macro_block(MacroRegime.RUN)
        rev_block = self.macro_block(MacroRegime.REVERSAL)
        macro = np.empty((flat.shape[0], N_MACRO_STATES), dtype=np.float64)
        macro[:, 0] = flat[:, list(_BLOCK_STATES[run_block])].sum(axis=1)
        macro[:, 1] = flat[:, list(_BLOCK_STATES[rev_block])].sum(axis=1)
        return macro

    def predict_proba_macro(self, observations, mode: PosteriorMode = PosteriorMode.FILTER, **kw) -> np.ndarray:
        """Convenience: macro (Run/Reversal) posterior, shape ``(T, 2)``."""
        return self.predict_proba(observations, HHMMLevel.MACRO, mode, **kw)

    def predict_proba_production(self, observations, mode: PosteriorMode = PosteriorMode.FILTER, **kw) -> np.ndarray:
        """Convenience: production-state posterior, shape ``(T, 4)``."""
        return self.predict_proba(observations, HHMMLevel.PRODUCTION, mode, **kw)

    def event_times(self, observations: ZigzagObservations) -> np.ndarray:
        """Clock timestamps at which each zigzag completed — for downstream alignment."""
        if not isinstance(observations, ZigzagObservations):
            raise TypeError("event_times requires a ZigzagObservations instance.")
        return observations.event_times

    # ------------------------------------------------------------------
    # Session policy
    # ------------------------------------------------------------------
    def _derive_segments(self, n: int, event_times: Optional[np.ndarray],
                         session_breaks: Optional[Sequence[int]], apply_policy: bool) -> list[tuple[int, int]]:
        if n == 0:
            return []
        if session_breaks is not None:
            bounds = sorted({0, *[int(b) for b in session_breaks if 0 < b < n], n})
            return list(zip(bounds[:-1], bounds[1:]))
        policy = getattr(self, "session_policy", SessionPolicy.CONTINUOUS)
        if apply_policy and policy is SessionPolicy.HARD and event_times is not None:
            breaks = self._calendar_day_breaks(event_times)
            bounds = sorted({0, *breaks, n})
            return list(zip(bounds[:-1], bounds[1:]))
        return [(0, n)]

    @staticmethod
    def _calendar_day_breaks(event_times: np.ndarray) -> list[int]:
        """
        Positions where the UTC calendar day changes — the HARD default when no explicit
        ``session_breaks`` are given.

        This is a heuristic proxy: real exchange sessions rarely align to UTC midnight (index
        futures roll at 17:00 ET, cash equities run 09:30-16:00, etc.). It is causal (breaks in a
        prefix never move when future data arrives) but it is *not* a substitute for the venue's true
        session calendar. Pass explicit ``session_breaks`` to :meth:`predict_proba` for real HARD
        gating; rely on this default only for ~24h UTC-aligned data.
        """
        days = np.asarray(event_times, dtype="datetime64[D]")
        return [int(i) for i in range(1, len(days)) if days[i] != days[i - 1]]

    def _apply_soft_policy(self, flat: np.ndarray, event_times: np.ndarray) -> np.ndarray:
        policy = getattr(self, "session_policy", SessionPolicy.CONTINUOUS)
        windows = getattr(self, "low_liquidity_windows", ())
        downweight = getattr(self, "soft_downweight", 1.0)
        if policy is not SessionPolicy.SOFT or not windows or downweight >= 1.0:
            return flat
        mask = self._low_liquidity_mask(event_times, windows)
        if not mask.any():
            return flat
        uniform = np.full(N_STATES, 1.0 / N_STATES)
        out = flat.copy()
        out[mask] = downweight * flat[mask] + (1.0 - downweight) * uniform
        return out

    @staticmethod
    def _low_liquidity_mask(event_times: np.ndarray, windows) -> np.ndarray:
        tod = np.asarray(event_times).astype("datetime64[ns]")
        minutes = (tod - tod.astype("datetime64[D]")) / np.timedelta64(1, "m")
        mask = np.zeros(len(tod), dtype=bool)
        for start, end in windows:
            s = start.hour * 60 + start.minute
            e = end.hour * 60 + end.minute
            if s <= e:
                mask |= (minutes >= s) & (minutes < e)
            else:  # wraps midnight
                mask |= (minutes >= s) | (minutes < e)
        return mask

    def attach_session_policy(self, policy: SessionPolicy, *, low_liquidity_windows=(),
                              soft_downweight: float = 1.0) -> "HierarchicalHMM":
        """Attach session-policy parameters used by :meth:`predict_proba` gating."""
        self.session_policy = SessionPolicy(policy)
        self.low_liquidity_windows = tuple(low_liquidity_windows)
        self.soft_downweight = float(soft_downweight)
        return self

    # ------------------------------------------------------------------
    # Abstract-method implementations
    # ------------------------------------------------------------------
    def get_aic_bic(self, observations: Union[ZigzagObservations, np.ndarray]) -> tuple[float, float]:
        """AIC/BIC using the free-parameter count of the masked topology + emissions."""
        X, _ = self._resolve_input(observations, None)
        ll = float(self.log_likelihood(X))
        n_trans = int(self._mask.sum()) - N_STATES          # free transition params (rows sum to 1)
        n_starts = N_STATES - 1
        if self.is_categorical:
            per_state = (SUB_ALPHABET_SIZE - 1) * max(self.n_components, 1)
            per_state += (self.n_components - 1) if self.is_mixture else 0
        else:
            n_features = X.shape[1]
            per_state = self.n_components * (n_features + self._cov_param_count(n_features))
            per_state += (self.n_components - 1) if self.is_mixture else 0
        n_params = n_starts + n_trans + N_STATES * per_state
        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + n_params * np.log(max(len(X), 1))
        return float(aic), float(bic)

    def train(self, observations: Union[ZigzagObservations, np.ndarray],
              lengths: Optional[Sequence[int]] = None, **kwargs) -> "HierarchicalHMM":
        """
        Fit the model. The HHMM topology is fixed (4 states), so there is no state-count search;
        ``train`` exists to satisfy the base contract and simply delegates to :meth:`fit`.
        """
        return self.fit(observations, lengths=lengths, **kwargs)

    # ------------------------------------------------------------------
    # Flat-parameter export / import (used by persistence)
    # ------------------------------------------------------------------
    def flat_params(self) -> dict:
        """
        Export the fitted model as exact flat arrays: ``starts (4,)``, ``edges (4, 4)`` and, for the
        categorical model, ``emissions (4, 18)``. Lossless for the categorical HHMM.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted.")
        starts = self._model.starts
        if hasattr(starts, "detach"):
            starts = starts.detach().cpu().numpy()
        starts = np.asarray(starts, dtype=np.float64)
        if np.any(starts < 0):
            starts = np.exp(starts)
        out = {
            "starts": (starts / starts.sum()).tolist(),
            "edges": self.transition_prob().tolist(),
            "macro_labels": {int(b): r.value for b, r in self.macro_labels_.items()},
            "distribution_type": self.distribution_type.name,
            "n_components": self.n_components,
        }
        if self.is_categorical:
            out["emissions"] = np.stack([self._emission_probs(s) for s in range(N_STATES)]).tolist()
        return out

    def _raw_starts_edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Raw (un-renormalised) start, transition and end probabilities from the fitted DenseHMM.

        Unlike :meth:`transition_prob`, this does not divide rows by their sum, so feeding these back
        into a fresh DenseHMM reproduces the original's internal log-parameters as closely as the
        backend allows. ``ends`` is included because forward-backward (SMOOTHER) uses it — used for
        high-fidelity persistence reload.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted.")

        def _to_prob(arr):
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            arr = np.asarray(arr, dtype=np.float64)
            return np.exp(arr) if np.any(arr < 0) else arr

        return _to_prob(self._model.starts), _to_prob(self._model.edges), _to_prob(self._model.ends)

    @classmethod
    def from_flat_params(cls, *, starts, edges, emissions, macro_labels, ends=None,
                         distribution_type: DistType = DistType.CATEGORICAL, n_components: int = 1,
                         random_state: int = 100) -> "HierarchicalHMM":
        """Rebuild a ready-to-infer categorical HHMM from exact flat arrays (no EM)."""
        if isinstance(distribution_type, str):
            distribution_type = DistType[distribution_type]
        model = cls(distribution_type, n_components=n_components, random_state=random_state)
        model._build_from_flat(np.asarray(starts, dtype=np.float64), np.asarray(edges, dtype=np.float64),
                               emissions, ends)
        model.macro_labels_ = {int(b): MacroRegime(r) for b, r in macro_labels.items()}
        model._fitted = True
        return model

    def _build_from_flat(self, starts: np.ndarray, edges: np.ndarray, emissions, ends=None) -> None:
        if not self.is_categorical:
            raise NotImplementedError("from_flat_params currently supports CATEGORICAL emissions only.")
        emissions = np.asarray(emissions, dtype=np.float64)
        dists = [Categorical(probs=emissions[s].reshape(1, -1), dtype=torch.float64) for s in range(N_STATES)]
        ends_arr = np.asarray(ends, dtype=np.float64) if ends is not None else None
        self._model = DenseHMM(distributions=dists, edges=edges, starts=starts, ends=ends_arr,
                               max_iter=self.max_iter, tol=1e-6, dtype=torch.float64)

    def __repr__(self) -> str:
        return (
            f"HierarchicalHMM(topology={TOPOLOGY_NAME}, distribution={self.distribution_type.name}, "
            f"n_components={self.n_components}, fitted={self._fitted})"
        )
