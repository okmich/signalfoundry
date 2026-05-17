"""Univariate HMM threshold distiller."""
from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import adjusted_rand_score

from okmich_quant_ml.hmm import InferenceMode
from okmich_quant_ml.posterior_inference import entropy, top_prob

from .._config import build_hmm
from .config import (
    EmissionFamily,
    ModelSelectionMetric,
    StateOrdering,
    ThresholdMethod,
    UnivariateHmmThresholdConfig,
)
from .result import CandidateFit, ThresholdBoundary, UnivariateHmmThresholdResult
from .separability import build_pairwise_separability, build_state_summaries


FAMILY_TO_ALGO = {
    EmissionFamily.LAMBDA: "hmm_lambda",
    EmissionFamily.GAUSSIAN: "hmm_pmgnt",
    EmissionFamily.STUDENT_T: "hmm_student",
}

# Non-mixture algos in FAMILY_TO_ALGO ignore mm_n_components, but build_hmm requires
# the keyword. Use 1 to be honest about the single-component intent.
_SINGLE_COMPONENT = 1


class UnivariateHmmThresholdDistiller:
    """Fit univariate HMM candidates and distill static raw-feature thresholds."""

    def __init__(self, config: UnivariateHmmThresholdConfig):
        self.config = config

    def fit_distill(self, x: NDArray) -> UnivariateHmmThresholdResult:
        """Fit the candidate grid on ``x`` and return distilled thresholds."""
        x_1d = self._validate_x(x)
        X = x_1d.reshape(-1, 1)
        fitted: list[tuple[object, CandidateFit, NDArray]] = []
        candidates: list[CandidateFit] = []
        for n_states in self.config.n_states_grid:
            for family in self.config.emission_families:
                for random_state in self.config.random_states:
                    model, candidate, gamma_cand = self._fit_candidate(X, n_states, family, random_state)
                    candidates.append(candidate)
                    if model is not None:
                        fitted.append((model, candidate, gamma_cand))
        if not fitted:
            errors = "; ".join(c.error or "unknown error" for c in candidates)
            raise RuntimeError(f"all univariate HMM candidates failed: {errors}")

        model, selected_candidate, gamma = self._select_candidate(fitted)
        # gamma was computed in FILTERING mode inside _fit_candidate. Set mode defensively
        # so any downstream re-use of the model produces consistent posteriors.
        model.inference_mode = InferenceMode.FILTERING
        original_labels = np.argmax(gamma, axis=1).astype(int)
        top_probs = top_prob(gamma)
        state_order = self._state_order(model, x_1d, original_labels, selected_candidate.n_states)
        # Vectorised remapping: O(n_states + T) at C-speed instead of O(T) Python iterations.
        lookup = np.empty(selected_candidate.n_states, dtype=int)
        for ordered_idx, original_state_id in enumerate(state_order):
            lookup[original_state_id] = ordered_idx
        ordered_labels = lookup[original_labels]
        thresholds, non_monotonic_count = self._extract_thresholds(model, x_1d, ordered_labels, state_order)
        threshold_values = tuple(boundary.value for boundary in thresholds)
        threshold_labels = np.searchsorted(np.asarray(threshold_values), x_1d, side="right").astype(int)
        fidelity = float((threshold_labels == ordered_labels).mean())
        ari = float(adjusted_rand_score(ordered_labels, threshold_labels))
        state_summaries = build_state_summaries(x_1d, ordered_labels, original_labels, state_order, top_probs)
        separability = build_pairwise_separability(x_1d, ordered_labels, threshold_values, eps=self.config.eps)

        return UnivariateHmmThresholdResult(
            model=model,
            x=x_1d,
            gamma=gamma,
            original_labels=original_labels,
            ordered_labels=ordered_labels,
            ordered_state_to_original_state=state_order,
            thresholds=thresholds,
            state_summaries=state_summaries,
            separability=separability,
            selected_candidate=selected_candidate,
            candidates=tuple(candidates),
            threshold_labels=threshold_labels,
            threshold_fidelity=fidelity,
            adjusted_rand_index=ari,
            non_monotonic_count=non_monotonic_count,
            posterior_metrics=self._posterior_metrics(gamma),
        )

    def _fit_candidate(self, X: NDArray, n_states: int, family: EmissionFamily,
                       random_state: int | None) -> tuple[object | None, CandidateFit, NDArray | None]:
        algo = FAMILY_TO_ALGO[family]
        try:
            model = build_hmm(algo, n_states=n_states, mm_n_components=_SINGLE_COMPONENT,
                              random_state=random_state)
            model.fit(X)
            model.inference_mode = InferenceMode.FILTERING
            gamma = model.predict_proba(X)
            labels = np.argmax(gamma, axis=1)
            ll = float(model.log_likelihood(X))
            aic, bic = model.get_aic_bic(X)
            counts = np.bincount(labels, minlength=n_states)
            populated = counts[counts > 0]
            missing_states = int((counts == 0).sum())
            min_state_fraction = float(counts.min() / len(labels))
            # `populated` is filtered by `counts > 0`, so populated.min() >= 1 by construction —
            # no need to guard the divisor against zero.
            balance = float(populated.max() / populated.min()) if len(populated) else float("inf")
            valid = missing_states == 0 and min_state_fraction >= self.config.min_state_fraction
            candidate = CandidateFit(
                n_states=n_states,
                emission_family=family,
                random_state=random_state,
                log_likelihood=ll,
                aic=float(aic),
                bic=float(bic),
                selected_metric=self.config.selection_metric,
                selected_score=self._candidate_score(ll, float(aic), float(bic)),
                mean_top_prob=float(top_prob(gamma).mean()),
                state_balance_ratio=balance,
                min_state_fraction=min_state_fraction,
                missing_states=missing_states,
                valid=valid,
                error=None,
            )
            return model, candidate, gamma
        except Exception as exc:
            candidate = CandidateFit(
                n_states=n_states,
                emission_family=family,
                random_state=random_state,
                log_likelihood=float("nan"),
                aic=float("nan"),
                bic=float("nan"),
                selected_metric=self.config.selection_metric,
                selected_score=float("inf"),
                mean_top_prob=float("nan"),
                state_balance_ratio=float("nan"),
                min_state_fraction=float("nan"),
                missing_states=n_states,
                valid=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            return None, candidate, None

    def _select_candidate(self, fitted: list[tuple[object, CandidateFit, NDArray]]
                          ) -> tuple[object, CandidateFit, NDArray]:
        valid = [item for item in fitted if item[1].valid]
        if not valid:
            warnings.warn(
                "All HMM candidates produced invalid state structure (missing states or "
                "min_state_fraction below threshold). Selecting best-scoring candidate from "
                "the invalid pool — inspect `result.selected_candidate.valid` and "
                "`result.candidate_frame` before trusting the output.",
                UserWarning, stacklevel=2,
            )
        pool = valid if valid else fitted
        return min(pool, key=lambda item: item[1].selected_score)

    def _candidate_score(self, log_likelihood: float, aic: float, bic: float) -> float:
        if self.config.selection_metric == ModelSelectionMetric.AIC:
            return aic
        if self.config.selection_metric == ModelSelectionMetric.BIC:
            return bic
        return -log_likelihood

    def _state_order(self, model, x: NDArray, labels: NDArray, n_states: int) -> tuple[int, ...]:
        scores = []
        locations = self._emission_locations(model, n_states)
        for state in range(n_states):
            values = x[labels == state]
            if self.config.state_ordering == StateOrdering.EMISSION_LOCATION and np.isfinite(locations[state]):
                score = locations[state]
            elif self.config.state_ordering == StateOrdering.FEATURE_MEAN:
                score = float(values.mean()) if len(values) else float("inf")
            else:
                score = float(np.median(values)) if len(values) else float("inf")
            scores.append((score, state))
        return tuple(int(state) for _, state in sorted(scores))

    @staticmethod
    def _emission_locations(model, n_states: int) -> list[float]:
        try:
            params = model.parameters
        except Exception:
            return [float("nan")] * n_states
        locations: list[float] = []
        for item in params:
            for key in ("means", "locs", "mu"):
                if key in item:
                    locations.append(float(np.ravel(item[key])[0]))
                    break
            else:
                locations.append(float("nan"))
        # Defensive: caller indexes ``locations[state]`` for ``state in range(n_states)``,
        # so we must guarantee the list is at least ``n_states`` long even if the wrapper
        # exposes a different number of parameter items (e.g. mixture components per state).
        if len(locations) < n_states:
            locations.extend([float("nan")] * (n_states - len(locations)))
        elif len(locations) > n_states:
            locations = locations[:n_states]
        return locations

    def _extract_thresholds(self, model, x: NDArray, ordered_labels: NDArray,
                            state_order: tuple[int, ...]) -> tuple[tuple[ThresholdBoundary, ...], int]:
        """Extract threshold boundaries between adjacent ordered states.

        Returns ``(boundaries, non_monotonic_count)`` where ``non_monotonic_count``
        is the number of boundaries that had to be coerced upward by
        ``_monotonic_thresholds`` to maintain strict ordering — a real diagnostic
        of feature-value overlap between adjacent regimes.

        ``ThresholdBoundary.method`` reflects the method *actually* used for that
        boundary, which may differ from ``config.threshold_method`` if the chosen
        method fell back (e.g. ``EMISSION_CROSSING`` falls back to
        ``POSTERIOR_MAP_SWITCH`` when ``model._compute_log_emissions`` is unavailable).
        """
        raw_values: list[float] = []
        actual_methods: list[ThresholdMethod] = []
        for lower_state in range(len(state_order) - 1):
            if self.config.threshold_method == ThresholdMethod.EMISSION_CROSSING:
                value, actual = self._emission_crossing_threshold(model, x, state_order, ordered_labels, lower_state)
            elif self.config.threshold_method == ThresholdMethod.EMPIRICAL_SWITCH_QUANTILE:
                value, actual = self._empirical_switch_threshold(x, ordered_labels, lower_state, lower_state + 1)
            else:
                value = self._best_binary_threshold(x, ordered_labels, lower_state, lower_state + 1)
                actual = ThresholdMethod.POSTERIOR_MAP_SWITCH
            raw_values.append(value)
            actual_methods.append(actual)
        threshold_values, n_collapsed = self._monotonic_thresholds(raw_values)
        boundaries = tuple(
            ThresholdBoundary(
                lower_ordered_state=i,
                upper_ordered_state=i + 1,
                value=float(value),
                empirical_quantile=float(np.mean(x <= value)),
                method=actual_methods[i],
            )
            for i, value in enumerate(threshold_values)
        )
        return boundaries, n_collapsed

    def _emission_crossing_threshold(self, model, x: NDArray, state_order: tuple[int, ...],
                                     ordered_labels: NDArray, lower_state: int) -> tuple[float, ThresholdMethod]:
        lo, hi = np.quantile(x, [0.001, 0.999])
        if not np.isfinite(lo + hi) or hi <= lo:
            return float(np.median(x)), ThresholdMethod.POSTERIOR_MAP_SWITCH
        grid = np.linspace(float(lo), float(hi), self.config.emission_grid_size)
        try:
            log_emit = model._compute_log_emissions(grid.reshape(-1, 1))
        except Exception:
            return (self._best_binary_threshold(x, ordered_labels, lower_state, lower_state + 1),
                    ThresholdMethod.POSTERIOR_MAP_SWITCH)
        left_original = state_order[lower_state]
        right_original = state_order[lower_state + 1]
        diff = log_emit[:, left_original] - log_emit[:, right_original]
        crossings = np.flatnonzero(np.signbit(diff[:-1]) != np.signbit(diff[1:]))
        if len(crossings) == 0:
            return (self._best_binary_threshold(x, ordered_labels, lower_state, lower_state + 1),
                    ThresholdMethod.POSTERIOR_MAP_SWITCH)
        left_values = x[ordered_labels == lower_state]
        right_values = x[ordered_labels == lower_state + 1]
        target = 0.5 * (np.median(left_values) + np.median(right_values))
        crossing_values = []
        for idx in crossings:
            x0, x1 = grid[idx], grid[idx + 1]
            y0, y1 = diff[idx], diff[idx + 1]
            if abs(y1 - y0) <= self.config.eps:
                crossing_values.append(float(0.5 * (x0 + x1)))
            else:
                crossing_values.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
        best = min(crossing_values, key=lambda value: abs(value - target))
        return best, ThresholdMethod.EMISSION_CROSSING

    @staticmethod
    def _empirical_switch_threshold(x: NDArray, ordered_labels: NDArray, lower_state: int,
                                    upper_state: int) -> tuple[float, ThresholdMethod]:
        mask = (ordered_labels == lower_state) | (ordered_labels == upper_state)
        values = x[mask]
        labels = ordered_labels[mask]
        if len(values) == 0:
            return float(np.median(x)), ThresholdMethod.POSTERIOR_MAP_SWITCH
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_labels = labels[order]
        switch_idx = np.flatnonzero(sorted_labels[:-1] != sorted_labels[1:])
        if len(switch_idx) == 0:
            return (UnivariateHmmThresholdDistiller._best_binary_threshold(
                x, ordered_labels, lower_state, upper_state),
                ThresholdMethod.POSTERIOR_MAP_SWITCH)
        switch_values = 0.5 * (sorted_values[switch_idx] + sorted_values[switch_idx + 1])
        return float(np.median(switch_values)), ThresholdMethod.EMPIRICAL_SWITCH_QUANTILE

    @staticmethod
    def _best_binary_threshold(x: NDArray, ordered_labels: NDArray, lower_state: int, upper_state: int) -> float:
        mask = (ordered_labels == lower_state) | (ordered_labels == upper_state)
        values = x[mask]
        labels = ordered_labels[mask]
        if len(values) == 0:
            return float(np.median(x))
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_labels = labels[order]
        unique = np.unique(sorted_values)
        if len(unique) == 1:
            return float(unique[0])

        lower_hits = (sorted_labels == lower_state).astype(int)
        upper_hits = (sorted_labels == upper_state).astype(int)
        cum_lower = np.cumsum(lower_hits)
        cum_upper = np.cumsum(upper_hits)
        total_upper = int(cum_upper[-1])

        group_ends = np.flatnonzero(sorted_values[:-1] != sorted_values[1:])
        correct_if_split_after_group = cum_lower[group_ends] + (total_upper - cum_upper[group_ends])
        best_idx = int(group_ends[int(np.argmax(correct_if_split_after_group))])
        best_value = float(0.5 * (sorted_values[best_idx] + sorted_values[best_idx + 1]))
        return best_value

    def _monotonic_thresholds(self, thresholds: list[float]) -> tuple[tuple[float, ...], int]:
        """Coerce thresholds to strict monotonic order.

        Returns ``(values, n_collapsed)`` where ``n_collapsed`` is the count of
        boundaries whose raw value was below the running maximum + eps and had to
        be bumped up. A non-zero count is a real diagnostic of feature-value
        overlap between adjacent regimes — surfaced on the result so the caller
        can react rather than silently trusting a flat threshold sequence.
        """
        if not thresholds:
            return (), 0
        out = [float(thresholds[0])]
        n_collapsed = 0
        for value in thresholds[1:]:
            minimum = out[-1] + self.config.eps
            if float(value) < minimum:
                n_collapsed += 1
                out.append(minimum)
            else:
                out.append(float(value))
        return tuple(out), n_collapsed

    def _posterior_metrics(self, gamma: NDArray) -> dict[str, float]:
        tp = top_prob(gamma)
        h = entropy(gamma)
        metrics = {
            "mean_top_prob": float(tp.mean()),
            "median_top_prob": float(np.median(tp)),
            "mean_entropy": float(h.mean()),
            "mean_norm_entropy": float((h / np.log(gamma.shape[1])).mean()),
        }
        for threshold in self.config.posterior_confidence_thresholds:
            key = f"frac_top_prob_gt_{str(threshold).replace('.', 'p')}"
            metrics[key] = float((tp > threshold).mean())
        return metrics

    @staticmethod
    def _validate_x(x: NDArray) -> NDArray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 1:
            raise ValueError(f"x must be a 1D array or a (T, 1) array, got shape {arr.shape}")
        if len(arr) < 60:
            raise ValueError(f"x must contain at least 60 observations, got {len(arr)}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("x contains NaN or Inf values")
        return arr
