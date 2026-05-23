"""Diagnostic audits for posterior series.

Two families of audits, decoupled by intent:

* **Posterior dynamics** — ``summarize_posterior_dynamics`` returns a ``DynamicsReport`` of scalar metrics on bar-to-bar
  belief movement. With an HMM transition matrix supplied, the report includes ``flip_rate_excess`` and ``dwell_length_ratio``
  (observed minus the model's own prior expectation), the only fields giving a principled "smooth or not" reference.
  Without it, the report is purely descriptive.
* **Posterior calibration** — ``posterior_calibration_report`` returns a ``CalibrationReport`` of ECE, multi-class Brier,
  NLL, reliability-curve data, and per-class one-vs-rest decomposition. Requires ground-truth labels. For pure HMMs without
  exogenous label grounding, calibration is not well-posed and the audit should not be run against the model's own argmax.

``recommend_smoothing`` / ``recommend_calibration`` map a report to a typed verdict from ``SmoothingRecommendation`` /
``CalibrationRecommendation``. The verdict layer is intentionally separate from the report — researchers can inspect the
raw fields and disagree with the default thresholds without monkey-patching the report itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from .features import _validate_posterior_matrix, _validate_window, dwell_length, rolling_entropy_std,\
    rolling_max_prob_std, step_kl


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _autocorr_lag1(x: NDArray) -> float:
    """Pearson autocorrelation at lag 1 of a 1-D series. Returns 0 for degenerate inputs.

    The full series can have non-zero variance while one of the lagged slices (``arr[:-1]`` or ``arr[1:]``) is constant —
    e.g. ``[0, 1, 1]`` — which makes ``np.corrcoef`` return NaN. Check both slices independently.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size < 3:
        return 0.0
    left, right = arr[:-1], arr[1:]
    if float(left.std()) < 1e-12 or float(right.std()) < 1e-12:
        return 0.0
    return float(np.corrcoef(right, left)[0, 1])


def _validate_transmat(transmat: NDArray, n_states: int, name: str = "transmat") -> NDArray:
    """Validate a row-stochastic transition matrix of shape ``(K, K)``."""
    a = np.asarray(transmat, dtype=float)
    if a.shape != (n_states, n_states):
        raise ValueError(f"{name} must have shape ({n_states}, {n_states}), got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    if a.min() < -1e-9:
        raise ValueError(f"{name} contains negative values (min={a.min():.3e}).")
    row_sums = a.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(f"{name} rows must sum to 1, got row sums {row_sums.tolist()}.")
    return np.maximum(a, 0.0)


def _stationary_distribution(transmat: NDArray) -> NDArray:
    """Stationary distribution ``π`` solving ``π A = π``, ``Σπ = 1``.

    Computed from the left eigenvector of ``A`` (= right eigenvector of ``A.T``) associated with the eigenvalue 1.
    Requires an irreducible chain — eigenvalue 1 must have algebraic multiplicity exactly 1; otherwise the stationary
    distribution is non-unique and the eigenvector chosen by ``np.linalg.eig`` is implementation-dependent, making the
    downstream expected-dynamics fields arbitrary. Reducible chains raise with guidance to pass an ergodic transmat or
    compute the comparison manually with an explicit state-occupancy prior.
    """
    eigvals, eigvecs = np.linalg.eig(transmat.T)
    eig_one_tol = 1e-6
    near_one_count = int(np.sum(np.abs(eigvals - 1.0) < eig_one_tol))
    if near_one_count > 1:
        raise ValueError(
            f"transmat is reducible: eigenvalue 1 has multiplicity {near_one_count} (closed "
            f"communicating classes), so the stationary distribution is non-unique. The "
            f"observed-vs-expected comparison is not well-defined here — pass an irreducible "
            f"(ergodic) transmat, or compute the expected fields manually using an explicit "
            f"state-occupancy prior."
        )
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi = np.abs(np.real(eigvecs[:, idx]))
    total = pi.sum()
    if total <= 0.0:
        raise ValueError(
            "transmat: stationary distribution is degenerate (sum of |eigenvector| is non-positive); "
            "the chain is numerically ill-conditioned."
        )
    return pi / total


def _require_simplex_rows(probs: NDArray, func_name: str, row_sum_tol: float = 1e-6) -> None:
    """Reject row-sum drift beyond ``row_sum_tol``.

    ``_validate_posterior_matrix`` with ``normalize=False`` enforces non-negativity and finiteness
    but does NOT check that rows sum to 1. For a diagnostic audit this is the wrong default — silently
    accepting non-simplex rows would let entropy / KL / top-prob volatility be computed on corrupted
    upstream output. The audits use this helper after validation to surface the problem instead.
    """
    if probs.size == 0:
        return
    row_sums = probs.sum(axis=1)
    max_dev = float(np.abs(row_sums - 1.0).max())
    if max_dev > row_sum_tol:
        raise ValueError(
            f"{func_name}: posterior rows must sum to 1 (max deviation {max_dev:.3e} > {row_sum_tol}); "
            f"this is a diagnostic audit and silently normalising bad input would mask upstream "
            f"model corruption."
        )


def _bin_indices(values: NDArray, n_bins: int) -> NDArray:
    """Map values in ``[0, 1]`` to integer bin indices in ``[0, n_bins - 1]``.

    Uses uniform bins over ``[0, 1]`` with the last edge nudged past 1 so that
    ``value == 1.0`` lands in the final bin under ``np.digitize`` right-open
    semantics. Values outside ``[0, 1]`` (which shouldn't occur for validated
    probabilities) are clipped to the valid range.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] = np.nextafter(1.0, 2.0)
    idx = np.digitize(values, edges, right=False) - 1
    return np.clip(idx, 0, n_bins - 1)


def _per_bin_means(values: NDArray, bin_idx: NDArray, n_bins: int) -> tuple[NDArray, NDArray]:
    """Per-bin counts and per-bin means of ``values``. NaN where a bin is empty."""
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.int64)
    sums = np.bincount(bin_idx, weights=values, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    return counts, means


# ---------------------------------------------------------------------------
# Family 1 — posterior dynamics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DynamicsReport:
    """Scalar summary of posterior bar-to-bar dynamics.

    Direct fields are time-averages or robust summaries computed from
    ``features.py`` primitives. Transmat-conditional fields are ``None`` when
    the caller does not supply a transition matrix — only those fields give a
    principled smoothing threshold (observed dynamics vs. the model's own
    prior expectation).
    """
    n_bars: int
    n_states: int
    window: int

    mean_flip_rate: float
    mean_dwell_length: float
    median_dwell_length: float
    flip_autocorr_lag1: float

    mean_step_kl: float
    std_step_kl: float

    mean_top_prob_std: float
    mean_entropy_std: float

    expected_change_rate: float | None
    expected_dwell_length: float | None
    flip_rate_excess: float | None
    dwell_length_ratio: float | None


def summarize_posterior_dynamics(probs: NDArray, window: int = 20,
                                 transmat: NDArray | None = None) -> DynamicsReport:
    """Compute a single-shot dynamics summary over a posterior series.

    ``window`` drives the rolling-stdev fields (``mean_top_prob_std``,
    ``mean_entropy_std``); the other fields are window-free.

    ``transmat`` (row-stochastic, shape ``(K, K)``) unlocks the
    expected-vs-observed comparison. ``expected_change_rate`` is
    ``Σ π_i (1 - a_ii)`` and ``expected_dwell_length`` is ``Σ π_i / (1 - a_ii)``,
    both stationary-weighted; ``π`` is the chain's stationary distribution.
    Absorbing states (``a_ii == 1``) are rejected because the expected dwell
    is then infinite.
    """
    p = _validate_posterior_matrix(probs, "summarize_posterior_dynamics")
    _require_simplex_rows(p, "summarize_posterior_dynamics")
    _validate_window(window, "summarize_posterior_dynamics")
    T, K = p.shape
    if T < 2:
        raise ValueError(
            f"summarize_posterior_dynamics requires at least 2 bars for any temporal metric, got T={T}."
        )

    argmax = np.argmax(p, axis=1)
    flips = (argmax[1:] != argmax[:-1]).astype(float)
    mean_flip_rate = float(flips.mean())
    flip_autocorr = _autocorr_lag1(flips)

    dwells = dwell_length(p)
    mean_dwell_length = float(dwells.mean())
    median_dwell_length = float(np.median(dwells))

    skl_full = step_kl(p)
    skl = skl_full[1:]
    mean_step_kl = float(skl.mean()) if skl.size > 0 else 0.0
    std_step_kl = float(skl.std()) if skl.size > 0 else 0.0

    mean_top_prob_std = float(rolling_max_prob_std(p, window).mean())
    mean_entropy_std = float(rolling_entropy_std(p, window).mean())

    if transmat is None:
        return DynamicsReport(
            n_bars=int(T), n_states=int(K), window=int(window),
            mean_flip_rate=mean_flip_rate, mean_dwell_length=mean_dwell_length,
            median_dwell_length=median_dwell_length, flip_autocorr_lag1=flip_autocorr,
            mean_step_kl=mean_step_kl, std_step_kl=std_step_kl,
            mean_top_prob_std=mean_top_prob_std, mean_entropy_std=mean_entropy_std,
            expected_change_rate=None, expected_dwell_length=None,
            flip_rate_excess=None, dwell_length_ratio=None,
        )

    a = _validate_transmat(transmat, K)
    diag = np.diag(a)
    if (diag >= 1.0 - 1e-12).any():
        raise ValueError(
            "summarize_posterior_dynamics: transmat has an absorbing state (a_ii >= 1 - 1e-12); "
            "expected dwell length is infinite and the comparison is undefined."
        )
    pi = _stationary_distribution(a)
    expected_change_rate = float(np.sum(pi * (1.0 - diag)))
    expected_dwell_length = float(np.sum(pi / (1.0 - diag)))
    flip_rate_excess = mean_flip_rate - expected_change_rate
    dwell_length_ratio = mean_dwell_length / expected_dwell_length

    return DynamicsReport(
        n_bars=int(T), n_states=int(K), window=int(window),
        mean_flip_rate=mean_flip_rate, mean_dwell_length=mean_dwell_length,
        median_dwell_length=median_dwell_length, flip_autocorr_lag1=flip_autocorr,
        mean_step_kl=mean_step_kl, std_step_kl=std_step_kl,
        mean_top_prob_std=mean_top_prob_std, mean_entropy_std=mean_entropy_std,
        expected_change_rate=expected_change_rate, expected_dwell_length=expected_dwell_length,
        flip_rate_excess=flip_rate_excess, dwell_length_ratio=dwell_length_ratio,
    )


# ---------------------------------------------------------------------------
# Family 2 — posterior calibration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationReport:
    """Calibration audit of a posterior series against hard labels.

    Requires ground-truth ``y_idx``. For pure HMMs without exogenous label
    grounding, the audit is not well-posed — running it against the model's
    own argmax is circular and yields trivially-perfect calibration.

    ``brier_score`` is the multi-class Brier ``mean_n Σ_k (p_nk - y_nk)^2``;
    the decomposition fields (``brier_reliability``, ``brier_resolution``,
    ``brier_uncertainty``) are the Murphy 1973 decomposition applied to the
    *top-prob calibration* binary problem (forecast = ``max(p)``, outcome =
    ``argmax(p) == y``). The decomposition therefore answers "is the model's
    declared confidence well-calibrated against its hit rate" — a related but
    distinct question from the multi-class Brier.

    Murphy's decomposition is exact only when forecasts within each bin are
    constant. With finite-width bins it is approximate — the missing term is
    the within-bin forecast variance ``Σ_k (n_k / N) · Var(f | bin k)``. The
    components remain individually meaningful, but the identity
    ``brier_reliability − brier_resolution + brier_uncertainty == binary Brier``
    only holds exactly when the forecast is bin-constant.

    ``ece_class_dispersion`` (``per_class_max_ece - per_class_mean_ece``) is
    the Platt-vs-Temperature discriminator: small dispersion means
    miscalibration is uniform across classes (Temperature suffices); large
    dispersion means class-dependent miscalibration (Platt's per-class
    parameters are needed).
    """
    n_samples: int
    n_classes: int
    n_bins: int

    ece: float
    brier_score: float
    nll: float

    brier_reliability: float
    brier_resolution: float
    brier_uncertainty: float

    top_prob_histogram: NDArray
    top_prob_bin_edges: NDArray

    reliability_bin_centers: NDArray
    reliability_predicted_confidence: NDArray
    reliability_empirical_accuracy: NDArray
    reliability_bin_counts: NDArray

    per_class_ece: NDArray
    per_class_brier: NDArray
    per_class_max_ece: float
    per_class_mean_ece: float
    ece_class_dispersion: float

    def __post_init__(self) -> None:
        for arr in (self.top_prob_histogram, self.top_prob_bin_edges, self.reliability_bin_centers,
                    self.reliability_predicted_confidence, self.reliability_empirical_accuracy,
                    self.reliability_bin_counts, self.per_class_ece, self.per_class_brier):
            arr.setflags(write=False)


def posterior_calibration_report(probs: NDArray, y_idx: NDArray, n_bins: int = 10) -> CalibrationReport:
    """Compute calibration audit against hard labels.

    Rejects empty calibration sets, label-length mismatches, out-of-range
    ``y_idx``, single-class ``y_idx`` (degenerate), and ``n_bins < 2`` — the
    same set of degeneracies that ``TemperatureScalingTransformer.fit`` and
    ``PlattScalingTransformer.fit`` reject.

    NaN entries in the reliability-curve arrays mark empty bins; downstream
    plotting code should treat them as gaps rather than zeros.
    """
    p = _validate_posterior_matrix(probs, "posterior_calibration_report")
    _require_simplex_rows(p, "posterior_calibration_report")
    T, K = p.shape
    y_raw = np.asarray(y_idx)
    if y_raw.ndim != 1:
        raise ValueError(f"posterior_calibration_report: y_idx must be 1-D, got shape {y_raw.shape}.")
    if not np.issubdtype(y_raw.dtype, np.integer):
        raise ValueError(
            f"posterior_calibration_report: y_idx must be an integer array (got dtype {y_raw.dtype}); "
            f"silent float-to-int truncation would corrupt the audit."
        )
    y = y_raw.astype(np.int64)
    if T == 0 or len(y) == 0:
        raise ValueError("posterior_calibration_report: empty calibration set; calibration is undefined.")
    if len(y) != T:
        raise ValueError(
            f"posterior_calibration_report: y_idx length must equal number of rows in probs, "
            f"got {len(y)} vs {T}."
        )
    if (y < 0).any() or (y >= K).any():
        raise ValueError(f"posterior_calibration_report: y_idx must be in [0, K-1] where K={K}.")
    if len(np.unique(y)) < 2:
        raise ValueError(
            "posterior_calibration_report: degenerate calibration set — y_idx contains only one class; "
            "calibration is undefined."
        )
    if n_bins < 2:
        raise ValueError(f"posterior_calibration_report: n_bins must be >= 2, got {n_bins}.")

    top_prob = p.max(axis=1)
    pred_class = p.argmax(axis=1)
    is_correct = (pred_class == y).astype(float)

    # --- Top-prob histogram + reliability curve (uniform [0, 1] bins) ----
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = _bin_indices(top_prob, n_bins)
    counts, mean_top_prob = _per_bin_means(top_prob, bin_idx, n_bins)
    _, mean_correct = _per_bin_means(is_correct, bin_idx, n_bins)

    # --- ECE on top-prob ---------------------------------------------------
    weights = counts.astype(float) / float(T)
    gaps = np.where(counts > 0, np.abs(mean_top_prob - mean_correct), 0.0)
    ece = float(np.sum(weights * gaps))

    # --- Multi-class Brier -------------------------------------------------
    y_onehot = np.zeros((T, K), dtype=float)
    y_onehot[np.arange(T), y] = 1.0
    brier_score = float(np.mean(np.sum((p - y_onehot) ** 2, axis=1)))

    # --- Murphy decomposition on top-prob calibration ----------------------
    overall_correct = float(is_correct.mean())
    brier_uncertainty = overall_correct * (1.0 - overall_correct)
    rel_terms = np.where(counts > 0, counts * (mean_top_prob - mean_correct) ** 2, 0.0)
    res_terms = np.where(counts > 0, counts * (mean_correct - overall_correct) ** 2, 0.0)
    brier_reliability = float(np.sum(rel_terms) / T)
    brier_resolution = float(np.sum(res_terms) / T)

    # --- NLL (cross-entropy of true class) ---------------------------------
    eps = 1e-12
    chosen = np.clip(p[np.arange(T), y], eps, 1.0)
    nll = float(-np.mean(np.log(chosen)))

    # --- Per-class one-vs-rest ECE and Brier -------------------------------
    per_class_ece = np.zeros(K, dtype=float)
    per_class_brier = np.zeros(K, dtype=float)
    for k in range(K):
        pk = p[:, k]
        yk = (y == k).astype(float)
        bin_idx_k = _bin_indices(pk, n_bins)
        counts_k, mean_pk = _per_bin_means(pk, bin_idx_k, n_bins)
        _, mean_yk = _per_bin_means(yk, bin_idx_k, n_bins)
        weights_k = counts_k.astype(float) / float(T)
        gaps_k = np.where(counts_k > 0, np.abs(mean_pk - mean_yk), 0.0)
        per_class_ece[k] = float(np.sum(weights_k * gaps_k))
        per_class_brier[k] = float(np.mean((pk - yk) ** 2))

    per_class_max_ece = float(per_class_ece.max())
    per_class_mean_ece = float(per_class_ece.mean())
    ece_class_dispersion = per_class_max_ece - per_class_mean_ece

    return CalibrationReport(
        n_samples=int(T), n_classes=int(K), n_bins=int(n_bins),
        ece=ece, brier_score=brier_score, nll=nll,
        brier_reliability=brier_reliability, brier_resolution=brier_resolution,
        brier_uncertainty=brier_uncertainty,
        top_prob_histogram=counts.copy(), top_prob_bin_edges=bin_edges,
        reliability_bin_centers=bin_centers,
        reliability_predicted_confidence=mean_top_prob, reliability_empirical_accuracy=mean_correct,
        reliability_bin_counts=counts.copy(),
        per_class_ece=per_class_ece, per_class_brier=per_class_brier,
        per_class_max_ece=per_class_max_ece, per_class_mean_ece=per_class_mean_ece,
        ece_class_dispersion=ece_class_dispersion,
    )


# ---------------------------------------------------------------------------
# Typed recommendations
# ---------------------------------------------------------------------------

class SmoothingRecommendation(StrEnum):
    """Verdict from a ``DynamicsReport`` given a transmat-derived reference.

    ``UNKNOWN`` indicates the report lacks transmat-conditional fields, so a
    principled threshold is unavailable. The remaining bands map to
    progressively heavier smoothing:

    * ``LIGHT``    — EMA ``alpha ≈ 0.5`` or RollingMean ``window ≈ 3``
    * ``MODERATE`` — EMA ``alpha ≈ 0.2`` or RollingMean ``window ≈ 5–7``
    * ``HEAVY``    — EMA ``alpha ≈ 0.1`` or RollingMean ``window ≈ 10+``

    Bands are advisory; the right operating point depends on the downstream
    consumer's tolerance for lag.
    """
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    UNKNOWN = "unknown"


class CalibrationRecommendation(StrEnum):
    """Verdict from a ``CalibrationReport``.

    * ``NONE``        — overall ECE below threshold; identity transform is fine.
    * ``TEMPERATURE`` — uniform miscalibration; a single scalar fix suffices.
    * ``PLATT``       — class-dependent miscalibration; per-class scaling needed.
    """
    NONE = "none"
    TEMPERATURE = "temperature"
    PLATT = "platt"


def recommend_smoothing(report: DynamicsReport, *, flip_rate_excess_threshold: float = 0.05,
                        dwell_ratio_threshold: float = 0.5) -> SmoothingRecommendation:
    """Map a ``DynamicsReport`` to a typed smoothing verdict.

    Severity is the **max** of two normalised excesses:

    * flip-rate excess above expected, divided by ``flip_rate_excess_threshold``
    * dwell shortfall ``(1 - observed/expected)`` divided by ``dwell_ratio_threshold``

    Both default thresholds are conservative ML practice and should be tuned
    against the consumer's own tolerance. Returns ``UNKNOWN`` when the report
    lacks transmat-conditional fields (``transmat`` was not supplied to
    ``summarize_posterior_dynamics``).
    """
    if flip_rate_excess_threshold <= 0.0:
        raise ValueError(
            f"recommend_smoothing: flip_rate_excess_threshold must be > 0, got {flip_rate_excess_threshold}."
        )
    if dwell_ratio_threshold <= 0.0:
        raise ValueError(
            f"recommend_smoothing: dwell_ratio_threshold must be > 0, got {dwell_ratio_threshold}."
        )
    if report.flip_rate_excess is None or report.dwell_length_ratio is None:
        return SmoothingRecommendation.UNKNOWN

    excess_severity = max(0.0, report.flip_rate_excess) / flip_rate_excess_threshold
    short_severity = max(0.0, 1.0 - report.dwell_length_ratio) / dwell_ratio_threshold
    severity = max(excess_severity, short_severity)

    if severity < 0.5:
        return SmoothingRecommendation.NONE
    if severity < 1.0:
        return SmoothingRecommendation.LIGHT
    if severity < 2.0:
        return SmoothingRecommendation.MODERATE
    return SmoothingRecommendation.HEAVY


def recommend_calibration(report: CalibrationReport, *, ece_threshold: float = 0.05,
                          dispersion_threshold: float = 0.03) -> CalibrationRecommendation:
    """Map a ``CalibrationReport`` to a calibrator choice.

    Decision rule:

    1. If overall ``ece <= ece_threshold``, return ``NONE`` — the posterior is
       already calibrated enough.
    2. Else if ``ece_class_dispersion >= dispersion_threshold``, return
       ``PLATT`` — class-dependent miscalibration cannot be fixed by a single
       scalar.
    3. Otherwise return ``TEMPERATURE`` — uniform miscalibration; the simpler
       1-parameter fix is preferred to avoid overfitting on small calibration
       sets.
    """
    if ece_threshold <= 0.0:
        raise ValueError(f"recommend_calibration: ece_threshold must be > 0, got {ece_threshold}.")
    if dispersion_threshold <= 0.0:
        raise ValueError(
            f"recommend_calibration: dispersion_threshold must be > 0, got {dispersion_threshold}."
        )
    if report.ece <= ece_threshold:
        return CalibrationRecommendation.NONE
    if report.ece_class_dispersion >= dispersion_threshold:
        return CalibrationRecommendation.PLATT
    return CalibrationRecommendation.TEMPERATURE
