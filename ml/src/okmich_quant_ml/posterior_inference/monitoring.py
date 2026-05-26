from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp

# The four underscore-prefixed helpers below are deliberately shared with this module
# (cross-module private utilities). They are not part of the public posterior_inference
# surface; do not import them from outside the package.
from .features import _rolling_mean_1d, _rolling_mean_2d, _validate_posterior_matrix, _validate_window, entropy,\
    rolling_flip_rate

# Floor on rolling-mean stdev at fit time: below this the gate would divide by an
# effectively-zero std and produce garbage z-scores. 1e-10 is well below any plausible
# live-posterior / log-lik signal but well above FP-cumsum residual noise (~1e-14 on
# T~200 constant inputs).
_MIN_BASELINE_STD = 1e-10

# Floor on |old_value| in audit_refit_metrics below which relative-delta is treated as
# undefined. Below this threshold the ratio (new - old) / old explodes; absolute-delta
# gate (if set) still applies.
_REL_DELTA_OLD_EPS = 1e-12


def _validate_feature_matrix(X: NDArray, func_name: str) -> NDArray:
    """Validate a feature matrix ``(T, n_features)`` and return it cast to ``float``.

    Rejects non-2D shapes, zero-column matrices, and NaN/Inf entries. Does not
    enforce a row-count minimum: ``fit_feature_health_baselines`` and
    ``feature_ks_drift`` each impose their own minimum because KS requires
    non-empty samples on both sides.
    """
    x = np.asarray(X, dtype=float)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError(
            f"{func_name} requires feature matrix (T, n_features) with n_features >= 1, got shape={x.shape}"
        )
    if x.size > 0 and not np.isfinite(x.sum()):
        raise ValueError(f"{func_name}: feature matrix contains NaN or Inf values.")
    return x


def _validate_loglik_series(loglik_series: NDArray, func_name: str) -> NDArray:
    """Validate a per-bar log-likelihood series ``(T,)`` and return it cast to ``float``.

    Rejects non-1D shapes and NaN/Inf entries. Does not constrain the sign:
    continuous-density emissions can produce positive log-density values, and
    discrete emissions produce non-positive log-probability values.
    """
    s = np.asarray(loglik_series, dtype=float)
    if s.ndim != 1:
        raise ValueError(f"{func_name}: loglik_series must be 1-D, got shape={s.shape}")
    if s.size > 0 and not np.isfinite(s.sum()):
        raise ValueError(f"{func_name}: loglik_series contains NaN or Inf values.")
    return s


def _mask_warmup(arr: NDArray, window: int) -> NDArray:
    """Set the first ``window - 1`` entries of ``arr`` to NaN (rolling ``min_periods=window`` semantics).

    Mutates ``arr`` in place; intended for arrays the caller owns (freshly
    created in the same function). No-op when ``window <= 1`` or ``arr`` is
    empty.
    """
    if window <= 1 or arr.shape[0] == 0:
        return arr
    cutoff = min(window - 1, arr.shape[0])
    arr[:cutoff] = np.nan
    return arr


def entropy_staleness(entropy_series: NDArray, baseline_mean: float, baseline_std: float, window: int) -> NDArray:
    """Z-score of trailing rolling-mean entropy vs. a training baseline, shape ``(T,)``.

    Detects posterior over/under-confidence drift. ``baseline_mean`` and
    ``baseline_std`` must be the mean and stdev of the *same* rolling-mean
    statistic measured at training time on the same ``window``; otherwise the
    z-score is dimensionally inconsistent. Use ``fit_posterior_health_baselines``
    to compute them consistently.

    The first ``window - 1`` output values are NaN: the trailing window is
    under-populated there and the estimator's variance does not match the
    steady-state baseline, so a threshold against the steady-state ``baseline_std``
    would be miscalibrated. Positive output: model less certain than training
    showed; negative: more certain. Note that :func:`score_posterior_health`
    gates on ``abs(z) <= max_entropy_abs_z`` — both directions trigger
    (over-confidence drift / mode collapse is flagged as strongly as
    under-confidence drift).

    ``entropy_series`` is the ``(T,)`` output of ``features.entropy(probs)``.
    """
    _validate_window(window, "entropy_staleness")
    if baseline_std <= 0.0:
        raise ValueError(f"entropy_staleness: baseline_std must be positive, got {baseline_std}")
    e = np.asarray(entropy_series, dtype=float)
    if e.ndim != 1:
        raise ValueError(f"entropy_staleness: entropy_series must be 1-D, got shape={e.shape}")
    rolling_mean = _rolling_mean_1d(e, window)
    out = (rolling_mean - baseline_mean) / baseline_std
    return _mask_warmup(out, window)


def state_occupancy_drift(probs: NDArray, baseline_occupancy: NDArray, window: int) -> NDArray:
    """Trailing rolling L1 distance between observed soft-occupancy and a training baseline, shape ``(T,)``.

    At row ``t``, ``observed = mean(probs[max(0, t - window + 1) : t + 1], axis=0)``
    (shape ``(K,)``) and the output is ``sum(abs(observed - baseline_occupancy))``,
    bounded in ``[0, 2]``. Flags regime-mix drift: the posterior is increasingly
    favoring states under-represented at training time. ``baseline_occupancy``
    must be a length-``K`` simplex vector (non-negative, summing to ~1).

    The first ``window - 1`` output values are NaN (warmup region where the
    trailing window is under-populated).
    """
    p = _validate_posterior_matrix(probs, "state_occupancy_drift")
    _validate_window(window, "state_occupancy_drift")
    b = np.asarray(baseline_occupancy, dtype=float)
    if b.shape != (p.shape[1],):
        raise ValueError(
            f"state_occupancy_drift: baseline_occupancy must have shape ({p.shape[1]},), got {b.shape}"
        )
    if not np.all(b >= 0.0):
        raise ValueError(f"state_occupancy_drift: baseline_occupancy must be non-negative, got {b.tolist()}")
    if not np.isclose(b.sum(), 1.0, atol=1e-6):
        raise ValueError(f"state_occupancy_drift: baseline_occupancy must sum to 1, got sum={b.sum()}")
    if p.shape[0] == 0:
        return np.zeros(0, dtype=float)
    rolling = _rolling_mean_2d(p, window)
    out = np.sum(np.abs(rolling - b), axis=1)
    return _mask_warmup(out, window)


def flip_rate_drift(probs: NDArray, baseline_flip_rate: float, window: int) -> NDArray:
    """Signed difference between trailing rolling flip-rate and a training baseline, shape ``(T,)``.

    Wraps ``features.rolling_flip_rate(probs, window) - baseline_flip_rate``.
    Positive: posterior argmax flipping more than training (instability or
    regime-boundary period); negative: stickier than training (possible
    posterior collapse). Note that :func:`score_posterior_health` gates on
    ``abs(diff) <= max_flip_rate_drift_abs`` — both directions trigger
    (collapse is flagged as strongly as jitter).

    The first ``window - 1`` output values are NaN (warmup region).
    """
    if not 0.0 <= baseline_flip_rate <= 1.0:
        raise ValueError(f"flip_rate_drift: baseline_flip_rate must be in [0, 1], got {baseline_flip_rate}")
    out = rolling_flip_rate(probs, window) - baseline_flip_rate
    return _mask_warmup(out, window)


@dataclass(frozen=True)
class PosteriorHealthBaselines:
    """Training-time reference statistics for the posterior-health gate.

    Produced by ``fit_posterior_health_baselines`` so inference-time consumers
    receive self-consistent baselines (same window, same estimators) instead of
    composing them by hand. ``window`` is bound here to prevent mismatched
    train/inference window sizes from silently producing miscalibrated z-scores.

    ``occupancy`` is set read-only in ``__post_init__`` (``setflags(write=False)``)
    so the immutability guarantee extends through the ndarray contents.

    Round-trippable to/from plain dict via ``to_dict()`` / ``from_dict()`` for
    embedding in ``metadata.json`` under the ``monitoring_baselines`` block.
    """
    window: int
    entropy_mean: float
    entropy_std: float
    occupancy: NDArray
    flip_rate: float

    def __post_init__(self) -> None:
        self.occupancy.setflags(write=False)

    def to_dict(self) -> dict:
        return {
            "window": int(self.window),
            "entropy_mean": float(self.entropy_mean),
            "entropy_std": float(self.entropy_std),
            "occupancy": [float(x) for x in self.occupancy],
            "flip_rate": float(self.flip_rate),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> PosteriorHealthBaselines:
        required = ("window", "entropy_mean", "entropy_std", "occupancy", "flip_rate")
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(
                f"PosteriorHealthBaselines.from_dict: missing required keys {missing}"
            )
        return cls(window=int(payload["window"]), entropy_mean=float(payload["entropy_mean"]),
                   entropy_std=float(payload["entropy_std"]),
                   occupancy=np.asarray(payload["occupancy"], dtype=float),
                   flip_rate=float(payload["flip_rate"]))


def fit_posterior_health_baselines(probs: NDArray, window: int) -> PosteriorHealthBaselines:
    """Compute training-time baselines for ``score_posterior_health``.

    Computes:
      * ``entropy_mean`` / ``entropy_std``: mean and stdev (``ddof=0``) of the
        rolling-mean entropy series with the same ``window``, computed after
        skipping the first ``window - 1`` warmup rows.
      * ``occupancy``: time-average soft state occupancy ``mean(probs, axis=0)``.
      * ``flip_rate``: mean of the rolling flip-rate with the same ``window``,
        computed after skipping warmup rows.

    Skipping warmup ensures the baselines characterize the steady-state
    estimator, matching what ``score_posterior_health`` evaluates at inference
    time (where warmup values are NaN-masked).

    Requires ``probs.shape[0] >= window + 2`` so the post-warmup region has at
    least two rows for a meaningful stdev. Callers should pass substantially
    more for statistically stable baselines.
    """
    p = _validate_posterior_matrix(probs, "fit_posterior_health_baselines")
    _validate_window(window, "fit_posterior_health_baselines")
    if p.shape[0] < window + 2:
        raise ValueError(
            f"fit_posterior_health_baselines: probs must have at least window+2={window + 2} rows for the "
            f"post-warmup region to have >=2 samples (otherwise stdev is undefined or zero), got {p.shape[0]}"
        )
    e = entropy(p)
    rolling_e = _rolling_mean_1d(e, window)[window - 1:]
    rolling_flip = rolling_flip_rate(p, window)[window - 1:]
    entropy_std = float(rolling_e.std())
    if entropy_std < _MIN_BASELINE_STD:
        raise ValueError(
            f"fit_posterior_health_baselines: training posteriors have near-zero entropy variance "
            f"(rolling-mean entropy stdev = {entropy_std:.3e}). Baseline cannot be calibrated because "
            f"entropy_staleness would divide by an effectively-zero std. Inspect the upstream model — "
            f"typical causes are HMM state collapse, all-uniform posteriors, or a training period "
            f"too short relative to the window."
        )
    return PosteriorHealthBaselines(window=window, entropy_mean=float(rolling_e.mean()),
                                    entropy_std=entropy_std, occupancy=p.mean(axis=0),
                                    flip_rate=float(rolling_flip.mean()))


@dataclass(frozen=True)
class PosteriorHealthReport:
    """Point-in-time result of the posterior-health gate.

    All component checks are evaluated at the **final row** of the input series.
    The raw values are surfaced alongside the boolean verdicts so callers can log
    them or build their own composite policies.
    """
    overall_ok: bool
    entropy_staleness_z: float
    entropy_staleness_ok: bool
    occupancy_drift_l1: float
    occupancy_drift_ok: bool
    flip_rate_drift_signed: float
    flip_rate_drift_ok: bool


def score_posterior_health(probs: NDArray, baselines: PosteriorHealthBaselines, max_entropy_abs_z: float = 3.0,
                           max_occupancy_drift_l1: float = 0.2,
                           max_flip_rate_drift_abs: float = 0.1) -> PosteriorHealthReport:
    """Composite point-in-time gate over the three posterior-drift checks.

    Evaluates ``entropy_staleness``, ``state_occupancy_drift``, and
    ``flip_rate_drift`` at the final row of ``probs`` and compares each against
    its threshold. ``overall_ok`` is ``True`` iff all three component checks
    pass. Entropy and flip-rate checks fire on absolute deviation; occupancy
    drift is already non-negative.

    Requires ``probs.shape[0] >= baselines.window`` so the final-row index is
    in the post-warmup region and every component value is non-NaN by
    construction. Pass a trailing window slice (e.g., the last ``2 * window``
    rows) in hot-path inference to avoid recomputing full-history rolling
    statistics each bar.

    Thresholds default to conservative values that should be tuned against the
    baselines' own training-time spread before production use.
    """
    p = _validate_posterior_matrix(probs, "score_posterior_health")
    if p.shape[0] < baselines.window:
        raise ValueError(
            f"score_posterior_health: probs must have at least window={baselines.window} rows for the final row "
            f"to be in the post-warmup region, got {p.shape[0]}"
        )
    e = entropy(p)
    ent_z = float(entropy_staleness(e, baselines.entropy_mean, baselines.entropy_std, baselines.window)[-1])
    occ_l1 = float(state_occupancy_drift(p, baselines.occupancy, baselines.window)[-1])
    flip_diff = float(flip_rate_drift(p, baselines.flip_rate, baselines.window)[-1])
    ent_ok = abs(ent_z) <= max_entropy_abs_z
    occ_ok = occ_l1 <= max_occupancy_drift_l1
    flip_ok = abs(flip_diff) <= max_flip_rate_drift_abs
    return PosteriorHealthReport(overall_ok=ent_ok and occ_ok and flip_ok, entropy_staleness_z=ent_z,
                                 entropy_staleness_ok=ent_ok, occupancy_drift_l1=occ_l1, occupancy_drift_ok=occ_ok,
                                 flip_rate_drift_signed=flip_diff, flip_rate_drift_ok=flip_ok)


@dataclass(frozen=True)
class FeatureHealthBaselines:
    """Training-time reference feature samples for the feature-drift gate.

    Stores the *training feature matrix* itself (not a summary statistic) so
    inference-time consumers can run exact two-sample Kolmogorov-Smirnov tests
    without distributional assumptions. Memory is ``O(T_train * n_features)``
    floats — at 80K bars * 3 features that is ~2 MB, negligible for typical
    model sizes. ``feature_names`` is bound here so per-feature diagnostics in
    ``FeatureHealthReport`` can name the offending column.

    ``samples`` is set read-only in ``__post_init__`` (``setflags(write=False)``)
    so the immutability guarantee extends through the ndarray contents.

    **No ``to_dict`` / ``from_dict``** — by deliberate design, feature
    baselines are not persisted in ``metadata.json`` (they would balloon the
    artefact to ~MB-scale). The monitor re-derives them at runtime from the
    OOS window definition + persisted transform pipeline + OHLCV data lake.
    See ``okmich_quant_ml.posterior_inference.load_posterior_and_loglik_baselines_from_metadata``
    for the loader pattern; feature baselines must be constructed via
    :func:`fit_feature_health_baselines` against freshly-loaded OOS features.
    """
    samples: NDArray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        self.samples.setflags(write=False)

    @classmethod
    def from_dict(cls, payload: Mapping) -> FeatureHealthBaselines:
        raise NotImplementedError(
            "FeatureHealthBaselines.from_dict is intentionally not implemented. "
            "Feature baselines are not persisted in metadata.json (the raw samples are large "
            "and trivially re-derivable). Call fit_feature_health_baselines(oos_X, feature_names=...) "
            "directly at monitor runtime using OOS features replayed through the persisted "
            "transform_pipeline.joblib."
        )


def fit_feature_health_baselines(X: NDArray, feature_names: Sequence[str] | None = None) -> FeatureHealthBaselines:
    """Compute training-time feature baselines for ``score_feature_health``.

    Stores the input feature matrix verbatim as the reference sample for
    two-sample KS tests at inference time. Validates shape, rejects NaN/Inf,
    and requires at least 2 rows (KS needs non-empty samples on both sides).
    A defensive ``copy()`` is taken so the read-only flag set in
    ``__post_init__`` cannot be circumvented by a caller mutating the original
    array after the fit.

    ``feature_names`` is optional; when omitted, columns are auto-named
    ``"feature_0"``, ``"feature_1"``, etc. When provided, length must equal
    ``X.shape[1]``.
    """
    x = _validate_feature_matrix(X, "fit_feature_health_baselines")
    if x.shape[0] < 2:
        raise ValueError(f"fit_feature_health_baselines: X must have at least 2 rows, got {x.shape[0]}")
    n_features = x.shape[1]
    if feature_names is None:
        names = tuple(f"feature_{i}" for i in range(n_features))
    else:
        names = tuple(feature_names)
        if len(names) != n_features:
            raise ValueError(
                f"fit_feature_health_baselines: feature_names has length {len(names)} but X has {n_features} columns"
            )
    return FeatureHealthBaselines(samples=x.copy(), feature_names=names)


def feature_ks_drift(X_window: NDArray, baselines: FeatureHealthBaselines) -> tuple[NDArray, NDArray]:
    """Per-feature two-sample Kolmogorov-Smirnov drift test (window vs baseline).

    Returns ``(ks_statistics, p_values)``, each of shape ``(n_features,)``.
    The KS statistic is the supremum of the absolute difference between the
    empirical CDFs of the inference window and the training baseline; bounded
    in ``[0, 1]``, scale-free, and distribution-free. ``p_values`` are
    two-sided.

    KS p-values are sensitive to sample size: with windows of a few hundred
    bars the test will reject identity for almost any real distribution, while
    with very short windows even large effects can fail to reach significance.
    ``score_feature_health`` gates on the AND of effect size and significance
    to address both failure modes.

    Requires ``X_window`` to have at least 2 rows and the same column count as
    ``baselines.samples``.
    """
    x = _validate_feature_matrix(X_window, "feature_ks_drift")
    if x.shape[1] != baselines.samples.shape[1]:
        raise ValueError(
            f"feature_ks_drift: X_window has {x.shape[1]} columns but baselines has {baselines.samples.shape[1]}"
        )
    if x.shape[0] < 2:
        raise ValueError(f"feature_ks_drift: X_window must have at least 2 rows, got {x.shape[0]}")
    n_features = x.shape[1]
    ks_stats = np.empty(n_features, dtype=float)
    p_values = np.empty(n_features, dtype=float)
    for j in range(n_features):
        result = ks_2samp(x[:, j], baselines.samples[:, j], alternative="two-sided")
        ks_stats[j] = float(result.statistic)
        p_values[j] = float(result.pvalue)
    return ks_stats, p_values


@dataclass(frozen=True)
class FeatureHealthReport:
    """Point-in-time result of the feature-drift gate.

    KS is a sample-vs-sample test, so every component is computed over the
    entire ``X_window`` slice the caller passed (unlike the posterior-side
    report, which evaluates at the final row). Per-feature diagnostics are
    surfaced so consumers can identify which column drifted.

    ``per_feature_ok[j]`` is ``True`` when the feature passes the gate, i.e.
    NOT (``ks_statistics[j] > max_ks_statistic`` AND ``p_values[j] < alpha``).
    Drift is flagged only when both the effect size is large and the test is
    significant. ``overall_ok`` is ``True`` iff every feature passes (strict
    AND aggregation).
    """
    overall_ok: bool
    ks_statistics: NDArray
    p_values: NDArray
    per_feature_ok: NDArray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        self.ks_statistics.setflags(write=False)
        self.p_values.setflags(write=False)
        self.per_feature_ok.setflags(write=False)


def score_feature_health(X_window: NDArray, baselines: FeatureHealthBaselines, max_ks_statistic: float = 0.1,
                         alpha: float = 0.01) -> FeatureHealthReport:
    """Composite feature-drift gate over per-feature KS tests.

    A feature is flagged as drifted only when BOTH conditions fire:
      * KS statistic > ``max_ks_statistic`` (effect-size gate)
      * p-value < ``alpha`` (significance gate)

    This avoids the two failure modes of single-gate KS: large-N false alarms
    (significant but tiny effect) and small-N false negatives (large effect
    but underpowered). ``overall_ok`` is the strict AND across features — any
    single drifted feature fails the composite.

    Defaults are conservative production values: ``max_ks_statistic=0.1``
    flags a 10-percentile shift in any quantile of the empirical CDF;
    ``alpha=0.01`` is the standard one-percent significance gate. Tune both
    against the baselines' own training-time spread before live use.
    """
    if not 0.0 <= max_ks_statistic <= 1.0:
        raise ValueError(
            f"score_feature_health: max_ks_statistic must be in [0, 1], got {max_ks_statistic}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"score_feature_health: alpha must be in (0, 1), got {alpha}")
    ks_stats, p_values = feature_ks_drift(X_window, baselines)
    per_feature_drift = (ks_stats > max_ks_statistic) & (p_values < alpha)
    per_feature_ok = ~per_feature_drift
    return FeatureHealthReport(overall_ok=bool(per_feature_ok.all()), ks_statistics=ks_stats, p_values=p_values,
                               per_feature_ok=per_feature_ok, feature_names=baselines.feature_names)


def log_likelihood_drift(loglik_series: NDArray, baseline_mean: float, baseline_std: float, window: int) -> NDArray:
    """Z-score of trailing rolling-mean per-bar log-likelihood vs. a training baseline, shape ``(T,)``.

    Detects HMM generative-fit drift directly: when the current model is
    becoming a poor description of the data, the rolling mean of per-bar
    predictive log-likelihood drifts away from its training-time distribution.
    ``loglik_series`` is the per-bar predictive log-likelihood under the
    current model — for HMMs:

      * ``loglik[0] = logsumexp(log_alpha[0])`` (= ``log P(o_0)``, special case)
      * ``loglik[t] = logsumexp(log_alpha[t]) - logsumexp(log_alpha[t-1])`` for ``t >= 1``

    extracted from the forward pass. No sign assumption: continuous-density
    emissions can produce positive log-density values.

    ``baseline_mean`` and ``baseline_std`` must be the mean and stdev of the
    *same* rolling-mean statistic measured at training time on the same
    ``window``; otherwise the z-score is dimensionally inconsistent. Use
    ``fit_loglik_drift_baselines`` to compute them consistently.

    The first ``window - 1`` output values are NaN: the trailing window is
    under-populated there and the estimator's variance does not match the
    steady-state baseline, so a threshold against the steady-state
    ``baseline_std`` would be miscalibrated. Negative output: model fits worse
    than training showed (classical drift); positive: model fits better
    (possible over-fit, mode collapse, or pathological distribution narrowing).
    """
    _validate_window(window, "log_likelihood_drift")
    if baseline_std <= 0.0:
        raise ValueError(f"log_likelihood_drift: baseline_std must be positive, got {baseline_std}")
    s = _validate_loglik_series(loglik_series, "log_likelihood_drift")
    rolling_mean = _rolling_mean_1d(s, window)
    out = (rolling_mean - baseline_mean) / baseline_std
    return _mask_warmup(out, window)


@dataclass(frozen=True)
class LoglikDriftBaselines:
    """Training-time reference statistics for the log-likelihood drift gate.

    Produced by ``fit_loglik_drift_baselines`` so inference-time consumers
    receive self-consistent baselines (same window, same estimator) instead of
    composing them by hand. ``window`` is bound here to prevent mismatched
    train/inference window sizes from silently producing miscalibrated
    z-scores.

    Round-trippable to/from plain dict via ``to_dict()`` / ``from_dict()`` for
    embedding in ``metadata.json`` under the ``monitoring_baselines`` block.
    """
    window: int
    loglik_mean: float
    loglik_std: float

    def to_dict(self) -> dict:
        return {
            "window": int(self.window),
            "loglik_mean": float(self.loglik_mean),
            "loglik_std": float(self.loglik_std),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> LoglikDriftBaselines:
        required = ("window", "loglik_mean", "loglik_std")
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(
                f"LoglikDriftBaselines.from_dict: missing required keys {missing}"
            )
        return cls(window=int(payload["window"]), loglik_mean=float(payload["loglik_mean"]),
                   loglik_std=float(payload["loglik_std"]))


def fit_loglik_drift_baselines(loglik_series: NDArray, window: int) -> LoglikDriftBaselines:
    """Compute training-time baselines for ``score_loglik_health``.

    Computes the mean and stdev (``ddof=0``) of the rolling-mean
    log-likelihood series with the same ``window``, computed after skipping
    the first ``window - 1`` warmup rows. Skipping warmup ensures the baselines
    characterize the steady-state estimator, matching what
    ``score_loglik_health`` evaluates at inference time (where warmup values
    are NaN-masked).

    Requires ``loglik_series.shape[0] >= window + 2`` so the post-warmup
    region has at least two rows for a meaningful stdev. Callers should pass
    substantially more for statistically stable baselines. Refuses to fit when
    the rolling-mean stdev is below ``_MIN_BASELINE_STD`` (degenerate emission
    distribution or training window too short relative to ``window``) —
    surfaces the calibration failure here rather than producing miscalibrated
    z-scores later at the first ``score_loglik_health`` call.
    """
    s = _validate_loglik_series(loglik_series, "fit_loglik_drift_baselines")
    _validate_window(window, "fit_loglik_drift_baselines")
    if s.shape[0] < window + 2:
        raise ValueError(
            f"fit_loglik_drift_baselines: loglik_series must have at least window+2={window + 2} rows for the "
            f"post-warmup region to have >=2 samples (otherwise stdev is undefined or zero), got {s.shape[0]}"
        )
    rolling = _rolling_mean_1d(s, window)[window - 1:]
    loglik_std = float(rolling.std())
    if loglik_std < _MIN_BASELINE_STD:
        raise ValueError(
            f"fit_loglik_drift_baselines: training log-likelihood has near-zero variance "
            f"(rolling-mean stdev = {loglik_std:.3e}). Baseline cannot be calibrated because "
            f"log_likelihood_drift would divide by an effectively-zero std. Inspect the upstream model — "
            f"typical causes are degenerate emission distributions (state collapse to a delta), a training "
            f"period too short relative to the window, or constant-likelihood synthetic inputs."
        )
    return LoglikDriftBaselines(window=window, loglik_mean=float(rolling.mean()), loglik_std=loglik_std)


@dataclass(frozen=True)
class LoglikDriftReport:
    """Point-in-time result of the log-likelihood drift gate.

    Evaluated at the **final row** of the input series. The signed z-score is
    surfaced alongside the boolean verdict so callers can log it and
    distinguish drops (negative z, classical drift) from spikes (positive z,
    over-fit / mode collapse) when building composite policies.

    ``overall_ok`` is redundant with ``loglik_drift_ok`` for this single-component
    gate; it is exposed for API symmetry with ``PosteriorHealthReport`` and
    ``FeatureHealthReport`` so consumers composing multiple reports can read
    ``.overall_ok`` with the same semantics across all three.
    """
    overall_ok: bool
    loglik_drift_z: float
    loglik_drift_ok: bool


def score_loglik_health(loglik_series: NDArray, baselines: LoglikDriftBaselines,
                        max_abs_z: float = 3.0) -> LoglikDriftReport:
    """Composite point-in-time gate over the log-likelihood drift check.

    Evaluates ``log_likelihood_drift`` at the final row of ``loglik_series``
    and compares against the absolute-z threshold (two-sided gate). Drops
    indicate the model is becoming a poor generative fit (the classical drift
    signal); spikes indicate over-fit, mode collapse, or pathological
    distribution narrowing into a high-density region.

    Requires ``loglik_series.shape[0] >= baselines.window`` so the final-row
    index is in the post-warmup region and the component value is non-NaN by
    construction. Pass a trailing window slice (e.g., the last ``2 * window``
    rows) in hot-path inference to avoid recomputing full-history rolling
    statistics each bar.

    Default ``max_abs_z=3.0`` matches ``score_posterior_health``; tune against
    the baseline's own training-time spread before live use.
    """
    s = _validate_loglik_series(loglik_series, "score_loglik_health")
    if s.shape[0] < baselines.window:
        raise ValueError(
            f"score_loglik_health: loglik_series must have at least window={baselines.window} rows for the "
            f"final row to be in the post-warmup region, got {s.shape[0]}"
        )
    z = float(log_likelihood_drift(s, baselines.loglik_mean, baselines.loglik_std, baselines.window)[-1])
    ok = abs(z) <= max_abs_z
    return LoglikDriftReport(overall_ok=ok, loglik_drift_z=z, loglik_drift_ok=ok)


@dataclass(frozen=True)
class RefitMetricVerdict:
    """Per-metric outcome from a refit audit comparing old vs new model.

    ``absolute_delta = new_value - old_value`` (signed; consumer interprets
    direction). ``relative_delta = absolute_delta / old_value`` is ``None``
    when ``|old_value| < _REL_DELTA_OLD_EPS`` because the relative comparison
    is ill-defined against ~zero — the absolute-delta gate still applies.

    ``ok`` reflects only the gates that could fire: when ``relative_delta`` is
    ``None`` or the threshold's ``max_rel_delta`` is ``None``, only the
    absolute gate contributes; symmetrically for ``max_abs_delta=None``.
    """
    metric_name: str
    old_value: float
    new_value: float
    absolute_delta: float
    relative_delta: float | None
    ok: bool


@dataclass(frozen=True)
class RefitAuditReport:
    """Result of the refit promote gate comparing two fits on a common eval window.

    Designed for **signed axes** (trend direction, momentum score) where
    metrics from ``okmich_quant_labelling.utils.label_eval_util.evaluate_regime_returns_potentials``
    (Sharpe, total_return, net_return_after_costs, win_rate, profit_factor,
    persistence_score, etc.) are well-defined. Unsigned axes (volatility
    bucket, choppiness bucket) require different evaluation primitives not
    handled here.

    ``per_metric`` preserves the iteration order of the caller's ``thresholds``
    mapping; use ``get(metric_name)`` for named access. ``overall_ok`` is the
    strict AND across all gated metrics — any single drifted metric fails the
    composite, mirroring the aggregation choice used by ``score_feature_health``.
    """
    overall_ok: bool
    per_metric: tuple[RefitMetricVerdict, ...]

    def get(self, metric_name: str) -> RefitMetricVerdict:
        for v in self.per_metric:
            if v.metric_name == metric_name:
                return v
        raise KeyError(f"metric '{metric_name}' not in audit report")


def audit_refit_metrics(old_metrics: Mapping[str, float], new_metrics: Mapping[str, float],
                        thresholds: Mapping[str, tuple[float | None, float | None]]) -> RefitAuditReport:
    """Refit promote gate: compare per-axis metrics between an old fit and a candidate refit.

    For each metric in ``thresholds``, computes signed absolute and relative
    deltas between ``new_metrics[metric]`` and ``old_metrics[metric]``, then
    fires the per-metric gate when EITHER bound is met or exceeded:

      * ``|new - old| >= max_abs_delta`` (absolute drift), OR
      * ``|(new - old) / old| >= max_rel_delta`` (relative drift)

    A bound can be ``None`` to disable that side; at least one bound per
    metric must be set or the gate would be vacuous. ``max_*_delta`` must be
    strictly positive (zero would either silently never fire under strict
    inequality or fire on every non-identical pair under non-strict).
    Relative drift is undefined when ``|old| < _REL_DELTA_OLD_EPS``; the
    relative gate is silently skipped in that case and only the absolute gate
    contributes. Callers gating on relative-only thresholds for metrics that
    legitimately reach zero (e.g. win_rate on a no-trade fit) should set
    ``max_abs_delta`` as a safety net.

    ``overall_ok`` is the strict AND across all gated metrics.

    Intended use: at refit time, evaluate the old and new fits on a common
    recent observation window via
    ``okmich_quant_labelling.utils.label_eval_util.evaluate_regime_returns_potentials``
    (one call per model on the same DataFrame, using each model's labels),
    pass the two resulting metric mappings here as a promote gate. Designed
    for signed axes today; unsigned axes need different evaluators.

    Requires every metric in ``thresholds`` to be present in both
    ``old_metrics`` and ``new_metrics``, and every metric value to be finite.
    """
    if not thresholds:
        raise ValueError("audit_refit_metrics: thresholds must be a non-empty mapping")

    verdicts: list[RefitMetricVerdict] = []
    overall_ok = True
    for metric_name, bounds in thresholds.items():
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError(
                f"audit_refit_metrics: thresholds[{metric_name!r}] must be a (max_abs_delta, max_rel_delta) tuple, "
                f"got {bounds!r}"
            )
        max_abs, max_rel = bounds
        if max_abs is None and max_rel is None:
            raise ValueError(
                f"audit_refit_metrics: thresholds[{metric_name!r}] has both bounds None — at least one must be set"
            )
        if max_abs is not None and max_abs <= 0:
            raise ValueError(
                f"audit_refit_metrics: thresholds[{metric_name!r}].max_abs_delta must be > 0, got {max_abs}"
            )
        if max_rel is not None and max_rel <= 0:
            raise ValueError(
                f"audit_refit_metrics: thresholds[{metric_name!r}].max_rel_delta must be > 0, got {max_rel}"
            )
        if metric_name not in old_metrics:
            raise ValueError(f"audit_refit_metrics: metric '{metric_name}' missing from old_metrics")
        if metric_name not in new_metrics:
            raise ValueError(f"audit_refit_metrics: metric '{metric_name}' missing from new_metrics")

        old_value = float(old_metrics[metric_name])
        new_value = float(new_metrics[metric_name])
        if not np.isfinite(old_value):
            raise ValueError(
                f"audit_refit_metrics: old metric '{metric_name}' is not finite (got {old_metrics[metric_name]!r})"
            )
        if not np.isfinite(new_value):
            raise ValueError(
                f"audit_refit_metrics: new metric '{metric_name}' is not finite (got {new_metrics[metric_name]!r})"
            )

        abs_delta = new_value - old_value
        rel_delta: float | None = abs_delta / old_value if abs(old_value) >= _REL_DELTA_OLD_EPS else None

        abs_fired = max_abs is not None and abs(abs_delta) >= max_abs
        rel_fired = max_rel is not None and rel_delta is not None and abs(rel_delta) >= max_rel
        metric_ok = not (abs_fired or rel_fired)
        if not metric_ok:
            overall_ok = False

        verdicts.append(RefitMetricVerdict(metric_name=metric_name, old_value=old_value, new_value=new_value,
                                           absolute_delta=abs_delta, relative_delta=rel_delta, ok=metric_ok))

    return RefitAuditReport(overall_ok=overall_ok, per_metric=tuple(verdicts))
