from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .features import _rolling_mean_1d, _rolling_mean_2d, _validate_posterior_matrix, _validate_window, entropy,\
    rolling_flip_rate


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
    showed; negative: more certain.

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
    posterior collapse). Callers that gate on either direction should ``abs()``
    the result.

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
    """
    window: int
    entropy_mean: float
    entropy_std: float
    occupancy: NDArray
    flip_rate: float

    def __post_init__(self) -> None:
        self.occupancy.setflags(write=False)


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

    Requires ``probs.shape[0] >= window + 1`` so the post-warmup region has at
    least two rows for a meaningful stdev. Callers should pass substantially
    more for statistically stable baselines.
    """
    p = _validate_posterior_matrix(probs, "fit_posterior_health_baselines")
    _validate_window(window, "fit_posterior_health_baselines")
    if p.shape[0] < window + 1:
        raise ValueError(
            f"fit_posterior_health_baselines: probs must have at least window+1={window + 1} rows, "
            f"got {p.shape[0]}"
        )
    e = entropy(p)
    rolling_e = _rolling_mean_1d(e, window)[window - 1:]
    rolling_flip = rolling_flip_rate(p, window)[window - 1:]
    entropy_std = float(rolling_e.std())
    # Threshold guards against FP-accumulation noise in the rolling-mean cumsum
    # (constant entropy inputs leave ~1e-14 residual std at T~200). 1e-10 is well
    # below any plausible live-posterior signal but well above the FP-noise floor.
    if entropy_std < 1e-10:
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
