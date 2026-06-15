"""Posterior-first probabilistic inference abstractions.

All temporal primitives in this package (``step_kl``, ``posterior_delta``, ``rolling_flip_rate``, ``rolling_max_prob_std``,
``rolling_entropy_std``, ``dwell_length``, and the drift functions in ``monitoring``) plus the rolling transformers
(``EmaPosteriorTransformer``, ``RollingMeanPosteriorTransformer``) assume the input is a **single contiguous time series**.
They do not detect segment boundaries: concatenating multiple symbols, sessions, or training periods will leak state across
the boundary (deltas, rolling means, dwell counts, flip-rate moving averages will all span the seam). Callers spanning
segments must apply per-segment and concatenate the outputs.

``MaturationAlignTransformer`` fills the first ``lag`` rows with the uniform prior ``[1/K, ..., 1/K]``. This is chosen
over NaN because every gate inferer rejects NaN at validation — a NaN warmup would break pipeline composition with the
standard gates. The trade-off: aggregate metrics computed naively across the full alignment output will include ``1/K``
placeholder mass for the warmup region; callers reporting aggregates should slice off the warmup or use a NaN-aware
upstream.

**Streaming / online use:** this package is designed for **batch** calls — the caller passes a full ``(T, K)`` posterior
matrix (or a trailing-window slice) and gets back a ``(T,)`` or ``(T, K)`` output. Calling ``run()``/``transform()``/
``infer()`` row-by-row resets all temporal state every call: EMA recurrence restarts from ``probs[0]``, rolling means
compute over a single sample, ``StabilityGateInferer`` sees flip rate 0, and ``HOLD_LAST`` abstain cannot carry the prior
label across calls. Online deployments must either (a) pass a sufficient trailing window per call (the lambdanotebook's
pattern — ``O(window)`` memory, correct results) or (b) wait for a future ``transform_one`` / ``infer_one`` /``reset``
API to be added. Single-row calls without trailing context will produce results that materially diverge from batch.
"""

from .adaptive_lag import AdaptiveLagInferer, AdaptiveLagResult, StabilityCriterion,\
    compute_trajectories, lag_commitment_audit
from .asymmetry import ForwardOutcome, bartlett_hac_variance, forward_outcome_by_state
from .diagnostics import CalibrationRecommendation, CalibrationReport, DynamicsReport, SmoothingRecommendation,\
    posterior_calibration_report, recommend_calibration, recommend_smoothing, summarize_posterior_dynamics
from .features import dwell_length, entropy, margin, posterior_delta, rolling_entropy_std, rolling_flip_rate,\
    rolling_max_prob_std, step_kl, top_prob
from .inferers import AbstainMode, ArgmaxInferer, CompositeGateInferer, ConfidenceHysteresisInferer,\
    ConfidenceWeightedModeInferer, EntropyGateInferer, MarginGateInferer, StabilityGateInferer, ViterbiInferer
from .monitoring import FeatureHealthBaselines, FeatureHealthReport, LoglikDriftBaselines, LoglikDriftReport,\
    PosteriorHealthBaselines, PosteriorHealthReport, RefitAuditReport, RefitMetricVerdict, audit_refit_metrics,\
    entropy_staleness, feature_ks_drift, fit_feature_health_baselines, fit_loglik_drift_baselines,\
    fit_posterior_health_baselines, flip_rate_drift, log_likelihood_drift, score_feature_health, score_loglik_health,\
    score_posterior_health, state_occupancy_drift
from .monitoring_io import InferenceLogFrame, MonitoringCycleReport,\
    load_posterior_and_loglik_baselines_from_metadata, read_inference_log, run_streaming_gates

from .pipeline import PosteriorPipeline
from .protocols import PosteriorInferer, PosteriorTransformer
from .transformers import EmaPosteriorTransformer, KalmanPosteriorTransformer, MaturationAlignTransformer, \
    PlattScalingTransformer, RollingMeanPosteriorTransformer, TemperatureScalingTransformer

__all__ = [
    "PosteriorTransformer",
    "PosteriorInferer",
    "PosteriorPipeline",
    "AbstainMode",
    "ArgmaxInferer",
    "MarginGateInferer",
    "EntropyGateInferer",
    "CompositeGateInferer",
    "StabilityGateInferer",
    "ViterbiInferer",
    "ConfidenceWeightedModeInferer",
    "ConfidenceHysteresisInferer",
    "AdaptiveLagInferer",
    "AdaptiveLagResult",
    "StabilityCriterion",
    "compute_trajectories",
    "lag_commitment_audit",
    "ForwardOutcome",
    "forward_outcome_by_state",
    "bartlett_hac_variance",
    "EmaPosteriorTransformer",
    "RollingMeanPosteriorTransformer",
    "KalmanPosteriorTransformer",
    "TemperatureScalingTransformer",
    "PlattScalingTransformer",
    "MaturationAlignTransformer",
    "margin",
    "top_prob",
    "entropy",
    "step_kl",
    "posterior_delta",
    "rolling_flip_rate",
    "rolling_max_prob_std",
    "rolling_entropy_std",
    "dwell_length",
    "PosteriorHealthBaselines",
    "PosteriorHealthReport",
    "entropy_staleness",
    "state_occupancy_drift",
    "flip_rate_drift",
    "fit_posterior_health_baselines",
    "score_posterior_health",
    "FeatureHealthBaselines",
    "FeatureHealthReport",
    "feature_ks_drift",
    "fit_feature_health_baselines",
    "score_feature_health",
    "LoglikDriftBaselines",
    "LoglikDriftReport",
    "log_likelihood_drift",
    "fit_loglik_drift_baselines",
    "score_loglik_health",
    "RefitMetricVerdict",
    "RefitAuditReport",
    "audit_refit_metrics",
    "InferenceLogFrame",
    "MonitoringCycleReport",
    "read_inference_log",
    "run_streaming_gates",
    "load_posterior_and_loglik_baselines_from_metadata",
    "DynamicsReport",
    "CalibrationReport",
    "SmoothingRecommendation",
    "CalibrationRecommendation",
    "summarize_posterior_dynamics",
    "posterior_calibration_report",
    "recommend_smoothing",
    "recommend_calibration",
]
