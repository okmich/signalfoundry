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
compute over a single sample, ``StabilityGateInferer`` sees flip rate 0, and ``HOLD_LAST`` abstain cannot carry the
prior label across calls. Online deployments must either (a) pass a sufficient trailing window per call (the lambda
notebook's pattern — ``O(window)`` memory, correct results) or (b) wait for a future ``transform_one`` / ``infer_one`` /
``reset`` API to be added. Single-row calls without trailing context will produce results that materially diverge from
batch.
"""

from .adaptive_lag import AdaptiveLagInferer, AdaptiveLagResult, StabilityCriterion,\
    compute_trajectories, lag_commitment_audit
from .features import dwell_length, entropy, margin, posterior_delta, rolling_entropy_std, rolling_flip_rate,\
    rolling_max_prob_std, step_kl, top_prob
from .inferers import AbstainMode, ArgmaxInferer, CompositeGateInferer, EntropyGateInferer, MarginGateInferer,\
    StabilityGateInferer
from .monitoring import PosteriorHealthBaselines, PosteriorHealthReport, entropy_staleness,\
    fit_posterior_health_baselines, flip_rate_drift, score_posterior_health, state_occupancy_drift

from .pipeline import PosteriorPipeline
from .protocols import PosteriorInferer, PosteriorTransformer
from .transformers import EmaPosteriorTransformer, MaturationAlignTransformer, PlattScalingTransformer, \
    RollingMeanPosteriorTransformer, TemperatureScalingTransformer

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
    "AdaptiveLagInferer",
    "AdaptiveLagResult",
    "StabilityCriterion",
    "compute_trajectories",
    "lag_commitment_audit",
    "EmaPosteriorTransformer",
    "RollingMeanPosteriorTransformer",
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
]
