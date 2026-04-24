"""Posterior-first probabilistic inference abstractions."""

from .features import dwell_length, entropy, margin, rolling_entropy_std, rolling_flip_rate, rolling_max_prob_std, step_kl
from .inferers import AbstainMode, ArgmaxInferer, CompositeGateInferer, EntropyGateInferer, MarginGateInferer,\
    StabilityGateInferer

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
    "EmaPosteriorTransformer",
    "RollingMeanPosteriorTransformer",
    "TemperatureScalingTransformer",
    "PlattScalingTransformer",
    "MaturationAlignTransformer",
    "margin",
    "entropy",
    "step_kl",
    "rolling_flip_rate",
    "rolling_max_prob_std",
    "rolling_entropy_std",
    "dwell_length",
]
