"""Posterior-first probabilistic inference abstractions."""

from .features import entropy, margin
from .inferers import AbstainMode, ArgmaxInferer, MarginGateInferer
from .pipeline import PosteriorPipeline
from .protocols import PosteriorInferer, PosteriorTransformer
from .transformers import EmaPosteriorTransformer, TemperatureScalingTransformer

__all__ = [
    "PosteriorTransformer",
    "PosteriorInferer",
    "PosteriorPipeline",
    "AbstainMode",
    "ArgmaxInferer",
    "MarginGateInferer",
    "EmaPosteriorTransformer",
    "TemperatureScalingTransformer",
    "margin",
    "entropy",
]
