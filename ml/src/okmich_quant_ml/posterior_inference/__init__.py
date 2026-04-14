"""Posterior-first probabilistic inference abstractions."""

from .features import entropy, margin
from .inferers import ArgmaxInferer
from .pipeline import PosteriorPipeline
from .protocols import PosteriorInferer, PosteriorTransformer

__all__ = [
    "PosteriorTransformer",
    "PosteriorInferer",
    "PosteriorPipeline",
    "ArgmaxInferer",
    "margin",
    "entropy",
]
