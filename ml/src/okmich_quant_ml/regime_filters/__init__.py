"""
Regime filters Module

Modular library of post-processing filters for HMM decoded state sequences. These processors operate on
**hard label sequences** to smooth noisy labels and improve temporal stability while remaining responsive
to genuine regime transitions.

Posterior-aware operators (those that consume the ``(T, K)`` posterior matrix rather than just labels)
live in ``okmich_quant_ml.posterior_inference`` — including the inferer step that produces labels from
posteriors. The natural pipeline is:

    posterior (T, K)
        └── posterior_inference (transformers + an inferer)
                └── labels (T,)
                        └── regime_filters (label-only smoothing)

CAUSAL BY DEFAULT FOR LIVE TRADING SAFETY
==========================================
All processors default to causal mode (no look-ahead bias) to ensure:
- Realistic backtesting results
- Safe deployment in live trading systems
- No accidental use of future information

Causal Processors (Safe for Live Trading):
------------------------------------------
- MedianFilter (causal=True by default)
- MinimumDurationFilter (inherently causal)
- HysteresisProcessor (inherently causal — count-based only)
- TransitionRateLimiter (inherently causal)
- MarkovJumpProcessRegularizer (inherently causal)
"""

from .base import BasePostProcessor
from .filters import MedianFilter, MinimumDurationFilter
from .hysteresis import HysteresisProcessor
from .markov_jump import MarkovJumpProcessRegularizer
from .pipeline import ProcessorPipeline
from .rate_limiter import TransitionRateLimiter

__all__ = [
    # Base
    "BasePostProcessor",
    "ProcessorPipeline",

    "MinimumDurationFilter",
    "MedianFilter",
    "HysteresisProcessor",

    "TransitionRateLimiter",
    "MarkovJumpProcessRegularizer",
]
