"""
Regime filters Module

Modular library of post-processing filters for Hidden Markov Model (HMM) decoded state sequences. These processors smooth
noisy state labels to improve temporal stability and persistence while maintaining responsiveness to genuine regime transitions.

Primary use case: Regime detection for quantitative trading systems.

CAUSAL BY DEFAULT FOR LIVE TRADING SAFETY
==========================================
All processors default to causal mode (no look-ahead bias) to ensure:
- Realistic backtesting results
- Safe deployment in live trading systems
- No accidental use of future information

Causal Processors (Safe for Live Trading):
------------------------------------------
- MedianFilter (causal=True by default)
- ModeFilterWithConfidence (causal=True by default)
- MinimumDurationFilter (inherently causal)
- HysteresisProcessor (inherently causal)
- TransitionRateLimiter (inherently causal)
- MarkovJumpProcessRegularizer (inherently causal)
- AdaptiveKalmanStyleSmoother (inherently causal)

Non-Causal Processor (Research Only):
-------------------------------------
- ConditionalRandomFieldRefiner (Viterbi algorithm - NOT for live trading)

Deprecated Non-Causal Modes:
---------------------------
Setting causal=False on MedianFilter or ModeFilterWithConfidence is deprecated and will be removed in a future version.
Use causal=True for all production work.
"""

from .base import BasePostProcessor
from .confidence_filter import ModeFilterWithConfidence
from .crf_refiner import ConditionalRandomFieldRefiner
from .filters import MedianFilter, MinimumDurationFilter
from .hysteresis import HysteresisProcessor
from .kalman_smoother import AdaptiveKalmanStyleSmoother
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

    "ModeFilterWithConfidence",
    "TransitionRateLimiter",

    "MarkovJumpProcessRegularizer",
    "AdaptiveKalmanStyleSmoother",
    "ConditionalRandomFieldRefiner",
]
