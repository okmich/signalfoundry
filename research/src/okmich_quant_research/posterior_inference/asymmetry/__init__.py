"""Posterior asymmetry — offline research stack: discovery, axis construction, and walk-forward validation.

Depends on the ``okmich_quant_ml.posterior_inference`` processing library (the dependency runs one way). These functions
consume *forward-looking* outcomes, so they are offline-only and must never be called inside a live signal path.

  * ``profiler``   — per-(axis, state) posterior-weighted contrast + Bartlett-HAC overlap correction.
  * ``forward_axes`` — forward outcome + trailing baseline per market axis.
  * ``validation`` — the source-agnostic judge (PosteriorStream -> ValidationReport / verdict).
"""
from .forward_axes import MarketAxis, build_forward_outcomes, forward_axis_series
from .profiler import ForwardOutcome, bartlett_hac_variance, forward_outcome_by_state
from .validation import AxisProbe, PosteriorStream, ValidationReport, ValidationVerdict,\
    incremental_residual, validate_outcomes, validate_stream

__all__ = [
    "ForwardOutcome",
    "forward_outcome_by_state",
    "bartlett_hac_variance",
    "MarketAxis",
    "forward_axis_series",
    "build_forward_outcomes",
    "PosteriorStream",
    "ValidationVerdict",
    "ValidationReport",
    "AxisProbe",
    "incremental_residual",
    "validate_outcomes",
    "validate_stream",
]
