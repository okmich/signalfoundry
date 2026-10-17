"""Regime diagnostics for causal regime labels: one call, every pass (see README.md).

    from okmich_quant_labelling.utils.regime_diagnostics import regime_diagnostics
    diag = regime_diagnostics(close, label, first_test_start="2024-07-01", condition=vol_state, out_dir="...")

The parts (``forward_separability``, ``conditional_analysis``, ``tell_story``) stay importable, but the factory is the
way in: it runs the descriptive pass, the forward pass with tails, the conditioning and the story, and writes the
report.
"""
from .clock import DEFAULT_EXCLUDE_WINDOWS, BarClock, build_clock
from .conditional import ConditionalResult, conditional_analysis
from .core import (ForwardSeparabilityResult, LabelVerdict, Layer, StateVerdict, Thresholds, View,
                   forward_separability)
from .cost import CostBucket, seasonal_cost
from .diagnostics import RegimeDiagnostics, regime_diagnostics
from .story import tell_story
from .targets import Axis, TargetKey, forward_targets, har_regressors

__all__ = [
    "Axis", "BarClock", "ConditionalResult", "CostBucket", "DEFAULT_EXCLUDE_WINDOWS", "ForwardSeparabilityResult",
    "LabelVerdict", "Layer", "RegimeDiagnostics", "StateVerdict", "TargetKey", "Thresholds", "View", "build_clock",
    "conditional_analysis", "forward_separability", "forward_targets", "har_regressors", "regime_diagnostics",
    "seasonal_cost", "tell_story",
]
