"""
Regression tests for validate_causality in diagnostics/returns/utils/validators.py.

Covers:
  - Causal targets (CUMULATIVE_RETURN pattern) → no violations
  - Forward-return-pattern targets (converge to zero at segment end) → violations recorded
  - Constant-segment-value targets (AMPLITUDE_PER_BAR pattern) → violations recorded
  - Mixed segments → only violating segments flagged
  - Edge cases: segment too short, zero start value, NaN targets
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.diagnostics.returns.utils.validators import validate_causality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg_info(*segments):
    """Build a segment_info DataFrame from (start, end, direction) tuples."""
    rows = [{"start_idx": s, "end_idx": e, "direction": d} for s, e, d in segments]
    return pd.DataFrame(rows)


def _cumulative_return_targets(n, drift=0.001):
    """Simulate causal CUMULATIVE_RETURN targets: monotonically growing within segment."""
    returns = np.random.default_rng(0).normal(drift, 0.01, n)
    return pd.Series(np.cumsum(returns))


def _forward_return_targets(n, start_val=0.05):
    """
    Simulate FORWARD_RETURN targets: value at bar i = remaining return to segment end.
    Starts at `start_val`, declines linearly to ~0 at bar n-1.
    """
    vals = np.linspace(start_val, 0.0, n)
    return pd.Series(vals)


def _amplitude_per_bar_targets(n, value=0.03):
    """Simulate AMPLITUDE_PER_BAR targets: all bars get the same constant value."""
    return pd.Series([value] * n)


# ---------------------------------------------------------------------------
# Tests — causal targets (should NOT trigger violations)
# ---------------------------------------------------------------------------

class TestCausalTargets:
    def test_cumulative_return_is_causal(self):
        """Monotonically growing returns within a segment are NOT flagged as violations."""
        n = 50
        targets = _cumulative_return_targets(n)
        seg_info = _seg_info((0, n - 1, 1))

        result = validate_causality(targets, seg_info)

        assert result["is_causal"] is True
        assert result["violations"] == []
        assert result["total_segments"] == 1

    def test_noisy_causal_targets_are_causal(self):
        """Non-monotonic causal targets (e.g., MOMENTUM) do not hit either pattern."""
        rng = np.random.default_rng(7)
        targets = pd.Series(rng.normal(0.001, 0.01, 80))
        seg_info = _seg_info((0, 79, 1))

        result = validate_causality(targets, seg_info)

        assert result["is_causal"] is True
        assert result["violations"] == []

    def test_multiple_clean_segments(self):
        """Multiple segments with causal targets → all clean."""
        n = 100
        targets = _cumulative_return_targets(n)
        # Three non-overlapping segments
        seg_info = _seg_info((0, 30, 1), (31, 60, -1), (61, 99, 1))

        result = validate_causality(targets, seg_info)

        assert result["is_causal"] is True
        assert result["total_segments"] == 3

    def test_single_bar_segment_skipped(self):
        """Segments with end_idx == start_idx are skipped (too short)."""
        targets = pd.Series([0.01, 0.02, 0.03])
        seg_info = _seg_info((1, 1, 1))  # length 0 after slice

        result = validate_causality(targets, seg_info)

        assert result["is_causal"] is True
        assert result["violations"] == []


# ---------------------------------------------------------------------------
# Tests — forward-return-pattern (FORWARD_RETURN / RETURN_TO_EXTREME)
# ---------------------------------------------------------------------------

class TestForwardReturnPattern:
    def test_forward_return_target_flagged(self):
        """Target that starts large and converges to ~0 at segment end is a violation."""
        n = 40
        targets = _forward_return_targets(n, start_val=0.08)
        seg_info = _seg_info((0, n - 1, 1))

        result = validate_causality(targets, seg_info)

        assert result["is_causal"] is False
        assert len(result["violations"]) == 1
        v = result["violations"][0]
        assert v["reason"] == "forward_return_pattern"
        assert v["start_idx"] == 0
        assert v["end_idx"] == n - 1

    def test_negative_forward_return_target_flagged(self):
        """Negative forward-return (downtrend) also detected."""
        n = 30
        targets = pd.Series(np.linspace(-0.06, 0.0, n))
        seg_info = _seg_info((0, n - 1, -1))

        result = validate_causality(targets, seg_info)

        assert result["is_causal"] is False
        assert any(v["reason"] == "forward_return_pattern" for v in result["violations"])

    def test_violation_carries_correct_values(self):
        """Violation dict carries t_start and t_end values."""
        targets = pd.Series(np.linspace(0.10, 0.0, 20))
        seg_info = _seg_info((0, 19, 1))

        result = validate_causality(targets, seg_info)

        v = result["violations"][0]
        assert abs(v["t_start"] - 0.10) < 1e-6
        assert abs(v["t_end"] - 0.0) < 1e-6


# ---------------------------------------------------------------------------
# Tests — constant-segment-value pattern (AMPLITUDE_PER_BAR)
# ---------------------------------------------------------------------------

class TestConstantSegmentValuePattern:
    def test_constant_segment_flagged(self):
        """All bars in a segment sharing the same value is a violation."""
        n = 20
        targets = _amplitude_per_bar_targets(n, value=0.05)
        seg_info = _seg_info((0, n - 1, 1))

        result = validate_causality(targets, seg_info)

        assert result["is_causal"] is False
        assert len(result["violations"]) == 1
        v = result["violations"][0]
        assert v["reason"] == "constant_segment_value"
        assert abs(v["value"] - 0.05) < 1e-12

    def test_near_constant_with_noise_not_flagged(self):
        """Target with genuine variation (not constant) is not flagged as constant."""
        rng = np.random.default_rng(99)
        targets = pd.Series(0.03 + rng.normal(0, 0.005, 30))
        seg_info = _seg_info((0, 29, 1))

        result = validate_causality(targets, seg_info)

        # No constant-segment violation (std is large relative to mean)
        constant_violations = [v for v in result["violations"] if v["reason"] == "constant_segment_value"]
        assert len(constant_violations) == 0


# ---------------------------------------------------------------------------
# Tests — mixed: some segments causal, some not
# ---------------------------------------------------------------------------

class TestMixedSegments:
    def test_only_violating_segments_recorded(self):
        """Only forward-return segments are recorded; clean segments are not."""
        n = 60
        # First segment: causal cumulative return
        causal = _cumulative_return_targets(30, drift=0.002)
        # Second segment: forward-return (leaking)
        leaking = _forward_return_targets(30, start_val=0.07)
        targets = pd.concat([causal, leaking], ignore_index=True)

        seg_info = _seg_info((0, 29, 1), (30, 59, 1))

        result = validate_causality(targets, seg_info)

        assert result["is_causal"] is False
        assert result["total_segments"] == 2
        assert len(result["violations"]) == 1
        assert result["violations"][0]["start_idx"] == 30

    def test_total_segments_count_correct(self):
        """total_segments reflects the number of rows in segment_info."""
        targets = pd.Series(np.random.randn(100))
        seg_info = _seg_info((0, 24, 1), (25, 49, -1), (50, 74, 1), (75, 99, -1))

        result = validate_causality(targets, seg_info)

        assert result["total_segments"] == 4


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_segment_info(self):
        """Empty segment_info returns is_causal=True with zero violations."""
        targets = pd.Series([0.01, 0.02, 0.03])
        seg_info = pd.DataFrame(columns=["start_idx", "end_idx", "direction"])

        result = validate_causality(targets, seg_info)

        assert result["is_causal"] is True
        assert result["violations"] == []
        assert result["total_segments"] == 0

    def test_nan_targets_in_segment_handled_gracefully(self):
        """Segment with all-NaN targets (after dropna) is skipped without error."""
        targets = pd.Series([np.nan] * 20)
        seg_info = _seg_info((0, 19, 1))

        result = validate_causality(targets, seg_info)

        assert result["is_causal"] is True

    def test_zero_start_value_not_flagged_as_forward_return(self):
        """If t_start ≈ 0, the forward-return heuristic does not fire (no division by zero)."""
        targets = pd.Series(np.linspace(0.0, 0.0, 15))  # All zero
        seg_info = _seg_info((0, 14, 1))

        result = validate_causality(targets, seg_info)

        # Zero-start doesn't trigger forward_return_pattern (guard: abs(t_start) > 1e-8)
        forward_violations = [v for v in result["violations"] if v["reason"] == "forward_return_pattern"]
        assert len(forward_violations) == 0
