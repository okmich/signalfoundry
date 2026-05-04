"""Tests for spec §1 target catalogue and §7 fold-edge censoring."""

from __future__ import annotations

import numpy as np
import pytest

from okmich_quant_labelling.diagnostics.ruptures import (
    CpDistanceTarget,
    LabeledPosteriors,
    UnivariateCost,
    censor_fold_edge_segments,
    cp_distance,
    is_boundary,
    within_segment_position,
)


def _make_labeled(breakpoints: list[int], t_total: int) -> LabeledPosteriors:
    bp = np.asarray(breakpoints, dtype=np.int64)
    seg_ids = np.empty(t_total, dtype=np.int64)
    start = 0
    for sid, end in enumerate(bp):
        seg_ids[start:end] = sid
        start = end
    posterior = np.zeros((t_total, 2), dtype=np.float64)
    posterior[:, 0] = 1.0
    return LabeledPosteriors(
        breakpoints=bp, segment_ids=seg_ids, posterior=posterior,
        cost_model=UnivariateCost.L2, penalty=10.0, hazard_rate=0.01,
        r_max=2, min_size=2, warm_up_length=0, observation_model_class="Stub",
    )


class TestIsBoundary:
    def test_marks_window_around_each_interior_breakpoint(self):
        labeled = _make_labeled([10, 20, 30], t_total=30)
        out = is_boundary(labeled, w=2)
        # Interior breakpoints are at 10, 20 (30 is the trailing endpoint)
        assert out[8] == 1 and out[9] == 1 and out[10] == 1 and out[11] == 1
        assert out[18] == 1 and out[19] == 1 and out[20] == 1 and out[21] == 1
        # Bars further from any boundary are 0
        assert out[5] == 0 and out[15] == 0 and out[25] == 0
        # Trailing endpoint (30) is NOT a boundary — it's just the array length
        assert out[28] == 0 and out[29] == 0

    def test_no_interior_breakpoints_yields_all_zeros(self):
        labeled = _make_labeled([50], t_total=50)
        out = is_boundary(labeled, w=3)
        assert (out == 0).all()

    def test_w_below_one_rejected(self):
        labeled = _make_labeled([10, 20], t_total=20)
        with pytest.raises(ValueError, match="w must be >= 1"):
            is_boundary(labeled, w=0)

    def test_dtype_is_int8(self):
        labeled = _make_labeled([10, 20], t_total=20)
        assert is_boundary(labeled, w=2).dtype == np.int8

    def test_window_clipped_to_array_bounds(self):
        labeled = _make_labeled([2, 50], t_total=50)
        out = is_boundary(labeled, w=10)
        # Breakpoint at 2: window [2-10, 2+10) = [-8, 12) clipped to [0, 12)
        assert out[0] == 1 and out[11] == 1
        # 50 is trailing endpoint, not a boundary
        assert out[49] == 0


class TestCpDistance:
    def test_distance_to_next_interior_breakpoint(self):
        labeled = _make_labeled([10, 20, 30], t_total=30)
        target = cp_distance(labeled, horizon=15)
        assert isinstance(target, CpDistanceTarget)
        # At t=0: next interior breakpoint is 10, d_raw=10, d=10, event=1
        assert target.distance[0] == 10 and target.event[0] == 1
        # At t=9: d_raw=1, d=1, event=1
        assert target.distance[9] == 1 and target.event[9] == 1
        # At t=15: next interior breakpoint is 20, d_raw=5, d=5, event=1
        assert target.distance[15] == 5 and target.event[15] == 1
        # At t=20 (boundary itself): next is none beyond 20 (since 30 is endpoint),
        #   d_raw=+inf, d=horizon=15, event=0
        assert target.distance[20] == 15 and target.event[20] == 0
        assert target.horizon == 15

    def test_right_censoring_caps_distance(self):
        labeled = _make_labeled([100, 200], t_total=200)
        target = cp_distance(labeled, horizon=10)
        # At t=0: d_raw=100, capped to 10, event=0
        assert target.distance[0] == 10 and target.event[0] == 0
        # At t=95: d_raw=5, d=5, event=1
        assert target.distance[95] == 5 and target.event[95] == 1

    def test_no_interior_breakpoints_all_censored(self):
        labeled = _make_labeled([50], t_total=50)
        target = cp_distance(labeled, horizon=12)
        np.testing.assert_array_equal(target.distance, np.full(50, 12, dtype=np.int64))
        assert (target.event == 0).all()

    def test_horizon_below_one_rejected(self):
        labeled = _make_labeled([10, 20], t_total=20)
        with pytest.raises(ValueError, match="horizon"):
            cp_distance(labeled, horizon=0)


class TestWithinSegmentPosition:
    def test_position_runs_zero_to_one_per_segment(self):
        labeled = _make_labeled([5, 10], t_total=10)
        out = within_segment_position(labeled)
        # Segment 0: indices 0..4, length 5, denom = 4
        np.testing.assert_allclose(out[0:5], np.array([0, 1, 2, 3, 4]) / 4.0)
        # Segment 1: indices 5..9, length 5, denom = 4
        np.testing.assert_allclose(out[5:10], np.array([0, 1, 2, 3, 4]) / 4.0)

    def test_singleton_segment_yields_zero(self):
        # min_size in PELT prevents singletons in real flows, but the function
        # must behave for any input shape.
        bp = np.array([1, 5], dtype=np.int64)
        seg_ids = np.array([0, 1, 1, 1, 1], dtype=np.int64)
        posterior = np.zeros((5, 2), dtype=np.float64)
        posterior[:, 0] = 1.0
        labeled = LabeledPosteriors(
            breakpoints=bp, segment_ids=seg_ids, posterior=posterior,
            cost_model=UnivariateCost.L2, penalty=10.0, hazard_rate=0.01,
            r_max=2, min_size=2, warm_up_length=0, observation_model_class="Stub",
        )
        out = within_segment_position(labeled)
        assert out[0] == 0.0  # singleton segment: denom = max(0, 1) = 1, so 0/1 = 0


class TestCensorFoldEdgeSegments:
    def test_drops_both_edge_segments(self):
        labeled = _make_labeled([5, 10, 15, 20], t_total=20)
        mask = censor_fold_edge_segments(labeled)
        # Segment 0 occupies [0, 5), segment 3 occupies [15, 20)
        assert mask[0:5].all()
        assert mask[15:20].all()
        assert not mask[5:15].any()

    def test_single_segment_drops_everything(self):
        labeled = _make_labeled([20], t_total=20)
        mask = censor_fold_edge_segments(labeled)
        # leftmost == rightmost, so all bars are dropped
        assert mask.all()

    def test_returns_bool_dtype(self):
        labeled = _make_labeled([5, 10, 15], t_total=15)
        assert censor_fold_edge_segments(labeled).dtype == bool
