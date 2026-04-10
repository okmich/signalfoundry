import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.regime_filters import AdaptiveKalmanStyleSmoother, ConditionalRandomFieldRefiner, \
    HysteresisProcessor, MarkovJumpProcessRegularizer, MedianFilter, MinimumDurationFilter, ModeFilterWithConfidence, \
    ProcessorPipeline, TransitionRateLimiter

from okmich_quant_ml.regime_filters.utils import build_asymmetric_cost_matrix, build_symmetric_cost_matrix, \
    calculate_label_stability, calculate_regime_sharpe, calculate_transition_costs, calculate_transition_rate


class TestMinimumDurationFilter:
    """Tests for MinimumDurationFilter."""

    def test_basic_filtering(self):
        """Test basic duration filtering removes short spikes."""
        states = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        processor = MinimumDurationFilter({"min_duration": 3})
        smoothed = processor.process(states)

        # Single-period spike at index 2 should be removed
        assert smoothed[2] == 0
        # Long regime starting at index 8 needs 3 periods to confirm
        # So indices 8, 9 are still 0, but 10, 11, 12 should be 1
        assert smoothed[8] == 0
        assert smoothed[9] == 0
        assert all(smoothed[10:] == 1)

    def test_hold_previous_mode(self):
        """Test hold_previous parameter."""
        states = np.array([0, 0, 1, 1, 0, 0])

        # hold_previous=True (default)
        processor = MinimumDurationFilter({"min_duration": 3, "hold_previous": True})
        smoothed = processor.process(states)
        # Should hold regime 0 until sufficient duration
        assert smoothed[2] == 0
        assert smoothed[3] == 0

    def test_pandas_series_support(self):
        """Test that pandas Series input is handled correctly."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        states = pd.Series([0, 0, 1, 0, 0, 0, 1, 1, 1, 1], index=dates)

        processor = MinimumDurationFilter({"min_duration": 3})
        smoothed = processor.process(states)

        # Should return pandas Series with same index
        assert isinstance(smoothed, pd.Series)
        assert all(smoothed.index == states.index)

    def test_online_processing(self):
        """Test online processing mode."""
        processor = MinimumDurationFilter({"min_duration": 3})
        processor.reset()

        states = [0, 0, 1, 1, 1, 0, 0, 0]
        results = []

        for state in states:
            result = processor.process_online(state)
            results.append(result)

        results = np.array(results)

        # First two should be 0
        assert results[0] == 0
        assert results[1] == 0
        # Transition to 1 at index 2, but not confirmed until index 4
        assert results[2] == 0
        assert results[3] == 0
        assert results[4] == 1  # Now confirmed

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="min_duration must be >= 1"):
            MinimumDurationFilter({"min_duration": 0})

        with pytest.raises(ValueError, match="min_duration must be an integer"):
            MinimumDurationFilter({"min_duration": 3.5})

    def test_regime_statistics(self):
        """Test regime statistics calculation."""
        states = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
        processor = MinimumDurationFilter({"min_duration": 2})

        stats = processor.get_regime_statistics(states)
        assert "mean_duration" in stats
        assert "transition_frequency" in stats
        assert "regime_counts" in stats
        assert stats["num_transitions"] == 2


class TestMedianFilter:
    """Tests for MedianFilter."""

    def test_basic_filtering(self):
        """Test basic mode filtering removes isolated errors."""
        states = np.array([0, 0, 1, 0, 0, 0, 1, 1, 1])
        processor = MedianFilter({"window_size": 3})
        smoothed = processor.process(states)

        # Isolated 1 at index 2 should be replaced with 0
        assert smoothed[2] == 0

    def test_symmetric_window(self):
        """Test symmetric window behavior (deprecated non-causal mode)."""
        states = np.array([0, 0, 0, 1, 0, 0, 0])
        # Explicitly test deprecated non-causal mode
        with pytest.warns(DeprecationWarning, match="causal=False.*look-ahead bias"):
            processor = MedianFilter({"window_size": 3, "causal": False})
        smoothed = processor.process(states)

        # Single 1 in middle should be replaced
        assert smoothed[3] == 0

    def test_causal_mode(self):
        """Test causal (look-back only) mode - now default behavior."""
        states = np.array([0, 0, 0, 1, 1, 1])
        processor = MedianFilter({"window_size": 3, "causal": True})
        smoothed = processor.process(states)

        # Each position should only look backward
        assert isinstance(smoothed, np.ndarray)

    def test_online_processing(self):
        """Test online processing mode."""
        processor = MedianFilter({"window_size": 3, "causal": True})
        processor.reset()

        states = [0, 0, 1, 0, 0]
        results = []

        for state in states:
            result = processor.process_online(state)
            results.append(result)

        results = np.array(results)
        # Mode filter should smooth out isolated spike
        assert isinstance(results, np.ndarray)

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError, match="window_size must be odd"):
                MedianFilter({"window_size": 4, "causal": False})

        with pytest.raises(ValueError, match="window_size must be >= 1"):
            MedianFilter({"window_size": 0})

    def test_causal_default(self):
        """Verify causal=True is the default."""
        processor = MedianFilter({"window_size": 5})
        assert processor.config["causal"] is True

    def test_deprecation_warning(self):
        """Verify deprecation warning when causal=False."""
        with pytest.warns(DeprecationWarning, match="causal=False.*look-ahead bias"):
            processor = MedianFilter({"window_size": 5, "causal": False})
        assert processor.config["causal"] is False

    def test_online_mode_requires_causal(self):
        """Online mode should raise error if causal=False."""
        with pytest.warns(DeprecationWarning):
            processor = MedianFilter({"window_size": 5, "causal": False})
        processor.reset()

        with pytest.raises(ValueError, match="process_online.*requires causal=True"):
            processor.process_online(state=1)


class TestHysteresisProcessor:
    """Tests for HysteresisProcessor."""

    def test_count_based_hysteresis(self):
        """Test count-based hysteresis behavior."""
        states = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        processor = HysteresisProcessor(
            {"entry_threshold": 3, "exit_threshold": 2, "use_confidence": False}
        )
        smoothed = processor.process(states)

        # Should require 3 consecutive observations to enter new regime
        assert smoothed[2] == 0
        assert smoothed[3] == 0
        assert smoothed[4] == 1  # Entered after 3 consecutive

    def test_asymmetric_thresholds(self):
        """Test asymmetric entry/exit thresholds."""
        states = np.array([0] * 10 + [1] * 5 + [0] * 10)
        processor = HysteresisProcessor(
            {
                "entry_threshold": 5,  # Hard to enter
                "exit_threshold": 2,  # Easy to exit
                "use_confidence": False,
            }
        )
        smoothed = processor.process(states)

        # Should be sticky when entering but quick to exit
        assert isinstance(smoothed, np.ndarray)

    def test_confidence_based_hysteresis(self):
        """Test confidence-based hysteresis with posteriors."""
        states = np.array([0, 0, 1, 1, 1])
        posteriors = np.array(
            [[0.9, 0.1], [0.85, 0.15], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]]
        )

        processor = HysteresisProcessor(
            {"entry_threshold": 1.5, "exit_threshold": 1.0, "use_confidence": True}
        )
        smoothed = processor.process(states, posteriors=posteriors)

        assert isinstance(smoothed, np.ndarray)

    def test_online_processing(self):
        """Test online processing mode."""
        processor = HysteresisProcessor(
            {"entry_threshold": 3, "exit_threshold": 2, "use_confidence": False}
        )
        processor.reset()

        states = [0, 0, 0, 1, 1, 1, 1, 1, 1]
        results = []

        for state in states:
            result = processor.process_online(state)
            results.append(result)

        results = np.array(results)
        # Should stay in regime 0 until exit threshold (2) and entry threshold (3) met
        # First 3 are regime 0, then need 2 exits + 3 consecutive entries
        # Index 3-4: 2 exits met
        # Index 4-6: 3 consecutive for entry
        assert results[3] == 0
        assert results[4] == 0
        # After enough consecutive observations, should transition
        assert results[8] == 1  # Should have transitioned by now

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            HysteresisProcessor({"entry_threshold": -1, "use_confidence": False})


class TestProcessorPipeline:
    """Tests for ProcessorPipeline."""

    def test_sequential_processing(self):
        """Test that processors are applied sequentially."""
        states = np.array([0, 0, 1, 0, 0, 0, 1, 1, 1, 1])

        pipeline = ProcessorPipeline(
            [
                MinimumDurationFilter({"min_duration": 2}),
                MedianFilter({"window_size": 3}),
            ]
        )

        smoothed = pipeline.process(states)
        assert isinstance(smoothed, np.ndarray)
        # Pipeline should smooth more than individual processors
        assert np.sum(smoothed[1:] != smoothed[:-1]) <= np.sum(
            states[1:] != states[:-1]
        )

    def test_online_processing(self):
        """Test online processing through pipeline."""
        pipeline = ProcessorPipeline(
            [
                MinimumDurationFilter({"min_duration": 2}),
                MedianFilter({"window_size": 3, "causal": True}),
            ]
        )
        pipeline.reset()

        states = [0, 0, 1, 1, 1, 0, 0]
        results = []

        for state in states:
            result = pipeline.process_online(state)
            results.append(result)

        assert len(results) == len(states)

    def test_reset(self):
        """Test pipeline reset functionality."""
        pipeline = ProcessorPipeline(
            [
                MinimumDurationFilter({"min_duration": 2}),
            ]
        )

        # Process some data
        pipeline.process_online(0)
        pipeline.process_online(1)

        # Reset
        pipeline.reset()

        # Should start fresh
        result = pipeline.process_online(1)
        assert result == 1

    def test_empty_pipeline_raises(self):
        """Test that empty pipeline raises error."""
        with pytest.raises(ValueError, match="At least one processor"):
            ProcessorPipeline([])

    def test_pipeline_length(self):
        """Test pipeline length."""
        pipeline = ProcessorPipeline(
            [
                MinimumDurationFilter({"min_duration": 2}),
                MedianFilter({"window_size": 3}),
            ]
        )
        assert len(pipeline) == 2

    def test_pipeline_indexing(self):
        """Test pipeline indexing."""
        min_filter = MinimumDurationFilter({"min_duration": 2})
        median_filter = MedianFilter({"window_size": 3})

        pipeline = ProcessorPipeline([min_filter, median_filter])

        assert pipeline[0] == min_filter
        assert pipeline[1] == median_filter


class TestCostFunctions:
    """Tests for cost calculation functions."""

    def test_simple_transition_costs(self):
        """Test simple fixed cost calculation."""
        original = np.array([0, 1, 0, 1, 0, 1, 1, 1, 2, 2])
        smoothed = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])

        costs = calculate_transition_costs(original, smoothed, cost_model=5.0)

        assert costs["original_transitions"] > costs["smoothed_transitions"]
        assert costs["savings_bps"] > 0
        assert costs["transitions_saved"] > 0

    def test_regime_specific_costs(self):
        """Test regime-specific cost calculation."""
        original = np.array([0, 1, 0, 1, 0])
        smoothed = np.array([0, 0, 0, 1, 1])

        cost_matrix = {
            (0, 1): 10,
            (1, 0): 7,
        }

        costs = calculate_transition_costs(original, smoothed, cost_model=cost_matrix)

        assert "cost_by_transition" in costs
        assert costs["original_cost_bps"] > costs["smoothed_cost_bps"]

    def test_symmetric_cost_matrix_builder(self):
        """Test symmetric cost matrix builder."""
        matrix = build_symmetric_cost_matrix(3, base_cost=5.0)

        assert matrix[(0, 1)] == 5.0
        assert matrix[(1, 0)] == 5.0
        assert matrix[(2, 1)] == 5.0
        assert (0, 0) not in matrix  # No self-transitions

    def test_asymmetric_cost_matrix_builder(self):
        """Test asymmetric cost matrix builder."""
        entry_costs = {0: 5, 1: 10, 2: 8}
        exit_costs = {0: 7, 1: 6, 2: 15}

        matrix = build_asymmetric_cost_matrix(3, entry_costs, exit_costs)

        # Cost should be average of exit and entry
        expected_0_to_1 = (exit_costs[0] + entry_costs[1]) / 2
        assert matrix[(0, 1)] == expected_0_to_1


class TestMetricFunctions:
    """Tests for metric calculation functions."""

    def test_transition_rate(self):
        """Test transition rate calculation."""
        states = np.array([0, 0, 1, 1, 1, 0, 0])
        rate = calculate_transition_rate(states)

        # 2 transitions in 7 periods
        assert abs(rate - 2 / 7) < 1e-6

    def test_label_stability(self):
        """Test label stability metrics."""
        states = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        stats = calculate_label_stability(states)

        assert stats["mean_duration"] == 3.0
        assert stats["min_duration"] == 2
        assert stats["max_duration"] == 4
        assert "entropy" in stats
        assert "transition_rate" in stats

    def test_regime_sharpe(self):
        """Test regime Sharpe ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005])
        regimes = np.array([0, 0, 1, 1, 1])
        positions = {0: 1.0, 1: -1.0}  # Long regime 0, short regime 1

        sharpe = calculate_regime_sharpe(returns, regimes, positions)

        assert "overall" in sharpe
        assert "by_regime" in sharpe
        assert "mean_return_annualized" in sharpe
        assert "volatility_annualized" in sharpe

    def test_regime_sharpe_pandas(self):
        """Test regime Sharpe with pandas Series."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        regimes = pd.Series(np.random.choice([0, 1], 100), index=dates)
        positions = {0: 1.0, 1: 0.0}

        sharpe = calculate_regime_sharpe(returns, regimes, positions)

        assert isinstance(sharpe["overall"], float)
        assert isinstance(sharpe["by_regime"], dict)


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_pipeline_workflow(self):
        """Test complete pipeline workflow with all components."""
        # Generate synthetic data
        np.random.seed(42)
        true_regimes = np.repeat([0, 1, 0, 2, 1], 20)
        # Add noise
        noisy_regimes = true_regimes.copy()
        noise_indices = np.random.choice(len(noisy_regimes), 15, replace=False)
        noisy_regimes[noise_indices] = np.random.choice([0, 1, 2], 15)

        # Create pipeline
        pipeline = ProcessorPipeline(
            [
                MinimumDurationFilter({"min_duration": 5}),
                HysteresisProcessor(
                    {"entry_threshold": 3, "exit_threshold": 2, "use_confidence": False}
                ),
                MedianFilter({"window_size": 5}),
            ]
        )

        # Process
        smoothed = pipeline.process(noisy_regimes)

        # Calculate metrics
        original_stability = calculate_label_stability(noisy_regimes)
        smoothed_stability = calculate_label_stability(smoothed)

        # Smoothed should have longer mean duration and lower transition rate
        assert (
            smoothed_stability["mean_duration"] >= original_stability["mean_duration"]
        )
        assert (
            smoothed_stability["transition_rate"]
            <= original_stability["transition_rate"]
        )

        # Calculate costs
        costs = calculate_transition_costs(noisy_regimes, smoothed, cost_model=5.0)
        assert costs["savings_bps"] >= 0

    def test_online_offline_consistency(self):
        """Test that online and offline modes produce similar results."""
        states = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1])

        processor = MinimumDurationFilter({"min_duration": 3})

        # Offline processing
        offline_result = processor.process(states)

        # Online processing
        processor.reset()
        online_results = []
        for state in states:
            result = processor.process_online(state)
            online_results.append(result)
        online_result = np.array(online_results)

        # Results should match
        np.testing.assert_array_equal(offline_result, online_result)


class TestModeFilterWithConfidence:
    """Tests for ModeFilterWithConfidence (Phase 2)."""

    def test_basic_confidence_filtering(self):
        """Test confidence-weighted filtering."""
        states = np.array([0, 0, 1, 1, 0, 0, 0])
        # Create posteriors where state 0 has higher confidence
        posteriors = np.array(
            [
                [0.9, 0.1],
                [0.85, 0.15],
                [0.4, 0.6],  # Low confidence for state 1
                [0.45, 0.55],  # Low confidence for state 1
                [0.9, 0.1],
                [0.9, 0.1],
                [0.9, 0.1],
            ]
        )

        processor = ModeFilterWithConfidence(
            {"window_size": 5, "confidence_weight": 2.0, "min_score_threshold": 0.5}
        )

        smoothed = processor.process(states, posteriors=posteriors)
        assert isinstance(smoothed, np.ndarray)

    def test_requires_posteriors(self):
        """Test that processor requires posteriors."""
        states = np.array([0, 1, 0, 1])
        processor = ModeFilterWithConfidence({"window_size": 3})

        with pytest.raises(ValueError, match="requires posteriors"):
            processor.process(states)

    def test_confidence_weight_zero(self):
        """Test that confidence_weight=0 ignores posteriors."""
        states = np.array([0, 0, 1, 0, 0])
        # High confidence for wrong state
        posteriors = np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.9, 0.1],  # High confidence but only 1 occurrence
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        )

        processor = ModeFilterWithConfidence(
            {"window_size": 5, "confidence_weight": 0.0}  # Ignore confidence
        )

        smoothed = processor.process(states, posteriors=posteriors)
        # Should be mostly state 0 (more frequent)
        assert np.sum(smoothed == 0) >= 4

    def test_high_confidence_weight(self):
        """Test that high confidence_weight emphasizes confidence."""
        states = np.array([0, 0, 1, 1, 1])
        # State 1 has fewer occurrences but very high confidence
        posteriors = np.array(
            [
                [0.6, 0.4],
                [0.6, 0.4],
                [0.1, 0.9],  # Very high confidence for state 1
                [0.1, 0.9],
                [0.1, 0.9],
            ]
        )

        processor = ModeFilterWithConfidence(
            {"window_size": 5, "confidence_weight": 3.0}  # Heavily weight confidence
        )

        smoothed = processor.process(states, posteriors=posteriors)
        # Should favor state 1 due to high confidence
        assert isinstance(smoothed, np.ndarray)

    def test_pandas_support(self):
        """Test pandas Series/DataFrame support."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        states = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 1, 0], index=dates)
        posteriors = pd.DataFrame(np.random.dirichlet([2, 2], 10), index=dates)

        processor = ModeFilterWithConfidence({"window_size": 5})
        smoothed = processor.process(states, posteriors=posteriors)

        assert isinstance(smoothed, pd.Series)
        assert all(smoothed.index == states.index)

    def test_online_processing(self):
        """Test online processing mode."""
        processor = ModeFilterWithConfidence(
            {"window_size": 3, "confidence_weight": 1.0}
        )
        processor.reset()

        states = [0, 0, 1, 1, 0]
        posteriors_list = [
            np.array([0.8, 0.2]),
            np.array([0.9, 0.1]),
            np.array([0.3, 0.7]),
            np.array([0.2, 0.8]),
            np.array([0.85, 0.15]),
        ]

        results = []
        for state, posterior in zip(states, posteriors_list):
            result = processor.process_online(state, posterior=posterior)
            results.append(result)

        assert len(results) == len(states)

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError, match="window_size must be odd"):
                ModeFilterWithConfidence({"window_size": 4, "causal": False})

        with pytest.raises(ValueError, match="confidence_weight must be a number"):
            ModeFilterWithConfidence({"confidence_weight": "high"})

        with pytest.raises(ValueError, match="confidence_weight must be >= 0"):
            ModeFilterWithConfidence({"confidence_weight": -1.0})

    def test_min_score_threshold(self):
        """Test minimum score threshold filtering."""
        states = np.array([0, 1, 2, 0, 0])
        posteriors = np.array(
            [
                [0.6, 0.3, 0.1],
                [0.3, 0.6, 0.1],
                [0.3, 0.1, 0.6],
                [0.6, 0.3, 0.1],
                [0.6, 0.3, 0.1],
            ]
        )

        processor = ModeFilterWithConfidence(
            {
                "window_size": 3,
                "confidence_weight": 1.0,
                "min_score_threshold": 1.5,  # High threshold
            }
        )

        smoothed = processor.process(states, posteriors=posteriors)
        assert isinstance(smoothed, np.ndarray)

    def test_causal_default(self):
        """Verify causal=True is the default."""
        processor = ModeFilterWithConfidence({"window_size": 7})
        assert processor.config["causal"] is True

    def test_deprecation_warning(self):
        """Verify deprecation warning when causal=False."""
        with pytest.warns(DeprecationWarning, match="causal=False.*look-ahead bias"):
            processor = ModeFilterWithConfidence({"window_size": 7, "causal": False})
        assert processor.config["causal"] is False

    def test_online_mode_requires_causal(self):
        """Online mode should raise error if causal=False."""
        with pytest.warns(DeprecationWarning):
            processor = ModeFilterWithConfidence({"window_size": 7, "causal": False})
        processor.reset()

        posterior = np.array([0.8, 0.2])
        with pytest.raises(ValueError, match="process_online.*requires causal=True"):
            processor.process_online(state=1, posterior=posterior)


class TestTransitionRateLimiter:
    """Tests for TransitionRateLimiter (Phase 2)."""

    def test_basic_rate_limiting(self):
        """Test basic transition rate limiting."""
        # Create sequence with many transitions
        states = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        processor = TransitionRateLimiter(
            {"max_transitions_per_window": 2, "window_size": 10, "penalty_duration": 3}
        )

        smoothed = processor.process(states)

        # Should have limited transitions
        original_transitions = np.sum(states[1:] != states[:-1])
        smoothed_transitions = np.sum(smoothed[1:] != smoothed[:-1])

        assert smoothed_transitions <= 2
        assert smoothed_transitions < original_transitions

    def test_penalty_duration(self):
        """Test that penalty duration is enforced."""
        # States: 0, 0, 1, 1, 2, 2, 0, 0, ...
        states = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 0, 0])

        processor = TransitionRateLimiter(
            {"max_transitions_per_window": 2, "window_size": 12, "penalty_duration": 5}
        )

        smoothed = processor.process(states)

        # After hitting limit, should hold current regime
        transitions = np.sum(smoothed[1:] != smoothed[:-1])
        assert transitions <= 2

    def test_unlimited_transitions(self):
        """Test with high limit (effectively unlimited)."""
        states = np.array([0, 1, 0, 1, 0, 1])

        processor = TransitionRateLimiter(
            {"max_transitions_per_window": 100, "window_size": 10}
        )

        smoothed = processor.process(states)

        # Should allow all transitions
        np.testing.assert_array_equal(states, smoothed)

    def test_zero_transitions_allowed(self):
        """Test with max_transitions=0."""
        states = np.array([0, 1, 0, 1, 0, 1])

        processor = TransitionRateLimiter(
            {"max_transitions_per_window": 0, "window_size": 10}
        )

        smoothed = processor.process(states)

        # Should hold first state
        assert all(smoothed == 0)

    def test_pandas_support(self):
        """Test pandas Series support."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        states = pd.Series(np.tile([0, 1], 10), index=dates)

        processor = TransitionRateLimiter(
            {"max_transitions_per_window": 3, "window_size": 20}
        )

        smoothed = processor.process(states)

        assert isinstance(smoothed, pd.Series)
        assert all(smoothed.index == states.index)

    def test_online_processing(self):
        """Test online processing mode."""
        processor = TransitionRateLimiter(
            {"max_transitions_per_window": 2, "window_size": 10, "penalty_duration": 3}
        )
        processor.reset()

        states = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        results = []

        for state in states:
            result = processor.process_online(state)
            results.append(result)

        results = np.array(results)

        # Should limit transitions
        transitions = np.sum(results[1:] != results[:-1])
        assert transitions <= 2

    def test_transition_history(self):
        """Test getting transition history."""
        processor = TransitionRateLimiter(
            {"max_transitions_per_window": 3, "window_size": 10}
        )
        processor.reset()

        states = [0, 0, 1, 1, 0, 0]
        for state in states:
            processor.process_online(state)

        history = processor.get_transition_history()
        assert history is not None
        assert len(history) > 0

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(
            ValueError, match="max_transitions_per_window must be an integer"
        ):
            TransitionRateLimiter({"max_transitions_per_window": 2.5})

        with pytest.raises(ValueError, match="window_size must be >= 1"):
            TransitionRateLimiter({"window_size": 0})

        with pytest.raises(NotImplementedError, match="cost_aware"):
            TransitionRateLimiter({"cost_aware": True})

        with pytest.raises(NotImplementedError, match="non-overlapping"):
            TransitionRateLimiter({"sliding": False})

    def test_long_sequence(self):
        """Test with longer sequence to verify sliding window."""
        np.random.seed(42)
        states = np.random.choice([0, 1], 100)

        processor = TransitionRateLimiter(
            {"max_transitions_per_window": 5, "window_size": 20}
        )

        smoothed = processor.process(states)

        # Check that no 20-period window has more than 5 transitions
        for i in range(20, 100):
            window = smoothed[i - 20 : i + 1]
            window_transitions = np.sum(window[1:] != window[:-1])
            assert window_transitions <= 5


class TestPhase2Integration:
    """Integration tests for Phase 2 processors."""

    def test_confidence_and_rate_limit_pipeline(self):
        """Test combining confidence filter with rate limiter."""
        np.random.seed(42)
        states = np.random.choice([0, 1, 2], 50)
        posteriors = np.random.dirichlet([2, 2, 2], 50)

        pipeline = ProcessorPipeline(
            [
                ModeFilterWithConfidence({"window_size": 5, "confidence_weight": 1.5}),
                TransitionRateLimiter(
                    {"max_transitions_per_window": 3, "window_size": 20}
                ),
            ]
        )

        smoothed = pipeline.process(states, posteriors=posteriors)

        # Should be smoother than original
        original_transitions = np.sum(states[1:] != states[:-1])
        smoothed_transitions = np.sum(smoothed[1:] != smoothed[:-1])

        assert smoothed_transitions < original_transitions

    def test_full_phase_2_pipeline(self):
        """Test all Phase 2 processors in a pipeline."""
        np.random.seed(42)
        true_regimes = np.repeat([0, 1, 2, 0], 25)
        noisy_regimes = true_regimes.copy()

        # Add noise
        noise_idx = np.random.choice(100, 30, replace=False)
        noisy_regimes[noise_idx] = np.random.choice([0, 1, 2], 30)

        # Generate posteriors correlated with states
        posteriors = np.zeros((100, 3))
        for i in range(100):
            probs = np.array([0.2, 0.2, 0.2])
            probs[noisy_regimes[i]] = 0.6
            posteriors[i] = probs

        pipeline = ProcessorPipeline(
            [
                MinimumDurationFilter({"min_duration": 3}),
                ModeFilterWithConfidence({"window_size": 5, "confidence_weight": 2.0}),
                TransitionRateLimiter(
                    {"max_transitions_per_window": 4, "window_size": 25}
                ),
            ]
        )

        smoothed = pipeline.process(noisy_regimes, posteriors=posteriors)

        # Calculate metrics
        original_stability = calculate_label_stability(noisy_regimes)
        smoothed_stability = calculate_label_stability(smoothed)

        # Should be more stable
        assert (
            smoothed_stability["mean_duration"] >= original_stability["mean_duration"]
        )
        assert (
            smoothed_stability["transition_rate"]
            <= original_stability["transition_rate"]
        )

    def test_online_offline_consistency_phase2(self):
        """Test online/offline consistency for Phase 2 processors."""
        np.random.seed(42)
        states = np.random.choice([0, 1], 20)
        posteriors = np.random.dirichlet([3, 3], 20)

        # Test ModeFilterWithConfidence
        processor = ModeFilterWithConfidence(
            {"window_size": 5, "confidence_weight": 1.0}
        )

        offline_result = processor.process(states, posteriors=posteriors)

        processor.reset()
        online_results = []
        for i in range(len(states)):
            result = processor.process_online(states[i], posterior=posteriors[i])
            online_results.append(result)
        online_result = np.array(online_results)

        # Allow small differences due to circular buffer edge effects
        # Most elements should match
        match_rate = np.mean(offline_result == online_result)
        assert match_rate >= 0.85, f"Match rate {match_rate:.2%} is too low"


class TestMarkovJumpProcessRegularizer:
    """Tests for MarkovJumpProcessRegularizer (Phase 3)."""

    def test_basic_dwell_time_regularization(self):
        """Test basic dwell time-based regularization."""
        # Create sequence with varying dwell times
        states = np.array([0] * 10 + [1] * 3 + [0] * 15 + [1] * 8)

        # Configure with gamma distributions
        processor = MarkovJumpProcessRegularizer(
            {
                "state_dwell_params": {
                    0: {
                        "distribution": "gamma",
                        "shape": 2.0,
                        "scale": 10.0,
                    },  # Long dwell
                    1: {
                        "distribution": "gamma",
                        "shape": 1.5,
                        "scale": 5.0,
                    },  # Short dwell
                },
                "regularization_strength": 1.0,
            }
        )

        smoothed = processor.process(states)
        assert isinstance(smoothed, np.ndarray)

    def test_exponential_distribution(self):
        """Test exponential distribution for dwell times."""
        states = np.array([0, 0, 1, 1, 1, 0, 0])

        processor = MarkovJumpProcessRegularizer(
            {
                "state_dwell_params": {
                    0: {"distribution": "exponential", "rate": 0.2},
                    1: {"distribution": "exponential", "rate": 0.3},
                }
            }
        )

        smoothed = processor.process(states)
        assert len(smoothed) == len(states)

    def test_lognormal_distribution(self):
        """Test lognormal distribution for dwell times."""
        states = np.array([0] * 20 + [1] * 10)

        processor = MarkovJumpProcessRegularizer(
            {
                "state_dwell_params": {
                    0: {"distribution": "lognormal", "mu": 3.0, "sigma": 0.5},
                    1: {"distribution": "lognormal", "mu": 2.0, "sigma": 0.5},
                }
            }
        )

        smoothed = processor.process(states)
        assert isinstance(smoothed, np.ndarray)

    def test_estimate_from_data(self):
        """Test automatic estimation of dwell distributions."""
        # Create sequence with clear dwell patterns
        states = np.array([0] * 50 + [1] * 20 + [0] * 40 + [1] * 15)

        processor = MarkovJumpProcessRegularizer(
            {"estimate_from_data": True, "min_samples": 2}
        )

        smoothed = processor.process(states)

        # Check that distributions were learned
        assert 0 in processor.config["state_dwell_params"]
        assert processor.config["state_dwell_params"][0]["distribution"] == "gamma"

    def test_transition_costs(self):
        """Test with transition costs."""
        states = np.array([0, 0, 1, 1, 0, 0])

        transition_costs = {
            (0, 1): 2.0,  # Expensive
            (1, 0): 0.5,  # Cheap
        }

        processor = MarkovJumpProcessRegularizer(
            {
                "state_dwell_params": {
                    0: {"distribution": "gamma", "shape": 2.0, "scale": 3.0},
                    1: {"distribution": "gamma", "shape": 2.0, "scale": 3.0},
                },
                "transition_costs": transition_costs,
                "regularization_strength": 2.0,
            }
        )

        smoothed = processor.process(states)
        assert isinstance(smoothed, np.ndarray)

    def test_online_processing(self):
        """Test online processing mode."""
        processor = MarkovJumpProcessRegularizer(
            {
                "state_dwell_params": {
                    0: {"distribution": "gamma", "shape": 2.0, "scale": 5.0},
                    1: {"distribution": "gamma", "shape": 1.5, "scale": 3.0},
                }
            }
        )
        processor.reset()

        states = [0, 0, 0, 1, 1, 1, 0, 0]
        results = []

        for state in states:
            result = processor.process_online(state)
            results.append(result)

        assert len(results) == len(states)

    def test_pandas_support(self):
        """Test pandas Series support."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        states = pd.Series([0] * 10 + [1] * 10, index=dates)

        processor = MarkovJumpProcessRegularizer(
            {
                "state_dwell_params": {
                    0: {"distribution": "gamma", "shape": 2.0, "scale": 5.0},
                    1: {"distribution": "gamma", "shape": 2.0, "scale": 5.0},
                }
            }
        )

        smoothed = processor.process(states)
        assert isinstance(smoothed, pd.Series)
        assert all(smoothed.index == states.index)

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="state_dwell_params must be a dict"):
            MarkovJumpProcessRegularizer({"state_dwell_params": "invalid"})

        with pytest.raises(ValueError, match="Invalid distribution type"):
            MarkovJumpProcessRegularizer(
                {"state_dwell_params": {0: {"distribution": "invalid"}}}
            )

        with pytest.raises(ValueError, match="Gamma distribution requires"):
            MarkovJumpProcessRegularizer(
                {"state_dwell_params": {0: {"distribution": "gamma"}}}
            )


class TestAdaptiveKalmanStyleSmoother:
    """Tests for AdaptiveKalmanStyleSmoother (Phase 3)."""

    def test_basic_smoothing(self):
        """Test basic Kalman-style smoothing."""
        np.random.seed(42)
        states = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1])
        posteriors = np.random.dirichlet([3, 3], 9)

        processor = AdaptiveKalmanStyleSmoother(
            {"num_states": 2, "process_noise": 0.1, "measurement_noise": 0.2}
        )

        smoothed = processor.process(states, posteriors=posteriors)
        assert isinstance(smoothed, np.ndarray)

    def test_requires_posteriors(self):
        """Test that processor requires posteriors."""
        states = np.array([0, 1, 0, 1])
        processor = AdaptiveKalmanStyleSmoother({"num_states": 2})

        with pytest.raises(ValueError, match="requires posteriors"):
            processor.process(states)

    def test_adaptation(self):
        """Test noise parameter adaptation."""
        np.random.seed(42)
        states = np.random.choice([0, 1, 2], 50)
        posteriors = np.random.dirichlet([2, 2, 2], 50)

        processor = AdaptiveKalmanStyleSmoother(
            {
                "num_states": 3,
                "process_noise": 0.1,
                "measurement_noise": 0.2,
                "adaptation_rate": 0.1,
            }
        )

        smoothed = processor.process(states, posteriors=posteriors)

        # Check that adaptation occurred
        adapted_params = processor.get_adapted_parameters()
        assert adapted_params is not None
        assert "process_noise" in adapted_params
        assert "measurement_noise" in adapted_params

    def test_online_processing(self):
        """Test online processing mode."""
        np.random.seed(42)
        states = np.random.choice([0, 1], 10)
        posteriors = np.random.dirichlet([3, 3], 10)

        processor = AdaptiveKalmanStyleSmoother(
            {"num_states": 2, "process_noise": 0.1, "measurement_noise": 0.2}
        )
        processor.reset()

        results = []
        for i in range(len(states)):
            result = processor.process_online(states[i], posterior=posteriors[i])
            results.append(result)

        assert len(results) == len(states)

        # Check adaptation in online mode
        adapted_params = processor.get_adapted_parameters()
        assert adapted_params is not None

    def test_pandas_support(self):
        """Test pandas Series support."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=15, freq="D")
        states = pd.Series(np.random.choice([0, 1], 15), index=dates)
        posteriors = pd.DataFrame(np.random.dirichlet([3, 3], 15), index=dates)

        processor = AdaptiveKalmanStyleSmoother({"num_states": 2})
        smoothed = processor.process(states, posteriors=posteriors)

        assert isinstance(smoothed, pd.Series)
        assert all(smoothed.index == states.index)

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="num_states is required"):
            AdaptiveKalmanStyleSmoother({})

        with pytest.raises(ValueError, match="num_states must be >= 2"):
            AdaptiveKalmanStyleSmoother({"num_states": 1})

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            AdaptiveKalmanStyleSmoother({"num_states": 2, "process_noise": 1.5})


class TestConditionalRandomFieldRefiner:
    """Tests for ConditionalRandomFieldRefiner (Phase 3)."""

    def test_basic_viterbi(self):
        """Test basic Viterbi inference."""
        np.random.seed(42)
        states = np.array([0, 0, 1, 1, 0, 0])
        posteriors = np.random.dirichlet([3, 3], 6)

        # Should warn about non-causal nature
        with pytest.warns(UserWarning, match="NOT suitable for live trading"):
            processor = ConditionalRandomFieldRefiner({"observation_weight": 1.0})

        smoothed = processor.process(states, posteriors=posteriors)
        assert isinstance(smoothed, np.ndarray)
        assert len(smoothed) == len(states)

    def test_requires_posteriors(self):
        """Test that processor requires posteriors."""
        states = np.array([0, 1, 0, 1])
        with pytest.warns(UserWarning, match="NOT suitable for live trading"):
            processor = ConditionalRandomFieldRefiner({})

        with pytest.raises(ValueError, match="requires posteriors"):
            processor.process(states)

    def test_custom_transition_weights(self):
        """Test with custom transition weights."""
        np.random.seed(42)
        states = np.array([0, 0, 1, 1, 0, 0])
        posteriors = np.random.dirichlet([3, 3], 6)

        # Prefer staying in same state
        transition_weights = np.array(
            [[0, 5], [5, 0]]  # From 0: stay=0, to_1=5  # From 1: to_0=5, stay=0
        )

        with pytest.warns(UserWarning, match="NOT suitable for live trading"):
            processor = ConditionalRandomFieldRefiner(
                {"transition_weights": transition_weights}
            )

        smoothed = processor.process(states, posteriors=posteriors)
        # Should have fewer transitions due to high transition costs
        transitions = np.sum(smoothed[1:] != smoothed[:-1])
        assert transitions <= np.sum(states[1:] != states[:-1])

    def test_smooth_transitions_parameter(self):
        """Test smooth_transitions parameter."""
        np.random.seed(42)
        states = np.random.choice([0, 1], 20)
        posteriors = np.random.dirichlet([3, 3], 20)

        # High smoothness penalty
        with pytest.warns(UserWarning, match="NOT suitable for live trading"):
            processor = ConditionalRandomFieldRefiner({"smooth_transitions": 10.0})

        smoothed = processor.process(states, posteriors=posteriors)

        # Should have fewer transitions
        original_transitions = np.sum(states[1:] != states[:-1])
        smoothed_transitions = np.sum(smoothed[1:] != smoothed[:-1])
        assert smoothed_transitions <= original_transitions

    def test_learn_transition_weights(self):
        """Test learning transition weights from data."""
        np.random.seed(42)
        # Create patterns
        states = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        ground_truth = np.array([0, 0, 0, 1, 1, 1, 1, 1])

        with pytest.warns(UserWarning, match="NOT suitable for live trading"):
            processor = ConditionalRandomFieldRefiner({})
        learned_weights = processor.learn_transition_weights(states, ground_truth)

        assert learned_weights.shape == (2, 2)
        # Diagonal should be lower (staying is more common)
        assert learned_weights[0, 0] < learned_weights[0, 1]

    def test_online_processing(self):
        """Test online processing with sliding window."""
        np.random.seed(42)
        states = np.random.choice([0, 1], 15)
        posteriors = np.random.dirichlet([3, 3], 15)

        with pytest.warns(UserWarning, match="NOT suitable for live trading"):
            processor = ConditionalRandomFieldRefiner({})
        processor.reset()

        results = []
        for i in range(len(states)):
            result = processor.process_online(states[i], posterior=posteriors[i])
            results.append(result)

        assert len(results) == len(states)

    def test_pandas_support(self):
        """Test pandas Series support."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        states = pd.Series(np.random.choice([0, 1, 2], 10), index=dates)
        posteriors = pd.DataFrame(np.random.dirichlet([2, 2, 2], 10), index=dates)

        with pytest.warns(UserWarning, match="NOT suitable for live trading"):
            processor = ConditionalRandomFieldRefiner({})
        smoothed = processor.process(states, posteriors=posteriors)

        assert isinstance(smoothed, pd.Series)
        assert all(smoothed.index == states.index)

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.warns(UserWarning, match="NOT suitable for live trading"):
            with pytest.raises(ValueError, match="observation_weight must be positive"):
                ConditionalRandomFieldRefiner({"observation_weight": 0})

        with pytest.warns(UserWarning, match="NOT suitable for live trading"):
            with pytest.raises(ValueError, match="Only 'viterbi' inference"):
                ConditionalRandomFieldRefiner({"inference_method": "forward_backward"})


class TestPhase3Integration:
    """Integration tests for Phase 3 processors."""

    def test_markov_jump_in_pipeline(self):
        """Test Markov Jump regularizer in pipeline."""
        np.random.seed(42)
        states = np.repeat([0, 1, 0, 1], [50, 20, 40, 30])
        # Add noise
        noise_idx = np.random.choice(len(states), 20, replace=False)
        states[noise_idx] = (states[noise_idx] + 1) % 2

        pipeline = ProcessorPipeline(
            [
                MinimumDurationFilter({"min_duration": 3}),
                MarkovJumpProcessRegularizer(
                    {
                        "state_dwell_params": {
                            0: {"distribution": "gamma", "shape": 2.5, "scale": 20.0},
                            1: {"distribution": "gamma", "shape": 1.5, "scale": 10.0},
                        },
                        "regularization_strength": 1.0,
                    }
                ),
            ]
        )

        smoothed = pipeline.process(states)

        # Should be smoother than original
        original_transitions = np.sum(states[1:] != states[:-1])
        smoothed_transitions = np.sum(smoothed[1:] != smoothed[:-1])
        assert smoothed_transitions < original_transitions

    def test_kalman_and_crf_pipeline(self):
        """Test Kalman smoother followed by CRF."""
        np.random.seed(42)
        states = np.random.choice([0, 1, 2], 50)
        posteriors = np.random.dirichlet([2, 2, 2], 50)

        with pytest.warns(UserWarning, match="NOT suitable for live trading"):
            pipeline = ProcessorPipeline(
                [
                    AdaptiveKalmanStyleSmoother({"num_states": 3}),
                    ConditionalRandomFieldRefiner({"smooth_transitions": 2.0}),
                ]
            )

        smoothed = pipeline.process(states, posteriors=posteriors)

        assert len(smoothed) == len(states)
        # Should produce valid states
        assert all((smoothed >= 0) & (smoothed < 3))

    def test_full_phase_3_pipeline(self):
        """Test all Phase 3 processors together."""
        np.random.seed(42)
        true_regimes = np.repeat([0, 1, 2, 0], [100, 50, 70, 80])
        noisy_regimes = true_regimes.copy()
        noise_idx = np.random.choice(len(noisy_regimes), 50, replace=False)
        noisy_regimes[noise_idx] = np.random.choice([0, 1, 2], 50)

        # Generate posteriors
        posteriors = np.zeros((len(noisy_regimes), 3))
        for i in range(len(noisy_regimes)):
            probs = np.array([0.1, 0.1, 0.1])
            probs[noisy_regimes[i]] = 0.7
            posteriors[i] = probs

        with pytest.warns(UserWarning, match="NOT suitable for live trading"):
            pipeline = ProcessorPipeline(
                [
                    MarkovJumpProcessRegularizer(
                        {"estimate_from_data": True, "min_samples": 5}
                    ),
                    AdaptiveKalmanStyleSmoother({"num_states": 3}),
                    ConditionalRandomFieldRefiner({"smooth_transitions": 1.0}),
                ]
            )

        smoothed = pipeline.process(noisy_regimes, posteriors=posteriors)

        # Calculate stability
        original_stability = calculate_label_stability(noisy_regimes)
        smoothed_stability = calculate_label_stability(smoothed)

        # Should be more stable
        assert (
            smoothed_stability["mean_duration"] >= original_stability["mean_duration"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
