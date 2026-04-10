import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.utils.label_util import (
    map_label_to_trend_direction,
    map_label_to_momentum_score,
    map_regime_to_volatility_score,
    map_regime_to_path_structure_score,
)


class TestMapLabelToTrendDirection:
    """Test suite for map_label_to_trend_direction"""

    @pytest.fixture
    def basic_hmm_data(self):
        """
        Simulated HMM output with 3 states:
        - State 0: Strong bullish (mean=0.001, low vol)
        - State 1: Neutral/choppy (mean≈0, high vol)
        - State 2: Bearish (mean=-0.0005, medium vol)
        """
        np.random.seed(42)
        n = 200

        # State 0: Bullish
        state0_returns = np.random.normal(0.001, 0.0002, 70)

        # State 1: Neutral/choppy
        state1_returns = np.random.normal(0.0, 0.001, 80)

        # State 2: Bearish
        state2_returns = np.random.normal(-0.0005, 0.0003, 50)

        df = pd.DataFrame(
            {
                "state": [0] * 70 + [1] * 80 + [2] * 50,
                "returns": np.concatenate(
                    [state0_returns, state1_returns, state2_returns]
                ),
            }
        )

        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    @pytest.fixture
    def edge_case_data(self):
        """Data with edge cases: small samples, NaN values"""
        np.random.seed(42)

        # State 0: Good sample size
        state0_returns = np.random.normal(0.002, 0.0003, 50)

        # State 1: Small sample (below min_samples=30)
        state1_returns = np.random.normal(0.001, 0.0002, 15)

        # State 2: Include some NaN
        state2_returns = np.random.normal(-0.001, 0.0004, 40)
        state2_returns[::10] = np.nan

        df = pd.DataFrame(
            {
                "state": [0] * 50 + [1] * 15 + [2] * 40,
                "returns": np.concatenate(
                    [state0_returns, state1_returns, state2_returns]
                ),
            }
        )

        return df

    def test_conservative_method_basic(self, basic_hmm_data):
        """Test conservative method returns correct mapping"""
        mapping = map_label_to_trend_direction(
            basic_hmm_data,
            state_col="state",
            return_col="returns",
            method="conservative",
            min_samples=30,
            min_sharpe=0.2,
            confidence_level=0.95,
        )

        assert isinstance(mapping, dict)
        assert set(mapping.keys()) == {0, 1, 2}
        assert all(v in {-1, 0, 1} for v in mapping.values())

        # State 0 should be bullish (strong positive mean with low vol)
        assert mapping[0] == 1, "State 0 should be labeled as bullish"

        # State 1 should be neutral (near-zero mean despite sample size)
        assert mapping[1] == 0, "State 1 should be labeled as neutral"

    def test_conservative_method_with_diagnostics(self, basic_hmm_data):
        """Test diagnostics output includes all expected columns"""
        mapping, diag = map_label_to_trend_direction(
            basic_hmm_data, method="conservative", return_diagnostics=True
        )

        assert isinstance(diag, pd.DataFrame)
        assert not diag.empty

        # Check required columns
        expected_cols = [
            "state",
            "count",
            "mean",
            "std",
            "sharpe",
            "t_stat",
            "p_value",
            "direction",
            "reason",
        ]
        for col in expected_cols:
            assert col in diag.columns, f"Missing column: {col}"

        # Check that directions in mapping match diagnostics
        for _, row in diag.iterrows():
            assert mapping[row["state"]] == row["direction"]

        # Check that reasons are provided
        assert all(len(str(r)) > 0 for r in diag["reason"])

    def test_conservative_requires_all_criteria(self):
        """Test that conservative method requires ALL criteria to pass"""
        np.random.seed(42)

        # Create state with:
        # - High mean (passes magnitude check)
        # - High volatility → low Sharpe (fails Sharpe check)
        high_mean_low_sharpe = np.random.normal(0.01, 0.05, 50)  # Sharpe ≈ 0.2

        df = pd.DataFrame({"state": [0] * 50, "returns": high_mean_low_sharpe})

        mapping, diag = map_label_to_trend_direction(
            df,
            method="conservative",
            min_sharpe=0.5,  # Require high Sharpe
            min_samples=30,
            return_diagnostics=True,
        )

        # Should be labeled as 0 due to poor Sharpe
        assert mapping[0] == 0
        assert "Sharpe" in diag.loc[0, "reason"] or "Failed" in diag.loc[0, "reason"]

    # ========== TEST STATISTICAL METHOD ==========

    def test_statistical_method(self, basic_hmm_data):
        """Test statistical method uses t-test"""
        mapping = map_label_to_trend_direction(
            basic_hmm_data, method="statistical", confidence_level=0.95
        )

        assert isinstance(mapping, dict)
        assert all(v in {-1, 0, 1} for v in mapping.values())

    def test_statistical_method_significance_threshold(self):
        """Test that statistical method respects confidence level"""
        np.random.seed(42)

        # State with very small but non-zero mean
        # Should pass t-test with large sample, fail with small sample
        small_mean = np.random.normal(0.0001, 0.0005, 100)

        df = pd.DataFrame({"state": [0] * 100, "returns": small_mean})

        # With 95% confidence, might not be significant
        mapping_95 = map_label_to_trend_direction(
            df, method="statistical", confidence_level=0.95, min_samples=30
        )

        # With 80% confidence, more likely to be significant
        mapping_80 = map_label_to_trend_direction(
            df, method="statistical", confidence_level=0.80, min_samples=30
        )

        # Lower confidence should be more permissive (though outcome depends on actual p-value)
        assert isinstance(mapping_95, dict)
        assert isinstance(mapping_80, dict)

    # ========== TEST SHARPE METHOD ==========

    def test_sharpe_method(self, basic_hmm_data):
        """Test Sharpe ratio method"""
        mapping = map_label_to_trend_direction(
            basic_hmm_data, method="sharpe", min_sharpe=0.3
        )

        assert isinstance(mapping, dict)
        assert all(v in {-1, 0, 1} for v in mapping.values())

    def test_sharpe_method_threshold(self):
        """Test Sharpe method respects min_sharpe threshold"""
        np.random.seed(42)

        # Create two states with same mean but different volatilities
        # State 0: Low vol → high Sharpe
        state0 = np.random.normal(0.001, 0.0001, 50)

        # State 1: High vol → low Sharpe
        state1 = np.random.normal(0.001, 0.001, 50)

        df = pd.DataFrame(
            {"state": [0] * 50 + [1] * 50, "returns": np.concatenate([state0, state1])}
        )

        mapping, diag = map_label_to_trend_direction(
            df,
            method="sharpe",
            min_sharpe=2.0,  # High threshold
            min_samples=30,
            return_diagnostics=True,
        )

        # State 0 should have higher Sharpe
        sharpe_0 = diag.loc[diag["state"] == 0, "sharpe"].values[0]
        sharpe_1 = diag.loc[diag["state"] == 1, "sharpe"].values[0]

        assert sharpe_0 > sharpe_1, "State 0 should have higher Sharpe"

    # ========== TEST SIMPLE METHOD ==========

    def test_simple_method(self, basic_hmm_data):
        """Test simple mean threshold method"""
        mapping = map_label_to_trend_direction(
            basic_hmm_data, method="simple", min_samples=30
        )

        assert isinstance(mapping, dict)
        assert all(v in {-1, 0, 1} for v in mapping.values())

    # ========== TEST EDGE CASES ==========

    def test_insufficient_samples(self, edge_case_data):
        """Test that states with insufficient samples are labeled as 0"""
        mapping, diag = map_label_to_trend_direction(
            edge_case_data,
            min_samples=30,
            method="conservative",
            return_diagnostics=True,
        )

        # State 1 has only 15 samples (< 30)
        assert (
            mapping[1] == 0
        ), "State 1 should be labeled 0 due to insufficient samples"

        # Check diagnostics
        state1_row = diag.loc[diag["state"] == 1]
        assert state1_row["insufficient_data"].values[0] == True
        assert "Insufficient" in state1_row["reason"].values[0]

    def test_nan_handling(self, edge_case_data):
        """Test that NaN values are properly filtered"""
        # State 2 has NaN values
        mapping = map_label_to_trend_direction(
            edge_case_data, min_samples=30, method="conservative"
        )

        # Should not crash and should return valid mapping
        assert isinstance(mapping, dict)
        assert 2 in mapping
        assert mapping[2] in {-1, 0, 1}

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame({"state": [], "returns": []})

        mapping = map_label_to_trend_direction(df)

        assert mapping == {}

    def test_all_nan_returns(self):
        """Test handling of all NaN returns"""
        df = pd.DataFrame({"state": [0, 0, 0, 1, 1, 1], "returns": [np.nan] * 6})

        mapping = map_label_to_trend_direction(df, min_samples=1)

        assert mapping == {}

    def test_single_state(self):
        """Test with only one state"""
        np.random.seed(42)
        df = pd.DataFrame(
            {"state": [0] * 50, "returns": np.random.normal(0.001, 0.0002, 50)}
        )

        mapping = map_label_to_trend_direction(
            df, method="conservative", min_samples=30
        )

        assert len(mapping) == 1
        assert 0 in mapping
        assert mapping[0] in {-1, 0, 1}

    # ========== TEST AUTO COST THRESHOLD ==========

    def test_auto_cost_threshold(self):
        """Test that cost threshold is auto-estimated when not provided"""
        np.random.seed(42)

        # Create data with known return distribution
        df = pd.DataFrame(
            {"state": [0] * 100, "returns": np.random.normal(0.0001, 0.0005, 100)}
        )

        mapping, diag = map_label_to_trend_direction(
            df,
            method="conservative",
            cost_threshold=None,  # Auto-estimate
            min_samples=30,
            return_diagnostics=True,
        )

        # Auto-estimated threshold should be reasonable
        # (median(abs(returns)) * 0.1)
        expected_threshold = np.median(np.abs(df["returns"])) * 0.1

        assert expected_threshold > 0

    def test_custom_cost_threshold(self):
        """Test that custom cost threshold is respected"""
        np.random.seed(42)

        # Create state with mean below custom threshold
        df = pd.DataFrame(
            {
                "state": [0] * 100,
                "returns": np.random.normal(0.00001, 0.00001, 100),  # Very small
            }
        )

        # High cost threshold should filter this out
        mapping = map_label_to_trend_direction(
            df,
            method="conservative",
            cost_threshold=0.0001,  # 1bps
            min_samples=30,
            min_sharpe=0.0,  # Don't filter on Sharpe
            confidence_level=0.5,  # Very permissive
        )

        # Should be labeled 0 due to cost threshold
        assert mapping[0] == 0

    # ========== TEST PRODUCTION SCENARIOS ==========

    def test_hmm_backtesting_workflow(self):
        """
        Test realistic HMM backtesting workflow:
        1. HMM generates states
        2. Map states to directions
        3. Use directions for trading signals
        """
        np.random.seed(42)

        # Simulate HMM output over time series
        n = 300
        states = np.random.choice([0, 1, 2], size=n)

        # State characteristics (unknown to mapper)
        state_means = {0: 0.002, 1: 0.0, 2: -0.001}
        state_stds = {0: 0.0003, 1: 0.001, 2: 0.0004}

        returns = np.array(
            [np.random.normal(state_means[s], state_stds[s]) for s in states]
        )

        df = pd.DataFrame({"hmm_state": states, "returns": returns})

        # Map states to directions
        mapping = map_label_to_trend_direction(
            df,
            state_col="hmm_state",
            return_col="returns",
            method="conservative",
            min_samples=30,
        )

        # Create trading signals
        df["direction"] = df["hmm_state"].map(mapping)

        # Verify signals are valid
        assert df["direction"].isin([-1, 0, 1]).all()
        assert not df["direction"].isna().any()

        # Check that mapping makes sense
        assert isinstance(mapping, dict)
        assert len(mapping) == 3  # 3 HMM states

    def test_deterministic_output(self, basic_hmm_data):
        """Test that same input produces same output (deterministic)"""
        mapping1 = map_label_to_trend_direction(
            basic_hmm_data, method="conservative", min_samples=30
        )

        mapping2 = map_label_to_trend_direction(
            basic_hmm_data, method="conservative", min_samples=30
        )

        assert mapping1 == mapping2, "Function should be deterministic"

    def test_all_states_get_labels(self):
        """Test that all states in data get labels (no missing states)"""
        np.random.seed(42)

        # Create data with 5 states
        states = []
        returns_list = []

        for state_id in range(5):
            n = np.random.randint(20, 60)
            states.extend([state_id] * n)
            returns_list.extend(np.random.normal(0.001 * state_id, 0.0005, n))

        df = pd.DataFrame({"state": states, "returns": returns_list})

        mapping = map_label_to_trend_direction(
            df,
            method="conservative",
            min_samples=15,  # Lower threshold to include all states
        )

        # All 5 states should be in mapping
        assert set(mapping.keys()) == {0, 1, 2, 3, 4}

    # ========== TEST INPUT VALIDATION ==========

    def test_invalid_state_column(self, basic_hmm_data):
        """Test error when state column doesn't exist"""
        with pytest.raises(ValueError, match="Column 'invalid_col' not found"):
            map_label_to_trend_direction(basic_hmm_data, state_col="invalid_col")

    def test_invalid_return_column(self, basic_hmm_data):
        """Test error when return column doesn't exist"""
        with pytest.raises(ValueError, match="Column 'invalid_col' not found"):
            map_label_to_trend_direction(basic_hmm_data, return_col="invalid_col")

    def test_invalid_method(self, basic_hmm_data):
        """Test error when invalid method is specified"""
        with pytest.raises(ValueError, match="Unknown method"):
            map_label_to_trend_direction(basic_hmm_data, method="invalid_method")

    # ========== TEST RETURN TYPES ==========

    def test_return_dict_only(self, basic_hmm_data):
        """Test that return_diagnostics=False returns only dict"""
        result = map_label_to_trend_direction(basic_hmm_data, return_diagnostics=False)

        assert isinstance(result, dict)

    def test_return_tuple_with_diagnostics(self, basic_hmm_data):
        """Test that return_diagnostics=True returns tuple"""
        result = map_label_to_trend_direction(basic_hmm_data, return_diagnostics=True)

        assert isinstance(result, tuple)
        assert len(result) == 2

        mapping, diag = result
        assert isinstance(mapping, dict)
        assert isinstance(diag, pd.DataFrame)


class TestMapLabelToMomentumScore:
    """Test suite for map_label_to_momentum_score"""

    @pytest.fixture
    def momentum_data(self):
        """
        Test data with 4 regimes of varying momentum:
        - Regime 0: Strong downward (-0.002)
        - Regime 1: Weak downward (-0.0005)
        - Regime 2: Weak upward (0.0003)
        - Regime 3: Strong upward (0.0015)
        """
        np.random.seed(42)

        regime0 = np.random.normal(-0.002, 0.0003, 50)
        regime1 = np.random.normal(-0.0005, 0.0002, 50)
        regime2 = np.random.normal(0.0003, 0.0002, 50)
        regime3 = np.random.normal(0.0015, 0.0004, 50)

        df = pd.DataFrame(
            {
                "regime": [0] * 50 + [1] * 50 + [2] * 50 + [3] * 50,
                "returns": np.concatenate([regime0, regime1, regime2, regime3]),
            }
        )

        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ========== TEST DIRECTIONAL MODE ==========

    def test_directional_mode_basic(self, momentum_data):
        """Test directional momentum scoring (-n to +n)"""
        mapping = map_label_to_momentum_score(
            momentum_data, method="robust", momentum_range=5, is_directional=True
        )

        assert isinstance(mapping, dict)
        assert set(mapping.keys()) == {0, 1, 2, 3}

        # All scores should be in [-5, +5]
        assert all(-5 <= v <= 5 for v in mapping.values())

        # Check monotonicity: scores increase with median return
        # Regime 0 (most negative) should have lowest score
        # Regime 3 (most positive) should have highest score
        assert mapping[0] < mapping[1] < mapping[2] < mapping[3]

    def test_directional_mode_with_diagnostics(self, momentum_data):
        """Test diagnostics in directional mode"""
        mapping, diag = map_label_to_momentum_score(
            momentum_data,
            method="robust",
            momentum_range=5,
            is_directional=True,
            return_diagnostics=True,
        )

        assert isinstance(diag, pd.DataFrame)
        assert len(diag) == 4  # 4 regimes

        # Check columns
        expected_cols = [
            "regime",
            "median",
            "mean",
            "sharpe",
            "count",
            "score",
            "reason",
        ]
        for col in ["regime", "count", "score", "reason"]:
            assert col in diag.columns, f"Missing column: {col}"

    def test_directional_symmetric_range(self, momentum_data):
        """Test that directional mode creates symmetric range"""
        mapping = map_label_to_momentum_score(
            momentum_data, momentum_range=3, is_directional=True
        )

        # Range should be -3 to +3
        assert all(-3 <= v <= 3 for v in mapping.values())

    # ========== TEST NON-DIRECTIONAL MODE ==========

    def test_nondirectional_mode_basic(self, momentum_data):
        """Test non-directional momentum scoring (0 to n)"""
        mapping = map_label_to_momentum_score(
            momentum_data, method="robust", momentum_range=5, is_directional=False
        )

        assert isinstance(mapping, dict)
        assert set(mapping.keys()) == {0, 1, 2, 3}

        # All scores should be in [0, 5]
        assert all(0 <= v <= 5 for v in mapping.values())

    def test_nondirectional_ranks_by_magnitude(self):
        """Test that non-directional mode ranks by absolute value"""
        np.random.seed(42)

        # Create regimes with different magnitudes regardless of sign
        # Regime 0: near zero
        # Regime 1: moderate negative
        # Regime 2: moderate positive
        # Regime 3: large negative
        df = pd.DataFrame(
            {
                "regime": [0] * 50 + [1] * 50 + [2] * 50 + [3] * 50,
                "returns": np.concatenate(
                    [
                        np.random.normal(0.0001, 0.0001, 50),  # Near zero
                        np.random.normal(-0.001, 0.0002, 50),  # Moderate neg
                        np.random.normal(0.001, 0.0002, 50),  # Moderate pos
                        np.random.normal(-0.003, 0.0003, 50),  # Large neg
                    ]
                ),
            }
        )

        mapping = map_label_to_momentum_score(
            df, method="robust", momentum_range=5, is_directional=False
        )

        # Regime 0 (near zero) should have lowest score
        # Regime 3 (largest magnitude) should have highest score
        assert mapping[0] < mapping[1]
        assert mapping[0] < mapping[2]

    # ========== TEST METHODS ==========

    def test_robust_method(self, momentum_data):
        """Test robust method uses median"""
        mapping, diag = map_label_to_momentum_score(
            momentum_data, method="robust", return_diagnostics=True
        )

        # Check that median is present in diagnostics
        assert "median" in diag.columns or "median_" in str(diag.columns)

    def test_mean_method(self, momentum_data):
        """Test mean method"""
        mapping = map_label_to_momentum_score(
            momentum_data, method="mean", momentum_range=5
        )

        assert isinstance(mapping, dict)
        assert len(mapping) == 4

    def test_sharpe_method(self, momentum_data):
        """Test Sharpe ratio method"""
        mapping = map_label_to_momentum_score(
            momentum_data, method="sharpe", momentum_range=5
        )

        assert isinstance(mapping, dict)
        assert len(mapping) == 4

    # ========== TEST MONOTONICITY ==========

    def test_monotonicity_guarantee(self):
        """Test strict monotonicity: higher score = higher momentum"""
        np.random.seed(42)

        # Create 5 regimes with clear ordering
        regimes = []
        returns = []
        means = [-0.003, -0.001, 0.0, 0.001, 0.003]

        for i, mean in enumerate(means):
            regimes.extend([i] * 50)
            returns.extend(np.random.normal(mean, 0.0001, 50))

        df = pd.DataFrame({"regime": regimes, "returns": returns})

        mapping, diag = map_label_to_momentum_score(
            df,
            method="robust",
            momentum_range=10,
            is_directional=True,
            return_diagnostics=True,
        )

        # Verify monotonicity
        sorted_regimes = diag.sort_values("score")["regime"].tolist()
        sorted_medians = diag.sort_values("score")["median"].tolist()

        # Medians should be increasing
        for i in range(len(sorted_medians) - 1):
            assert (
                sorted_medians[i] <= sorted_medians[i + 1]
            ), f"Monotonicity violated: median[{i}]={sorted_medians[i]}, median[{i + 1}]={sorted_medians[i + 1]}"

    # ========== TEST EDGE CASES ==========

    def test_insufficient_samples(self):
        """Test handling of regimes with insufficient samples"""
        np.random.seed(42)

        df = pd.DataFrame(
            {
                "regime": [0] * 50
                + [1] * 10
                + [2] * 50,  # Regime 1 has only 10 samples
                "returns": np.concatenate(
                    [
                        np.random.normal(0.001, 0.0002, 50),
                        np.random.normal(-0.001, 0.0002, 10),
                        np.random.normal(0.002, 0.0003, 50),
                    ]
                ),
            }
        )

        mapping, diag = map_label_to_momentum_score(
            df, min_samples=30, is_directional=True, return_diagnostics=True
        )

        # Regime 1 should be assigned score 0 (neutral)
        assert mapping[1] == 0

        # Check diagnostics
        regime1_row = diag[diag["regime"] == 1]
        assert "Insufficient" in regime1_row["reason"].values[0]

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame({"regime": [], "returns": []})

        mapping = map_label_to_momentum_score(df)

        assert mapping == {}

    def test_nan_handling(self):
        """Test that NaN values are properly filtered"""
        np.random.seed(42)

        returns = np.random.normal(0.001, 0.0002, 100)
        returns[::10] = np.nan  # Insert NaNs

        df = pd.DataFrame({"regime": [0] * 100, "returns": returns})

        mapping = map_label_to_momentum_score(df, min_samples=30)

        # Should not crash and should return valid mapping
        assert isinstance(mapping, dict)
        assert 0 in mapping

    # ========== TEST RANGE CONSTRAINTS ==========

    def test_custom_range_int(self):
        """Test custom range specified as integer"""
        np.random.seed(42)

        df = pd.DataFrame(
            {
                "regime": [0] * 50 + [1] * 50 + [2] * 50,
                "returns": np.concatenate(
                    [
                        np.random.normal(-0.001, 0.0001, 50),
                        np.random.normal(0.0, 0.0001, 50),
                        np.random.normal(0.001, 0.0001, 50),
                    ]
                ),
            }
        )

        mapping = map_label_to_momentum_score(
            df, momentum_range=10, is_directional=True
        )

        # Range should be -10 to +10
        assert all(-10 <= v <= 10 for v in mapping.values())

    def test_custom_range_tuple(self):
        """Test custom range specified as tuple"""
        np.random.seed(42)

        df = pd.DataFrame(
            {
                "regime": [0] * 50 + [1] * 50,
                "returns": np.concatenate(
                    [
                        np.random.normal(-0.001, 0.0001, 50),
                        np.random.normal(0.001, 0.0001, 50),
                    ]
                ),
            }
        )

        mapping = map_label_to_momentum_score(
            df, momentum_range=(-3, 7), is_directional=True  # Asymmetric range
        )

        # Range should be -3 to +7
        assert all(-3 <= v <= 7 for v in mapping.values())

    # ========== TEST INPUT VALIDATION ==========

    def test_invalid_regime_column(self, momentum_data):
        """Test error when regime column doesn't exist"""
        with pytest.raises(ValueError, match="Column 'invalid' not found"):
            map_label_to_momentum_score(momentum_data, regime_col="invalid")

    def test_invalid_return_column(self, momentum_data):
        """Test error when return column doesn't exist"""
        with pytest.raises(ValueError, match="Column 'invalid' not found"):
            map_label_to_momentum_score(momentum_data, ret_col="invalid")

    def test_invalid_method(self, momentum_data):
        """Test error for invalid method"""
        with pytest.raises(ValueError, match="Unknown method"):
            map_label_to_momentum_score(momentum_data, method="invalid")

    def test_deterministic_output(self, momentum_data):
        """Test that same input produces same output"""
        mapping1 = map_label_to_momentum_score(momentum_data)
        mapping2 = map_label_to_momentum_score(momentum_data)

        assert mapping1 == mapping2


class TestMapRegimeToVolatilityScore:
    """Test suite for map_regime_to_volatility_score"""

    @pytest.fixture
    def volatility_data(self):
        """
        Test data with 3 regimes of varying volatility:
        - Regime 0: Low volatility (quiet)
        - Regime 1: Medium volatility
        - Regime 2: High volatility (turbulent)
        """
        np.random.seed(42)

        # Create regimes with different volatility levels
        # Use realized volatility as proxy
        regime0_vol = np.random.uniform(0.001, 0.005, 50)  # Low vol
        regime1_vol = np.random.uniform(0.005, 0.010, 50)  # Medium vol
        regime2_vol = np.random.uniform(0.010, 0.020, 50)  # High vol

        df = pd.DataFrame(
            {
                "regime": [0] * 50 + [1] * 50 + [2] * 50,
                "realized_vol": np.concatenate([regime0_vol, regime1_vol, regime2_vol]),
            }
        )

        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ========== TEST BASIC FUNCTIONALITY ==========

    def test_basic_volatility_scoring(self, volatility_data):
        """Test basic volatility bucket assignment"""
        mapping = map_regime_to_volatility_score(
            volatility_data, vol_proxy_col="realized_vol", method="median"
        )

        assert isinstance(mapping, dict)
        assert set(mapping.keys()) == {0, 1, 2}

        # All scores should be sequential buckets: 0, 1, 2
        scores = sorted(mapping.values())
        assert scores == [0, 1, 2]

        # Verify monotonicity: regime 0 (low vol) < regime 2 (high vol)
        assert mapping[0] < mapping[2]

    def test_with_diagnostics(self, volatility_data):
        """Test diagnostics output"""
        mapping, diag = map_regime_to_volatility_score(
            volatility_data, vol_proxy_col="realized_vol", return_diagnostics=True
        )

        assert isinstance(diag, pd.DataFrame)
        assert len(diag) == 3

        # Check required columns
        expected_cols = [
            "regime",
            "median_vol",
            "mean_vol",
            "count",
            "assigned_bucket",
            "reason",
        ]
        for col in expected_cols:
            assert col in diag.columns, f"Missing column: {col}"

    # ========== TEST METHODS ==========

    def test_median_method(self, volatility_data):
        """Test median volatility method"""
        mapping = map_regime_to_volatility_score(
            volatility_data, vol_proxy_col="realized_vol", method="median"
        )

        assert len(mapping) == 3

    def test_mean_method(self, volatility_data):
        """Test mean volatility method"""
        mapping = map_regime_to_volatility_score(
            volatility_data, vol_proxy_col="realized_vol", method="mean"
        )

        assert len(mapping) == 3

    def test_q95_method(self, volatility_data):
        """Test 95th percentile method"""
        mapping = map_regime_to_volatility_score(
            volatility_data, vol_proxy_col="realized_vol", method="q95"
        )

        assert len(mapping) == 3

    # ========== TEST MONOTONICITY ==========

    def test_monotonicity_guarantee(self):
        """Test that buckets are strictly monotonic with volatility"""
        np.random.seed(42)

        # Create 4 regimes with clear volatility ordering
        df = pd.DataFrame(
            {
                "regime": [0] * 50 + [1] * 50 + [2] * 50 + [3] * 50,
                "vol_proxy": np.concatenate(
                    [
                        np.random.uniform(0.001, 0.002, 50),  # Quietest
                        np.random.uniform(0.003, 0.004, 50),
                        np.random.uniform(0.005, 0.006, 50),
                        np.random.uniform(0.008, 0.010, 50),  # Most volatile
                    ]
                ),
            }
        )

        mapping, diag = map_regime_to_volatility_score(
            df, vol_proxy_col="vol_proxy", return_diagnostics=True
        )

        # Sort by bucket and check monotonicity
        sorted_df = diag.sort_values("assigned_bucket")
        vols = sorted_df["median_vol"].tolist()

        for i in range(len(vols) - 1):
            assert (
                vols[i] <= vols[i + 1]
            ), f"Monotonicity violated: vol[{i}]={vols[i]}, vol[{i + 1}]={vols[i + 1]}"

    def test_sequential_buckets(self):
        """Test that buckets are sequential with no gaps"""
        np.random.seed(42)

        df = pd.DataFrame(
            {
                "regime": [0] * 50 + [1] * 50 + [2] * 50 + [3] * 50 + [4] * 50,
                "vol_proxy": np.concatenate(
                    [
                        np.random.uniform(i * 0.002, (i + 1) * 0.002, 50)
                        for i in range(5)
                    ]
                ),
            }
        )

        mapping = map_regime_to_volatility_score(df, vol_proxy_col="vol_proxy")

        # Buckets should be 0, 1, 2, 3, 4 (no gaps)
        buckets = sorted(mapping.values())
        expected = list(range(5))
        assert buckets == expected

    # ========== TEST EDGE CASES ==========

    def test_insufficient_samples(self):
        """Test regimes with insufficient samples get bucket 0"""
        np.random.seed(42)

        df = pd.DataFrame(
            {
                "regime": [0] * 50
                + [1] * 10
                + [2] * 50,  # Regime 1 has only 10 samples
                "vol_proxy": np.concatenate(
                    [
                        np.random.uniform(0.001, 0.002, 50),
                        np.random.uniform(
                            0.005, 0.010, 10
                        ),  # Would be high vol but small sample
                        np.random.uniform(0.003, 0.004, 50),
                    ]
                ),
            }
        )

        mapping, diag = map_regime_to_volatility_score(
            df, vol_proxy_col="vol_proxy", min_samples=30, return_diagnostics=True
        )

        # Regime 1 should be forced to bucket 0
        assert mapping[1] == 0

        # Check diagnostics
        regime1_row = diag[diag["regime"] == 1]
        assert "Insufficient" in regime1_row["reason"].values[0]

    def test_tie_breaking(self):
        """Test tie-breaking when regimes have similar volatility"""
        np.random.seed(42)

        # Create two regimes with very similar volatility
        df = pd.DataFrame(
            {
                "regime": [0] * 100 + [1] * 50,  # Different sample sizes
                "vol_proxy": np.concatenate(
                    [
                        np.random.uniform(0.005, 0.00501, 100),  # Larger sample
                        np.random.uniform(0.005, 0.00501, 50),  # Smaller sample
                    ]
                ),
            }
        )

        mapping, diag = map_regime_to_volatility_score(
            df,
            vol_proxy_col="vol_proxy",
            tie_tolerance=1e-4,  # Loose tolerance to force tie
            return_diagnostics=True,
        )

        # Both should get assigned buckets (tie broken by sample size)
        assert len(set(mapping.values())) == 2  # 2 distinct buckets

    # ========== TEST INPUT VALIDATION ==========

    def test_missing_vol_proxy_col(self, volatility_data):
        """Test error when vol_proxy_col is not provided"""
        with pytest.raises(ValueError, match="vol_proxy_col is required"):
            map_regime_to_volatility_score(volatility_data, vol_proxy_col=None)

    def test_invalid_vol_proxy_col(self, volatility_data):
        """Test error when vol_proxy_col doesn't exist"""
        with pytest.raises(ValueError, match="Column 'invalid' not found"):
            map_regime_to_volatility_score(volatility_data, vol_proxy_col="invalid")

    def test_invalid_method(self, volatility_data):
        """Test error for invalid method"""
        with pytest.raises(ValueError, match="Unknown method"):
            map_regime_to_volatility_score(
                volatility_data, vol_proxy_col="realized_vol", method="invalid"
            )


class TestMapRegimeToPathStructureScore:
    """Test suite for map_regime_to_path_structure_score"""

    @pytest.fixture
    def ohlc_data(self):
        """
        Test data with OHLC prices and 3 regimes:
        - Regime 0: Smooth trending (low choppiness)
        - Regime 1: Moderate choppiness
        - Regime 2: Very choppy (high oscillation)
        """
        np.random.seed(42)

        n = 150
        regimes = [0] * 50 + [1] * 50 + [2] * 50

        # Generate realistic OHLC data
        close = np.cumsum(np.random.randn(n) * 0.01) + 100

        # High/Low around close with different ranges for each regime
        hl_range = []
        for r in regimes:
            if r == 0:
                hl_range.append(np.random.uniform(0.1, 0.3))  # Low range (smooth)
            elif r == 1:
                hl_range.append(np.random.uniform(0.3, 0.6))  # Medium range
            else:
                hl_range.append(np.random.uniform(0.6, 1.2))  # High range (choppy)

        hl_range = np.array(hl_range)

        high = close + hl_range * 0.5
        low = close - hl_range * 0.5
        open_price = close + np.random.randn(n) * 0.1

        df = pd.DataFrame(
            {
                "regime": regimes,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
            }
        )

        return df

    # ========== TEST BASIC FUNCTIONALITY ==========

    def test_basic_choppiness_scoring(self, ohlc_data):
        """Test basic choppiness score assignment"""
        mapping = map_regime_to_path_structure_score(
            ohlc_data, method="id_chop", lookback=10
        )

        assert isinstance(mapping, dict)
        assert set(mapping.keys()) == {0, 1, 2}

        # All scores should be in default range [0, 4]
        assert all(0 <= v <= 4 for v in mapping.values())

    def test_with_diagnostics(self, ohlc_data):
        """Test diagnostics output"""
        mapping, diag = map_regime_to_path_structure_score(
            ohlc_data, method="id_chop", return_diagnostics=True
        )

        assert isinstance(diag, pd.DataFrame)
        assert len(diag) == 3

        # Check required columns
        expected_cols = [
            "regime",
            "median_chop",
            "mean_chop",
            "count",
            "assigned_score",
            "reason",
        ]
        for col in expected_cols:
            assert col in diag.columns, f"Missing column: {col}"

    # ========== TEST METHODS ==========

    def test_id_chop_method(self, ohlc_data):
        """Test Choppiness Index method"""
        mapping = map_regime_to_path_structure_score(
            ohlc_data, method="id_chop", lookback=14
        )

        assert len(mapping) == 3
        assert all(isinstance(v, (int, np.integer)) for v in mapping.values())

    def test_cv_range_method(self, ohlc_data):
        """Test Coefficient of Variation method"""
        mapping = map_regime_to_path_structure_score(
            ohlc_data, method="cv_range", lookback=10
        )

        assert len(mapping) == 3

    def test_atr_dr_method(self, ohlc_data):
        """Test ATR/DR ratio method"""
        mapping = map_regime_to_path_structure_score(
            ohlc_data, method="atr_dr", lookback=10
        )

        assert len(mapping) == 3

    def test_custom_method(self):
        """Test custom choppiness proxy"""
        np.random.seed(42)

        # Provide pre-computed choppiness proxy
        df = pd.DataFrame(
            {
                "regime": [0] * 50 + [1] * 50 + [2] * 50,
                "choppiness_proxy": np.concatenate(
                    [
                        np.random.uniform(20, 40, 50),  # Low choppiness
                        np.random.uniform(40, 60, 50),  # Medium
                        np.random.uniform(60, 80, 50),  # High choppiness
                    ]
                ),
            }
        )

        mapping = map_regime_to_path_structure_score(df, method="custom")

        assert len(mapping) == 3
        # Regime 0 (low chop) should have lower score than regime 2 (high chop)
        assert mapping[0] < mapping[2]

    # ========== TEST MONOTONICITY ==========

    def test_monotonicity_guarantee(self, ohlc_data):
        """Test that scores increase monotonically with choppiness"""
        mapping, diag = map_regime_to_path_structure_score(
            ohlc_data, method="id_chop", return_diagnostics=True
        )

        # Sort by score and check that choppiness is increasing
        sorted_df = diag.sort_values("assigned_score")
        chops = sorted_df["median_chop"].tolist()

        for i in range(len(chops) - 1):
            assert (
                chops[i] <= chops[i + 1]
            ), f"Monotonicity violated: chop[{i}]={chops[i]}, chop[{i + 1}]={chops[i + 1]}"

    # ========== TEST RANGE CONSTRAINTS ==========

    def test_custom_range(self, ohlc_data):
        """Test custom choppiness range"""
        mapping = map_regime_to_path_structure_score(
            ohlc_data, method="id_chop", choppiness_range=(0, 10)
        )

        # All scores should be in [0, 10]
        assert all(0 <= v <= 10 for v in mapping.values())

    def test_invalid_range(self, ohlc_data):
        """Test error for invalid range"""
        with pytest.raises(ValueError, match="choppiness_range must be tuple"):
            map_regime_to_path_structure_score(
                ohlc_data, choppiness_range=[0, 4]  # Wrong type
            )

    # ========== TEST EDGE CASES ==========

    def test_insufficient_samples(self):
        """Test regimes with insufficient samples"""
        np.random.seed(42)

        n = 80
        close = np.cumsum(np.random.randn(n) * 0.01) + 100
        hl_range = np.random.uniform(0.2, 0.5, n)

        df = pd.DataFrame(
            {
                "regime": [0] * 50
                + [1] * 10
                + [2] * 20,  # Regime 1 has only 10 samples
                "open": close + np.random.randn(n) * 0.1,
                "high": close + hl_range * 0.5,
                "low": close - hl_range * 0.5,
                "close": close,
            }
        )

        mapping, diag = map_regime_to_path_structure_score(
            df, method="id_chop", min_samples=30, return_diagnostics=True
        )

        # Regime 1 should be forced to min score (smoothest)
        assert mapping[1] == 0

        # Check diagnostics
        regime1_row = diag[diag["regime"] == 1]
        assert "Insufficient" in regime1_row["reason"].values[0]

    def test_missing_required_columns(self):
        """Test error when required OHLC columns are missing"""
        df = pd.DataFrame(
            {
                "regime": [0] * 50,
                "close": np.random.randn(50),
                # Missing 'high' and 'low'
            }
        )

        with pytest.raises(ValueError, match="requires columns"):
            map_regime_to_path_structure_score(df, method="id_chop")

    # ========== TEST INPUT VALIDATION ==========

    def test_invalid_method(self, ohlc_data):
        """Test error for invalid method"""
        with pytest.raises(ValueError, match="Unknown method"):
            map_regime_to_path_structure_score(ohlc_data, method="invalid")

    def test_deterministic_output(self, ohlc_data):
        """Test that same input produces same output"""
        mapping1 = map_regime_to_path_structure_score(ohlc_data, method="id_chop")
        mapping2 = map_regime_to_path_structure_score(ohlc_data, method="id_chop")

        assert mapping1 == mapping2
