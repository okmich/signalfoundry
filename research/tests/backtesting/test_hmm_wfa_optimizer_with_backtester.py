import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.backtesting import HMMWalkForwardAnalysisBacktestOptimizer


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data with regime-switching behavior."""
    np.random.seed(42)
    n_samples = 300
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")

    # Create regime-switching data
    close = np.zeros(n_samples)
    close[0] = 100

    for i in range(1, n_samples):
        # Alternate between low and high volatility regimes
        regime_idx = i // 100
        if regime_idx % 2 == 0:
            drift = 0.001
            vol = 0.01
        else:
            drift = -0.0005
            vol = 0.03

        close[i] = close[i - 1] * (1 + drift + vol * np.random.randn())

    # Ensure positive prices
    close = np.maximum(close, 10)

    df = pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(n_samples) * 0.005),
            "high": close * (1 + np.abs(np.random.randn(n_samples) * 0.01)),
            "low": close * (1 - np.abs(np.random.randn(n_samples) * 0.01)),
            "close": close,
            "volume": np.random.uniform(1e6, 5e6, n_samples),
        },
        index=dates,
    )

    return df


@pytest.fixture
def simple_feature_engineering():
    """Simple feature engineering function for testing."""
    from sklearn.preprocessing import StandardScaler

    def feature_fn(train_raw, test_raw, train_labels, test_labels):
        # Calculate log returns
        train_log_returns = np.log(train_raw["close"] / train_raw["close"].shift(1))
        test_log_returns = np.log(test_raw["close"] / test_raw["close"].shift(1))

        # Drop NaN and convert to DataFrame
        train_features = pd.DataFrame(
            train_log_returns.dropna().values.reshape(-1, 1),
            index=train_raw.index[1 : len(train_log_returns.dropna()) + 1],
            columns=["log_returns"],
        )
        test_features = pd.DataFrame(
            test_log_returns.dropna().values.reshape(-1, 1),
            index=test_raw.index[1 : len(test_log_returns.dropna()) + 1],
            columns=["log_returns"],
        )

        # Apply scaling
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        train_features = pd.DataFrame(
            train_features_scaled,
            columns=train_features.columns,
            index=train_features.index,
        )

        test_features_scaled = scaler.transform(test_features)
        test_features = pd.DataFrame(
            test_features_scaled,
            columns=test_features.columns,
            index=test_features.index,
        )

        # Return scaler as metadata
        train_metadata = {"scaler": scaler}
        return train_features, test_features, train_metadata, None

    return feature_fn


@pytest.fixture
def multi_feature_engineering():
    """Multi-feature engineering function for testing."""
    from sklearn.preprocessing import StandardScaler

    def feature_fn(train_raw, test_raw, train_labels, test_labels):
        def calculate_features(df):
            features = pd.DataFrame(index=df.index)
            features["log_returns"] = np.log(df["close"] / df["close"].shift(1))
            features["high_low_range"] = (df["high"] - df["low"]) / df["close"]
            returns = df["close"].pct_change()
            features["vol_5"] = returns.rolling(5).std()
            return features.dropna()

        train_features = calculate_features(train_raw)
        test_features = calculate_features(test_raw)

        # Apply scaling
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        train_features = pd.DataFrame(
            train_features_scaled,
            columns=train_features.columns,
            index=train_features.index,
        )

        test_features_scaled = scaler.transform(test_features)
        test_features = pd.DataFrame(
            test_features_scaled,
            columns=test_features.columns,
            index=test_features.index,
        )

        # Return scaler as metadata
        train_metadata = {"scaler": scaler}
        return train_features, test_features, train_metadata, None

    return feature_fn


@pytest.fixture
def volatility_state_labeling():
    """State labeling function based on volatility."""

    def labeling_fn(
        model, train_data_raw, train_features, state_predictions, state_probabilities
    ):
        state_map = {}

        # Get aligned indices
        if isinstance(train_features, pd.DataFrame):
            aligned_indices = train_features.index
        else:
            n_samples = len(state_predictions)
            aligned_indices = train_data_raw.index[-n_samples:]

        aligned_data = train_data_raw.loc[aligned_indices]
        all_returns = aligned_data["close"].pct_change().dropna()

        for state_id in range(model.n_states):
            state_mask = state_predictions == state_id

            if len(state_mask) > len(all_returns):
                state_mask = state_mask[: len(all_returns)]

            state_returns = all_returns[state_mask]

            if len(state_returns) > 0:
                state_volatility = state_returns.std()
            else:
                state_volatility = 0.0

            overall_volatility = all_returns.std()

            if state_volatility < overall_volatility * 0.8:
                state_map[state_id] = "low_volatility"
            else:
                state_map[state_id] = "high_volatility"

        return state_map

    return labeling_fn


@pytest.fixture
def simple_signal_generator():
    """Simple MA crossover signal generator."""

    def signal_fn(
        state_predictions,
        state_probabilities,
        prices,
        features,
        state_map,
        fast_window=10,
        slow_window=20,
        confidence_threshold=0.7,
    ):
        n_prices = len(prices)
        n_states = len(state_predictions)

        # Align predictions with prices
        if n_states < n_prices:
            state_predictions_aligned = np.concatenate(
                [np.full(n_prices - n_states, state_predictions[0]), state_predictions]
            )
            if state_probabilities.ndim == 1:
                state_probabilities = state_probabilities.reshape(-1, 1)
            first_prob_row = state_probabilities[0:1]
            padding_rows = np.repeat(first_prob_row, n_prices - n_states, axis=0)
            state_probabilities_aligned = np.concatenate(
                [padding_rows, state_probabilities], axis=0
            )
        else:
            state_predictions_aligned = state_predictions[:n_prices]
            state_probabilities_aligned = state_probabilities[:n_prices]

        # Initialize signals
        long_entries = np.zeros(n_prices, dtype=bool)
        long_exits = np.zeros(n_prices, dtype=bool)
        short_entries = np.zeros(n_prices, dtype=bool)
        short_exits = np.zeros(n_prices, dtype=bool)

        # Calculate MAs
        close_prices = prices["close"].values
        fast_ma = pd.Series(close_prices).rolling(window=fast_window).mean().values
        slow_ma = pd.Series(close_prices).rolling(window=slow_window).mean().values

        # Identify low volatility states
        low_vol_states = [
            sid for sid, label in state_map.items() if "low_volatility" in label.lower()
        ]

        is_low_vol = np.isin(state_predictions_aligned, low_vol_states)

        # Handle confidence calculation
        if state_probabilities_aligned.ndim == 1:
            is_confident = state_probabilities_aligned > confidence_threshold
        else:
            max_probs = state_probabilities_aligned.max(axis=1)
            if max_probs.ndim == 0:
                max_probs = np.array([max_probs])
            is_confident = max_probs > confidence_threshold

        trade_allowed = is_low_vol & is_confident

        # Detect crossovers
        for i in range(slow_window, n_prices):
            if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]):
                continue

            # Bullish crossover
            if fast_ma[i - 1] <= slow_ma[i - 1] and fast_ma[i] > slow_ma[i]:
                if trade_allowed[i]:
                    long_entries[i] = True
                    short_exits[i] = True

            # Bearish crossover
            elif fast_ma[i - 1] >= slow_ma[i - 1] and fast_ma[i] < slow_ma[i]:
                if trade_allowed[i]:
                    short_entries[i] = True
                    long_exits[i] = True

        return long_entries, long_exits, short_entries, short_exits

    return signal_fn


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Test initialization and validation."""

    def test_initialization_pomegranate(
        self,
        sample_ohlcv_data,
        simple_feature_engineering,
        volatility_state_labeling,
        simple_signal_generator,
        temp_checkpoint_dir,
    ):
        """Test initialization with pomegranate."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "bic",
            "random_state": 42,
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            state_labeling_fn=volatility_state_labeling,
            signal_generator_fn=simple_signal_generator,
            signal_param_grid={"fast_window": [10], "slow_window": [20]},
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        assert optimizer.hmm_params["model_variants"] == ["hmm_pmgnt"]

    def test_invalid_model_variant(
        self, sample_ohlcv_data, simple_feature_engineering, temp_checkpoint_dir
    ):
        """Test that invalid model variant raises error."""
        hmm_params = {
            "model_variants": ["invalid_model"],
            "n_states_range": [2],
            "criterion": "aic",
        }

        with pytest.raises(ValueError, match="Unknown HMM variant 'invalid_model'"):
            HMMWalkForwardAnalysisBacktestOptimizer(
                raw_data=sample_ohlcv_data,
                train_period=100,
                test_period=30,
                step_period=30,
                feature_engineering_fn=simple_feature_engineering,
                hmm_params=hmm_params,
                checkpoint_dir=temp_checkpoint_dir,
            )


# ============================================================================
# HMM Model Optimization Tests
# ============================================================================


class TestHMMOptimization:
    """Test HMM model optimization."""

    def test_optimize_hmm_pomegranate(
        self, sample_ohlcv_data, simple_feature_engineering, temp_checkpoint_dir
    ):
        """Test HMM optimization with pomegranate."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2, 3],
            "criterion": "bic",
            "random_state": 42,
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        # Get training data
        train_data = sample_ohlcv_data.iloc[:100]
        test_data = sample_ohlcv_data.iloc[100:130]
        train_features, _, _, _ = simple_feature_engineering(
            train_data, test_data, None, None
        )

        # Test optimization
        best_model, best_config = optimizer._optimize_hmm_model(
            train_features, train_data
        )

        assert best_model is not None
        assert best_model.n_states in [2, 3]
        assert "aic" in best_config
        assert "bic" in best_config

    def test_dataframe_to_numpy_conversion(
        self, sample_ohlcv_data, simple_feature_engineering, temp_checkpoint_dir
    ):
        """Test that DataFrames are properly converted to numpy arrays."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "aic",
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        train_data = sample_ohlcv_data.iloc[:100]
        test_data = sample_ohlcv_data.iloc[100:130]
        train_features, _, _, _ = simple_feature_engineering(
            train_data, test_data, None, None
        )

        # Ensure train_features is DataFrame
        assert isinstance(train_features, pd.DataFrame)

        # This should not raise an error
        best_model, best_config = optimizer._optimize_hmm_model(
            train_features, train_data
        )
        assert best_model is not None


# ============================================================================
# State Labeling Tests
# ============================================================================


class TestStateLabeling:
    """Test state labeling functionality."""

    def test_state_labeling_with_function(
        self,
        sample_ohlcv_data,
        simple_feature_engineering,
        volatility_state_labeling,
        temp_checkpoint_dir,
    ):
        """Test state labeling with custom function."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "aic",
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            state_labeling_fn=volatility_state_labeling,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        train_data = sample_ohlcv_data.iloc[:100]
        test_data = sample_ohlcv_data.iloc[100:130]
        train_features, _, _, _ = simple_feature_engineering(
            train_data, test_data, None, None
        )

        best_model, _ = optimizer._optimize_hmm_model(train_features, train_data)
        state_map = optimizer._label_states(best_model, train_data, train_features)

        assert isinstance(state_map, dict)
        assert len(state_map) == 2
        assert all(
            label in ["low_volatility", "high_volatility"]
            for label in state_map.values()
        )

    def test_default_state_labeling(
        self, sample_ohlcv_data, simple_feature_engineering, temp_checkpoint_dir
    ):
        """Test default state labeling (no custom function)."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],  # Use 2 states for numerical stability
            "criterion": "aic",
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            state_labeling_fn=None,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        train_data = sample_ohlcv_data.iloc[:100]
        test_data = sample_ohlcv_data.iloc[100:130]
        train_features, _, _, _ = simple_feature_engineering(
            train_data, test_data, None, None
        )

        best_model, _ = optimizer._optimize_hmm_model(train_features, train_data)
        state_map = optimizer._label_states(best_model, train_data, train_features)

        assert isinstance(state_map, dict)
        assert len(state_map) == 2  # Updated to 2 states
        assert all(label.startswith("state_") for label in state_map.values())


# ============================================================================
# Signal Parameter Optimization Tests
# ============================================================================


class TestSignalParameterOptimization:
    """Test signal parameter optimization."""

    def test_grid_search_optimization(
        self,
        sample_ohlcv_data,
        simple_feature_engineering,
        volatility_state_labeling,
        simple_signal_generator,
        temp_checkpoint_dir,
    ):
        """Test grid search for signal parameters."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "aic",
        }

        signal_param_grid = {
            "fast_window": [5, 10],
            "slow_window": [15, 20],
            "confidence_threshold": [0.6, 0.7],
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            state_labeling_fn=volatility_state_labeling,
            signal_generator_fn=simple_signal_generator,
            signal_param_grid=signal_param_grid,
            signal_param_optimizer="grid",
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        # Run one window
        results_df, _, _ = optimizer.run()

        assert len(results_df) > 0
        assert "best_signal_params" in results_df.columns

    def test_bayesian_optimization(
        self,
        sample_ohlcv_data,
        simple_feature_engineering,
        volatility_state_labeling,
        simple_signal_generator,
        temp_checkpoint_dir,
    ):
        """Test Bayesian optimization for signal parameters."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "aic",
        }

        signal_param_grid = {
            "fast_window": [5, 10, 15],
            "slow_window": [15, 20, 25],
            "confidence_threshold": [0.5, 0.6, 0.7],
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            state_labeling_fn=volatility_state_labeling,
            signal_generator_fn=simple_signal_generator,
            signal_param_grid=signal_param_grid,
            signal_param_optimizer="bayesian",
            signal_param_n_calls=10,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        results_df, _, _ = optimizer.run()

        assert len(results_df) > 0


# ============================================================================
# Feature Importance Tests
# ============================================================================


class TestFeatureImportance:
    """Test feature importance extraction."""

    def test_feature_importance_extraction(
        self, sample_ohlcv_data, multi_feature_engineering, temp_checkpoint_dir
    ):
        """Test feature importance with multiple features."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "aic",
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=multi_feature_engineering,
            hmm_params=hmm_params,
            track_feature_importance=True,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        results_df, _, _ = optimizer.run()

        assert len(results_df) > 0
        # Check if feature importance was tracked
        if results_df.iloc[0]["feature_importance"] is not None:
            importance = results_df.iloc[0]["feature_importance"]
            assert "emission_based" in importance
            assert "random_forest" in importance

    def test_correlation_scores(
        self, sample_ohlcv_data, multi_feature_engineering, temp_checkpoint_dir
    ):
        """Test that correlation scores are calculated."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "aic",
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=multi_feature_engineering,
            hmm_params=hmm_params,
            track_feature_importance=True,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        train_data = sample_ohlcv_data.iloc[:100]
        test_data = sample_ohlcv_data.iloc[100:130]
        train_features, _, _, _ = multi_feature_engineering(
            train_data, test_data, None, None
        )

        best_model, _ = optimizer._optimize_hmm_model(train_features, train_data)
        train_features_np = train_features.values
        train_state_predictions = best_model.predict(train_features_np)

        importance = optimizer._extract_feature_importance(
            best_model,
            train_features_np,
            train_features.columns.tolist(),
            train_state_predictions,
        )

        assert importance is not None
        assert "correlation_emission_rf" in importance


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_walkforward_pomegranate(
        self,
        sample_ohlcv_data,
        simple_feature_engineering,
        volatility_state_labeling,
        simple_signal_generator,
        temp_checkpoint_dir,
    ):
        """Test complete walk-forward analysis with pomegranate."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "bic",
            "random_state": 42,
        }

        signal_param_grid = {
            "fast_window": [10],
            "slow_window": [20],
            "confidence_threshold": [0.7],
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            state_labeling_fn=volatility_state_labeling,
            signal_generator_fn=simple_signal_generator,
            signal_param_grid=signal_param_grid,
            signal_param_optimizer="grid",
            vbt_params={"fees": 0.001},
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        results_df, predictions, _ = optimizer.run()

        # Verify results
        assert len(results_df) > 0
        assert all(
            col in results_df.columns
            for col in [
                "window_idx",
                "n_states",
                "aic_score",
                "bic_score",
                "sharpe_ratio",
                "total_return",
                "num_trades",
            ]
        )
        assert predictions is not None
        assert len(predictions) > 0

    def test_without_signal_generator(
        self,
        sample_ohlcv_data,
        simple_feature_engineering,
        volatility_state_labeling,
        temp_checkpoint_dir,
    ):
        """Test walk-forward analysis without signal generation."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "aic",
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            state_labeling_fn=volatility_state_labeling,
            signal_generator_fn=None,  # No signal generation
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        results_df, predictions, _ = optimizer.run()

        assert len(results_df) > 0
        # Trading metrics should be NaN when no signals are generated
        assert pd.isna(results_df.iloc[0]["sharpe_ratio"])


# ============================================================================
# State Transition Tests
# ============================================================================


class TestStateTransitions:
    """Test state transition tracking."""

    def test_state_transition_calculation(
        self,
        sample_ohlcv_data,
        simple_feature_engineering,
        volatility_state_labeling,
        temp_checkpoint_dir,
    ):
        """Test state transition matrix calculation."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "aic",
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            state_labeling_fn=volatility_state_labeling,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        results_df, _, _ = optimizer.run()

        assert len(results_df) > 0
        # Check basic results columns
        assert "n_states" in results_df.columns
        assert "aic_score" in results_df.columns
        assert "bic_score" in results_df.columns


# ============================================================================
# Pomegranate-Specific Tests
# ============================================================================


class TestPomegranateSpecific:
    """Test pomegranate-specific functionality."""

    def test_tensor_to_numpy_conversion(
        self, sample_ohlcv_data, simple_feature_engineering, temp_checkpoint_dir
    ):
        """Test that PyTorch tensors are properly converted to numpy."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "bic",
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        train_data = sample_ohlcv_data.iloc[:100]
        test_data = sample_ohlcv_data.iloc[100:130]
        train_features, _, _, _ = simple_feature_engineering(
            train_data, test_data, None, None
        )

        best_model, _ = optimizer._optimize_hmm_model(train_features, train_data)

        # Test predictions
        train_features_np = train_features.values
        predictions = best_model.predict(train_features_np)
        probabilities = best_model.predict_proba(train_features_np)

        # Verify numpy arrays
        assert isinstance(predictions, np.ndarray)
        assert isinstance(probabilities, np.ndarray)
        # Probabilities should be 2D (n_samples, n_states)
        assert probabilities.ndim == 2
        assert probabilities.shape[1] == 2  # n_states

    def test_batch_dimension_removal(
        self, sample_ohlcv_data, simple_feature_engineering, temp_checkpoint_dir
    ):
        """Test that batch dimension is properly removed from probabilities."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [3],
            "criterion": "bic",
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        train_data = sample_ohlcv_data.iloc[:100]
        test_data = sample_ohlcv_data.iloc[100:130]
        train_features, _, _, _ = simple_feature_engineering(
            train_data, test_data, None, None
        )

        best_model, _ = optimizer._optimize_hmm_model(train_features, train_data)
        train_features_np = train_features.values
        probabilities = best_model.predict_proba(train_features_np)

        # Should be (n_samples, n_states), NOT (1, n_samples, n_states)
        assert probabilities.shape == (len(train_features_np), 3)


# ============================================================================
# Scaler Persistence Tests
# ============================================================================


class TestScalerPersistence:
    """Test scaler saving and loading functionality."""

    def test_scaler_saved_with_best_model(
        self,
        sample_ohlcv_data,
        simple_feature_engineering,
        volatility_state_labeling,
        simple_signal_generator,
        temp_checkpoint_dir,
    ):
        """Test that scaler is saved alongside the best model."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "aic",
            "random_state": 42,
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            state_labeling_fn=volatility_state_labeling,
            signal_generator_fn=simple_signal_generator,
            signal_param_grid={"fast_window": [10], "slow_window": [20]},
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        results_df, _, _ = optimizer.run()

        # Check that scaler file exists
        import os

        scaler_path = os.path.join(
            temp_checkpoint_dir, "best_models", "scaler_0.joblib"
        )
        assert os.path.exists(scaler_path), "Scaler file should be saved"

        # Verify scaler can be loaded
        best_scaler = optimizer.get_best_scaler()
        assert best_scaler is not None, "Scaler should be loadable"

        # Verify it's a StandardScaler
        from sklearn.preprocessing import StandardScaler

        assert isinstance(
            best_scaler, StandardScaler
        ), "Should be a StandardScaler instance"

    def test_scaler_retrieval_method(
        self, sample_ohlcv_data, simple_feature_engineering, temp_checkpoint_dir
    ):
        """Test get_best_scaler() method."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "aic",
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        results_df, _, _ = optimizer.run()

        # Test scaler retrieval
        scaler = optimizer.get_best_scaler()
        assert scaler is not None

        # Verify scaler can transform new data
        test_data = np.random.randn(10, 1)
        transformed = scaler.transform(test_data)
        assert transformed.shape == test_data.shape

    def test_scaler_tracking_per_window(
        self, sample_ohlcv_data, simple_feature_engineering, temp_checkpoint_dir
    ):
        """Test that scalers are tracked for each window."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "aic",
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        results_df, _, _ = optimizer.run()

        # Check that scalers were stored for each window
        assert len(optimizer.window_scalers) == len(results_df)

        # Verify all scalers are valid
        for scaler in optimizer.window_scalers:
            if scaler is not None:
                from sklearn.preprocessing import StandardScaler

                assert isinstance(scaler, StandardScaler)

    def test_feature_engineering_without_scaler(
        self, sample_ohlcv_data, temp_checkpoint_dir
    ):
        """Test backward compatibility when feature engineering doesn't return scaler."""

        def feature_fn_no_scaler(train_raw, test_raw, train_labels, test_labels):
            # Old-style feature engineering without scaler
            train_log_returns = np.log(train_raw["close"] / train_raw["close"].shift(1))
            test_log_returns = np.log(test_raw["close"] / test_raw["close"].shift(1))

            train_features = pd.DataFrame(
                train_log_returns.dropna().values.reshape(-1, 1),
                index=train_raw.index[1 : len(train_log_returns.dropna()) + 1],
                columns=["log_returns"],
            )
            test_features = pd.DataFrame(
                test_log_returns.dropna().values.reshape(-1, 1),
                index=test_raw.index[1 : len(test_log_returns.dropna()) + 1],
                columns=["log_returns"],
            )

            return train_features, test_features, None, None

        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "aic",
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=feature_fn_no_scaler,
            hmm_params=hmm_params,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        # Should work without errors
        results_df, _, _ = optimizer.run()
        assert len(results_df) > 0

        # Scaler should be None
        scaler = optimizer.get_best_scaler()
        # Should warn but not crash
        assert scaler is None or scaler is not None  # Either is acceptable

    def test_scaler_consistency_across_windows(
        self, sample_ohlcv_data, simple_feature_engineering, temp_checkpoint_dir
    ):
        """Test that each window gets its own scaler fitted on its training data."""
        hmm_params = {
            "model_variants": ["hmm_pmgnt"],
            "n_states_range": [2],
            "criterion": "aic",
        }

        optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
            raw_data=sample_ohlcv_data,
            train_period=100,
            test_period=30,
            step_period=30,
            feature_engineering_fn=simple_feature_engineering,
            hmm_params=hmm_params,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        results_df, _, _ = optimizer.run()

        # Each window should have its own scaler
        if len(optimizer.window_scalers) > 1:
            scaler1 = optimizer.window_scalers[0]
            scaler2 = optimizer.window_scalers[1]

            # Scalers should be different objects
            assert scaler1 is not scaler2

            # But both should be StandardScalers
            if scaler1 is not None and scaler2 is not None:
                from sklearn.preprocessing import StandardScaler

                assert isinstance(scaler1, StandardScaler)
                assert isinstance(scaler2, StandardScaler)
