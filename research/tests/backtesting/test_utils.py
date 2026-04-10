import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from okmich_quant_research.backtesting import (
    generate_seq_feature_func_for_training,
    generate_seq_features_for_inference,
)


@pytest.fixture
def sample_raw_data():
    """Create sample raw price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=200, freq="1h")
    data = pd.DataFrame(
        {
            "open": 100 + np.cumsum(np.random.randn(200) * 0.5),
            "high": 101 + np.cumsum(np.random.randn(200) * 0.5),
            "low": 99 + np.cumsum(np.random.randn(200) * 0.5),
            "close": 100 + np.cumsum(np.random.randn(200) * 0.5),
            "volume": np.random.randint(1000, 10000, 200),
        },
        index=dates,
    )
    return data


@pytest.fixture
def sample_labels():
    """Create sample binary labels (buy/sell signals)."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=200, freq="1h")
    labels = pd.Series(np.random.choice([-1, 1], size=200), index=dates)
    return labels


@pytest.fixture
def mock_feature_engineering_func():
    """Mock feature engineering function that returns features with some NaN rows."""

    def feature_func(raw_data, feature_config):
        # Simulate feature engineering that drops first few rows due to indicators
        features = pd.DataFrame(
            {
                "feature_1": raw_data["close"].pct_change(),
                "feature_2": raw_data["close"].rolling(window=5).mean(),
                "feature_3": raw_data["volume"].pct_change(),
            },
            index=raw_data.index,
        )
        # Drop NaN rows (simulating indicator calculation)
        return features.dropna()

    return feature_func


@pytest.fixture
def feature_config():
    """Sample feature configuration."""
    return {"lookback_periods": [5, 10, 20], "use_volume": True}


@pytest.fixture
def selected_features():
    """List of selected feature names."""
    return ["feature_1", "feature_2", "feature_3"]


@pytest.fixture
def scaler():
    """Create a StandardScaler instance."""
    return StandardScaler()


# =============================================================================
# Tests for generate_seq_feature_func_for_training
# =============================================================================


class TestGenerateSeqFeatureFuncForTraining:
    """Test suite for generate_seq_feature_func_for_training function."""

    def test_basic_functionality(
        self,
        sample_raw_data,
        sample_labels,
        mock_feature_engineering_func,
        feature_config,
        selected_features,
        scaler,
    ):
        """Test basic functionality returns properly shaped 3D sequences."""
        sequence_length = 10

        # Split data into train and test
        split_idx = 150
        train_raw = sample_raw_data.iloc[:split_idx]
        test_raw = sample_raw_data.iloc[split_idx:]
        train_labels = sample_labels.iloc[:split_idx]
        test_labels = sample_labels.iloc[split_idx:]

        # Generate the feature engineering function
        feature_fn = generate_seq_feature_func_for_training(
            selected_features,
            mock_feature_engineering_func,
            feature_config,
            sequence_length,
            scaler,
        )

        # Apply the function
        X_train_3d, X_test_3d, y_train_seq, y_test_seq = feature_fn(
            train_raw, test_raw, train_labels, test_labels
        )

        # Verify shapes
        assert len(X_train_3d.shape) == 3, "Training features should be 3D"
        assert len(X_test_3d.shape) == 3, "Test features should be 3D"
        assert (
            X_train_3d.shape[1] == sequence_length
        ), f"Sequence length should be {sequence_length}"
        assert X_train_3d.shape[2] == len(
            selected_features
        ), "Number of features should match selected features"
        assert (
            X_test_3d.shape[1] == sequence_length
        ), "Test sequence length should match"
        assert X_test_3d.shape[2] == len(
            selected_features
        ), "Test features should match"

        # Verify labels are aligned
        assert len(y_train_seq.shape) == 1, "Training labels should be 1D"
        assert len(y_test_seq.shape) == 1, "Test labels should be 1D"
        assert (
            X_train_3d.shape[0] == y_train_seq.shape[0]
        ), "Training samples and labels should align"
        assert (
            X_test_3d.shape[0] == y_test_seq.shape[0]
        ), "Test samples and labels should align"

    def test_label_conversion_to_binary(
        self,
        sample_raw_data,
        sample_labels,
        mock_feature_engineering_func,
        feature_config,
        selected_features,
        scaler,
    ):
        """Test that labels are converted from -1/1 to 0/1."""
        sequence_length = 5

        split_idx = 150
        train_raw = sample_raw_data.iloc[:split_idx]
        test_raw = sample_raw_data.iloc[split_idx:]
        train_labels = sample_labels.iloc[:split_idx]
        test_labels = sample_labels.iloc[split_idx:]

        feature_fn = generate_seq_feature_func_for_training(
            selected_features,
            mock_feature_engineering_func,
            feature_config,
            sequence_length,
            scaler,
        )

        _, _, y_train_seq, y_test_seq = feature_fn(
            train_raw, test_raw, train_labels, test_labels
        )

        # Check that labels are only 0 or 1
        assert set(np.unique(y_train_seq)).issubset(
            {0, 1}
        ), "Training labels should only be 0 or 1"
        assert set(np.unique(y_test_seq)).issubset(
            {0, 1}
        ), "Test labels should only be 0 or 1"
        assert -1 not in y_train_seq, "Training labels should not contain -1"
        assert -1 not in y_test_seq, "Test labels should not contain -1"

    def test_scaler_fitted_on_training_data(
        self,
        sample_raw_data,
        sample_labels,
        mock_feature_engineering_func,
        feature_config,
        selected_features,
    ):
        """Test that scaler is fitted on training data only."""
        scaler = StandardScaler()
        sequence_length = 10

        split_idx = 150
        train_raw = sample_raw_data.iloc[:split_idx]
        test_raw = sample_raw_data.iloc[split_idx:]
        train_labels = sample_labels.iloc[:split_idx]
        test_labels = sample_labels.iloc[split_idx:]

        feature_fn = generate_seq_feature_func_for_training(
            selected_features,
            mock_feature_engineering_func,
            feature_config,
            sequence_length,
            scaler,
        )

        # Before calling the function, scaler should not be fitted
        assert not hasattr(scaler, "mean_"), "Scaler should not be fitted before use"

        # Apply the function
        feature_fn(train_raw, test_raw, train_labels, test_labels)

        # After calling, scaler should be fitted
        assert hasattr(scaler, "mean_"), "Scaler should be fitted after use"
        assert scaler.mean_.shape[0] == len(
            selected_features
        ), "Scaler should fit all features"

    def test_feature_selection_applied(
        self, sample_raw_data, sample_labels, feature_config, scaler
    ):
        """Test that only selected features are used."""
        selected_features = ["feature_1", "feature_2"]  # Only 2 features
        sequence_length = 5

        def feature_func(raw_data, config):
            return pd.DataFrame(
                {
                    "feature_1": raw_data["close"].pct_change(),
                    "feature_2": raw_data["close"].rolling(window=5).mean(),
                    "feature_3": raw_data[
                        "volume"
                    ].pct_change(),  # This should be excluded
                },
                index=raw_data.index,
            ).dropna()

        split_idx = 150
        train_raw = sample_raw_data.iloc[:split_idx]
        test_raw = sample_raw_data.iloc[split_idx:]
        train_labels = sample_labels.iloc[:split_idx]
        test_labels = sample_labels.iloc[split_idx:]

        feature_fn = generate_seq_feature_func_for_training(
            selected_features, feature_func, feature_config, sequence_length, scaler
        )

        X_train_3d, X_test_3d, _, _ = feature_fn(
            train_raw, test_raw, train_labels, test_labels
        )

        # Should only have 2 features, not 3
        assert X_train_3d.shape[2] == 2, "Should only have selected features"
        assert X_test_3d.shape[2] == 2, "Should only have selected features"

    def test_handles_nan_from_feature_engineering(
        self, sample_raw_data, sample_labels, feature_config, selected_features, scaler
    ):
        """Test that NaN rows from feature engineering are handled correctly."""
        sequence_length = 5

        def feature_func_with_nans(raw_data, config):
            # Create features with NaN in first 10 rows
            features = pd.DataFrame(
                {
                    "feature_1": raw_data["close"].pct_change(),
                    "feature_2": raw_data["close"].rolling(window=10).mean(),
                    "feature_3": raw_data["volume"].pct_change(),
                },
                index=raw_data.index,
            )
            return features.dropna()

        split_idx = 150
        train_raw = sample_raw_data.iloc[:split_idx]
        test_raw = sample_raw_data.iloc[split_idx:]
        train_labels = sample_labels.iloc[:split_idx]
        test_labels = sample_labels.iloc[split_idx:]

        feature_fn = generate_seq_feature_func_for_training(
            selected_features,
            feature_func_with_nans,
            feature_config,
            sequence_length,
            scaler,
        )

        X_train_3d, _, y_train_seq, _ = feature_fn(
            train_raw, test_raw, train_labels, test_labels
        )

        # Should not have any NaN values in final output
        assert not np.isnan(X_train_3d).any(), "Output should not contain NaN values"
        assert not np.isnan(y_train_seq).any(), "Labels should not contain NaN values"

    def test_label_alignment_with_features(
        self,
        sample_raw_data,
        sample_labels,
        mock_feature_engineering_func,
        feature_config,
        selected_features,
        scaler,
    ):
        """Test that labels are properly aligned with features after NaN removal."""
        sequence_length = 5

        split_idx = 150
        train_raw = sample_raw_data.iloc[:split_idx]
        test_raw = sample_raw_data.iloc[split_idx:]
        train_labels = sample_labels.iloc[:split_idx]
        test_labels = sample_labels.iloc[split_idx:]

        feature_fn = generate_seq_feature_func_for_training(
            selected_features,
            mock_feature_engineering_func,
            feature_config,
            sequence_length,
            scaler,
        )

        X_train_3d, X_test_3d, y_train_seq, y_test_seq = feature_fn(
            train_raw, test_raw, train_labels, test_labels
        )

        # Number of sequences should be (samples - sequence_length)
        # This verifies that labels are aligned properly
        assert X_train_3d.shape[0] == y_train_seq.shape[0]
        assert X_test_3d.shape[0] == y_test_seq.shape[0]

    def test_different_sequence_lengths(
        self,
        sample_raw_data,
        sample_labels,
        mock_feature_engineering_func,
        feature_config,
        selected_features,
        scaler,
    ):
        """Test with different sequence lengths."""
        for seq_len in [3, 5, 10, 20]:
            scaler_copy = StandardScaler()

            split_idx = 150
            train_raw = sample_raw_data.iloc[:split_idx]
            test_raw = sample_raw_data.iloc[split_idx:]
            train_labels = sample_labels.iloc[:split_idx]
            test_labels = sample_labels.iloc[split_idx:]

            feature_fn = generate_seq_feature_func_for_training(
                selected_features,
                mock_feature_engineering_func,
                feature_config,
                seq_len,
                scaler_copy,
            )

            X_train_3d, X_test_3d, _, _ = feature_fn(
                train_raw, test_raw, train_labels, test_labels
            )

            assert (
                X_train_3d.shape[1] == seq_len
            ), f"Sequence length should be {seq_len}"
            assert (
                X_test_3d.shape[1] == seq_len
            ), f"Test sequence length should be {seq_len}"


# =============================================================================
# Tests for generate_seq_features_for_inference
# =============================================================================


class TestGenerateSeqFeaturesForInference:
    """Test suite for generate_seq_features_for_inference function."""

    def test_basic_inference(
        self,
        sample_raw_data,
        mock_feature_engineering_func,
        feature_config,
        selected_features,
    ):
        """Test basic inference functionality."""
        sequence_length = 10
        scaler = StandardScaler()

        # First, fit the scaler on some training data
        train_features_2d = mock_feature_engineering_func(
            sample_raw_data.iloc[:150], feature_config
        )
        train_features_2d = train_features_2d[selected_features]
        scaler.fit(train_features_2d.values)

        # Now use inference function
        X_inference_3d = generate_seq_features_for_inference(
            sample_raw_data.iloc[150:],
            selected_features,
            mock_feature_engineering_func,
            feature_config,
            sequence_length,
            scaler,
        )

        # Verify shape
        assert len(X_inference_3d.shape) == 3, "Output should be 3D"
        assert (
            X_inference_3d.shape[1] == sequence_length
        ), f"Sequence length should be {sequence_length}"
        assert X_inference_3d.shape[2] == len(
            selected_features
        ), "Should have correct number of features"

    def test_uses_fitted_scaler(
        self,
        sample_raw_data,
        mock_feature_engineering_func,
        feature_config,
        selected_features,
    ):
        """Test that the function uses a pre-fitted scaler without refitting."""
        sequence_length = 5
        scaler = StandardScaler()

        # Fit scaler on training data
        train_features_2d = mock_feature_engineering_func(
            sample_raw_data.iloc[:150], feature_config
        )
        train_features_2d = train_features_2d[selected_features]
        scaler.fit(train_features_2d.values)

        # Store the scaler mean to verify it doesn't change
        original_mean = scaler.mean_.copy()

        # Use inference function
        generate_seq_features_for_inference(
            sample_raw_data.iloc[150:],
            selected_features,
            mock_feature_engineering_func,
            feature_config,
            sequence_length,
            scaler,
        )

        # Scaler should not have been refitted (mean should be the same)
        np.testing.assert_array_equal(
            scaler.mean_,
            original_mean,
            err_msg="Scaler should not be refitted during inference",
        )

    def test_returns_correct_number_of_sequences(
        self,
        sample_raw_data,
        mock_feature_engineering_func,
        feature_config,
        selected_features,
    ):
        """Test that correct number of sequences are returned."""
        sequence_length = 10
        scaler = StandardScaler()

        # Fit scaler
        train_features_2d = mock_feature_engineering_func(
            sample_raw_data.iloc[:150], feature_config
        )
        train_features_2d = train_features_2d[selected_features]
        scaler.fit(train_features_2d.values)

        # Get inference data
        inference_raw = sample_raw_data.iloc[150:]
        inference_features_2d = mock_feature_engineering_func(
            inference_raw, feature_config
        )

        X_inference_3d = generate_seq_features_for_inference(
            inference_raw,
            selected_features,
            mock_feature_engineering_func,
            feature_config,
            sequence_length,
            scaler,
        )

        # Number of sequences should be (number of 2D samples - sequence_length)
        expected_sequences = len(inference_features_2d) - sequence_length
        assert (
            X_inference_3d.shape[0] == expected_sequences
        ), f"Should have {expected_sequences} sequences"

    def test_no_nan_values_in_output(
        self,
        sample_raw_data,
        mock_feature_engineering_func,
        feature_config,
        selected_features,
    ):
        """Test that output contains no NaN values."""
        sequence_length = 5
        scaler = StandardScaler()

        # Fit scaler
        train_features_2d = mock_feature_engineering_func(
            sample_raw_data.iloc[:150], feature_config
        )
        train_features_2d = train_features_2d[selected_features]
        scaler.fit(train_features_2d.values)

        X_inference_3d = generate_seq_features_for_inference(
            sample_raw_data.iloc[150:],
            selected_features,
            mock_feature_engineering_func,
            feature_config,
            sequence_length,
            scaler,
        )

        assert not np.isnan(
            X_inference_3d
        ).any(), "Output should not contain NaN values"

    def test_different_sequence_lengths_inference(
        self,
        sample_raw_data,
        mock_feature_engineering_func,
        feature_config,
        selected_features,
    ):
        """Test inference with different sequence lengths."""
        scaler = StandardScaler()

        # Fit scaler once
        train_features_2d = mock_feature_engineering_func(
            sample_raw_data.iloc[:150], feature_config
        )
        train_features_2d = train_features_2d[selected_features]
        scaler.fit(train_features_2d.values)

        for seq_len in [3, 5, 10, 15]:
            X_inference_3d = generate_seq_features_for_inference(
                sample_raw_data.iloc[150:],
                selected_features,
                mock_feature_engineering_func,
                feature_config,
                seq_len,
                scaler,
            )

            assert (
                X_inference_3d.shape[1] == seq_len
            ), f"Sequence length should be {seq_len}"

    def test_feature_selection_inference(self, sample_raw_data, feature_config):
        """Test that only selected features are used in inference."""
        selected_features = ["feature_1", "feature_2"]  # Only 2 features
        sequence_length = 5
        scaler = StandardScaler()

        def feature_func(raw_data, config):
            return pd.DataFrame(
                {
                    "feature_1": raw_data["close"].pct_change(),
                    "feature_2": raw_data["close"].rolling(window=5).mean(),
                    "feature_3": raw_data["volume"].pct_change(),  # Should be excluded
                },
                index=raw_data.index,
            ).dropna()

        # Fit scaler
        train_features_2d = feature_func(sample_raw_data.iloc[:150], feature_config)
        train_features_2d = train_features_2d[selected_features]
        scaler.fit(train_features_2d.values)

        X_inference_3d = generate_seq_features_for_inference(
            sample_raw_data.iloc[150:],
            selected_features,
            feature_func,
            feature_config,
            sequence_length,
            scaler,
        )

        # Should only have 2 features
        assert X_inference_3d.shape[2] == 2, "Should only have selected features"

    def test_scaled_values_in_reasonable_range(
        self,
        sample_raw_data,
        mock_feature_engineering_func,
        feature_config,
        selected_features,
    ):
        """Test that scaled values are in a reasonable range (z-scores)."""
        sequence_length = 5
        scaler = StandardScaler()

        # Fit scaler
        train_features_2d = mock_feature_engineering_func(
            sample_raw_data.iloc[:150], feature_config
        )
        train_features_2d = train_features_2d[selected_features]
        scaler.fit(train_features_2d.values)

        X_inference_3d = generate_seq_features_for_inference(
            sample_raw_data.iloc[150:],
            selected_features,
            mock_feature_engineering_func,
            feature_config,
            sequence_length,
            scaler,
        )

        # For standardized data, most values should be within [-3, 3]
        assert (
            np.abs(X_inference_3d).max() < 10
        ), "Scaled values should be in reasonable range (most within ±3 std devs)"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining training and inference."""

    def test_training_then_inference_pipeline(
        self,
        sample_raw_data,
        sample_labels,
        mock_feature_engineering_func,
        feature_config,
        selected_features,
    ):
        """Test the full pipeline: training then inference."""
        sequence_length = 10
        scaler = StandardScaler()

        # Step 1: Training
        split_idx = 150
        train_raw = sample_raw_data.iloc[:split_idx]
        test_raw = sample_raw_data.iloc[split_idx:180]
        train_labels = sample_labels.iloc[:split_idx]
        test_labels = sample_labels.iloc[split_idx:180]

        feature_fn = generate_seq_feature_func_for_training(
            selected_features,
            mock_feature_engineering_func,
            feature_config,
            sequence_length,
            scaler,
        )

        X_train_3d, X_test_3d, y_train_seq, y_test_seq = feature_fn(
            train_raw, test_raw, train_labels, test_labels
        )

        # Step 2: Inference on new data
        new_raw_data = sample_raw_data.iloc[180:]
        X_inference_3d = generate_seq_features_for_inference(
            new_raw_data,
            selected_features,
            mock_feature_engineering_func,
            feature_config,
            sequence_length,
            scaler,
        )

        # Verify shapes are consistent
        assert (
            X_train_3d.shape[1] == X_inference_3d.shape[1]
        ), "Sequence lengths should match"
        assert (
            X_train_3d.shape[2] == X_inference_3d.shape[2]
        ), "Number of features should match"
        assert (
            X_test_3d.shape[2] == X_inference_3d.shape[2]
        ), "All outputs should have same features"

    def test_consistent_scaling_across_train_and_inference(
        self,
        sample_raw_data,
        sample_labels,
        mock_feature_engineering_func,
        feature_config,
        selected_features,
    ):
        """Test that scaling is consistent between training and inference."""
        sequence_length = 5
        scaler = StandardScaler()

        # Training
        split_idx = 150
        train_raw = sample_raw_data.iloc[:split_idx]
        test_raw = sample_raw_data.iloc[split_idx:]
        train_labels = sample_labels.iloc[:split_idx]
        test_labels = sample_labels.iloc[split_idx:]

        feature_fn = generate_seq_feature_func_for_training(
            selected_features,
            mock_feature_engineering_func,
            feature_config,
            sequence_length,
            scaler,
        )

        X_train_3d, X_test_3d, _, _ = feature_fn(
            train_raw, test_raw, train_labels, test_labels
        )

        # Inference on same test data
        X_inference_3d = generate_seq_features_for_inference(
            test_raw,
            selected_features,
            mock_feature_engineering_func,
            feature_config,
            sequence_length,
            scaler,
        )

        # Test and inference outputs should be identical (same scaler, same data)
        np.testing.assert_array_almost_equal(
            X_test_3d,
            X_inference_3d,
            decimal=5,
            err_msg="Test and inference outputs should be identical for same data",
        )
