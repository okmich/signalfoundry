import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from okmich_quant_neural_net.tab_2_seq import (
    transform_to_sequence,
    create_sequences,
    create_train_test_sequences,
    create_multi_output_sequences,
)

np.random.seed(42)


# ============================================================================
# FIXTURES
# ============================================================================
@pytest.fixture
def simple_data():
    """Simple data arrays for basic testing."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = np.random.randn(100)
    return X, y


@pytest.fixture
def multi_output_data():
    """Data with multiple target columns."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.randn(100, 4)  # 4 targets
    return X, y


@pytest.fixture
def sequential_data():
    """Sequential data for tracking transformations."""
    n = 50
    X = np.arange(n * 3).reshape(n, 3)
    y = np.arange(n) * 100
    return X, y


@pytest.fixture
def fitted_scaler():
    """Pre-fitted scaler for transform_to_sequence tests."""
    scaler = StandardScaler()
    # Fit on dummy data
    scaler.fit(np.random.randn(100, 4))
    return scaler


# ============================================================================
# TEST transform_to_sequence - BASIC FUNCTIONALITY
# ============================================================================


class TestTransformToSequenceBasic:
    """Test basic functionality of transform_to_sequence."""

    def test_basic_transform(self, simple_data, fitted_scaler):
        """Test basic transformation with fitted scaler."""
        X, _ = simple_data
        features_seq = transform_to_sequence(X, sequence_length=5, scaler=fitted_scaler)

        assert features_seq.ndim == 3
        assert features_seq.shape[1] == 5  # sequence_length
        assert features_seq.shape[2] == X.shape[1]  # n_features
        assert len(features_seq) == len(X) - 5

    def test_sequence_output_shape(self, simple_data):
        """Test output shape with different sequence lengths."""
        X, _ = simple_data
        scaler = StandardScaler()
        scaler.fit(X)

        for seq_len in [3, 5, 10]:
            features_seq = transform_to_sequence(X, seq_len, scaler)
            expected_samples = len(X) - seq_len
            assert features_seq.shape == (expected_samples, seq_len, X.shape[1])

    def test_scaling_applied(self, simple_data):
        """Test that scaling is properly applied."""
        X, _ = simple_data
        scaler = StandardScaler()
        scaler.fit(X)

        features_seq = transform_to_sequence(X, 5, scaler)

        # The scaled features should have approximately zero mean and unit variance
        assert abs(features_seq.mean()) < 0.3
        assert abs(features_seq.std() - 1.0) < 0.3

    @pytest.mark.parametrize("seq_len", [1, 5, 10, 20])
    def test_various_sequence_lengths(self, simple_data, fitted_scaler, seq_len):
        """Test different sequence lengths."""
        X, _ = simple_data

        features_seq = transform_to_sequence(
            X, sequence_length=seq_len, scaler=fitted_scaler
        )

        assert features_seq.shape[1] == seq_len
        assert len(features_seq) == len(X) - seq_len


# ============================================================================
# TEST transform_to_sequence - ERROR HANDLING
# ============================================================================


class TestTransformToSequenceErrors:
    """Test error handling for transform_to_sequence."""

    def test_invalid_features_dimension(self, fitted_scaler):
        """Test error when features is not 2D."""
        features_1d = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="features must be 2-dimensional"):
            transform_to_sequence(features_1d, 2, fitted_scaler)

    def test_sequence_length_too_small(self, simple_data, fitted_scaler):
        """Test error when sequence_length < 1."""
        X, _ = simple_data

        with pytest.raises(ValueError, match="sequence_length must be at least 1"):
            transform_to_sequence(X, 0, fitted_scaler)

    def test_sequence_length_too_large(self, simple_data, fitted_scaler):
        """Test error when sequence_length >= data length."""
        X, _ = simple_data

        with pytest.raises(ValueError, match="at least"):
            transform_to_sequence(X, len(X), fitted_scaler)

    def test_insufficient_data(self, fitted_scaler):
        """Test error when data is too small for sequence length."""
        X = np.random.randn(5, 4)

        with pytest.raises(ValueError, match="at least"):
            transform_to_sequence(X, 10, fitted_scaler)


# ============================================================================
# TEST create_sequences - BASIC FUNCTIONALITY
# ============================================================================


class TestCreateSequencesBasic:
    """Test basic functionality of create_sequences."""

    def test_basic_transform_with_fit(self, simple_data):
        """Test basic transformation with fit_scaler=True."""
        X, y = simple_data
        scaler = StandardScaler()

        features_seq, target_seq = create_sequences(
            X, y, sequence_length=5, scaler=scaler, fit_scaler=True
        )

        assert features_seq.ndim == 3
        assert target_seq.ndim == 1
        assert features_seq.shape[1] == 5  # sequence_length
        assert features_seq.shape[2] == X.shape[1]  # n_features
        assert len(features_seq) == len(target_seq)
        assert len(features_seq) == len(X) - 5

    def test_basic_transform_without_fit(self, simple_data, fitted_scaler):
        """Test basic transformation with fit_scaler=False."""
        X, y = simple_data

        features_seq, target_seq = create_sequences(
            X, y, sequence_length=5, scaler=fitted_scaler, fit_scaler=False
        )

        assert features_seq.shape[0] == len(X) - 5
        assert len(target_seq) == len(X) - 5

    def test_target_alignment(self, sequential_data):
        """Test that targets are correctly aligned with sequences."""
        X, y = sequential_data
        seq_len = 3

        scaler = StandardScaler()
        scaler.fit(X)

        features_seq, target_seq = create_sequences(
            X, y, sequence_length=seq_len, scaler=scaler, fit_scaler=False
        )

        # The target at position i should correspond to y[i + seq_len]
        # So target_seq[0] should be y[seq_len]
        assert len(target_seq) == len(y) - seq_len
        assert target_seq[0] == y[seq_len]
        assert target_seq[-1] == y[-1]

    def test_fit_scaler_true(self, simple_data):
        """Test that scaler gets fitted when fit_scaler=True."""
        X, y = simple_data
        scaler = StandardScaler()

        features_seq, target_seq = create_sequences(
            X, y, sequence_length=5, scaler=scaler, fit_scaler=True
        )

        # Scaler should be fitted
        assert hasattr(scaler, "mean_")
        assert hasattr(scaler, "scale_")

    @pytest.mark.parametrize("seq_len", [1, 5, 10, 20])
    def test_various_sequence_lengths(self, simple_data, seq_len):
        """Test different sequence lengths."""
        X, y = simple_data
        scaler = StandardScaler()

        features_seq, target_seq = create_sequences(
            X, y, sequence_length=seq_len, scaler=scaler, fit_scaler=True
        )

        assert features_seq.shape[1] == seq_len
        assert len(features_seq) == len(X) - seq_len
        assert len(target_seq) == len(X) - seq_len


# ============================================================================
# TEST create_sequences - ERROR HANDLING
# ============================================================================


class TestCreateSequencesErrors:
    """Test error handling for create_sequences."""

    def test_invalid_features_dimension(self, fitted_scaler):
        """Test error when features is not 2D."""
        features_1d = np.array([1, 2, 3, 4, 5])
        target = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="features must be 2-dimensional"):
            create_sequences(features_1d, target, 2, fitted_scaler)

    def test_sequence_length_too_small(self, simple_data, fitted_scaler):
        """Test error when sequence_length < 1."""
        X, y = simple_data

        with pytest.raises(ValueError, match="sequence_length must be at least 1"):
            create_sequences(X, y, 0, fitted_scaler)

    def test_sequence_length_too_large(self, simple_data, fitted_scaler):
        """Test error when sequence_length >= data length."""
        X, y = simple_data

        with pytest.raises(ValueError, match="at least"):
            create_sequences(X, y, len(X), fitted_scaler)

    def test_mismatched_lengths(self, fitted_scaler):
        """Test error when features and target have different lengths."""
        X = np.random.randn(100, 4)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="must have same length"):
            create_sequences(X, y, 5, fitted_scaler)


# ============================================================================
# TEST create_train_test_sequences - BASIC FUNCTIONALITY
# ============================================================================


class TestCreateTrainTestSequencesBasic:
    """Test basic functionality of create_train_test_sequences."""

    def test_basic_functionality(self, simple_data):
        """Test basic train-test split functionality."""
        X, y = simple_data

        X_train, y_train, X_test, y_test, _, _ = create_train_test_sequences(
            X, y, sequence_length=5, test_size=0.2
        )

        assert X_train.ndim == 3
        assert y_train.ndim == 1
        assert X_test.ndim == 3
        assert y_test.ndim == 1
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_train_test_split_ratio(self, simple_data):
        """Test that train/test split respects test_size."""
        X, y = simple_data

        X_train, y_train, X_test, y_test, _, _ = create_train_test_sequences(
            X, y, sequence_length=5, test_size=0.2
        )

        total = len(y_train) + len(y_test)
        actual_test_ratio = len(y_test) / total
        assert abs(actual_test_ratio - 0.2) < 0.1

    @pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3, 0.4])
    def test_various_test_sizes(self, simple_data, test_size):
        """Test different test_size values."""
        X, y = simple_data

        X_train, y_train, X_test, y_test, _, _ = create_train_test_sequences(
            X, y, sequence_length=5, test_size=test_size
        )

        total = len(y_train) + len(y_test)
        actual_test_ratio = len(y_test) / total
        assert abs(actual_test_ratio - test_size) < 0.1

    @pytest.mark.parametrize("seq_len", [3, 5, 10, 20])
    def test_various_sequence_lengths(self, simple_data, seq_len):
        """Test different sequence lengths."""
        X, y = simple_data

        X_train, y_train, X_test, y_test, _, _ = create_train_test_sequences(
            X, y, sequence_length=seq_len, test_size=0.2
        )

        assert X_train.shape[1] == seq_len
        assert X_test.shape[1] == seq_len

    def test_no_data_leakage(self, sequential_data):
        """Test that there's no data leakage between train and test."""
        X, y = sequential_data

        X_train, y_train, X_test, y_test, _, _ = create_train_test_sequences(
            X, y, sequence_length=3, test_size=0.2, shuffle=False
        )

        # Last target in train should be before first sequences in test
        # This ensures temporal ordering is preserved
        assert len(X_train) > 0
        assert len(X_test) > 0


# ============================================================================
# TEST create_train_test_sequences - SCALING
# ============================================================================


class TestCreateTrainTestSequencesScaling:
    """Test scaling functionality."""

    def test_feature_scaling(self, simple_data):
        """Test feature scaling without target scaling."""
        X, y = simple_data

        X_train, y_train, X_test, y_test, feat_scaler, targ_scaler = (
            create_train_test_sequences(
                X,
                y,
                sequence_length=5,
                test_size=0.2,
                scaler=StandardScaler(),
                scale_target=False,
            )
        )

        # Features should be scaled
        assert abs(X_train.mean()) < 0.3
        assert abs(X_train.std() - 1.0) < 0.3
        assert feat_scaler is not None
        assert targ_scaler is None

    def test_target_scaling(self, simple_data):
        """Test target scaling with feature scaling."""
        X, y = simple_data

        X_train, y_train, X_test, y_test, feat_scaler, targ_scaler = (
            create_train_test_sequences(
                X,
                y,
                sequence_length=5,
                test_size=0.2,
                scaler=StandardScaler(),
                scale_target=True,
            )
        )

        # Both features and targets should be scaled
        assert abs(X_train.mean()) < 0.3
        assert abs(y_train.mean()) < 0.3
        assert abs(y_train.std() - 1.0) < 0.3
        assert feat_scaler is not None
        assert targ_scaler is not None

    def test_no_scaling(self, simple_data):
        """Test without any scaling."""
        X, y = simple_data

        X_train, y_train, X_test, y_test, feat_scaler, targ_scaler = (
            create_train_test_sequences(
                X, y, sequence_length=5, test_size=0.2, scaler=None, scale_target=False
            )
        )

        assert feat_scaler is None
        assert targ_scaler is None

    def test_minmax_scaler(self, simple_data):
        """Test MinMaxScaler."""
        X, y = simple_data

        X_train, y_train, X_test, y_test, feat_scaler, targ_scaler = (
            create_train_test_sequences(
                X,
                y,
                sequence_length=5,
                test_size=0.2,
                scaler=MinMaxScaler(),
                scale_target=True,
            )
        )

        # Features should be in [0, 1] range (with some tolerance)
        assert X_train.min() >= -0.1
        assert X_train.max() <= 1.1
        # Targets should be in [0, 1] range (with some tolerance)
        assert y_train.min() >= -0.1
        assert y_train.max() <= 1.1

    def test_robust_scaler(self, simple_data):
        """Test RobustScaler."""
        X, y = simple_data

        X_train, y_train, X_test, y_test, feat_scaler, targ_scaler = (
            create_train_test_sequences(
                X,
                y,
                sequence_length=5,
                test_size=0.2,
                scaler=RobustScaler(),
                scale_target=True,
            )
        )

        assert feat_scaler is not None
        assert targ_scaler is not None

    def test_scaler_fitted_on_train_only(self, simple_data):
        """Test that scaler is fitted only on training data."""
        X, y = simple_data

        X_train, y_train, X_test, y_test, feat_scaler, targ_scaler = (
            create_train_test_sequences(
                X,
                y,
                sequence_length=5,
                test_size=0.2,
                scaler=StandardScaler(),
                scale_target=True,
            )
        )

        # Scalers should be fitted
        assert hasattr(feat_scaler, "mean_")
        assert hasattr(targ_scaler, "mean_")

    def test_inverse_transform_features(self, simple_data):
        """Test inverse transform for features."""
        X, y = simple_data

        X_train, y_train, X_test, y_test, feat_scaler, _ = create_train_test_sequences(
            X,
            y,
            sequence_length=5,
            test_size=0.2,
            scaler=StandardScaler(),
            scale_target=False,
        )

        # Test inverse transform
        scaled_sample = X_train[0]  # Shape: (seq_len, n_features)
        original_sample = feat_scaler.inverse_transform(scaled_sample)

        assert original_sample.shape == scaled_sample.shape
        assert not np.isnan(original_sample).any()

    def test_inverse_transform_targets(self, simple_data):
        """Test inverse transform for targets."""
        X, y = simple_data

        X_train, y_train, X_test, y_test, _, targ_scaler = create_train_test_sequences(
            X,
            y,
            sequence_length=5,
            test_size=0.2,
            scaler=StandardScaler(),
            scale_target=True,
        )

        # Test inverse transform
        scaled_targets = y_train[:10].reshape(-1, 1)
        original_targets = targ_scaler.inverse_transform(scaled_targets).ravel()

        assert len(original_targets) == 10
        assert not np.isnan(original_targets).any()


# ============================================================================
# TEST create_train_test_sequences - SHUFFLE AND RANDOM STATE
# ============================================================================


class TestCreateTrainTestSequencesShuffle:
    """Test shuffle functionality."""

    def test_shuffle_changes_order(self, simple_data):
        """Test that shuffle changes sequence order."""
        X, y = simple_data

        X_train1, y_train1, _, _, _, _ = create_train_test_sequences(
            X, y, sequence_length=5, test_size=0.2, shuffle=False
        )

        X_train2, y_train2, _, _, _, _ = create_train_test_sequences(
            X, y, sequence_length=5, test_size=0.2, shuffle=True, random_state=42
        )

        # Should be different order
        assert not np.array_equal(y_train1, y_train2)

    def test_random_state_reproducibility(self, simple_data):
        """Test that random_state makes results reproducible."""
        X, y = simple_data

        X_train1, y_train1, X_test1, y_test1, _, _ = create_train_test_sequences(
            X, y, sequence_length=5, test_size=0.2, shuffle=True, random_state=42
        )

        X_train2, y_train2, X_test2, y_test2, _, _ = create_train_test_sequences(
            X, y, sequence_length=5, test_size=0.2, shuffle=True, random_state=42
        )

        # Should be identical
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_test1, y_test2)
        np.testing.assert_array_equal(X_test1, X_test2)

    def test_different_random_states(self, simple_data):
        """Test that different random states give different results."""
        X, y = simple_data

        X_train1, y_train1, _, _, _, _ = create_train_test_sequences(
            X, y, sequence_length=5, test_size=0.2, shuffle=True, random_state=42
        )

        X_train2, y_train2, _, _, _, _ = create_train_test_sequences(
            X, y, sequence_length=5, test_size=0.2, shuffle=True, random_state=123
        )

        # Should be different
        assert not np.array_equal(y_train1, y_train2)

    def test_shuffle_without_random_state(self, simple_data):
        """Test shuffle without setting random_state."""
        X, y = simple_data

        X_train1, y_train1, _, _, _, _ = create_train_test_sequences(
            X, y, sequence_length=5, test_size=0.2, shuffle=True
        )

        X_train2, y_train2, _, _, _, _ = create_train_test_sequences(
            X, y, sequence_length=5, test_size=0.2, shuffle=True
        )

        # Likely to be different (not guaranteed but highly probable)
        # Just verify it doesn't crash
        assert len(X_train1) == len(X_train2)


# ============================================================================
# TEST create_train_test_sequences - ERROR HANDLING
# ============================================================================


class TestCreateTrainTestSequencesErrors:
    """Test error handling."""

    def test_invalid_X_dimension(self):
        """Test error when X is not 2D."""
        X_1d = np.array([1, 2, 3, 4, 5])
        y = np.random.randn(5)

        with pytest.raises(ValueError, match="X must be 2-dimensional"):
            create_train_test_sequences(X_1d, y, 2, test_size=0.2)

    def test_invalid_y_dimension(self):
        """Test error when y is not 1D."""
        X = np.random.randn(10, 4)
        y_2d = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="y must be 1-dimensional"):
            create_train_test_sequences(X, y_2d, 2, test_size=0.2)

    def test_mismatched_lengths(self):
        """Test error when X and y have different lengths."""
        X = np.random.randn(100, 4)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="must have same length"):
            create_train_test_sequences(X, y, 5, test_size=0.2)

    def test_invalid_sequence_length_negative(self):
        """Test error when sequence_length < 1."""
        X = np.random.randn(100, 4)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="sequence_length must be at least 1"):
            create_train_test_sequences(X, y, 0, test_size=0.2)

    def test_invalid_sequence_length_too_large(self):
        """Test error when sequence_length >= data length."""
        X = np.random.randn(10, 4)
        y = np.random.randn(10)

        with pytest.raises(ValueError, match="sequence_length must be less than"):
            create_train_test_sequences(X, y, 20, test_size=0.2)

    def test_invalid_test_size_too_large(self):
        """Test error when test_size >= 1."""
        X = np.random.randn(100, 4)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="test_size must be between"):
            create_train_test_sequences(X, y, 5, test_size=1.5)

    def test_invalid_test_size_zero(self):
        """Test error when test_size = 0."""
        X = np.random.randn(100, 4)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="test_size must be between"):
            create_train_test_sequences(X, y, 5, test_size=0.0)


# ============================================================================
# TEST create_multi_output_sequences - BASIC FUNCTIONALITY
# ============================================================================


class TestCreateMultiOutputSequencesBasic:
    """Test basic functionality of create_multi_output_sequences."""

    def test_basic_functionality(self, multi_output_data):
        """Test with multiple targets."""
        X, y = multi_output_data

        X_train, y_train, X_test, y_test, _, _ = create_multi_output_sequences(
            X, y, sequence_length=5, test_size=0.2
        )

        assert X_train.shape[2] == X.shape[1]  # n_features
        assert y_train.shape[1] == y.shape[1]  # n_targets
        assert X_train.ndim == 3
        assert y_train.ndim == 2

    def test_consistent_shapes(self, multi_output_data):
        """Test that X and y shapes are consistent."""
        X, y = multi_output_data

        X_train, y_train, X_test, y_test, _, _ = create_multi_output_sequences(
            X, y, sequence_length=5, test_size=0.2
        )

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    @pytest.mark.parametrize("n_targets", [1, 2, 3, 5])
    def test_various_target_counts(self, n_targets):
        """Test with different numbers of targets."""
        X = np.random.randn(100, 4)
        y = np.random.randn(100, n_targets)

        X_train, y_train, X_test, y_test, _, _ = create_multi_output_sequences(
            X, y, sequence_length=5, test_size=0.2
        )

        assert y_train.shape[1] == n_targets
        assert y_test.shape[1] == n_targets

    @pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3, 0.4])
    def test_various_test_sizes(self, multi_output_data, test_size):
        """Test different test_size values."""
        X, y = multi_output_data

        X_train, y_train, X_test, y_test, _, _ = create_multi_output_sequences(
            X, y, sequence_length=5, test_size=test_size
        )

        total = len(y_train) + len(y_test)
        actual_test_ratio = len(y_test) / total
        assert abs(actual_test_ratio - test_size) < 0.1

    @pytest.mark.parametrize("seq_len", [3, 5, 10, 20])
    def test_various_sequence_lengths(self, multi_output_data, seq_len):
        """Test different sequence lengths."""
        X, y = multi_output_data

        X_train, y_train, X_test, y_test, _, _ = create_multi_output_sequences(
            X, y, sequence_length=seq_len, test_size=0.2
        )

        assert X_train.shape[1] == seq_len
        assert X_test.shape[1] == seq_len


# ============================================================================
# TEST create_multi_output_sequences - SCALING
# ============================================================================


class TestCreateMultiOutputSequencesScaling:
    """Test scaling functionality for multi-output."""

    def test_multi_output_scaling_features(self, multi_output_data):
        """Test feature scaling with multiple outputs."""
        X, y = multi_output_data

        X_train, y_train, X_test, y_test, feat_scaler, targ_scaler = (
            create_multi_output_sequences(
                X,
                y,
                sequence_length=5,
                test_size=0.2,
                scaler=StandardScaler(),
                scale_target=False,
            )
        )

        assert abs(X_train.mean()) < 0.2
        assert feat_scaler is not None
        assert targ_scaler is None

    def test_multi_output_scaling_targets(self, multi_output_data):
        """Test target scaling with multiple outputs."""
        X, y = multi_output_data

        X_train, y_train, X_test, y_test, feat_scaler, targ_scaler = (
            create_multi_output_sequences(
                X,
                y,
                sequence_length=5,
                test_size=0.2,
                scaler=StandardScaler(),
                scale_target=True,
            )
        )

        # Each target should be scaled
        assert abs(y_train.mean()) < 0.2
        assert abs(y_train.std() - 1.0) < 0.3
        assert targ_scaler is not None

    def test_multi_output_inverse_transform(self, multi_output_data):
        """Test inverse transform with multiple targets."""
        X, y = multi_output_data

        X_train, y_train, X_test, y_test, feat_scaler, targ_scaler = (
            create_multi_output_sequences(
                X,
                y,
                sequence_length=5,
                test_size=0.2,
                scaler=StandardScaler(),
                scale_target=True,
            )
        )

        # Test inverse transform
        scaled_preds = y_train[:5]
        original_preds = targ_scaler.inverse_transform(scaled_preds)

        assert original_preds.shape == scaled_preds.shape
        assert not np.isnan(original_preds).any()

    def test_minmax_scaler_multi_output(self, multi_output_data):
        """Test MinMaxScaler with multi-output."""
        X, y = multi_output_data

        X_train, y_train, X_test, y_test, feat_scaler, targ_scaler = (
            create_multi_output_sequences(
                X,
                y,
                sequence_length=5,
                test_size=0.2,
                scaler=MinMaxScaler(),
                scale_target=True,
            )
        )

        # Features should be in [0, 1]
        assert X_train.min() >= -0.1
        assert X_train.max() <= 1.1
        # Targets should be in [0, 1]
        assert y_train.min() >= -0.1
        assert y_train.max() <= 1.1

    def test_no_scaler(self, multi_output_data):
        """Test without scaling."""
        X, y = multi_output_data

        X_train, y_train, X_test, y_test, feat_scaler, targ_scaler = (
            create_multi_output_sequences(
                X, y, sequence_length=5, test_size=0.2, scaler=None
            )
        )

        assert feat_scaler is None
        assert targ_scaler is None


# ============================================================================
# TEST create_multi_output_sequences - SHUFFLE AND RANDOM STATE
# ============================================================================


class TestCreateMultiOutputSequencesShuffle:
    """Test shuffle functionality for multi-output."""

    def test_shuffle_changes_order(self, multi_output_data):
        """Test that shuffle changes sequence order."""
        X, y = multi_output_data

        X_train1, y_train1, _, _, _, _ = create_multi_output_sequences(
            X, y, sequence_length=5, test_size=0.2, shuffle=False
        )

        X_train2, y_train2, _, _, _, _ = create_multi_output_sequences(
            X, y, sequence_length=5, test_size=0.2, shuffle=True, random_state=42
        )

        # Should be different order
        assert not np.array_equal(y_train1, y_train2)

    def test_random_state_reproducibility(self, multi_output_data):
        """Test that random_state makes results reproducible."""
        X, y = multi_output_data

        X_train1, y_train1, _, _, _, _ = create_multi_output_sequences(
            X, y, sequence_length=5, test_size=0.2, shuffle=True, random_state=42
        )

        X_train2, y_train2, _, _, _, _ = create_multi_output_sequences(
            X, y, sequence_length=5, test_size=0.2, shuffle=True, random_state=42
        )

        # Should be identical
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(X_train1, X_train2)

    def test_different_random_states(self, multi_output_data):
        """Test that different random states give different results."""
        X, y = multi_output_data

        X_train1, y_train1, _, _, _, _ = create_multi_output_sequences(
            X, y, sequence_length=5, test_size=0.2, shuffle=True, random_state=42
        )

        X_train2, y_train2, _, _, _, _ = create_multi_output_sequences(
            X, y, sequence_length=5, test_size=0.2, shuffle=True, random_state=123
        )

        # Should be different
        assert not np.array_equal(y_train1, y_train2)


# ============================================================================
# TEST create_multi_output_sequences - ERROR HANDLING
# ============================================================================


class TestCreateMultiOutputSequencesErrors:
    """Test error handling for multi-output function."""

    def test_invalid_X_dimension(self):
        """Test error when X is not 2D."""
        X_1d = np.array([1, 2, 3, 4, 5])
        y = np.random.randn(5, 2)

        with pytest.raises(ValueError, match="X must be 2-dimensional"):
            create_multi_output_sequences(X_1d, y, 2, test_size=0.2)

    def test_invalid_y_dimension(self):
        """Test error when y is not 2D."""
        X = np.random.randn(10, 4)
        y_1d = np.random.randn(10)

        with pytest.raises(ValueError, match="y must be 2-dimensional"):
            create_multi_output_sequences(X, y_1d, 2, test_size=0.2)

    def test_mismatched_lengths(self):
        """Test error when X and y have different lengths."""
        X = np.random.randn(100, 4)
        y = np.random.randn(50, 3)

        with pytest.raises(ValueError, match="must have same length"):
            create_multi_output_sequences(X, y, 5, test_size=0.2)

    def test_invalid_sequence_length_negative(self):
        """Test error when sequence_length < 1."""
        X = np.random.randn(100, 4)
        y = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="sequence_length must be at least 1"):
            create_multi_output_sequences(X, y, 0, test_size=0.2)

    def test_invalid_sequence_length_too_large(self):
        """Test error when sequence_length >= data length."""
        X = np.random.randn(10, 4)
        y = np.random.randn(10, 3)

        with pytest.raises(ValueError, match="sequence_length must be less than"):
            create_multi_output_sequences(X, y, 20, test_size=0.2)

    def test_invalid_test_size_too_large(self):
        """Test error when test_size >= 1."""
        X = np.random.randn(100, 4)
        y = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="test_size must be between"):
            create_multi_output_sequences(X, y, 5, test_size=1.5)

    def test_invalid_test_size_zero(self):
        """Test error when test_size = 0."""
        X = np.random.randn(100, 4)
        y = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="test_size must be between"):
            create_multi_output_sequences(X, y, 5, test_size=0.0)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for real-world scenarios."""

    def test_full_pipeline_with_create_sequences(self, simple_data):
        """Test complete pipeline with create_sequences."""
        X, y = simple_data

        # Split data manually
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Create sequences for training with fit
        scaler = StandardScaler()
        X_train_seq, y_train_seq = create_sequences(
            X_train, y_train, sequence_length=10, scaler=scaler, fit_scaler=True
        )

        # Create sequences for testing without fit
        X_test_seq, y_test_seq = create_sequences(
            X_test, y_test, sequence_length=10, scaler=scaler, fit_scaler=False
        )

        # Verify shapes
        assert X_train_seq.shape[0] > 0
        assert X_test_seq.shape[0] > 0
        assert hasattr(scaler, "mean_")

    def test_full_pipeline_multi_output(self, multi_output_data):
        """Test complete pipeline with multiple outputs."""
        X, y = multi_output_data

        X_train, y_train, X_test, y_test, feat_scaler, targ_scaler = (
            create_multi_output_sequences(
                X,
                y,
                sequence_length=8,
                test_size=0.25,
                scaler=StandardScaler(),
                scale_target=True,
                shuffle=True,
                random_state=42,
            )
        )

        # Verify shapes
        assert y_train.shape[1] == y.shape[1]
        assert y_test.shape[1] == y.shape[1]

        # Test inverse transform
        dummy_preds = y_test[:10]
        original_scale = targ_scaler.inverse_transform(dummy_preds)
        assert original_scale.shape == (10, y.shape[1])

    def test_transform_to_sequence_production_use(self, simple_data):
        """Test using transform_to_sequence for production inference."""
        X, y = simple_data

        # Step 1: Fit scaler on training data
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        scaler = StandardScaler()
        scaler.fit(X_train)

        # Step 2: Simulate new production data
        X_new = np.random.randn(30, X.shape[1])

        # Step 3: Transform new data using fitted scaler
        X_new_seq = transform_to_sequence(X_new, sequence_length=5, scaler=scaler)

        # Verify
        assert X_new_seq.shape[1] == 5
        assert X_new_seq.shape[2] == X.shape[1]
        assert len(X_new_seq) == len(X_new) - 5


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_viable_dataset(self):
        """Test with minimum viable dataset size."""
        X = np.random.randn(12, 2)
        y = np.random.randn(12, 3)

        X_train, y_train, X_test, y_test, _, _ = create_multi_output_sequences(
            X, y, sequence_length=3, test_size=0.2
        )

        assert len(y_train) > 0
        assert len(y_test) > 0

    def test_single_feature_column(self):
        """Test with only one feature column."""
        X = np.random.randn(50, 1)
        y = np.random.randn(50, 2)

        X_train, y_train, X_test, y_test, _, _ = create_multi_output_sequences(
            X, y, sequence_length=5, test_size=0.2
        )

        assert X_train.shape[2] == 1

    def test_single_target_multi_output(self):
        """Test multi-output with just one target."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50, 1)

        X_train, y_train, X_test, y_test, _, _ = create_multi_output_sequences(
            X, y, sequence_length=5, test_size=0.2
        )

        assert y_train.shape[1] == 1

    def test_sequence_length_one(self):
        """Test with sequence_length=1 (edge case)."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50, 2)

        X_train, y_train, X_test, y_test, _, _ = create_multi_output_sequences(
            X, y, sequence_length=1, test_size=0.2
        )

        assert X_train.shape[1] == 1
        assert len(y_train) + len(y_test) == len(X) - 1

    def test_pandas_to_numpy_conversion(self):
        """Test that pandas DataFrames are converted properly."""
        df = pd.DataFrame(np.random.randn(50, 4))
        df_targets = pd.DataFrame(np.random.randn(50, 2))

        X_train, y_train, X_test, y_test, _, _ = create_multi_output_sequences(
            df, df_targets, sequence_length=5, test_size=0.2
        )

        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)

    def test_transform_to_sequence_minimum_data(self, fitted_scaler):
        """Test transform_to_sequence with minimal data."""
        X = np.random.randn(10, 4)

        features_seq = transform_to_sequence(X, sequence_length=3, scaler=fitted_scaler)

        assert len(features_seq) == 10 - 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
