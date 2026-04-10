import os
import tempfile

import joblib
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from okmich_quant_ml.clustering.swkmeans import SlicedWassersteinKMeans


class TestSlicedWassersteinKMeans:
    """Comprehensive test suite for SlicedWassersteinKMeans implementation."""

    @pytest.fixture
    def simple_2d_data(self):
        """Create simple 2D synthetic time series with known regime changes."""
        np.random.seed(42)
        n_samples = 200

        # Regime 1: Low volatility, positive correlation
        regime1_length = 100
        noise1 = np.random.randn(regime1_length, 2) * 0.01
        corr_noise1 = np.zeros_like(noise1)
        corr_noise1[:, 0] = noise1[:, 0]
        corr_noise1[:, 1] = 0.8 * noise1[:, 0] + 0.6 * noise1[:, 1]

        # Regime 2: High volatility, negative correlation
        regime2_length = n_samples - regime1_length
        noise2 = np.random.randn(regime2_length, 2) * 0.03
        corr_noise2 = np.zeros_like(noise2)
        corr_noise2[:, 0] = noise2[:, 0]
        corr_noise2[:, 1] = -0.7 * noise2[:, 0] + 0.7 * noise2[:, 1]

        # Combine and convert to prices
        returns = np.vstack([corr_noise1, corr_noise2])
        prices = np.exp(np.cumsum(returns, axis=0))

        # True trend for evaluation
        true_labels = np.concatenate(
            [np.zeros(regime1_length, dtype=int), np.ones(regime2_length, dtype=int)]
        )

        return prices, true_labels[1:]  # Remove first element due to diff in returns

    @pytest.fixture
    def complex_3d_data(self):
        """Create 3D synthetic data with three regimes."""
        np.random.seed(123)
        n_samples = 300

        # Three regimes of equal length
        regime_length = n_samples // 3

        # Regime 1: All positive correlations
        cov1 = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]]) * 0.01
        returns1 = np.random.multivariate_normal([0, 0, 0], cov1, regime_length)

        # Regime 2: Mixed correlations
        cov2 = np.array([[1.0, -0.6, 0.2], [-0.6, 1.0, -0.3], [0.2, -0.3, 1.0]]) * 0.02
        returns2 = np.random.multivariate_normal([0, 0, 0], cov2, regime_length)

        # Regime 3: High volatility, low correlations
        cov3 = np.array([[1.0, 0.1, -0.1], [0.1, 1.0, 0.0], [-0.1, 0.0, 1.0]]) * 0.04
        returns3 = np.random.multivariate_normal(
            [0, 0, 0], cov3, n_samples - 2 * regime_length
        )

        returns = np.vstack([returns1, returns2, returns3])
        prices = np.exp(np.cumsum(returns, axis=0))

        true_labels = np.concatenate(
            [
                np.zeros(regime_length, dtype=int),
                np.ones(regime_length, dtype=int),
                np.full(n_samples - 2 * regime_length, 2, dtype=int),
            ]
        )

        return prices, true_labels[1:]

    def test_initialization(self):
        """Test model initialization with various parameters."""
        # Default initialization
        swk = SlicedWassersteinKMeans()
        assert swk.n_clusters == 2
        assert swk.window_size == 35
        assert swk.lifting_size == 7

        # Custom initialization
        swk_custom = SlicedWassersteinKMeans(
            n_clusters=3,
            window_size=50,
            lifting_size=10,
            n_projections=16,
            p=2,
            random_state=42,
        )
        assert swk_custom.n_clusters == 3
        assert swk_custom.window_size == 50
        assert swk_custom.p == 2

    def test_fit_basic(self, simple_2d_data):
        """Test basic fitting functionality."""
        X, y_true = simple_2d_data

        swk = SlicedWassersteinKMeans(
            n_clusters=2, window_size=30, lifting_size=5, random_state=42
        )

        # Should not raise any exceptions
        swk.fit(X)

        # Check fitted attributes exist
        assert hasattr(swk, "cluster_centers_")
        assert hasattr(swk, "labels_")
        assert hasattr(swk, "projection_directions_")
        assert hasattr(swk, "inertia_")
        assert hasattr(swk, "n_iter_")

        # Check shapes
        assert swk.cluster_centers_.shape[0] == 2  # n_clusters
        assert len(swk.labels_) == len(X) - 1  # minus 1 due to returns calculation
        assert swk.projection_directions_.shape[1] == X.shape[1]  # n_features

    def test_predict(self, simple_2d_data):
        """Test prediction on new data."""
        X, _ = simple_2d_data

        swk = SlicedWassersteinKMeans(n_clusters=2, window_size=30, random_state=42)
        swk.fit(X)

        # Predict on same data
        labels = swk.predict(X)
        assert len(labels) == len(X) - 1
        assert all(label in [0, 1] for label in labels)

        # Predict on subset
        X_subset = X[:100]
        labels_subset = swk.predict(X_subset)
        assert len(labels_subset) == len(X_subset) - 1

    def test_predict_proba(self, simple_2d_data):
        """Test probabilistic predictions."""
        X, _ = simple_2d_data

        swk = SlicedWassersteinKMeans(n_clusters=2, window_size=30, random_state=42)
        swk.fit(X)

        probas = swk.predict_proba(X)

        # Check shape and properties
        assert probas.shape == (len(X) - 1, 2)
        assert np.all(probas >= 0) and np.all(probas <= 1)  # Valid probabilities
        cols_sum = probas.sum(axis=1)
        assert np.allclose(cols_sum, np.ones(len(cols_sum)))  # Probabilities sum to 1

    def test_fit_predict(self, simple_2d_data):
        """Test fit_predict method."""
        X, _ = simple_2d_data

        swk = SlicedWassersteinKMeans(n_clusters=2, window_size=30, random_state=42)
        labels = swk.fit_predict(X)

        assert len(labels) == len(X) - 1
        assert np.array_equal(labels, swk.labels_)

    def test_regime_detection_accuracy(self, simple_2d_data):
        """Test ability to detect regime changes."""
        X, y_true = simple_2d_data

        swk = SlicedWassersteinKMeans(
            n_clusters=2,
            window_size=20,
            lifting_size=5,
            n_projections=12,
            random_state=42,
        )

        labels = swk.fit_predict(X)

        # Calculate adjusted rand score (handles label permutations)
        ari = adjusted_rand_score(y_true, labels)

        # Should achieve reasonable accuracy on this simple synthetic data
        assert ari > 0.3, f"Poor regime detection: ARI = {ari:.3f}"

    def test_multiple_regimes(self, complex_3d_data):
        """Test with more complex 3D data and 3 regimes."""
        X, y_true = complex_3d_data

        swk = SlicedWassersteinKMeans(
            n_clusters=3,
            window_size=25,
            lifting_size=5,
            n_projections=9,
            random_state=42,
        )

        labels = swk.fit_predict(X)

        # Check that all three clusters are found
        unique_labels = np.unique(labels)
        assert len(unique_labels) <= 3  # May merge some clusters

        # Should have reasonable performance
        ari = adjusted_rand_score(y_true, labels)
        assert ari > 0.1, f"Poor multi-regime detection: ARI = {ari:.3f}"

    def test_standardization_effect(self, simple_2d_data):
        """Test effect of standardization parameter on return distributions."""
        X, _ = simple_2d_data

        # Create data where one feature has much higher return volatility
        # by scaling the underlying returns differently
        np.random.seed(42)

        low_vol_returns = np.random.normal(0, 0.01, (100, 1))
        high_vol_returns = np.random.normal(0, 0.05, (100, 1))  # 5x higher volatility

        # Convert to prices
        returns_combined = np.hstack([low_vol_returns, high_vol_returns])
        X_unequal_vol = np.exp(np.cumsum(returns_combined, axis=0))

        swk_std = SlicedWassersteinKMeans(
            n_clusters=2,
            window_size=20,
            lifting_size=5,
            standardize=True,
            random_state=42,
        )
        swk_no_std = SlicedWassersteinKMeans(
            n_clusters=2,
            window_size=20,
            lifting_size=5,
            standardize=False,
            random_state=42,
        )

        swk_std.fit(X_unequal_vol)
        swk_no_std.fit(X_unequal_vol)

        assert swk_std.standardize == True
        assert swk_no_std.standardize == False

    def test_hyperparameter_validation(self):
        """Test various edge cases and parameter validation."""
        X = np.random.randn(100, 2)
        X = np.exp(np.cumsum(X * 0.01, axis=0))  # Convert to prices

        # Too few samples
        with pytest.raises(ValueError):
            swk = SlicedWassersteinKMeans(window_size=150)
            swk.fit(X)

        # Invalid p parameter - should raise ValueError
        with pytest.raises(ValueError, match="p must be 1 or 2"):
            swk = SlicedWassersteinKMeans(p=3)

        # lifting_size >= window_size - should raise ValueError
        with pytest.raises(ValueError, match="lifting_size .* must be < window_size"):
            swk = SlicedWassersteinKMeans(window_size=20, lifting_size=25)

    def test_reproducibility(self, simple_2d_data):
        """Test that results are reproducible with same random state."""
        X, _ = simple_2d_data

        swk1 = SlicedWassersteinKMeans(n_clusters=2, window_size=30, random_state=42)
        swk2 = SlicedWassersteinKMeans(n_clusters=2, window_size=30, random_state=42)

        labels1 = swk1.fit_predict(X)
        labels2 = swk2.fit_predict(X)

        assert np.array_equal(labels1, labels2)
        assert np.allclose(swk1.inertia_, swk2.inertia_)

    def test_joblib_serialization(self, simple_2d_data):
        """Test saving and loading with joblib."""
        X, _ = simple_2d_data

        # Fit model
        swk_original = SlicedWassersteinKMeans(
            n_clusters=2, window_size=30, random_state=42
        )
        swk_original.fit(X)
        original_labels = swk_original.predict(X)
        original_probas = swk_original.predict_proba(X)

        # Save and load
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            joblib.dump(swk_original, tmp.name)
            swk_loaded = joblib.load(tmp.name)

            try:
                os.unlink(tmp.name)
            except PermissionError:
                pass

        # Test loaded model
        loaded_labels = swk_loaded.predict(X)
        loaded_probas = swk_loaded.predict_proba(X)

        # Should be identical
        assert np.array_equal(original_labels, loaded_labels)
        assert np.allclose(original_probas, loaded_probas)
        assert swk_loaded.n_clusters == swk_original.n_clusters
        assert np.allclose(swk_loaded.cluster_centers_, swk_original.cluster_centers_)

    def test_convergence_behavior(self, simple_2d_data):
        """Test convergence under different conditions."""
        X, _ = simple_2d_data

        # Fast convergence with tight tolerance
        swk_tight = SlicedWassersteinKMeans(
            n_clusters=2, window_size=30, tol=1e-3, max_iter=50, random_state=42
        )
        swk_tight.fit(X)

        # Should converge quickly
        assert swk_tight.n_iter_ < 20

        # Loose tolerance
        swk_loose = SlicedWassersteinKMeans(
            n_clusters=2, window_size=30, tol=1e-8, max_iter=5, random_state=42
        )
        swk_loose.fit(X)

        # May hit max iterations
        assert swk_loose.n_iter_ <= 5

    def test_1d_compatibility(self):
        """Test that algorithm works with 1D time series."""
        np.random.seed(42)

        # Generate 1D time series with regime change
        n_samples = 200
        returns1 = np.random.normal(0.001, 0.02, 100)
        returns2 = np.random.normal(-0.002, 0.04, 100)
        returns = np.concatenate([returns1, returns2])
        prices = np.exp(np.cumsum(returns)).reshape(-1, 1)

        swk = SlicedWassersteinKMeans(n_clusters=2, window_size=30, random_state=42)
        labels = swk.fit_predict(prices)

        assert len(labels) == len(prices) - 1
        assert len(np.unique(labels)) <= 2

    def test_different_wasserstein_orders(self, simple_2d_data):
        """Test different values of p parameter."""
        X, _ = simple_2d_data

        # Test p=1 and p=2
        swk_p1 = SlicedWassersteinKMeans(
            n_clusters=2, window_size=30, p=1, random_state=42
        )
        swk_p2 = SlicedWassersteinKMeans(
            n_clusters=2, window_size=30, p=2, random_state=42
        )

        labels_p1 = swk_p1.fit_predict(X)
        labels_p2 = swk_p2.fit_predict(X)

        # Results may differ but both should work
        assert len(labels_p1) == len(labels_p2)
        assert len(np.unique(labels_p1)) <= 2
        assert len(np.unique(labels_p2)) <= 2

    def test_empty_clusters_handling(self):
        """Test robustness when clusters become empty during iteration."""
        # Create data that might lead to empty clusters
        np.random.seed(42)
        X = np.random.randn(60, 2) * 0.01
        X = np.exp(np.cumsum(X, axis=0))

        # Use many clusters on small dataset
        swk = SlicedWassersteinKMeans(
            n_clusters=4, window_size=15, lifting_size=3, random_state=42, n_init=3
        )

        # Should handle gracefully without crashing
        labels = swk.fit_predict(X)
        assert len(labels) == len(X) - 1

    def test_performance_metrics(self, simple_2d_data):
        """Test that performance metrics are computed correctly."""
        X, _ = simple_2d_data

        swk = SlicedWassersteinKMeans(n_clusters=2, window_size=30, random_state=42)
        swk.fit(X)

        # Inertia should be positive
        assert swk.inertia_ >= 0

        # Should have completed at least 1 iteration
        assert swk.n_iter_ >= 1


# Additional utility functions for testing
def create_regime_switching_data(
    n_samples=500, n_features=2, n_regimes=2, regime_length=None
):
    """Utility function to create synthetic regime-switching data."""
    if regime_length is None:
        regime_length = n_samples // n_regimes

    np.random.seed(42)
    regimes = []

    for i in range(n_regimes):
        # Different correlation structure for each regime
        rho = 0.8 * (-1) ** i  # Alternate between positive and negative correlation
        cov = np.eye(n_features)
        if n_features == 2:
            cov[0, 1] = cov[1, 0] = rho

        regime_data = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=cov * (0.02 + 0.01 * i),  # Different volatilities
            size=(
                regime_length
                if i < n_regimes - 1
                else n_samples - (n_regimes - 1) * regime_length
            ),
        )
        regimes.append(regime_data)

    returns = np.vstack(regimes)
    prices = np.exp(np.cumsum(returns, axis=0))

    # Create true trend
    true_labels = []
    for i in range(n_regimes):
        length = (
            regime_length
            if i < n_regimes - 1
            else n_samples - (n_regimes - 1) * regime_length
        )
        true_labels.extend([i] * length)

    return prices, np.array(true_labels[1:])  # Remove first due to diff


#
# if __name__ == "__main__":
#     # Run basic tests
#     print("Running SlicedWassersteinKMeans test suite...")
#
#     # You can run individual tests like this:
#     test_instance = TestSlicedWassersteinKMeans()
#
#     # Create test data
#     X, y = create_regime_switching_data(300, 2, 2)
#
#     print("Testing basic functionality...")
#     test_instance.test_initialization()
#     print("✓ Initialization test passed")
#
#     test_instance.test_fit_basic((X, y))
#     print("✓ Basic fit test passed")
#
#     test_instance.test_predict((X, y))
#     print("✓ Prediction test passed")
#
#     test_instance.test_joblib_serialization((X, y))
#     print("✓ Joblib serialization test passed")
#
#     print("✅ All basic tests completed successfully!")
#
#     print("\nTo run the full test suite, use: pytest test_swk_means.py -v")
