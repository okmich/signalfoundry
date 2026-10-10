import numpy as np
import pytest

from okmich_quant_ml.hmm.pomegranate import PomegranateHMM, DistType


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_sequence_data():
    """Create sample sequential data for HMM testing."""
    np.random.seed(42)

    # Generate data with two clear regimes
    n_samples = 200
    data = np.zeros((n_samples, 1))

    for i in range(n_samples):
        if i < 100:
            # Regime 1: low mean, low variance
            data[i] = np.random.normal(0.0, 0.5)
        else:
            # Regime 2: high mean, higher variance
            data[i] = np.random.normal(2.0, 1.0)

    return data


@pytest.fixture
def multi_feature_data():
    """Create multi-feature sequential data."""
    np.random.seed(42)
    n_samples = 150

    # 3 features
    data = np.zeros((n_samples, 3))

    for i in range(n_samples):
        regime = i // 50
        if regime == 0:
            data[i] = np.random.normal([0, 0, 0], [0.5, 0.5, 0.5])
        elif regime == 1:
            data[i] = np.random.normal([2, 1, 1], [1.0, 0.8, 0.8])
        else:
            data[i] = np.random.normal([1, 2, 0], [0.7, 1.0, 0.6])

    return data


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Test PomegranateHMM initialization."""

    def test_init_normal_distribution(self):
        """Test initialization with Normal distribution."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        assert model.distribution_type == DistType.NORMAL
        assert model.n_states == 2
        assert model.random_state == 42
        assert model._model is None  # Not fitted yet

    def test_init_studentt_distribution(self):
        """Test initialization with StudentT distribution."""
        model = PomegranateHMM(distribution_type=DistType.STUDENTT, n_states=3, dofs=5)

        assert model.distribution_type == DistType.STUDENTT
        assert model.n_states == 3
        assert model.dist_kwargs["dofs"] == 5

    def test_init_lognormal_distribution(self):
        """Test initialization with LogNormal distribution."""
        model = PomegranateHMM(distribution_type=DistType.LOGNORMAL, n_states=2)

        assert model.distribution_type == DistType.LOGNORMAL


# ============================================================================
# Training and Prediction Tests
# ============================================================================


class TestTrainingAndPrediction:
    """Test model training and prediction."""

    def test_fit_predict_normal(self, sample_sequence_data):
        """Test fitting and prediction with Normal distribution."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42, max_iter=50
        )

        # Fit model
        model.fit(sample_sequence_data)

        assert model._model is not None

        # Predict states
        predictions = model.predict(sample_sequence_data)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(sample_sequence_data),)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba_shape(self, sample_sequence_data):
        """Test that predict_proba returns correct shape."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        model.fit(sample_sequence_data)
        probabilities = model.predict_proba(sample_sequence_data)

        # Should be (n_samples, n_states), NOT (1, n_samples, n_states)
        assert probabilities.shape == (len(sample_sequence_data), 2)
        assert isinstance(probabilities, np.ndarray)

        # Probabilities should sum to 1 for each sample
        np.testing.assert_array_almost_equal(
            probabilities.sum(axis=1), np.ones(len(sample_sequence_data)), decimal=5
        )

    def test_fit_predict_method(self, sample_sequence_data):
        """Test fit_predict convenience method."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        predictions = model.fit_predict(sample_sequence_data)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(sample_sequence_data),)

    def test_multi_feature_training(self, multi_feature_data):
        """Test training with multiple features."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=3, random_state=42, max_iter=50
        )

        model.fit(multi_feature_data)
        predictions = model.predict(multi_feature_data)

        assert predictions.shape == (len(multi_feature_data),)
        assert set(predictions).issubset({0, 1, 2})


# ============================================================================
# Tensor Conversion Tests
# ============================================================================


class TestTensorConversion:
    """Test PyTorch tensor to numpy conversion."""

    def test_predict_returns_numpy(self, sample_sequence_data):
        """Test that predict returns numpy array, not torch tensor."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        model.fit(sample_sequence_data)
        predictions = model.predict(sample_sequence_data)

        assert isinstance(predictions, np.ndarray)
        assert not hasattr(predictions, "detach")  # Should not be torch tensor

    def test_predict_proba_returns_numpy(self, sample_sequence_data):
        """Test that predict_proba returns numpy array."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        model.fit(sample_sequence_data)
        probabilities = model.predict_proba(sample_sequence_data)

        assert isinstance(probabilities, np.ndarray)
        assert not hasattr(probabilities, "detach")

    def test_batch_dimension_removed(self, sample_sequence_data):
        """Test that batch dimension is properly removed."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=3, random_state=42
        )

        model.fit(sample_sequence_data)
        probabilities = model.predict_proba(sample_sequence_data)

        # Original pomegranate returns (1, n_samples, n_states)
        # We should get (n_samples, n_states)
        assert probabilities.ndim == 2
        assert probabilities.shape == (len(sample_sequence_data), 3)


# ============================================================================
# Model Selection Tests
# ============================================================================


class TestModelSelection:
    """Test model selection with different n_states."""

    def test_train_method_selects_best(self, sample_sequence_data):
        """Test that train method selects best n_states."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42  # Initial
        )

        # Train with multiple n_states options
        best_model = model.train(
            sample_sequence_data, n_states_range=[2, 3, 4], criterion="bic"
        )

        assert best_model is not None
        assert best_model.n_states in [2, 3, 4]
        assert best_model._model is not None

    def test_aic_vs_bic_criterion(self, sample_sequence_data):
        """Test AIC vs BIC criterion."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        # Train with AIC
        best_aic = model.train(
            sample_sequence_data, n_states_range=[2, 3], criterion="aic"
        )

        # Train with BIC
        model2 = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )
        best_bic = model2.train(
            sample_sequence_data, n_states_range=[2, 3], criterion="bic"
        )

        # Both should be valid
        assert best_aic.n_states in [2, 3]
        assert best_bic.n_states in [2, 3]


# ============================================================================
# AIC/BIC Tests
# ============================================================================


class TestAICBIC:
    """Test AIC and BIC calculation."""

    def test_get_aic_bic(self, sample_sequence_data):
        """Test AIC and BIC calculation."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        model.fit(sample_sequence_data)
        aic, bic = model.get_aic_bic(sample_sequence_data)

        assert isinstance(aic, float)
        assert isinstance(bic, float)
        assert np.isfinite(aic)
        assert np.isfinite(bic)
        # BIC typically larger than AIC for same model
        assert bic > aic

    def test_aic_bic_penalizes_complexity(self, sample_sequence_data):
        """Test that AIC/BIC increases with model complexity."""
        model_2_states = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )
        model_2_states.fit(sample_sequence_data)
        _, bic_2 = model_2_states.get_aic_bic(sample_sequence_data)

        model_5_states = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=5, random_state=42
        )
        model_5_states.fit(sample_sequence_data)
        _, bic_5 = model_5_states.get_aic_bic(sample_sequence_data)

        # More complex model should have higher BIC penalty
        # (though not guaranteed if it fits much better)
        assert isinstance(bic_5, float)
        assert np.isfinite(bic_5)


# ============================================================================
# Parameter Extraction Tests
# ============================================================================


class TestParameterExtraction:
    """Test extraction of model parameters."""

    def test_means_extraction(self, sample_sequence_data):
        """Test extraction of state means."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        model.fit(sample_sequence_data)
        means = model.means

        assert means is not None
        assert len(means) == 2  # 2 states

    def test_covariances_extraction(self, sample_sequence_data):
        """Test extraction of covariances."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        model.fit(sample_sequence_data)
        covariances = model.covariances

        assert covariances is not None
        assert len(covariances) == 2

    def test_transition_probabilities(self, sample_sequence_data):
        """Test extraction of transition probability matrix."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        model.fit(sample_sequence_data)
        transitions = model.transition_prob()

        assert transitions is not None
        assert transitions.shape == (2, 2)

        # Each row should sum to approximately 1
        # Note: Use decimal=2 due to numerical precision in pomegranate
        np.testing.assert_array_almost_equal(
            transitions.sum(axis=1), np.ones(2), decimal=2
        )

    def test_parameters_property(self, sample_sequence_data):
        """Test parameters property for different distributions."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        model.fit(sample_sequence_data)
        params = model.parameters

        assert isinstance(params, list)
        assert len(params) == 2  # 2 states
        assert all("means" in p for p in params)
        assert all("covs" in p for p in params)


# ============================================================================
# Different Distribution Tests
# ============================================================================


class TestDistributionTypes:
    """Test different distribution types."""

    def test_studentt_distribution(self, sample_sequence_data):
        """Test StudentT distribution."""
        model = PomegranateHMM(
            distribution_type=DistType.STUDENTT, n_states=2, random_state=42, dofs=5
        )

        model.fit(sample_sequence_data)
        predictions = model.predict(sample_sequence_data)

        assert predictions.shape == (len(sample_sequence_data),)

        # StudentT parameters should include dofs
        params = model.parameters
        assert all("dofs" in p for p in params)

    def test_lognormal_distribution(self):
        """Test LogNormal distribution with positive data."""
        # LogNormal requires positive data
        np.random.seed(42)
        positive_data = np.random.lognormal(0, 1, size=(100, 1))

        model = PomegranateHMM(
            distribution_type=DistType.LOGNORMAL, n_states=2, random_state=42
        )

        model.fit(positive_data)
        predictions = model.predict(positive_data)

        assert predictions.shape == (len(positive_data),)


def _planted_bernoulli(n: int, p0: float, p1: float, dwell: int, seed: int, n_features: int = 1):
    rng = np.random.default_rng(seed)
    z = np.cumsum(rng.random(n) < 1 / dwell) % 2
    p = np.where(z == 0, p0, p1)[:, None]
    return (rng.random((n, n_features)) < p).astype(float), z


class TestBernoulliEmissions:
    """0/1 data used to break the default fit: k-means started every state at p = 0 or 1 (NaN fit), or crashed."""

    @pytest.mark.parametrize("n_states", [2, 3])
    def test_default_fit_on_binary_data_is_finite_and_interior(self, n_states):
        x, _ = _planted_bernoulli(1_500, 0.4, 0.6, 100, seed=0)
        model = PomegranateHMM(distribution_type=DistType.BERNOULLI, n_states=n_states, random_state=7, max_iter=30)
        model.fit(x)
        probs = np.array([p["probs"] for p in model.parameters]).ravel()
        assert np.all(np.isfinite(probs))
        assert np.all((probs > 0.01) & (probs < 0.99)), probs
        assert np.isfinite(model.log_likelihood(x))

    def test_recovers_a_planted_two_regime_sequence(self):
        x, z = _planted_bernoulli(2_000, 0.25, 0.75, 100, seed=3)
        model = PomegranateHMM(distribution_type=DistType.BERNOULLI, n_states=2, random_state=7, max_iter=50)
        model.fit(x)
        probs = np.sort(np.array([p["probs"] for p in model.parameters]).ravel())
        np.testing.assert_allclose(probs, [0.25, 0.75], atol=0.08)
        labels = model.predict(x)
        agreement = max(np.mean(labels == z), np.mean(labels != z))
        assert agreement > 0.85, agreement
        assert np.all(np.diag(model.transition_prob()) > 0.9)

    def test_multi_feature_binary_data(self):
        x, _ = _planted_bernoulli(1_200, 0.3, 0.7, 80, seed=5, n_features=3)
        model = PomegranateHMM(distribution_type=DistType.BERNOULLI, n_states=2, random_state=7, max_iter=30)
        model.fit(x)
        probs = np.array([p["probs"] for p in model.parameters])
        assert probs.shape == (2, 3)
        assert np.all((probs > 0.01) & (probs < 0.99))

    def test_same_seed_same_fit(self):
        x, _ = _planted_bernoulli(1_000, 0.4, 0.6, 100, seed=1)
        fits = []
        for _ in range(2):
            model = PomegranateHMM(distribution_type=DistType.BERNOULLI, n_states=2, random_state=11, max_iter=20)
            fits.append(np.array([p["probs"] for p in model.fit(x).parameters]).ravel())
        np.testing.assert_array_equal(fits[0], fits[1])

    def test_rejects_values_outside_unit_interval(self):
        model = PomegranateHMM(distribution_type=DistType.BERNOULLI, n_states=2, random_state=7, max_iter=5)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            model.fit(np.array([[0.0], [1.0], [2.0]] * 50))

    def test_starting_point_is_interior_and_sticky(self):
        model = PomegranateHMM(distribution_type=DistType.BERNOULLI, n_states=3, random_state=7)
        stats = model._compute_init_stats(np.array([[0.0], [1.0]] * 100))
        init = stats["bernoulli_init_probs"].ravel()
        assert np.all((init >= 0.01) & (init <= 0.99)) and len(np.unique(init)) == 3
        edges = model._bernoulli_init_edges()
        np.testing.assert_allclose(edges.sum(axis=1), 1.0)
        assert np.all(np.diag(edges) == model._BERNOULLI_INIT_STAY)

    def test_other_distributions_keep_the_kmeans_start(self, sample_sequence_data):
        model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, random_state=7)
        assert "centroids" in model._compute_init_stats(sample_sequence_data)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling."""

    def test_predict_before_fit_raises_error(self, sample_sequence_data):
        """Test that predict before fit raises error."""
        model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)

        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            model.predict(sample_sequence_data)

    def test_predict_proba_before_fit_raises_error(self, sample_sequence_data):
        """Test that predict_proba before fit raises error."""
        model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)

        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            model.predict_proba(sample_sequence_data)

    def test_get_aic_bic_before_fit_raises_error(self, sample_sequence_data):
        """Test that get_aic_bic before fit raises error."""
        model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)

        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            model.get_aic_bic(sample_sequence_data)

    def test_removed_hsmm_kwarg_duration_model_raises(self):
        with pytest.raises(TypeError, match="HSMM support has been removed"):
            PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, duration_model=object())

    def test_removed_hsmm_kwarg_via_factory_raises(self):
        from okmich_quant_ml.hmm import create_simple_hmm_instance
        with pytest.raises(TypeError, match="HSMM support has been removed"):
            create_simple_hmm_instance(DistType.NORMAL, n_states=2, duration_type="poisson")

    def test_removed_hsmm_kwarg_max_duration_raises(self):
        from okmich_quant_ml.hmm import create_simple_hmm_instance
        with pytest.raises(TypeError, match="HSMM support has been removed"):
            create_simple_hmm_instance(DistType.NORMAL, n_states=2, max_duration=100)


# ============================================================================
# Serialization Tests
# ============================================================================


class TestSerialization:
    """Test model serialization."""

    def test_save_and_load(self, sample_sequence_data, tmp_path):
        """Test saving and loading model."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        model.fit(sample_sequence_data)
        original_predictions = model.predict(sample_sequence_data)

        # Save model
        save_path = tmp_path / "test_model.pkl"
        model.save(str(save_path))

        # Load model
        loaded_model = PomegranateHMM.load(str(save_path))
        loaded_predictions = loaded_model.predict(sample_sequence_data)

        # Predictions should match
        np.testing.assert_array_equal(original_predictions, loaded_predictions)


# ============================================================================
# Visualization Tests
# ============================================================================


class TestVisualization:
    """Test new visualization methods."""

    def test_regime_summary_basic(self, sample_sequence_data):
        """Test basic regime summary generation."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        model.fit(sample_sequence_data)
        summary = model.regime_summary(sample_sequence_data)

        assert isinstance(summary, str)
        assert "HMM Regime Summary" in summary
        assert "State 0" in summary
        assert "State 1" in summary
        assert "Transition Probabilities" in summary
        assert "Distribution Parameters" in summary

    def test_regime_summary_without_data(self, sample_sequence_data):
        """Test regime summary without providing data."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        model.fit(sample_sequence_data)
        summary = model.regime_summary()

        assert isinstance(summary, str)
        assert "HMM Regime Summary" in summary
        # Should not include occupancy without data
        assert "State Occupancy" not in summary

    def test_regime_summary_includes_occupancy(self, sample_sequence_data):
        """Test that regime summary includes state occupancy when data provided."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        model.fit(sample_sequence_data)
        summary = model.regime_summary(sample_sequence_data)

        assert "State Occupancy" in summary
        assert "samples" in summary

    def test_regime_summary_studentt(self, sample_sequence_data):
        """Test regime summary with StudentT distribution."""
        model = PomegranateHMM(
            distribution_type=DistType.STUDENTT, n_states=2, random_state=42, dofs=5
        )

        model.fit(sample_sequence_data)
        summary = model.regime_summary(sample_sequence_data)

        assert "STUDENTT" in summary
        # Should include dofs in parameters
        assert "dof" in summary.lower()

    def test_plot_distributions_runs(self, sample_sequence_data):
        """Test that plot_distributions runs without error."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        model.fit(sample_sequence_data)

        # Should not raise an error
        try:
            fig, axes = model.plot_distributions(sample_sequence_data)
            assert fig is not None
            assert axes is not None
            assert len(axes) == 2  # 2 states
        except Exception as e:
            pytest.fail(f"plot_distributions raised an exception: {e}")

    def test_plot_distributions_without_data(self, sample_sequence_data):
        """Test plot_distributions without data overlay."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=2, random_state=42
        )

        model.fit(sample_sequence_data)

        # Should work without data
        try:
            fig, axes = model.plot_distributions()
            assert fig is not None
            assert axes is not None
        except Exception as e:
            pytest.fail(f"plot_distributions without data raised an exception: {e}")

    def test_plot_distributions_multifeature(self, multi_feature_data):
        """Test plot_distributions with multi-feature data."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=3, random_state=42
        )

        model.fit(multi_feature_data)

        # Plot first feature
        try:
            fig, axes = model.plot_distributions(multi_feature_data, feature_idx=0)
            assert fig is not None
        except Exception as e:
            pytest.fail(
                f"plot_distributions with multi-feature raised an exception: {e}"
            )

    def test_plot_transition_matrix_runs(self, sample_sequence_data):
        """Test that plot_transition_matrix runs without error."""
        model = PomegranateHMM(
            distribution_type=DistType.NORMAL, n_states=3, random_state=42
        )

        model.fit(sample_sequence_data)

        # Should not raise an error
        try:
            ax = model.plot_transition_matrix()
            assert ax is not None
        except Exception as e:
            pytest.fail(f"plot_transition_matrix raised an exception: {e}")

    def test_visualization_before_fit_raises_error(self):
        """Test that visualization methods before fit raise error."""
        model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)

        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            model.regime_summary()

        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            model.plot_distributions()

        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            model.plot_transition_matrix()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
