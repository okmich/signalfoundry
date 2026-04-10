import numpy as np
import pytest

from okmich_quant_ml.hmm.pomegranate import DistType
from okmich_quant_ml.hmm.pomegranate_mm import PomegranateMixtureHMM


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_sequence_data():
    """Create sample sequential data for HMM testing."""
    np.random.seed(42)

    # Generate data with two clear regimes, each with multi-modal behavior
    n_samples = 200
    data = np.zeros((n_samples, 1))

    for i in range(n_samples):
        if i < 100:
            # Regime 1: bimodal (mixture of two modes)
            if i % 2 == 0:
                data[i] = np.random.normal(0.0, 0.5)
            else:
                data[i] = np.random.normal(1.0, 0.5)
        else:
            # Regime 2: bimodal with different centers
            if i % 2 == 0:
                data[i] = np.random.normal(3.0, 0.7)
            else:
                data[i] = np.random.normal(4.5, 0.7)

    return data


@pytest.fixture
def multi_feature_data():
    """Create multi-feature sequential data with mixtures."""
    np.random.seed(42)
    n_samples = 150

    # 2 features
    data = np.zeros((n_samples, 2))

    for i in range(n_samples):
        regime = i // 75
        if regime == 0:
            # Bimodal regime 1
            if i % 2 == 0:
                data[i] = np.random.normal([0, 0], [0.5, 0.5])
            else:
                data[i] = np.random.normal([1, 1], [0.5, 0.5])
        else:
            # Bimodal regime 2
            if i % 2 == 0:
                data[i] = np.random.normal([3, 2], [0.7, 0.7])
            else:
                data[i] = np.random.normal([4, 3], [0.7, 0.7])

    return data


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Test PomegranateMixtureHMM initialization."""

    def test_init_normal_mixture(self):
        """Test initialization with Normal mixture."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=3,
            random_state=42,
        )

        assert model.distribution_type == DistType.NORMAL
        assert model.n_states == 2
        assert model.n_components == 3
        assert model.random_state == 42
        assert model._model is None  # Not fitted yet

    def test_init_studentt_mixture(self):
        """Test initialization with StudentT mixture."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.STUDENTT, n_states=3, n_components=2, dofs=5
        )

        assert model.distribution_type == DistType.STUDENTT
        assert model.n_states == 3
        assert model.n_components == 2
        assert model.dist_kwargs["dofs"] == 5

    def test_init_lognormal_mixture(self):
        """Test initialization with LogNormal mixture."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.LOGNORMAL, n_states=2, n_components=2
        )

        assert model.distribution_type == DistType.LOGNORMAL
        assert model.n_components == 2


# ============================================================================
# Training and Prediction Tests
# ============================================================================


class TestTrainingAndPrediction:
    """Test model training and prediction."""

    def test_fit_predict_normal_mixture(self, sample_sequence_data):
        """Test fitting and prediction with Normal mixture."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
            max_iter=50,
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
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
        )

        model.fit(sample_sequence_data)
        probabilities = model.predict_proba(sample_sequence_data)

        # Should be (n_samples, n_states)
        assert probabilities.shape == (len(sample_sequence_data), 2)
        assert isinstance(probabilities, np.ndarray)

        # Probabilities should sum to 1 for each sample
        np.testing.assert_array_almost_equal(
            probabilities.sum(axis=1), np.ones(len(sample_sequence_data)), decimal=5
        )

    def test_fit_predict_method(self, sample_sequence_data):
        """Test fit_predict convenience method."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
        )

        predictions = model.fit_predict(sample_sequence_data)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(sample_sequence_data),)

    def test_multi_feature_training(self, multi_feature_data):
        """Test training with multiple features."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
            max_iter=50,
        )

        model.fit(multi_feature_data)
        predictions = model.predict(multi_feature_data)

        assert predictions.shape == (len(multi_feature_data),)
        assert set(predictions).issubset({0, 1})


# ============================================================================
# Model Selection Tests
# ============================================================================


class TestModelSelection:
    """Test model selection with different n_states."""

    def test_train_method_selects_best(self, sample_sequence_data):
        """Test that train method selects best n_states."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
        )

        # Train with multiple n_states options
        best_model = model.train(
            sample_sequence_data,
            n_states_range=[2, 3],
            n_criteria_range=[2],  # Fix n_components to preserve original value
            criterion="bic",
        )

        assert best_model is not None
        assert best_model.n_states in [2, 3]
        assert best_model._model is not None
        assert best_model.n_components == 2  # Should preserve n_components

    def test_aic_vs_bic_criterion(self, sample_sequence_data):
        """Test AIC vs BIC criterion."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
        )

        # Train with AIC
        best_aic = model.train(
            sample_sequence_data,
            n_states_range=[2, 3],
            n_criteria_range=[2],  # Fix n_components to preserve original value
            criterion="aic",
        )

        # Train with BIC
        model2 = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
        )
        best_bic = model2.train(
            sample_sequence_data,
            n_states_range=[2, 3],
            n_criteria_range=[2],  # Fix n_components to preserve original value
            criterion="bic",
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
        """Test AIC and BIC calculation for mixture models."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
        )

        model.fit(sample_sequence_data)
        aic, bic = model.get_aic_bic(sample_sequence_data)

        assert isinstance(aic, float)
        assert isinstance(bic, float)
        assert np.isfinite(aic)
        assert np.isfinite(bic)
        # BIC typically larger than AIC
        assert bic > aic

    def test_aic_bic_accounts_for_mixture_complexity(self, sample_sequence_data):
        """Test that AIC/BIC accounts for mixture component complexity."""
        # Model with 2 components
        model_2_comp = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
        )
        model_2_comp.fit(sample_sequence_data)
        _, bic_2 = model_2_comp.get_aic_bic(sample_sequence_data)

        # Model with 4 components (more complex)
        model_4_comp = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=4,
            random_state=42,
        )
        model_4_comp.fit(sample_sequence_data)
        _, bic_4 = model_4_comp.get_aic_bic(sample_sequence_data)

        # Both should be valid
        assert isinstance(bic_2, float)
        assert isinstance(bic_4, float)
        assert np.isfinite(bic_2)
        assert np.isfinite(bic_4)


# ============================================================================
# Mixture Parameter Extraction Tests
# ============================================================================


class TestMixtureParameters:
    """Test extraction of mixture-specific parameters."""

    def test_get_mixture_parameters_structure(self, sample_sequence_data):
        """Test structure of get_mixture_parameters output."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=3,
            random_state=42,
        )

        model.fit(sample_sequence_data)
        params = model.get_mixture_parameters()

        assert isinstance(params, list)
        assert len(params) == 2  # 2 states

        for state_idx, state_params in enumerate(params):
            assert "state" in state_params
            assert state_params["state"] == state_idx
            assert "weights" in state_params
            assert "components" in state_params
            assert len(state_params["components"]) == 3  # 3 components

    def test_mixture_weights_sum_to_one(self, sample_sequence_data):
        """Test that mixture weights sum to 1 for each state."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=3,
            random_state=42,
        )

        model.fit(sample_sequence_data)
        params = model.get_mixture_parameters()

        for state_params in params:
            weights = state_params["weights"]
            if weights is not None:
                np.testing.assert_almost_equal(weights.sum(), 1.0, decimal=5)

    def test_component_parameters_extracted(self, sample_sequence_data):
        """Test that component parameters are correctly extracted."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
        )

        model.fit(sample_sequence_data)
        params = model.get_mixture_parameters()

        for state_params in params:
            components = state_params["components"]
            for comp in components:
                assert "component" in comp
                assert "mean" in comp
                assert "cov" in comp
                # Check that extracted params are numpy arrays
                assert isinstance(comp["mean"], np.ndarray)
                assert isinstance(comp["cov"], np.ndarray)


# ============================================================================
# Transition Probability Tests
# ============================================================================


class TestTransitionProbabilities:
    """Test transition probability extraction."""

    def test_transition_probabilities(self, sample_sequence_data):
        """Test extraction of transition probability matrix."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
        )

        model.fit(sample_sequence_data)
        transitions = model.transition_prob()

        assert transitions is not None
        assert transitions.shape == (2, 2)

        # Each row should sum to approximately 1
        np.testing.assert_array_almost_equal(
            transitions.sum(axis=1), np.ones(2), decimal=2
        )


# ============================================================================
# Different Distribution Tests
# ============================================================================


class TestDistributionTypes:
    """Test different distribution types for mixtures."""

    def test_studentt_mixture(self, sample_sequence_data):
        """Test StudentT mixture."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.STUDENTT,
            n_states=2,
            n_components=2,
            random_state=42,
            dofs=5,
        )

        model.fit(sample_sequence_data)
        predictions = model.predict(sample_sequence_data)

        assert predictions.shape == (len(sample_sequence_data),)

        # StudentT components should include dofs
        params = model.get_mixture_parameters()
        for state_params in params:
            for comp in state_params["components"]:
                if "dof" in comp:
                    assert comp["dof"] is not None

    def test_lognormal_mixture(self):
        """Test LogNormal mixture with positive data."""
        # LogNormal requires positive data
        np.random.seed(42)
        positive_data = np.random.lognormal(0, 1, size=(100, 1))

        model = PomegranateMixtureHMM(
            distribution_type=DistType.LOGNORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
        )

        model.fit(positive_data)
        predictions = model.predict(positive_data)

        assert predictions.shape == (len(positive_data),)


# ============================================================================
# Visualization Tests
# ============================================================================


class TestVisualization:
    """Test visualization methods."""

    def test_regime_summary_basic(self, sample_sequence_data):
        """Test basic regime summary generation."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
        )

        model.fit(sample_sequence_data)
        summary = model.regime_summary(sample_sequence_data)

        assert isinstance(summary, str)
        assert "Mixture HMM Regime Summary" in summary
        assert "State 0" in summary
        assert "State 1" in summary
        assert "Transition Probabilities" in summary
        assert "Component 0" in summary
        assert "Component 1" in summary

    def test_regime_summary_includes_occupancy(self, sample_sequence_data):
        """Test that regime summary includes state occupancy."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
        )

        model.fit(sample_sequence_data)
        summary = model.regime_summary(sample_sequence_data)

        assert "State Occupancy" in summary
        assert "samples" in summary

    def test_plot_component_weights_runs(self, sample_sequence_data):
        """Test that plot_component_weights runs without error."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=3,
            random_state=42,
        )

        model.fit(sample_sequence_data)

        # Should not raise an error
        try:
            ax = model.plot_component_weights()
            assert ax is not None
        except Exception as e:
            pytest.fail(f"plot_component_weights raised an exception: {e}")

    def test_plot_distributions_runs(self, sample_sequence_data):
        """Test that plot_distributions runs without error."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
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


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling."""

    def test_predict_before_fit_raises_error(self, sample_sequence_data):
        """Test that predict before fit raises error."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL, n_states=2, n_components=2
        )

        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            model.predict(sample_sequence_data)

    def test_predict_proba_before_fit_raises_error(self, sample_sequence_data):
        """Test that predict_proba before fit raises error."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL, n_states=2, n_components=2
        )

        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            model.predict_proba(sample_sequence_data)

    def test_get_aic_bic_before_fit_raises_error(self, sample_sequence_data):
        """Test that get_aic_bic before fit raises error."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL, n_states=2, n_components=2
        )

        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            model.get_aic_bic(sample_sequence_data)

    def test_get_mixture_parameters_before_fit_raises_error(self):
        """Test that get_mixture_parameters before fit raises error."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL, n_states=2, n_components=2
        )

        with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
            model.get_mixture_parameters()


# ============================================================================
# Serialization Tests
# ============================================================================


class TestSerialization:
    """Test model serialization."""

    def test_save_and_load(self, sample_sequence_data, tmp_path):
        """Test saving and loading model."""
        model = PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=2,
            n_components=2,
            random_state=42,
        )

        model.fit(sample_sequence_data)
        original_predictions = model.predict(sample_sequence_data)

        # Save model
        save_path = tmp_path / "test_mixture_model.pkl"
        model.save(str(save_path))

        # Load model
        loaded_model = PomegranateMixtureHMM.load(str(save_path))
        loaded_predictions = loaded_model.predict(sample_sequence_data)

        # Predictions should match
        np.testing.assert_array_equal(original_predictions, loaded_predictions)


class TestFitRetryOnCholesky:
    """Test that fit() retries on Cholesky / covariance errors."""

    def test_fit_retry_recovers_from_cholesky_failure(self):
        """Near-collinear data should succeed via retry (was 22% failure rate without retry)."""
        rng = np.random.default_rng(2)  # seed=2 triggered Cholesky failure before retry
        x1 = rng.normal(0, 1, size=100)
        x2 = x1 + rng.normal(0, 0.01, size=100)
        x3 = rng.normal(0, 0.001, size=100)
        X = np.column_stack([x1, x2, x3])
        model = PomegranateMixtureHMM(distribution_type=DistType.NORMAL, n_states=3, n_components=2, random_state=2)
        model.fit(X)  # should not raise
        assert model._model is not None

    def test_fit_retry_preserves_random_state_on_success(self):
        """After successful fit (with or without retries), random_state is restored to original."""
        rng = np.random.default_rng(2)
        x1 = rng.normal(0, 1, size=100)
        x2 = x1 + rng.normal(0, 0.01, size=100)
        x3 = rng.normal(0, 0.001, size=100)
        X = np.column_stack([x1, x2, x3])
        original_seed = 2
        model = PomegranateMixtureHMM(distribution_type=DistType.NORMAL, n_states=3, n_components=2, random_state=original_seed)
        model.fit(X)
        # random_state must be restored to original for reproducibility
        assert model.random_state == original_seed
        assert model._model is not None
        # Should produce valid predictions
        labels = model.predict(X)
        assert labels.shape == (100,)
        assert np.all((labels >= 0) & (labels < 3))

    def test_fit_non_covariance_runtime_error_not_retried(self):
        """Non-covariance RuntimeErrors should propagate immediately, not be retried."""
        model = PomegranateMixtureHMM(distribution_type=DistType.NORMAL, n_states=2, n_components=2, random_state=42)
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(100, 2))
        from unittest.mock import MagicMock
        call_count = 0

        def patched_build():
            mock_model = MagicMock()
            def bad_fit(*a, **kw):
                nonlocal call_count
                call_count += 1
                raise RuntimeError("some unrelated bug")
            mock_model.fit = bad_fit
            return mock_model

        model._build_model = patched_build
        model._kmeans_stats = model._compute_kmeans_init(X)
        with pytest.raises(RuntimeError, match="some unrelated bug"):
            model.fit(X)
        assert call_count == 1

    def test_fit_non_covariance_attribute_error_not_retried(self):
        """Non-covariance AttributeErrors should propagate immediately, not be retried."""
        model = PomegranateMixtureHMM(distribution_type=DistType.NORMAL, n_states=2, n_components=2, random_state=42)
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(100, 2))
        from unittest.mock import MagicMock
        call_count = 0

        def patched_build():
            mock_model = MagicMock()
            def bad_fit(*a, **kw):
                nonlocal call_count
                call_count += 1
                raise AttributeError("'Foo' object has no attribute 'bar'")
            mock_model.fit = bad_fit
            return mock_model

        model._build_model = patched_build
        model._kmeans_stats = model._compute_kmeans_init(X)
        with pytest.raises(AttributeError, match="bar"):
            model.fit(X)
        assert call_count == 1


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
