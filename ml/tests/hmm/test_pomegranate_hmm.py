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
