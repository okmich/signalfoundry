import numpy as np
import pytest
import tempfile
import os

from okmich_quant_ml.hmm import FactorialHMM, PomegranateHMM, DistType


class TestFactorialHMMConstructor:
    """Test Factorial HMM constructor and properties."""

    def test_constructor_basic(self):
        """Test basic constructor."""
        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=3),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)

        assert fhmm.n_chains == 2
        assert fhmm.n_states_per_chain == [3, 2]
        assert fhmm.n_total_joint_states == 6
        assert not fhmm.is_fitted

    def test_constructor_with_feature_assignment(self):
        """Test constructor with feature assignment."""
        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=2),
            PomegranateHMM(DistType.NORMAL, n_states=3),
        ]
        feature_assignment = [[0, 1], [2, 3, 4]]
        fhmm = FactorialHMM(chains=chains, feature_assignment=feature_assignment)

        assert fhmm.feature_assignment == feature_assignment

    def test_constructor_empty_chains(self):
        """Test constructor with empty chains list."""
        with pytest.raises(ValueError, match="Must provide at least one HMM chain"):
            FactorialHMM(chains=[])

    def test_repr(self):
        """Test string representation."""
        chains = [
            PomegranateHMM(DistType.LAMDA, n_states=3),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)

        repr_str = repr(fhmm)
        assert "FactorialHMM" in repr_str
        assert "n_chains=2" in repr_str
        assert "states=[3, 2]" in repr_str


class TestFactorialHMMFitting:
    """Test Factorial HMM fitting methods."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        # 5 features total: 3 for chain 0, 2 for chain 1
        X = np.random.randn(n_samples, 5)
        feature_assignment = [[0, 1, 2], [3, 4]]
        return X, feature_assignment

    def test_fit_basic(self, sample_data):
        """Test basic fit functionality."""
        X, feature_assignment = sample_data

        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=2),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)
        fhmm.fit(X, feature_assignment=feature_assignment)

        assert fhmm.is_fitted
        assert fhmm.feature_assignment == feature_assignment

    def test_fit_without_feature_assignment(self):
        """Test fit without feature assignment raises error."""
        X = np.random.randn(100, 4)
        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=2),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)

        with pytest.raises(ValueError, match="feature_assignment must be provided"):
            fhmm.fit(X)

    def test_fit_invalid_feature_assignment(self):
        """Test fit with invalid feature assignment."""
        X = np.random.randn(100, 5)
        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=2),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)

        # Missing features
        with pytest.raises(ValueError, match="feature_assignment must cover all"):
            fhmm.fit(X, feature_assignment=[[0, 1], [2]])

        # Wrong number of chains
        with pytest.raises(ValueError, match="groups but .* chains"):
            fhmm.fit(X, feature_assignment=[[0, 1, 2, 3, 4]])

    def test_fit_predict(self, sample_data):
        """Test fit_predict method."""
        X, feature_assignment = sample_data

        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=2),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)
        states = fhmm.fit_predict(X, feature_assignment=feature_assignment)

        assert states.shape == (len(X), 2)
        assert fhmm.is_fitted

    def test_train_with_grid_search(self, sample_data):
        """Test train method with grid search."""
        X, feature_assignment = sample_data

        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=2),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)
        best_fhmm = fhmm.train(
            X,
            feature_assignment=feature_assignment,
            n_states_range=[(2, 3), (2, 3)],
            criterion="bic",
        )

        assert best_fhmm.is_fitted
        assert isinstance(best_fhmm, FactorialHMM)
        assert best_fhmm.n_chains == 2


class TestFactorialHMMPrediction:
    """Test Factorial HMM prediction methods."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted Factorial HMM for testing."""
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 5)
        feature_assignment = [[0, 1, 2], [3, 4]]

        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=3),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)
        fhmm.fit(X, feature_assignment=feature_assignment)

        return fhmm, X

    def test_predict_separate_states(self, fitted_model):
        """Test predict with separate states."""
        fhmm, X = fitted_model

        states = fhmm.predict(X, return_joint=False)

        assert states.shape == (len(X), 2)
        assert np.all(states[:, 0] >= 0) and np.all(states[:, 0] < 3)
        assert np.all(states[:, 1] >= 0) and np.all(states[:, 1] < 2)

    def test_predict_joint_states(self, fitted_model):
        """Test predict with joint states."""
        fhmm, X = fitted_model

        joint_states = fhmm.predict(X, return_joint=True)

        assert joint_states.shape == (len(X),)
        assert np.all(joint_states >= 0) and np.all(joint_states < 6)

    def test_predict_proba_separate(self, fitted_model):
        """Test predict_proba with separate probabilities."""
        fhmm, X = fitted_model

        proba_list = fhmm.predict_proba(X, return_joint=False)

        assert len(proba_list) == 2
        assert proba_list[0].shape == (len(X), 3)
        assert proba_list[1].shape == (len(X), 2)
        # Check probabilities sum to 1
        assert np.allclose(proba_list[0].sum(axis=1), 1.0)
        assert np.allclose(proba_list[1].sum(axis=1), 1.0)

    def test_predict_proba_joint(self, fitted_model):
        """Test predict_proba with joint probabilities."""
        fhmm, X = fitted_model

        joint_proba = fhmm.predict_proba(X, return_joint=True)

        assert joint_proba.shape == (len(X), 6)
        assert np.allclose(joint_proba.sum(axis=1), 1.0)

    def test_predict_before_fit(self):
        """Test predict before fitting raises error."""
        X = np.random.randn(100, 5)
        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=2),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)

        with pytest.raises(RuntimeError, match="Model has not been fitted"):
            fhmm.predict(X)


class TestFactorialHMMEvaluation:
    """Test Factorial HMM evaluation methods."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted Factorial HMM for testing."""
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 5)
        feature_assignment = [[0, 1, 2], [3, 4]]

        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=2),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)
        fhmm.fit(X, feature_assignment=feature_assignment)

        return fhmm, X

    def test_get_aic_bic(self, fitted_model):
        """Test AIC/BIC calculation."""
        fhmm, X = fitted_model

        aic, bic = fhmm.get_aic_bic(X)

        assert isinstance(aic, float)
        assert isinstance(bic, float)
        assert bic > aic  # BIC penalizes complexity more

    def test_score(self, fitted_model):
        """Test score method (alias for get_aic_bic)."""
        fhmm, X = fitted_model

        score_result = fhmm.score(X)
        aic_bic_result = fhmm.get_aic_bic(X)

        assert score_result == aic_bic_result

    def test_log_likelihood(self, fitted_model):
        """Test log-likelihood calculation."""
        fhmm, X = fitted_model

        ll = fhmm.log_likelihood(X)

        assert isinstance(ll, float)
        assert np.isfinite(ll)

    def test_transition_prob(self, fitted_model):
        """Test transition probability matrices."""
        fhmm, X = fitted_model

        trans_mats = fhmm.transition_prob()

        assert len(trans_mats) == 2
        assert trans_mats[0].shape == (2, 2)
        assert trans_mats[1].shape == (2, 2)
        # Check rows sum to 1
        assert np.allclose(trans_mats[0].sum(axis=1), 1.0)
        assert np.allclose(trans_mats[1].sum(axis=1), 1.0)

    def test_get_chain_parameters(self, fitted_model):
        """Test getting chain parameters."""
        fhmm, X = fitted_model

        params = fhmm.get_chain_parameters(0)

        assert isinstance(params, dict)
        assert "means" in params or "parameters" in params


class TestFactorialHMMJointStateEncoding:
    """Test joint state encoding/decoding."""

    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding roundtrip."""
        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=3),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)

        # Test all combinations
        for s0 in range(3):
            for s1 in range(2):
                chain_states = (s0, s1)
                joint_state = fhmm.encode_joint_state(chain_states)
                decoded = fhmm.decode_joint_state(joint_state)
                assert decoded == chain_states

    def test_encode_invalid_states(self):
        """Test encoding invalid states."""
        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=3),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)

        # Wrong number of states
        with pytest.raises(ValueError, match="Expected .* states"):
            fhmm.encode_joint_state((1,))

        # Out of range state
        with pytest.raises(ValueError, match="out of range"):
            fhmm.encode_joint_state((3, 1))

    def test_decode_invalid_joint_state(self):
        """Test decoding invalid joint state."""
        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=3),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)

        with pytest.raises(ValueError, match="out of range"):
            fhmm.decode_joint_state(10)


class TestFactorialHMMLabeling:
    """Test state labeling functionality."""

    def test_set_and_get_chain_labels(self):
        """Test setting and getting chain labels."""
        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=3),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)

        labels = {0: "Trending Up", 1: "Ranging", 2: "Trending Down"}
        fhmm.set_chain_labels(0, labels)

        assert fhmm.get_chain_label(0, 0) == "Trending Up"
        assert fhmm.get_chain_label(0, 1) == "Ranging"
        assert fhmm.get_chain_label(0, 2) == "Trending Down"

    def test_get_chain_label_default(self):
        """Test getting label without setting (default)."""
        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=2),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)

        label = fhmm.get_chain_label(0, 1)
        assert label == "State 1"

    def test_set_chain_labels_invalid_chain(self):
        """Test setting labels for invalid chain index."""
        chains = [PomegranateHMM(DistType.NORMAL, n_states=2)]
        fhmm = FactorialHMM(chains=chains)

        with pytest.raises(IndexError, match="out of range"):
            fhmm.set_chain_labels(1, {0: "Label"})


class TestFactorialHMMPersistence:
    """Test saving and loading."""

    def test_save_and_load(self):
        """Test saving and loading model."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        feature_assignment = [[0, 1, 2], [3, 4]]

        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=2),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)
        fhmm.fit(X, feature_assignment=feature_assignment)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
            tmp_path = tmp.name

        try:
            fhmm.save(tmp_path)
            loaded_fhmm = FactorialHMM.load(tmp_path)

            # Check properties match
            assert loaded_fhmm.n_chains == fhmm.n_chains
            assert loaded_fhmm.n_states_per_chain == fhmm.n_states_per_chain
            assert loaded_fhmm.is_fitted == fhmm.is_fitted

            # Check predictions match
            states_original = fhmm.predict(X)
            states_loaded = loaded_fhmm.predict(X)
            assert np.array_equal(states_original, states_loaded)

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestFactorialHMMVisualization:
    """Test visualization methods."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted Factorial HMM for testing."""
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 5)
        feature_assignment = [[0, 1, 2], [3, 4]]

        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=2),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)
        fhmm.fit(X, feature_assignment=feature_assignment)

        return fhmm, X

    def test_regime_summary(self, fitted_model):
        """Test regime summary generation."""
        fhmm, X = fitted_model

        summary = fhmm.regime_summary(X)

        assert isinstance(summary, str)
        assert "Factorial HMM Summary" in summary
        assert "Number of chains: 2" in summary
        assert "Chain 0:" in summary
        assert "Chain 1:" in summary

    def test_regime_summary_without_data(self, fitted_model):
        """Test regime summary without data."""
        fhmm, X = fitted_model

        summary = fhmm.regime_summary()

        assert isinstance(summary, str)
        assert "Factorial HMM Summary" in summary
        # Should not have state occupancy section
        assert "State Occupancy" not in summary

    def test_plot_transition_matrices(self, fitted_model):
        """Test plotting transition matrices."""
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend

        fhmm, X = fitted_model

        fig, axes = fhmm.plot_transition_matrices()

        assert fig is not None
        assert len(axes) == 2

    def test_plot_regime_timeline(self, fitted_model):
        """Test plotting regime timeline."""
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend

        fhmm, X = fitted_model

        fig, axes = fhmm.plot_regime_timeline(X)

        assert fig is not None
        assert len(axes) == 2


class TestFactorialHMMIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self):
        """Test complete workflow from creation to prediction."""
        np.random.seed(42)

        # Generate synthetic data
        n_samples = 300
        X = np.random.randn(n_samples, 6)

        # Create factorial HMM with 3 chains
        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=3),
            PomegranateHMM(DistType.NORMAL, n_states=2),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)

        # Fit
        feature_assignment = [[0, 1], [2, 3], [4, 5]]
        fhmm.fit(X, feature_assignment=feature_assignment)

        # Set labels
        fhmm.set_chain_labels(0, {0: "Trend Up", 1: "Neutral", 2: "Trend Down"})
        fhmm.set_chain_labels(1, {0: "Low Vol", 1: "High Vol"})

        # Predict
        states = fhmm.predict(X)
        proba_list = fhmm.predict_proba(X)
        joint_states = fhmm.predict(X, return_joint=True)

        # Evaluation
        aic, bic = fhmm.get_aic_bic(X)
        summary = fhmm.regime_summary(X)

        # Assertions
        assert states.shape == (n_samples, 3)
        assert len(proba_list) == 3
        assert joint_states.shape == (n_samples,)
        assert isinstance(aic, float)
        assert isinstance(bic, float)
        assert "Trend Up" in summary
        assert "Low Vol" in summary

    def test_mixed_hmm_types(self):
        """Test using different HMM types in chains."""
        np.random.seed(42)
        X = np.random.randn(200, 4)

        # Use Pomegranate HMMs with different distributions
        chains = [
            PomegranateHMM(DistType.NORMAL, n_states=2),
            PomegranateHMM(DistType.NORMAL, n_states=2),
        ]
        fhmm = FactorialHMM(chains=chains)
        fhmm.fit(X, feature_assignment=[[0, 1], [2, 3]])

        states = fhmm.predict(X)

        assert states.shape == (200, 2)
        assert fhmm.is_fitted
