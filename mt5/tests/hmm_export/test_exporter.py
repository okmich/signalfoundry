"""The generated header is the only thing MQL5 sees - it has to be exactly right."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from okmich_quant_mt5.hmm_export import (FeatureSpec, Mql5Feature, StateOrder, build_params, params_from_model,
                                         params_from_mqh, params_hash, parse_mqh, write_mqh)

from .conftest import COVARIANCES, MEANS, SPEC, STARTS, TRANSITIONS


class FakeDistribution:
    def __init__(self, means, covs):
        self.means = np.asarray(means)
        self.covs = np.asarray(covs)


class FakeDenseHMM:
    def __init__(self):
        self.starts = STARTS
        self.edges = TRANSITIONS
        self.distributions = [FakeDistribution(MEANS[k], COVARIANCES[k]) for k in range(3)]


class FakeDistType:
    name = "NORMAL"


class FakePomegranateHMM:
    """Duck-type of the attributes the exporter reads off a fitted PomegranateHMM."""

    def __init__(self, dist_name="NORMAL"):
        self._model = FakeDenseHMM()
        self.distribution_type = type("D", (), {"name": dist_name})


class TestEmissionConstants:
    def test_reproduce_scipy_gaussian_logpdf(self, params):
        """log_norm - 0.5*q must equal the real multivariate normal log-density."""
        rng = np.random.default_rng(1)
        for _ in range(20):
            x = MEANS[1] + rng.normal(0.0, 0.3, 2) * np.array([1.0, 1e-4])
            expected = [multivariate_normal(MEANS[k], COVARIANCES[k]).logpdf(x) for k in range(3)]
            np.testing.assert_allclose(params.log_emissions(x), expected, rtol=1e-10)

    def test_sigma_inv_is_the_inverse(self, params):
        for k in range(params.n_states):
            np.testing.assert_allclose(params.sigma_inv[k] @ COVARIANCES[k], np.eye(2), atol=1e-8)

    def test_rejects_non_positive_definite_covariance(self):
        singular = np.array([[[1.0, 1.0], [1.0, 1.0]]] * 3)
        with pytest.raises(ValueError, match="not positive definite"):
            build_params(model_id="bad", spec=SPEC, starts=STARTS, edges=TRANSITIONS, means=MEANS,
                         covariances=singular)

    def test_accepts_diagonal_covariance(self):
        diag = np.array([[0.9, 4.0e-8], [0.35, 1.0e-8], [1.1, 6.0e-8]])
        built = build_params(model_id="diag", spec=SPEC, starts=STARTS, edges=TRANSITIONS, means=MEANS,
                            covariances=diag, state_order=StateOrder.NONE)

        assert built.sigma_inv.shape == (3, 2, 2)
        np.testing.assert_allclose(built.sigma_inv[0], np.diag(1.0 / diag[0]), rtol=1e-10)


class TestLogParameterHandling:
    def test_probabilities_are_logged(self, params):
        np.testing.assert_allclose(np.exp(params.log_pi), STARTS, rtol=1e-12)
        np.testing.assert_allclose(np.exp(params.log_a), TRANSITIONS, rtol=1e-12)

    def test_already_logged_input_is_left_alone(self):
        log_a = np.log(TRANSITIONS)
        built = build_params(model_id="m", spec=SPEC, starts=np.log(STARTS), edges=log_a, means=MEANS,
                            covariances=COVARIANCES, state_order=StateOrder.NONE)
        np.testing.assert_allclose(built.log_a, log_a, rtol=1e-12)

    def test_zero_probability_is_floored_not_infinite(self):
        """Matches the 1e-300 floor in _extract_hmm_parameters, so MQL5 never sees -inf."""
        edges = np.array([[1.0, 0.0, 0.0], [0.05, 0.90, 0.05], [0.03, 0.07, 0.90]])
        built = build_params(model_id="m", spec=SPEC, starts=STARTS, edges=edges, means=MEANS,
                            covariances=COVARIANCES, state_order=StateOrder.NONE)

        assert np.all(np.isfinite(built.log_a))
        assert built.log_a[0, 1] == pytest.approx(np.log(1e-300))


class TestStateOrdering:
    def test_by_feature_mean_sorts_ascending(self):
        built = build_params(model_id="m", spec=SPEC, starts=STARTS, edges=TRANSITIONS, means=MEANS,
                            covariances=COVARIANCES, state_order=StateOrder.BY_FEATURE_MEAN, order_feature=0)

        assert list(built.mu[:, 0]) == sorted(MEANS[:, 0])

    def test_permutation_is_applied_consistently(self):
        """Relabelling must move pi, both axes of A, means and covariances together."""
        shuffled = np.array([2, 0, 1])
        built = build_params(model_id="m", spec=SPEC, starts=STARTS[shuffled], edges=TRANSITIONS[np.ix_(shuffled, shuffled)],
                            means=MEANS[shuffled], covariances=COVARIANCES[shuffled],
                            state_order=StateOrder.BY_FEATURE_MEAN, order_feature=0)
        canonical = build_params(model_id="m", spec=SPEC, starts=STARTS, edges=TRANSITIONS, means=MEANS,
                                covariances=COVARIANCES, state_order=StateOrder.BY_FEATURE_MEAN, order_feature=0)

        np.testing.assert_allclose(built.log_pi, canonical.log_pi, rtol=1e-12)
        np.testing.assert_allclose(built.log_a, canonical.log_a, rtol=1e-12)
        np.testing.assert_allclose(built.mu, canonical.mu, rtol=1e-12)
        np.testing.assert_allclose(built.sigma_inv, canonical.sigma_inv, rtol=1e-10)

    def test_none_preserves_the_fitted_order(self, params):
        np.testing.assert_allclose(params.mu, MEANS, rtol=1e-12)

    def test_rejects_out_of_range_order_feature(self):
        with pytest.raises(ValueError, match="order_feature"):
            build_params(model_id="m", spec=SPEC, starts=STARTS, edges=TRANSITIONS, means=MEANS,
                         covariances=COVARIANCES, order_feature=9)


class TestValidation:
    def test_rejects_feature_count_mismatch(self):
        spec = FeatureSpec(names=(Mql5Feature.MACD_ATR,))
        with pytest.raises(ValueError, match="declares 1 features"):
            build_params(model_id="m", spec=spec, starts=STARTS, edges=TRANSITIONS, means=MEANS,
                         covariances=COVARIANCES)

    def test_rejects_transition_matrix_shape_mismatch(self):
        with pytest.raises(ValueError, match=r"edges must have shape \(3, 3\)"):
            build_params(model_id="m", spec=SPEC, starts=STARTS, edges=np.eye(2), means=MEANS,
                         covariances=COVARIANCES)

    def test_rejects_non_positive_standardisation_scale(self):
        with pytest.raises(ValueError, match="feat_sd must be strictly positive"):
            build_params(model_id="m", spec=SPEC, starts=STARTS, edges=TRANSITIONS, means=MEANS,
                         covariances=COVARIANCES, feat_sd=np.array([1.0, 0.0]))


class TestParamsFromModel:
    def test_reads_a_fitted_model(self):
        built = params_from_model(FakePomegranateHMM(), SPEC, "from_model", state_order=StateOrder.NONE)

        np.testing.assert_allclose(built.mu, MEANS, rtol=1e-12)
        np.testing.assert_allclose(np.exp(built.log_a), TRANSITIONS, rtol=1e-12)

    def test_rejects_non_gaussian_emissions(self):
        with pytest.raises(NotImplementedError, match="Gaussian emissions only"):
            params_from_model(FakePomegranateHMM(dist_name="STUDENTT"), SPEC, "m")

    def test_rejects_unfitted_model(self):
        unfitted = FakePomegranateHMM()
        unfitted._model = None
        with pytest.raises(TypeError, match="call fit\\(\\) before exporting"):
            params_from_model(unfitted, SPEC, "m")


class TestHeaderGeneration:
    def test_round_trips_through_the_generated_header(self, params, tmp_path):
        path = write_mqh(params, tmp_path / "HmmParams.mqh")
        restored = params_from_mqh(path)

        np.testing.assert_allclose(restored.log_pi, params.log_pi, rtol=1e-15)
        np.testing.assert_allclose(restored.log_a, params.log_a, rtol=1e-15)
        np.testing.assert_allclose(restored.mu, params.mu, rtol=1e-15)
        np.testing.assert_allclose(restored.sigma_inv, params.sigma_inv, rtol=1e-15)
        np.testing.assert_allclose(restored.log_norm, params.log_norm, rtol=1e-15)
        assert restored.spec == params.spec
        assert restored.model_id == params.model_id

    def test_round_trip_is_bit_exact(self, params, tmp_path):
        """repr() round-trips float64 exactly; anything less would silently shift emissions."""
        restored = params_from_mqh(write_mqh(params, tmp_path / "HmmParams.mqh"))
        assert params_hash(restored) == params_hash(params)

    def test_emits_the_symbols_the_mqh_side_references(self, params, tmp_path):
        text = (write_mqh(params, tmp_path / "HmmParams.mqh")).read_text(encoding="ascii")

        for symbol in ("HMM_K", "HMM_D", "HMM_MODEL_ID", "HMM_FEATURE_NAMES", "HMM_MACD_FAST", "HMM_MACD_SLOW",
                       "HMM_MACD_SIGNAL", "HMM_ATR_PERIOD", "HMM_LOG_PI", "HMM_LOG_A", "HMM_MU", "HMM_SIGINV",
                       "HMM_LOGNORM", "HMM_FEAT_MEAN", "HMM_FEAT_SD"):
            assert symbol in text, f"generated header is missing {symbol}"

        assert "#ifndef HMM_PARAMS_MQH" in text and "#endif" in text
        assert "DO NOT EDIT" in text

    def test_array_lengths_match_the_declared_dimensions(self, params, tmp_path):
        raw = parse_mqh(write_mqh(params, tmp_path / "HmmParams.mqh"))
        k, d = raw["HMM_K"], raw["HMM_D"]

        assert raw["HMM_LOG_PI"].size == k
        assert raw["HMM_LOG_A"].size == k * k
        assert raw["HMM_MU"].size == k * d
        assert raw["HMM_SIGINV"].size == k * d * d
        assert raw["HMM_LOGNORM"].size == k
        assert len(raw["HMM_FEATURE_NAMES"]) == d

    def test_header_is_ascii_only(self, params, tmp_path):
        """MQL5 consoles are cp1252; a stray non-ASCII glyph breaks the build or the log."""
        data = (write_mqh(params, tmp_path / "HmmParams.mqh")).read_bytes()
        assert all(byte < 128 for byte in data)

    def test_feature_periods_travel_with_the_model(self, tmp_path):
        spec = FeatureSpec(names=SPEC.names, macd_fast=5, macd_slow=13, macd_signal=3, atr_period=21)
        built = build_params(model_id="m", spec=spec, starts=STARTS, edges=TRANSITIONS, means=MEANS,
                            covariances=COVARIANCES)
        raw = parse_mqh(write_mqh(built, tmp_path / "HmmParams.mqh"))

        assert (raw["HMM_MACD_FAST"], raw["HMM_MACD_SLOW"]) == (5, 13)
        assert (raw["HMM_MACD_SIGNAL"], raw["HMM_ATR_PERIOD"]) == (3, 21)

    def test_warns_on_price_scaled_features(self, tmp_path, capsys):
        spec = FeatureSpec(names=(Mql5Feature.MACD, Mql5Feature.ATR))
        built = build_params(model_id="m", spec=spec, starts=STARTS, edges=TRANSITIONS, means=MEANS,
                            covariances=COVARIANCES)
        write_mqh(built, tmp_path / "HmmParams.mqh")

        assert "price-scaled features" in capsys.readouterr().out

    def test_hash_changes_when_a_parameter_changes(self, params):
        nudged = build_params(model_id="m", spec=SPEC, starts=STARTS, edges=TRANSITIONS, means=MEANS * 1.001,
                             covariances=COVARIANCES, state_order=StateOrder.NONE)
        assert params_hash(nudged) != params_hash(params)
