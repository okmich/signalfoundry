"""Tests for ``label_with_posteriors`` — the single public entry point."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.bocpd import BayesianOnlineChangepointDetector, NormalInverseGammaModel

from okmich_quant_labelling.diagnostics.ruptures import (
    LabeledPosteriors,
    UnivariateCost,
    label_with_posteriors,
)


def _new_nig() -> NormalInverseGammaModel:
    return NormalInverseGammaModel(mu_0=0.0, kappa_0=1.0, alpha_0=1.0, beta_0=1.0)


class TestLabelWithPosteriorsHappyPath:
    def test_returns_labeled_posteriors_with_expected_shapes(self, mean_shift_signal):
        result = label_with_posteriors(
            signal=mean_shift_signal, bocpd_observation_model=_new_nig(),
            hazard_rate=1 / 60, r_max=200, cost_model=UnivariateCost.L2, penalty=10.0, min_size=5,
        )
        assert isinstance(result, LabeledPosteriors)
        assert result.segment_ids.shape == (mean_shift_signal.size,)
        assert result.posterior.shape == (mean_shift_signal.size, 200)
        assert result.posterior.dtype == np.float64
        assert result.breakpoints[-1] == mean_shift_signal.size

    def test_segment_ids_are_consecutive_starting_from_zero(self, mean_shift_signal):
        result = label_with_posteriors(
            signal=mean_shift_signal, bocpd_observation_model=_new_nig(),
            hazard_rate=1 / 60, r_max=200,
        )
        unique = np.unique(result.segment_ids)
        np.testing.assert_array_equal(unique, np.arange(unique.size))

    def test_pelt_recovers_known_mean_shift_under_l2(self, mean_shift_signal):
        result = label_with_posteriors(
            signal=mean_shift_signal, bocpd_observation_model=_new_nig(),
            hazard_rate=1 / 60, r_max=200, cost_model=UnivariateCost.L2, penalty=10.0,
        )
        interior = result.breakpoints[:-1]
        assert interior.size >= 1
        closest = interior[np.argmin(np.abs(interior - 200))]
        assert abs(closest - 200) <= 5

    def test_posterior_rows_are_simplex_valid(self, two_shift_signal):
        result = label_with_posteriors(
            signal=two_shift_signal, bocpd_observation_model=_new_nig(),
            hazard_rate=1 / 80, r_max=160,
        )
        np.testing.assert_allclose(result.posterior.sum(axis=1), 1.0, atol=1e-9)
        assert (result.posterior >= 0).all()
        assert np.all(np.isfinite(result.posterior))

    def test_pandas_series_input_is_supported(self, mean_shift_signal):
        series = pd.Series(mean_shift_signal)
        result = label_with_posteriors(
            signal=series, bocpd_observation_model=_new_nig(),
            hazard_rate=1 / 60, r_max=100,
        )
        assert result.segment_ids.shape == (series.size,)


class TestLabelWithPosteriorsCausalityContract:
    def test_posterior_matches_freshly_run_detector_on_signal(self, two_shift_signal):
        result = label_with_posteriors(
            signal=two_shift_signal, bocpd_observation_model=_new_nig(),
            hazard_rate=1 / 80, r_max=160, warm_up_signal=None,
        )
        det = BayesianOnlineChangepointDetector(observation_model=_new_nig(), hazard_rate=1 / 80, r_max=160)
        expected = det.batch(two_shift_signal)
        np.testing.assert_allclose(result.posterior, expected, atol=1e-12)

    def test_warm_up_signal_advances_state_before_signal(self, two_shift_signal):
        warm = two_shift_signal[:120]
        eval_window = two_shift_signal[120:]

        warmed = label_with_posteriors(
            signal=eval_window, bocpd_observation_model=_new_nig(),
            hazard_rate=1 / 80, r_max=160, warm_up_signal=warm,
        )
        cold = label_with_posteriors(
            signal=eval_window, bocpd_observation_model=_new_nig(),
            hazard_rate=1 / 80, r_max=160, warm_up_signal=None,
        )

        det = BayesianOnlineChangepointDetector(observation_model=_new_nig(), hazard_rate=1 / 80, r_max=160)
        det.batch(warm)
        expected_warm = det.batch(eval_window)
        np.testing.assert_allclose(warmed.posterior, expected_warm, atol=1e-12)
        assert warmed.warm_up_length == warm.size
        assert cold.warm_up_length == 0
        assert not np.allclose(warmed.posterior[0], cold.posterior[0])

    def test_pelt_segments_only_signal_under_warm_up(self, two_shift_signal):
        warm = two_shift_signal[:120]
        eval_window = two_shift_signal[120:]
        result = label_with_posteriors(
            signal=eval_window, bocpd_observation_model=_new_nig(),
            hazard_rate=1 / 80, r_max=80, warm_up_signal=warm,
        )
        assert result.segment_ids.shape == (eval_window.size,)
        assert result.breakpoints[-1] == eval_window.size

    def test_empty_warm_up_signal_treated_as_none(self, mean_shift_signal):
        result = label_with_posteriors(
            signal=mean_shift_signal, bocpd_observation_model=_new_nig(),
            hazard_rate=1 / 60, r_max=80, warm_up_signal=np.empty(0, dtype=np.float64),
        )
        assert result.warm_up_length == 0


class TestLabelWithPosteriorsMetadata:
    def test_metadata_is_preserved_on_artefact(self, mean_shift_signal):
        result = label_with_posteriors(
            signal=mean_shift_signal, bocpd_observation_model=_new_nig(),
            hazard_rate=1 / 60, r_max=80, cost_model=UnivariateCost.L2, penalty=12.5, min_size=7,
        )
        assert result.cost_model is UnivariateCost.L2
        assert result.penalty == 12.5
        assert result.hazard_rate == pytest.approx(1 / 60)
        assert result.r_max == 80
        assert result.min_size == 7
        assert result.observation_model_class == "NormalInverseGammaModel"


class TestLabelWithPosteriorsValidation:
    @pytest.mark.parametrize("cost", [UnivariateCost.L1, UnivariateCost.L2,
                                      UnivariateCost.NORMAL, UnivariateCost.RANK, UnivariateCost.RBF])
    def test_all_univariate_costs_accepted(self, mean_shift_signal, cost):
        label_with_posteriors(
            signal=mean_shift_signal, bocpd_observation_model=_new_nig(),
            hazard_rate=1 / 60, r_max=40, cost_model=cost, penalty=10.0,
        )

    def test_string_cost_model_rejected(self, mean_shift_signal):
        with pytest.raises(TypeError):
            label_with_posteriors(
                signal=mean_shift_signal, bocpd_observation_model=_new_nig(),
                hazard_rate=1 / 60, r_max=40, cost_model="l2",
            )

    @pytest.mark.parametrize("hazard", [-0.1, 0.0, 1.0, 1.1])
    def test_hazard_rate_outside_open_unit_interval_rejected(self, mean_shift_signal, hazard):
        with pytest.raises(ValueError, match="hazard_rate"):
            label_with_posteriors(
                signal=mean_shift_signal, bocpd_observation_model=_new_nig(),
                hazard_rate=hazard, r_max=40,
            )

    def test_r_max_below_two_rejected(self, mean_shift_signal):
        with pytest.raises(ValueError, match="r_max"):
            label_with_posteriors(
                signal=mean_shift_signal, bocpd_observation_model=_new_nig(),
                hazard_rate=1 / 60, r_max=1,
            )

    def test_non_positive_penalty_rejected(self, mean_shift_signal):
        with pytest.raises(ValueError, match="penalty"):
            label_with_posteriors(
                signal=mean_shift_signal, bocpd_observation_model=_new_nig(),
                hazard_rate=1 / 60, r_max=40, penalty=0.0,
            )

    def test_min_size_below_two_rejected(self, mean_shift_signal):
        with pytest.raises(ValueError, match="min_size"):
            label_with_posteriors(
                signal=mean_shift_signal, bocpd_observation_model=_new_nig(),
                hazard_rate=1 / 60, r_max=40, min_size=1,
            )

    def test_signal_below_min_size_rejected(self):
        with pytest.raises(ValueError, match="below min_size"):
            label_with_posteriors(
                signal=np.array([1.0, 2.0]), bocpd_observation_model=_new_nig(),
                hazard_rate=1 / 60, r_max=4, min_size=5,
            )

    def test_multidimensional_signal_rejected(self):
        sig = np.zeros((50, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="signal must be 1-D"):
            label_with_posteriors(
                signal=sig, bocpd_observation_model=_new_nig(),
                hazard_rate=1 / 60, r_max=40,
            )

    def test_non_finite_signal_rejected(self):
        sig = np.linspace(0, 1, 50)
        sig[10] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            label_with_posteriors(
                signal=sig, bocpd_observation_model=_new_nig(),
                hazard_rate=1 / 60, r_max=40,
            )

    def test_non_finite_warm_up_signal_rejected(self, mean_shift_signal):
        warm = np.array([0.0, np.inf, 1.0])
        with pytest.raises(ValueError, match="warm_up_signal contains NaN or Inf"):
            label_with_posteriors(
                signal=mean_shift_signal, bocpd_observation_model=_new_nig(),
                hazard_rate=1 / 60, r_max=40, warm_up_signal=warm,
            )

    def test_multidimensional_warm_up_signal_rejected(self, mean_shift_signal):
        warm = np.zeros((10, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="warm_up_signal must be 1-D"):
            label_with_posteriors(
                signal=mean_shift_signal, bocpd_observation_model=_new_nig(),
                hazard_rate=1 / 60, r_max=40, warm_up_signal=warm,
            )
