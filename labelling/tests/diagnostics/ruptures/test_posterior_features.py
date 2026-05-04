"""Tests for spec §6 posterior summary features."""

from __future__ import annotations

import numpy as np
import pytest

from okmich_quant_labelling.diagnostics.ruptures import (
    UnivariateCost,
    cp_prob,
    entropy,
    expected_run_length,
    label_with_posteriors,
    map_run_length,
    mass_below_k,
    posterior_js_innovation,
)
from okmich_quant_labelling.diagnostics.ruptures.posterior_features import (
    _growth_predicted,
    _jsd,
)


def _delta_row(r_max: int, slot: int) -> np.ndarray:
    row = np.zeros(r_max, dtype=np.float64)
    row[slot] = 1.0
    return row


def _uniform_row(r_max: int) -> np.ndarray:
    return np.full(r_max, 1.0 / r_max, dtype=np.float64)


class TestCpProb:
    def test_extracts_first_column(self):
        post = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]], dtype=np.float64)
        np.testing.assert_array_equal(cp_prob(post), [0.7, 0.1])

    def test_returns_a_copy(self):
        post = np.array([[0.5, 0.5]], dtype=np.float64)
        out = cp_prob(post)
        out[0] = 999.0
        assert post[0, 0] == 0.5


class TestMapRunLength:
    def test_argmax_per_row(self):
        post = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]], dtype=np.float64)
        np.testing.assert_array_equal(map_run_length(post), [0, 1, 2])
        assert map_run_length(post).dtype == np.int64


class TestExpectedRunLength:
    def test_delta_at_slot_r_returns_r(self):
        for r in range(5):
            row = _delta_row(5, r)
            assert expected_run_length(row.reshape(1, -1))[0] == pytest.approx(float(r))

    def test_uniform_row_equals_mean_index(self):
        r_max = 8
        row = _uniform_row(r_max)
        out = expected_run_length(row.reshape(1, -1))
        assert out[0] == pytest.approx((r_max - 1) / 2.0)


class TestEntropy:
    def test_delta_has_zero_entropy(self):
        for r_max in [2, 5, 20]:
            for slot in range(r_max):
                out = entropy(_delta_row(r_max, slot).reshape(1, -1))
                assert out[0] == pytest.approx(0.0, abs=1e-12)

    def test_uniform_has_log_n_entropy(self):
        for r_max in [2, 5, 20]:
            row = _uniform_row(r_max)
            assert entropy(row.reshape(1, -1))[0] == pytest.approx(np.log(r_max), abs=1e-12)


class TestMassBelowK:
    def test_mass_below_k_sums_first_k_columns(self):
        post = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
        assert mass_below_k(post, 1)[0] == pytest.approx(0.1)
        assert mass_below_k(post, 3)[0] == pytest.approx(0.6)
        assert mass_below_k(post, 4)[0] == pytest.approx(1.0)

    @pytest.mark.parametrize("k", [0, -1, 5])
    def test_mass_below_k_rejects_out_of_range(self, k):
        post = np.zeros((1, 4), dtype=np.float64)
        post[0, 0] = 1.0
        with pytest.raises(ValueError):
            mass_below_k(post, k)


class TestPosteriorJsInnovation:
    def test_first_bar_is_nan(self):
        post = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float64)
        out = posterior_js_innovation(post, hazard_rate=0.1)
        assert np.isnan(out[0])
        assert np.isfinite(out[1])

    def test_growth_predicted_sums_to_one(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            r_max = rng.integers(2, 30)
            raw = rng.dirichlet(np.ones(r_max))
            for h in [0.01, 0.1, 0.5, 0.9]:
                g = _growth_predicted(raw, h)
                assert g.sum() == pytest.approx(1.0, abs=1e-12)
                assert (g >= 0).all()

    def test_growth_predicted_final_bin_folds_cap_mass(self):
        # If the cap holds non-zero mass, that mass must stay in the cap
        # multiplied by (1-h), in addition to mass advancing from R-2.
        p_prev = np.array([0.0, 0.0, 0.4, 0.6], dtype=np.float64)
        h = 0.2
        g = _growth_predicted(p_prev, h)
        assert g[0] == pytest.approx(h)
        assert g[1] == pytest.approx((1 - h) * 0.0)
        assert g[2] == pytest.approx((1 - h) * 0.0)
        assert g[3] == pytest.approx((1 - h) * (0.4 + 0.6))

    def test_jsd_zero_for_identical_distributions(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            p = rng.dirichlet(np.ones(8))
            assert _jsd(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_jsd_bounded_by_log_two(self):
        # JSD with natural log is bounded above by log(2).
        p = _delta_row(4, 0)
        q = _delta_row(4, 3)
        assert _jsd(p, q) <= np.log(2.0) + 1e-12

    def test_innovation_nonnegative_on_real_signal(self, two_shift_signal):
        from okmich_quant_ml.bocpd import NormalInverseGammaModel
        labeled = label_with_posteriors(
            signal=two_shift_signal,
            bocpd_observation_model=NormalInverseGammaModel(mu_0=0.0, kappa_0=1.0, alpha_0=1.0, beta_0=1.0),
            hazard_rate=1 / 80, r_max=120, cost_model=UnivariateCost.L2,
        )
        innov = posterior_js_innovation(labeled.posterior, hazard_rate=labeled.hazard_rate)
        assert np.all(innov[1:] >= -1e-12)
        assert np.all(np.isfinite(innov[1:]))


class TestPosteriorValidation:
    def test_one_d_input_rejected(self):
        with pytest.raises(ValueError, match="2-D"):
            cp_prob(np.array([0.5, 0.5]))

    def test_negative_entries_rejected(self):
        bad = np.array([[0.5, 0.6, -0.1]])
        with pytest.raises(ValueError, match="negative"):
            entropy(bad)

    def test_non_finite_entries_rejected(self):
        bad = np.array([[0.5, 0.5, np.nan]])
        with pytest.raises(ValueError, match="NaN or Inf"):
            cp_prob(bad)

    def test_r_max_below_two_rejected(self):
        with pytest.raises(ValueError, match="r_max"):
            cp_prob(np.array([[1.0]]))
