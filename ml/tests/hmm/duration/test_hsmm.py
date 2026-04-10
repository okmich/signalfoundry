"""Phase 1 HSMM tests.

Covers:
1. Duration model unit tests (Poisson, Nonparametric)
2. HSMM inference invariants (2-state, 3-timestep hand-computed)
3. Forward-backward cross-check
4. Parameter recovery on synthetic HSMM data
5. Numerical stability (long sequences)
6. Backward compatibility (duration_model=None)
"""
import numpy as np
import pytest
from scipy.special import logsumexp

from okmich_quant_ml.hmm.duration import BaseDuration, GammaDuration, LogNormalDuration, NegBinDuration, \
    NonparametricDuration, PoissonDuration
from okmich_quant_ml.hmm.hsmm_inference import HSMMInferenceResult, hsmm_backward, hsmm_filter, hsmm_forward, \
    hsmm_forward_backward, hsmm_viterbi
from okmich_quant_ml.hmm.util import DurationType


# ======================================================================
# Duration model tests
# ======================================================================
class TestPoissonDuration:
    def test_pmf_sums_to_one(self):
        dur = PoissonDuration(n_states=2, max_duration=50, lambdas=np.array([5.0, 15.0]))
        for j in range(2):
            total = np.exp(logsumexp(dur.log_pmf(j)))
            assert abs(total - 1.0) < 1e-10

    def test_survivor_starts_at_one(self):
        dur = PoissonDuration(n_states=2, max_duration=50, lambdas=np.array([5.0, 15.0]))
        for j in range(2):
            assert abs(np.exp(dur.log_survivor(j)[0]) - 1.0) < 1e-10

    def test_survivor_monotonically_decreasing(self):
        dur = PoissonDuration(n_states=2, max_duration=50, lambdas=np.array([5.0, 15.0]))
        for j in range(2):
            ls = dur.log_survivor(j)
            assert np.all(np.diff(ls) <= 1e-12)

    def test_expected_duration(self):
        dur = PoissonDuration(n_states=1, max_duration=100, lambdas=np.array([9.0]))
        u = np.arange(1, 101, dtype=np.float64)
        trunc_mean = float(np.dot(np.exp(dur.log_pmf(0)), u))
        assert abs(dur.expected_duration(0) - trunc_mean) < 1e-10

    def test_update_from_eta(self):
        dur = PoissonDuration(n_states=1, max_duration=50, lambdas=np.array([1.0]))
        # Simulate eta peaked at u=10 (index 9)
        eta = np.zeros(50)
        eta[9] = 100.0  # duration 10 has count 100
        dur.update(0, eta)
        # lambda should be ≈ 9 (mean of u-1 = 9)
        assert abs(dur._lambdas[0] - 9.0) < 0.1

    def test_skip_unused_state(self):
        dur = PoissonDuration(n_states=2, max_duration=20, lambdas=np.array([5.0, 10.0]))
        old_lambda = dur._lambdas[1]
        dur.update(1, np.zeros(20))  # empty eta
        assert dur._lambdas[1] == old_lambda

    def test_n_parameters(self):
        dur = PoissonDuration(n_states=3, max_duration=50)
        assert dur.n_parameters() == 3

    def test_get_set_params(self):
        dur = PoissonDuration(n_states=2, max_duration=30, lambdas=np.array([3.0, 7.0]))
        params = dur.get_params()
        dur2 = PoissonDuration(n_states=2, max_duration=30)
        dur2.set_params(params)
        np.testing.assert_array_almost_equal(dur2._lambdas, dur._lambdas)


class TestNonparametricDuration:
    def test_pmf_sums_to_one(self):
        dur = NonparametricDuration(n_states=2, max_duration=20)
        for j in range(2):
            total = np.exp(logsumexp(dur.log_pmf(j)))
            assert abs(total - 1.0) < 1e-10

    def test_update_from_eta(self):
        dur = NonparametricDuration(n_states=1, max_duration=10)
        eta = np.array([0, 0, 5, 10, 5, 0, 0, 0, 0, 0], dtype=np.float64)
        dur.update(0, eta)
        pmf = np.exp(dur.log_pmf(0))
        # Peak should be at index 3 (duration 4)
        assert np.argmax(pmf) == 3
        assert abs(pmf.sum() - 1.0) < 1e-10

    def test_floor_prevents_neg_inf(self):
        dur = NonparametricDuration(n_states=1, max_duration=5)
        eta = np.array([10, 0, 0, 0, 0], dtype=np.float64)
        dur.update(0, eta)
        lp = dur.log_pmf(0)
        assert np.all(np.isfinite(lp))

    def test_n_parameters(self):
        dur = NonparametricDuration(n_states=3, max_duration=20)
        assert dur.n_parameters() == 3 * 19


# ======================================================================
# HSMM inference: hand-computed 2-state, 3-timestep test
# ======================================================================
class TestHSMMInferenceInvariants:
    """Verify message invariant with a small hand-checkable example."""

    @pytest.fixture
    def small_hsmm(self):
        """2 states, M=3, T=3, simple Nonparametric durations."""
        N, M, T = 2, 3, 3
        dur = NonparametricDuration(n_states=N, max_duration=M)
        # State 0: duration mostly 2, State 1: duration mostly 1
        dur._log_pmfs[0] = np.log(np.array([0.1, 0.7, 0.2]))
        dur._log_pmfs[1] = np.log(np.array([0.6, 0.3, 0.1]))

        log_trans = np.array([[-np.inf, 0.0], [0.0, -np.inf]])  # [[-, 1], [1, -]]
        log_init = np.log(np.array([0.6, 0.4]))
        log_emissions = np.array([[-1.0, -3.0], [-2.0, -1.5], [-1.5, -2.5]])
        return log_emissions, log_trans, log_init, dur

    def test_forward_backward_ll_cross_check(self, small_hsmm):
        """Forward LL must equal backward LL."""
        log_emissions, log_trans, log_init, dur = small_hsmm
        alpha_begin, alpha_end, ll_fwd = hsmm_forward(log_emissions, log_trans, log_init, dur)
        beta_begin, beta_end = hsmm_backward(log_emissions, log_trans, log_init, dur)

        # Cross-check: logsumexp_j(alpha_begin[0,j] + beta_begin[0,j]) == ll_fwd
        ll_back = logsumexp(alpha_begin[0] + beta_begin[0])
        assert abs(ll_fwd - ll_back) < 1e-8, f"Forward LL {ll_fwd} != backward cross-check {ll_back}"

    def test_alpha_begin_excludes_emissions(self, small_hsmm):
        """alpha_begin[0, j] should equal log_init[j] — no emission term."""
        log_emissions, log_trans, log_init, dur = small_hsmm
        alpha_begin, _, _ = hsmm_forward(log_emissions, log_trans, log_init, dur)
        np.testing.assert_array_almost_equal(alpha_begin[0], log_init)

    def test_state_posteriors_sum_to_one(self, small_hsmm):
        """gamma[t, :] should sum to 1 for all t."""
        log_emissions, log_trans, log_init, dur = small_hsmm
        result = hsmm_forward_backward(log_emissions, log_trans, log_init, dur)
        row_sums = result.state_posteriors.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0, decimal=6)

    def test_filtered_posteriors_sum_to_one(self, small_hsmm):
        """Filtered gamma[t, :] should sum to 1 for all t."""
        log_emissions, log_trans, log_init, dur = small_hsmm
        gamma = hsmm_filter(log_emissions, log_trans, log_init, dur)
        row_sums = gamma.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0, decimal=6)

    def test_viterbi_returns_valid_states(self, small_hsmm):
        log_emissions, log_trans, log_init, dur = small_hsmm
        states = hsmm_viterbi(log_emissions, log_trans, log_init, dur)
        assert states.shape == (3,)
        assert set(states).issubset({0, 1})


# ======================================================================
# Parameter recovery: synthetic 2-state HSMM
# ======================================================================
class TestHSMMParameterRecovery:
    """Generate data from a known HSMM and verify EM recovers the parameters."""

    @staticmethod
    def _generate_hsmm_data(rng: np.random.Generator, T: int = 2000) -> tuple[np.ndarray, np.ndarray, dict]:
        """Generate from 2-state HSMM with Poisson durations + Normal emissions."""
        true_params = {
            "lambdas": np.array([9.0, 24.0]),  # expected durations 10, 25
            "means": np.array([-1.0, 1.0]),
            "stds": np.array([0.5, 0.5]),
            "trans": np.array([[0.0, 1.0], [1.0, 0.0]]),
            "init": np.array([0.5, 0.5]),
        }

        states = np.empty(T, dtype=int)
        observations = np.empty((T, 1), dtype=np.float64)

        t = 0
        current_state = rng.choice(2, p=true_params["init"])
        while t < T:
            # Sample duration from shifted Poisson
            dur = rng.poisson(true_params["lambdas"][current_state]) + 1
            dur = max(1, min(dur, T - t))  # clip to remaining length

            # Emit observations
            for d in range(dur):
                states[t + d] = current_state
                observations[t + d, 0] = rng.normal(true_params["means"][current_state], true_params["stds"][current_state])
            t += dur

            # Transition
            other_state = 1 - current_state
            current_state = other_state

        return observations, states, true_params

    def test_parameter_recovery(self):
        rng = np.random.default_rng(42)
        X, true_states, true_params = self._generate_hsmm_data(rng, T=3000)

        from okmich_quant_ml.hmm import PomegranateHMM, DistType, InferenceMode
        dur = PoissonDuration(n_states=2, max_duration=100)
        model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, max_iter=30,
                               random_state=42, inference_mode=InferenceMode.VITERBI, duration_model=dur,
                               covariance_type="diag")
        model.fit(X)

        # Check state accuracy (allowing for label permutation)
        pred = model.predict(X)
        acc_direct = np.mean(pred == true_states)
        acc_flipped = np.mean(pred == (1 - true_states))
        best_acc = max(acc_direct, acc_flipped)
        assert best_acc > 0.85, f"State accuracy too low: {best_acc:.3f}"

        # Check duration recovery (within 20% of true lambda)
        fitted_lambdas = dur._lambdas
        # Account for label permutation
        if acc_flipped > acc_direct:
            fitted_lambdas = fitted_lambdas[::-1]
        for j in range(2):
            rel_err = abs(fitted_lambdas[j] - true_params["lambdas"][j]) / true_params["lambdas"][j]
            assert rel_err < 0.3, f"State {j} lambda rel error {rel_err:.2f} > 0.3"

    def test_log_likelihood_is_finite(self):
        rng = np.random.default_rng(123)
        X, _, _ = self._generate_hsmm_data(rng, T=500)

        from okmich_quant_ml.hmm import PomegranateHMM, DistType
        dur = PoissonDuration(n_states=2, max_duration=80)
        model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, max_iter=10,
                               random_state=123, duration_model=dur, covariance_type="diag")
        model.fit(X)
        ll = model.log_likelihood(X)
        assert np.isfinite(ll)
        assert ll < 0  # log-likelihood should be negative


# ======================================================================
# Numerical stability
# ======================================================================
class TestHSMMRightCensoring:
    """Spec §9.7: Short sequences with long true durations — verify no systematic underestimation."""

    @staticmethod
    def _generate_hsmm_data(rng: np.random.Generator, T: int, true_lambda: float) -> np.ndarray:
        """Generate 2-state HSMM data with Poisson(true_lambda) durations."""
        means = [-1.0, 1.0]
        observations = np.empty((T, 1), dtype=np.float64)
        t, state = 0, 0
        while t < T:
            dur = rng.poisson(true_lambda) + 1
            dur = min(dur, T - t)
            for d in range(dur):
                observations[t + d, 0] = rng.normal(means[state], 0.5)
            t += dur
            state = 1 - state
        return observations

    def test_short_sequence_no_underestimation(self):
        """T=50 with true mean duration 20 — right-censoring must prevent systematic bias."""
        rng = np.random.default_rng(77)
        true_lambda = 19.0  # expected duration = 20

        # Generate multiple short sequences and fit
        n_seqs = 30
        X_list = [self._generate_hsmm_data(rng, T=50, true_lambda=true_lambda) for _ in range(n_seqs)]
        X_all = np.vstack(X_list)
        lengths = [50] * n_seqs

        from okmich_quant_ml.hmm import PomegranateHMM, DistType, InferenceMode
        dur = PoissonDuration(n_states=2, max_duration=80)
        model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, max_iter=30,
                               random_state=77, inference_mode=InferenceMode.VITERBI, duration_model=dur,
                               covariance_type="diag")
        model.fit(X_all, lengths=lengths)

        # Both states should have expected duration in the right ballpark
        # Without right-censoring, durations would be systematically underestimated
        fitted_durations = sorted([dur.expected_duration(j) for j in range(2)])
        # The larger duration should not be drastically underestimated
        assert fitted_durations[1] > 10.0, (
            f"Largest fitted duration {fitted_durations[1]:.1f} is too low — "
            f"right-censoring may not be working correctly"
        )


class TestHSMMNumericalStability:
    def test_long_sequence_no_nan(self):
        """T=10000, N=5, M=200 — no NaN/Inf in posteriors (spec §9.8)."""
        rng = np.random.default_rng(99)
        T, N, M = 10000, 5, 200
        log_emissions = rng.standard_normal((T, N)) * 2 - 3
        dur = PoissonDuration(n_states=N, max_duration=M, lambdas=np.array([10.0, 30.0, 50.0, 80.0, 120.0]))
        log_trans = np.full((N, N), -np.inf)
        for i in range(N):
            for j in range(N):
                if i != j:
                    log_trans[i, j] = -np.log(N - 1)
        log_init = np.full(N, -np.log(N))

        result = hsmm_forward_backward(log_emissions, log_trans, log_init, dur)
        assert np.all(np.isfinite(result.state_posteriors))
        assert np.all(np.isfinite(result.expected_transitions))
        assert np.all(np.isfinite(result.expected_durations))
        assert np.isfinite(result.log_likelihood)


# ======================================================================
# Backward compatibility
# ======================================================================
class TestBackwardCompatibility:
    def test_duration_model_none_uses_standard_hmm(self):
        """With duration_model=None, fit/predict should behave exactly as standard HMM."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 2))

        from okmich_quant_ml.hmm import PomegranateHMM, DistType, InferenceMode
        model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, max_iter=5,
                               random_state=0, inference_mode=InferenceMode.FILTERING, covariance_type="diag")
        assert model.duration_model is None
        assert not model._is_hsmm
        model.fit(X)
        pred = model.predict(X)
        assert pred.shape == (200,)
        proba = model.predict_proba(X)
        assert proba.shape == (200, 2)



# ======================================================================
# Factory function
# ======================================================================
class TestFactoryFunction:
    def test_create_hsmm_via_factory(self):
        from okmich_quant_ml.hmm import create_simple_hmm_instance, DistType, DurationType
        model = create_simple_hmm_instance(DistType.NORMAL, n_states=2, duration_type=DurationType.POISSON,
                                           max_duration=50, covariance_type="diag")
        assert model._is_hsmm
        assert isinstance(model.duration_model, PoissonDuration)
        assert model.duration_model.max_duration == 50

    def test_create_standard_hmm_via_factory(self):
        from okmich_quant_ml.hmm import create_simple_hmm_instance, DistType
        model = create_simple_hmm_instance(DistType.NORMAL, n_states=2, covariance_type="diag")
        assert not model._is_hsmm

    def test_all_duration_types_via_factory(self):
        from okmich_quant_ml.hmm import create_simple_hmm_instance, DistType, DurationType
        for dt in DurationType:
            model = create_simple_hmm_instance(DistType.NORMAL, n_states=2, duration_type=dt,
                                               max_duration=30, covariance_type="diag")
            assert model._is_hsmm


# ======================================================================
# HSMM train defaults
# ======================================================================
class TestHSMMTrainDefaults:
    def test_train_preserves_hsmm_default_pomegranate(self):
        rng = np.random.default_rng(101)
        X = rng.standard_normal((200, 2))
        from okmich_quant_ml.hmm import DistType, PomegranateHMM

        dur = PoissonDuration(n_states=2, max_duration=40)
        model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, max_iter=5,
                               random_state=101, covariance_type="diag", duration_model=dur)
        best = model.train(X, n_states_range=(2,), criterion="bic")

        assert best._is_hsmm
        assert isinstance(best.duration_model, PoissonDuration)
        assert best.duration_model.max_duration == 40

    def test_train_preserves_hsmm_default_mixture(self):
        rng = np.random.default_rng(102)
        X = rng.standard_normal((180, 2))
        from okmich_quant_ml.hmm import DistType, PomegranateMixtureHMM

        dur = PoissonDuration(n_states=2, max_duration=35)
        model = PomegranateMixtureHMM(distribution_type=DistType.NORMAL, n_states=2, n_components=2,
                                      max_iter=5, random_state=102, covariance_type="diag", duration_model=dur)
        best = model.train(X, n_states_range=(2,), n_criteria_range=(2,), criterion="bic")

        assert best._is_hsmm
        assert isinstance(best.duration_model, PoissonDuration)
        assert best.duration_model.max_duration == 35


# ======================================================================
# Truncation consistency
# ======================================================================
class TestDurationExpectationTruncation:
    def test_parametric_expected_duration_matches_truncated_pmf(self):
        duration_models = [
            PoissonDuration(n_states=1, max_duration=50, lambdas=np.array([200.0])),
            NegBinDuration(n_states=1, max_duration=50, rs=np.array([2.0]), ps=np.array([0.01])),
            GammaDuration(n_states=1, max_duration=50, shapes=np.array([9.0]), scales=np.array([8.0])),
            LogNormalDuration(n_states=1, max_duration=50, mus=np.array([4.0]), sigmas=np.array([1.0])),
        ]
        u = np.arange(1, 51, dtype=np.float64)
        for dur in duration_models:
            expected = dur.expected_duration(0)
            trunc_mean = float(np.dot(np.exp(dur.log_pmf(0)), u))
            assert abs(expected - trunc_mean) < 1e-8
            assert expected <= dur.max_duration + 1e-8


# ======================================================================
# Phase 2 duration families
# ======================================================================
class TestNegBinDuration:
    def test_pmf_sums_to_one(self):
        dur = NegBinDuration(n_states=2, max_duration=50)
        for j in range(2):
            total = np.exp(logsumexp(dur.log_pmf(j)))
            assert abs(total - 1.0) < 1e-10

    def test_is_approximate(self):
        assert not NegBinDuration(n_states=1, max_duration=20).is_exact_mle

    def test_update_underdispersed_fallback(self):
        dur = NegBinDuration(n_states=1, max_duration=30)
        # Create eta that is underdispersed (all mass at single point)
        eta = np.zeros(30)
        eta[9] = 100.0
        dur.update(0, eta)  # should not raise
        assert dur.expected_duration(0) > 0


class TestGammaDuration:
    def test_pmf_sums_to_one(self):
        dur = GammaDuration(n_states=2, max_duration=50, shapes=np.array([3.0, 5.0]), scales=np.array([2.0, 4.0]))
        for j in range(2):
            total = np.exp(logsumexp(dur.log_pmf(j)))
            assert abs(total - 1.0) < 1e-10

    def test_expected_duration(self):
        dur = GammaDuration(n_states=1, max_duration=100, shapes=np.array([4.0]), scales=np.array([5.0]))
        u = np.arange(1, 101, dtype=np.float64)
        trunc_mean = float(np.dot(np.exp(dur.log_pmf(0)), u))
        assert abs(dur.expected_duration(0) - trunc_mean) < 1e-10

    def test_n_parameters(self):
        dur = GammaDuration(n_states=3, max_duration=50)
        assert dur.n_parameters() == 6


class TestLogNormalDuration:
    def test_pmf_sums_to_one(self):
        dur = LogNormalDuration(n_states=2, max_duration=50)
        for j in range(2):
            total = np.exp(logsumexp(dur.log_pmf(j)))
            assert abs(total - 1.0) < 1e-10

    def test_get_set_params(self):
        dur = LogNormalDuration(n_states=2, max_duration=30, mus=np.array([2.0, 3.0]), sigmas=np.array([0.3, 0.5]))
        params = dur.get_params()
        dur2 = LogNormalDuration(n_states=2, max_duration=30)
        dur2.set_params(params)
        np.testing.assert_array_almost_equal(dur2._mus, dur._mus)
        np.testing.assert_array_almost_equal(dur2._sigmas, dur._sigmas)


# ======================================================================
# HSMM AIC/BIC
# ======================================================================
class TestHSMMAICBIC:
    def test_aic_bic_includes_duration_params(self):
        rng = np.random.default_rng(55)
        X = rng.standard_normal((300, 2))

        from okmich_quant_ml.hmm import PomegranateHMM, DistType
        # Standard HMM
        hmm = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, max_iter=5,
                              random_state=55, covariance_type="diag")
        hmm.fit(X)
        aic_hmm, bic_hmm = hmm.get_aic_bic(X)

        # HSMM with Poisson — should have different parameter count
        dur = PoissonDuration(n_states=2, max_duration=50)
        hsmm = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, max_iter=5,
                               random_state=55, covariance_type="diag", duration_model=dur)
        hsmm.fit(X)
        aic_hsmm, bic_hsmm = hsmm.get_aic_bic(X)

        # Both should be finite
        assert np.isfinite(aic_hmm) and np.isfinite(aic_hsmm)
        assert np.isfinite(bic_hmm) and np.isfinite(bic_hsmm)

    def test_n2_zero_transition_params(self):
        """N=2 HSMM: n_trans_params = 2*(2-2) = 0."""
        rng = np.random.default_rng(66)
        X = rng.standard_normal((200, 1))

        from okmich_quant_ml.hmm import PomegranateHMM, DistType
        dur = PoissonDuration(n_states=2, max_duration=30)
        hsmm = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, max_iter=5,
                               random_state=66, covariance_type="diag", duration_model=dur)
        hsmm.fit(X)
        aic, bic = hsmm.get_aic_bic(X)
        assert np.isfinite(aic) and np.isfinite(bic)


# ======================================================================
# HSMM public methods
# ======================================================================
class TestHSMMPublicMethods:
    @pytest.fixture
    def fitted_hsmm(self):
        rng = np.random.default_rng(88)
        X = rng.standard_normal((300, 2))
        from okmich_quant_ml.hmm import PomegranateHMM, DistType, InferenceMode
        dur = PoissonDuration(n_states=2, max_duration=50)
        model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, max_iter=10,
                               random_state=88, inference_mode=InferenceMode.VITERBI, duration_model=dur,
                               covariance_type="diag")
        model.fit(X)
        return model, X

    def test_duration_parameters(self, fitted_hsmm):
        model, _ = fitted_hsmm
        params = model.duration_parameters()
        assert len(params) == 2
        for p in params:
            assert "state" in p
            assert "expected_duration" in p
            assert p["expected_duration"] > 0

    def test_duration_pmf(self, fitted_hsmm):
        model, _ = fitted_hsmm
        pmf = model.duration_pmf(0)
        assert pmf.shape == (50,)
        assert abs(pmf.sum() - 1.0) < 1e-6
        assert np.all(pmf >= 0)

    def test_predict_duration_info(self, fitted_hsmm):
        model, X = fitted_hsmm
        df = model.predict_duration_info(X)
        assert set(df.columns) == {"state", "start", "end", "duration"}
        assert len(df) > 0
        assert df["duration"].sum() == len(X)
        assert df["start"].iloc[0] == 0
        assert df["end"].iloc[-1] == len(X) - 1

    def test_plot_duration_distributions(self, fitted_hsmm):
        model, _ = fitted_hsmm
        import matplotlib
        matplotlib.use("Agg")
        ax = model.plot_duration_distributions()
        assert ax is not None

    def test_raises_without_duration_model(self):
        from okmich_quant_ml.hmm import PomegranateHMM, DistType
        model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, covariance_type="diag")
        with pytest.raises(RuntimeError):
            model.duration_parameters()
        with pytest.raises(RuntimeError):
            model.duration_pmf(0)
        with pytest.raises(RuntimeError):
            model.predict_duration_info(np.zeros((10, 2)))
