"""AIC, BIC and ICL model selection on the pomegranate HMMs.

Verifies:
- The exact path entropy H(z_1:T | x) equals brute-force enumeration over all K^T paths, with the learned end
  probabilities, and the same enumeration reproduces pomegranate's own log-likelihood.
- The per-bar sum of smoothed-marginal entropies is an upper bound on it.
- It is ~0 for perfectly separated emissions, invariant to relabelling the states, and additive over sequences.
- AIC and BIC (and train() selection by them) are unchanged for families whose parameter count is unchanged.
- The corrected free-parameter counts (GAMMA, LAMDA, CATEGORICAL, mixture StudentT).
- train() accepts SelectionCriterion members or their strings, selects by ICL, and rejects unknown criteria.
"""

import itertools

import numpy as np
import pytest
import torch
from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM
from scipy.special import logsumexp

from okmich_quant_ml.hmm import (DistType, InferenceMode, PomegranateHMM, PomegranateMixtureHMM, SelectionCriterion,
                                 SelectionScores)


def _hand_model(pi, A, end, means, sd=1.0) -> PomegranateHMM:
    """A 1-feature Gaussian HMM with the given parameters (edges rows plus end probabilities sum to 1)."""
    dists = [Normal(means=[float(m)], covs=[sd ** 2], covariance_type="diag", dtype=torch.float64) for m in means]
    model = PomegranateHMM(DistType.NORMAL, n_states=len(means), covariance_type="diag")
    model._model = DenseHMM(distributions=dists, edges=np.asarray(A, dtype=np.float64),
                            starts=np.asarray(pi, dtype=np.float64), ends=np.asarray(end, dtype=np.float64),
                            dtype=torch.float64)
    return model


def _random_hand_model(rng, K=3, spread=1.5):
    pi = rng.dirichlet(np.ones(K))
    rows = rng.dirichlet(np.ones(K + 1) * 2, size=K)  # last column is the end probability
    return pi, rows[:, :-1], rows[:, -1], np.sort(rng.normal(0, spread, size=K))


def _brute_force(model: PomegranateHMM, X: np.ndarray) -> tuple[float, float]:
    """(entropy, log Z) of the posterior over every one of the K^T state paths."""
    log_pi, log_A, log_B = model._extract_hmm_parameters(X)
    log_end = model._log_end_probs()
    T, K = log_B.shape
    log_joint = []
    for path in itertools.product(range(K), repeat=T):
        lp = log_pi[path[0]] + log_B[0, path[0]] + log_end[path[-1]]
        for t in range(1, T):
            lp += log_A[path[t - 1], path[t]] + log_B[t, path[t]]
        log_joint.append(lp)
    log_joint = np.array(log_joint)
    log_z = logsumexp(log_joint)
    p = np.exp(log_joint - log_z)
    return float(-np.sum(p[p > 0] * np.log(p[p > 0]))), float(log_z)


def _regime_data(seed=0, n=300, d=2, dwell=50, n_regimes=3):
    rng = np.random.default_rng(seed)
    means = rng.uniform(-2, 2, size=(n_regimes, d))
    states = np.repeat(np.arange(n) // dwell % n_regimes, 1)
    return means[states] + rng.normal(0, 0.7, size=(n, d))


def _smoothed_per_bar_entropy(model: PomegranateHMM, X: np.ndarray) -> float:
    model.inference_mode = InferenceMode.SMOOTHING
    gamma = np.asarray(model.predict_proba(X), dtype=np.float64)
    model.inference_mode = InferenceMode.FILTERING
    return float(-np.sum(np.where(gamma > 0, gamma * np.log(np.where(gamma > 0, gamma, 1.0)), 0.0)))


@pytest.fixture(scope="module")
def regime_data():
    return _regime_data()


@pytest.fixture(scope="module")
def fitted_normal(regime_data):
    return PomegranateHMM(DistType.NORMAL, n_states=3, random_state=7, covariance_type="diag").fit(regime_data)


class TestPathEntropyExact:
    @pytest.mark.parametrize("seed", range(5))
    def test_matches_brute_force_on_hand_models(self, seed):
        rng = np.random.default_rng(seed)
        model = _hand_model(*_random_hand_model(rng))
        X = rng.normal(0, 1.5, size=(8, 1))
        h_brute, log_z = _brute_force(model, X)
        assert model.path_entropy(X) == pytest.approx(h_brute, abs=1e-9)
        # the enumeration is pomegranate's model: same likelihood, end probabilities included
        assert model.log_likelihood(X) == pytest.approx(log_z, abs=1e-9)

    def test_matches_brute_force_on_a_fitted_model(self, fitted_normal, regime_data):
        X = regime_data[45:52]  # 7 bars across a regime boundary, so the path is uncertain
        h_brute, log_z = _brute_force(fitted_normal, X)
        assert fitted_normal.path_entropy(X) == pytest.approx(h_brute, abs=1e-9)
        assert fitted_normal.log_likelihood(X) == pytest.approx(log_z, abs=1e-9)

    def test_forbidden_transitions_and_a_zero_start(self):
        A = np.array([[0.5, 0.0, 0.4], [0.2, 0.7, 0.0], [0.0, 0.3, 0.6]])
        model = _hand_model([0.0, 0.5, 0.5], A, 1.0 - A.sum(axis=1), [-1.0, 0.0, 1.0])
        X = np.random.default_rng(3).normal(0, 1, size=(8, 1))
        h_brute, _ = _brute_force(model, X)
        h = model.path_entropy(X)
        assert np.isfinite(h)
        assert h == pytest.approx(h_brute, abs=1e-9)

    def test_single_bar_is_the_marginal_entropy(self):
        model = _hand_model(*_random_hand_model(np.random.default_rng(11)))
        X = np.array([[0.3]])
        h_brute, _ = _brute_force(model, X)
        assert model.path_entropy(X) == pytest.approx(h_brute, abs=1e-12)

    def test_chunked_sum_equals_one_block(self, fitted_normal, regime_data):
        whole = fitted_normal.path_entropy(regime_data)
        fitted_normal._ENTROPY_CHUNK_BARS = 7
        try:
            chunked = fitted_normal.path_entropy(regime_data)
        finally:
            del fitted_normal._ENTROPY_CHUNK_BARS
        assert chunked == pytest.approx(whole, rel=1e-12)


class TestPathEntropyProperties:
    def test_per_bar_sum_is_an_upper_bound(self, fitted_normal, regime_data):
        exact = fitted_normal.path_entropy(regime_data)
        per_bar = _smoothed_per_bar_entropy(fitted_normal, regime_data)
        assert 0.0 <= exact <= per_bar + 1e-9

    @pytest.mark.parametrize("seed", range(3))
    def test_per_bar_sum_bounds_hand_models(self, seed):
        rng = np.random.default_rng(100 + seed)
        model = _hand_model(*_random_hand_model(rng, spread=1.0))
        X = rng.normal(0, 1.0, size=(400, 1))
        assert model.path_entropy(X) <= _smoothed_per_bar_entropy(model, X) + 1e-9

    def test_separated_emissions_give_zero_entropy(self):
        A = np.array([[0.89, 0.1], [0.1, 0.89]])
        model = _hand_model([0.5, 0.5], A, [0.01, 0.01], [0.0, 100.0])
        states = np.repeat([0, 1, 0, 1], 25)
        X = (np.array([0.0, 100.0])[states] + np.random.default_rng(0).normal(0, 1, size=100)).reshape(-1, 1)
        assert model.path_entropy(X) == pytest.approx(0.0, abs=1e-6)

    def test_invariant_to_relabelling_the_states(self):
        rng = np.random.default_rng(21)
        pi, A, end, means = _random_hand_model(rng)
        X = rng.normal(0, 1.5, size=(500, 1))
        perm = np.array([2, 0, 1])
        h = _hand_model(pi, A, end, means).path_entropy(X)
        h_perm = _hand_model(pi[perm], A[np.ix_(perm, perm)], end[perm], means[perm]).path_entropy(X)
        assert h_perm == pytest.approx(h, rel=1e-10)

    def test_lengths_sum_over_sequences(self, fitted_normal, regime_data):
        lengths = [120, 100, 80]
        pieces = np.split(regime_data, np.cumsum(lengths)[:-1])
        expected = sum(fitted_normal.path_entropy(p) for p in pieces)
        assert fitted_normal.path_entropy(regime_data, lengths) == pytest.approx(expected, rel=1e-12)
        scores = fitted_normal.selection_scores(regime_data, lengths)
        assert scores.path_entropy == pytest.approx(expected, rel=1e-12)
        assert scores.log_likelihood == pytest.approx(sum(fitted_normal.log_likelihood(p) for p in pieces), rel=1e-12)
        assert scores.n_obs == len(regime_data)

    def test_unfitted_model_raises(self, regime_data):
        model = PomegranateHMM(DistType.NORMAL, n_states=2)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.path_entropy(regime_data)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.selection_scores(regime_data)


class TestSelectionScores:
    def test_icl_is_bic_plus_twice_the_entropy(self, fitted_normal, regime_data):
        s = fitted_normal.selection_scores(regime_data)
        assert s.icl == pytest.approx(s.bic + 2 * fitted_normal.path_entropy(regime_data), rel=1e-12)
        assert s.icl >= s.bic
        assert s.score("icl") == s.icl and s.score(SelectionCriterion.BIC) == s.bic

    def test_aic_bic_unchanged_for_normal(self, regime_data):
        """The pre-change formula: ll of X as one sequence, nu = (K-1) + K(K-1) + K(d + cov)."""
        for cov, k in (("diag", 2), ("full", 3), ("sphere", 2)):
            model = PomegranateHMM(DistType.NORMAL, n_states=k, random_state=3, covariance_type=cov).fit(regime_data)
            n, d = regime_data.shape
            ll = float(model.log_likelihood(regime_data))
            n_params = (k - 1) + k * (k - 1) + k * (d + model._cov_param_count(d))
            aic, bic = model.get_aic_bic(regime_data)
            assert aic == -2 * ll + 2 * n_params
            assert bic == -2 * ll + n_params * np.log(n)

    def test_entropy_skipped_on_request(self, fitted_normal, regime_data):
        s = fitted_normal.selection_scores(regime_data, with_entropy=False)
        assert s.path_entropy is None
        with pytest.raises(ValueError, match="path entropy"):
            _ = s.icl

    def test_scores_record(self):
        s = SelectionScores(log_likelihood=-100.0, n_params=5, n_obs=50, path_entropy=2.0)
        assert s.aic == 210.0
        assert s.bic == pytest.approx(200.0 + 5 * np.log(50))
        assert s.icl == pytest.approx(s.bic + 4.0)


class TestFreeParameterCounts:
    @pytest.mark.parametrize("dist, kwargs, per_state", [
        (DistType.NORMAL, {"covariance_type": "diag"}, 2 * 2),
        (DistType.NORMAL, {"covariance_type": "full"}, 2 + 3),
        (DistType.NORMAL, {"covariance_type": "sphere"}, 2 + 1),
        (DistType.STUDENTT, {"covariance_type": "diag"}, 2 * 2),  # dofs is fixed, not learned
        (DistType.LOGNORMAL, {"covariance_type": "diag"}, 2 * 2),
        (DistType.GAMMA, {}, 2 * 2),
        (DistType.LAMDA, {}, 3 * 2),
        (DistType.CATEGORICAL, {"n_categories": 4}, 3 * 2),
        (DistType.EXPONENTIAL, {}, 2),
        (DistType.POISSON, {}, 2),
        (DistType.BERNOULLI, {}, 2),
    ])
    def test_plain_hmm(self, dist, kwargs, per_state):
        k, d = 3, 2
        model = PomegranateHMM(dist, n_states=k, **kwargs)
        assert model._n_free_params(d) == (k - 1) + k * (k - 1) + k * per_state

    @pytest.mark.parametrize("dist, kwargs, per_component", [
        (DistType.NORMAL, {"covariance_type": "diag"}, 2 * 2),
        (DistType.STUDENTT, {"covariance_type": "diag", "dofs": 5}, 2 * 2),
        (DistType.GAMMA, {}, 2 * 2),
        (DistType.LAMDA, {}, 3 * 2),
    ])
    def test_mixture_hmm(self, dist, kwargs, per_component):
        k, c, d = 3, 2, 2
        model = PomegranateMixtureHMM(dist, n_states=k, n_components=c, **kwargs)
        assert model._n_free_params(d) == (k - 1) + k * (k - 1) + k * (c * per_component + (c - 1))


class TestTrainCriterion:
    def _old_selection(self, X, criterion, k_range):
        """train() as it was before ICL: ll of X as one sequence, nu from the unchanged NORMAL count."""
        best, best_k = np.inf, None
        for k in k_range:
            for cov in ("diag", "full", "sphere"):
                inst = PomegranateHMM(DistType.NORMAL, n_states=k, random_state=5, covariance_type=cov).fit(X)
                n, d = X.shape
                ll = float(inst.log_likelihood(X))
                n_params = (k - 1) + k * (k - 1) + k * (d + inst._cov_param_count(d))
                score = -2 * ll + 2 * n_params if criterion == "aic" else -2 * ll + n_params * np.log(n)
                if score < best:
                    best, best_k = score, (k, cov)
        return best_k

    @pytest.mark.parametrize("criterion", ["aic", "bic", SelectionCriterion.AIC, SelectionCriterion.BIC])
    def test_aic_bic_selection_unchanged(self, regime_data, criterion):
        best = PomegranateHMM(DistType.NORMAL, random_state=5).train(regime_data, n_states_range=(2, 3),
                                                                     criterion=criterion)
        assert (best.n_states, best.dist_kwargs["covariance_type"]) == self._old_selection(regime_data, criterion,
                                                                                            (2, 3))

    @pytest.mark.parametrize("criterion", ["icl", SelectionCriterion.ICL])
    def test_icl_selects_the_lowest_icl(self, regime_data, criterion):
        base = PomegranateHMM(DistType.NORMAL, random_state=5, covariance_type="diag")
        best = base.train(regime_data, n_states_range=(2, 3, 4), criterion=criterion)
        icl = {k: PomegranateHMM(DistType.NORMAL, n_states=k, random_state=5, covariance_type="diag")
               .fit(regime_data).selection_scores(regime_data).icl for k in (2, 3, 4)}
        assert best.n_states == min(icl, key=icl.get)

    def test_mixture_accepts_icl(self, regime_data):
        best = PomegranateMixtureHMM(DistType.NORMAL, random_state=5, covariance_type="diag").train(
            regime_data, n_states_range=(2,), n_criteria_range=(2,), criterion=SelectionCriterion.ICL)
        assert best.n_states == 2 and best.n_components == 2

    @pytest.mark.parametrize("cls", [PomegranateHMM, PomegranateMixtureHMM])
    def test_unknown_criterion_raises(self, regime_data, cls):
        with pytest.raises(ValueError, match="Allowed values"):
            cls(DistType.NORMAL).train(regime_data, n_states_range=(2,), criterion="hqc")
