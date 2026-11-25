"""EM of the HHMM engine: monotone objective, closed-form optimum when the leaf path is observed, recovery of planted
parameters, structural zeros, emissions (Gaussian full, Student-t dof, tying), warm starts and exit schedules."""
import numpy as np
import pytest
from scipy import stats

from okmich_quant_ml.hmm.hierarchical.engine import (CategoricalEmission, CovarianceType, ExitSchedule,
                                                     GaussianEmission, HHMM, StateNode, StudentTEmission)

from ._trees import (categorical_pinned, gaussian_model, set_leaf_means, set_random_params, tayal_tree, three_level,
                     two_level)


def _planted_two_level(seed: int = 0) -> HHMM:
    model = gaussian_model(two_level())
    rng = np.random.default_rng(seed)
    model._init_params(rng.normal(size=(20, 1)), rng, 0)
    model.set_node_params("root", entry=[0.5, 0.5], horizontal=[[0.7, 0.3], [0.4, 0.6]])
    model.set_node_params("A", entry=[0.8, 0.2], horizontal=[[0.6, 0.4], [0.3, 0.7]], exit=[0.03, 0.08])
    model.set_node_params("B", entry=[0.3, 0.7], horizontal=[[0.5, 0.5], [0.2, 0.8]], exit=[0.06, 0.04])
    set_leaf_means(model, [-3.0, -1.0, 1.0, 3.0], var=0.25)
    return model


def _match_by_mean(fitted: HHMM, truth: HHMM) -> np.ndarray:
    """Leaf permutation: fitted leaf order sorted to the truth's by emission mean (distinct means)."""
    f = np.argsort(fitted.emissions[0].means.ravel())
    t = np.argsort(truth.emissions[0].means.ravel())
    perm = np.empty_like(f)
    perm[t] = f
    return perm


@pytest.mark.parametrize("prior", [0.0, 1e-2])
def test_em_objective_is_monotone(prior):
    truth = _planted_two_level()
    _, X = truth.sample(3000, seed=1)
    model = gaussian_model(two_level(), n_init=2, max_iter=60, transition_prior=prior, random_state=3).fit(X)
    assert np.all(np.diff(model.fit_report.history) > -1e-7 * len(X))


def test_em_is_monotone_for_categorical_studentt_and_schedules():
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.integers(0, 5, 2000), stats.t.rvs(4, size=2000, random_state=3)]).astype(float)
    sched = ExitSchedule.blocks(np.arange(2000) // 50)
    model = HHMM(two_level(), [CategoricalEmission(0, 5), StudentTEmission(1)], n_init=2, max_iter=40,
                 random_state=1).fit(X, schedule=sched)
    assert np.all(np.diff(model.fit_report.history) > -1e-7 * len(X))


def test_observed_leaf_path_gives_the_count_optimum():
    """With disjoint symbols per leaf and no root self-move, every transition's path is unique, so the EM optimum is
    the closed-form count estimate."""
    root = StateNode("root", list(two_level().children), transition_mask=[[False, True], [True, False]])
    truth = categorical_pinned(root, transition_prior=0.0)
    rng = np.random.default_rng(5)
    set_random_params(truth, rng, rng.integers(0, 8, size=(30, 1)).astype(float))
    leaves, X = truth.sample(4000, seed=7)
    model = categorical_pinned(root, transition_prior=0.0, n_init=1, max_iter=20, random_state=2).fit(X)
    counts = np.zeros((4, 4))
    np.add.at(counts, (leaves[:-1], leaves[1:]), 1.0)
    a, b = [0, 1], [2, 3]
    for block, other, name in ((a, b, "A"), (b, a, "B")):
        p = model.node_params(name)
        within = counts[np.ix_(block, block)]
        np.testing.assert_allclose(p["horizontal"], within / within.sum(axis=1, keepdims=True), atol=1e-8)
        out = counts[np.ix_(block, other)].sum(axis=1)
        np.testing.assert_allclose(p["exit"], out / counts[block].sum(axis=1), atol=1e-8)
        into = counts[np.ix_(other, block)].sum(axis=0)
        if leaves[0] in block:
            into[block.index(leaves[0])] += 1.0          # the first bar's entry
        np.testing.assert_allclose(p["entry"], into / into.sum(), atol=1e-8)


def test_planted_parameters_are_recovered():
    truth = _planted_two_level(seed=1)
    _, X = truth.sample(20000, seed=4)
    model = gaussian_model(two_level(), n_init=4, max_iter=300, random_state=6).fit(X)
    perm = _match_by_mean(model, truth)
    np.testing.assert_allclose(model.transition_prob()[np.ix_(perm, perm)], truth.transition_prob(), atol=0.03)
    assert model.log_likelihood(X) >= truth.log_likelihood(X) - 1e-6 * len(X)


def test_structural_zeros_survive_em():
    rng = np.random.default_rng(3)
    model = categorical_pinned(tayal_tree(), transition_prior=1e-2, n_init=2, max_iter=30, random_state=4)
    truth = categorical_pinned(tayal_tree())
    set_random_params(truth, rng, rng.integers(0, 8, size=(20, 1)).astype(float))
    _, X = truth.sample(1500, seed=2)
    model.fit(X)
    top = model.topology
    for q in top.internal:
        name = top.nodes[q].name
        p = model.node_params(name)
        assert np.all(p["entry"][~top.entry_mask(q)] == 0)
        assert np.all(p["horizontal"][~top.transition_mask(q)] == 0)
        assert np.all(p["exit"][~top.exit_mask(q)] == 0)
    assert model.node_params("R1")["horizontal"][1, 0] == 1.0      # the up leg of R1 must move to its down leg
    # Tayal's tree has 3 free transition parameters: the root's initial choice and one exit per regime
    assert model.n_free_params() - model.emissions[0].n_free_params() == 3


def test_tied_emission_shares_parameters_within_a_regime():
    rng = np.random.default_rng(0)
    model = HHMM(two_level(), [GaussianEmission([0], tie_level=1), GaussianEmission([1])], n_init=1, max_iter=5)
    X = rng.normal(size=(500, 2))
    model.fit(X)
    assert model.emissions[0].n_groups == 2 and model.emissions[1].n_groups == 4
    ld = model._log_emissions(np.column_stack([X[:, 0], np.full(500, np.nan)]))
    np.testing.assert_allclose(ld[:, 0], ld[:, 1])
    np.testing.assert_allclose(ld[:, 2], ld[:, 3])
    assert model.emissions[0].n_free_params() == 2 * 2


def test_full_covariance_single_state_equals_sample_moments():
    rng = np.random.default_rng(1)
    cov = np.array([[1.0, 0.6], [0.6, 2.0]])
    X = rng.multivariate_normal([1.0, -2.0], cov, size=5000)
    model = HHMM(StateNode("root", [StateNode.leaf("only")]),
                 [GaussianEmission([0, 1], covariance_type=CovarianceType.FULL, var_floor=1e-9)], n_init=1, max_iter=5)
    model.fit(X)
    np.testing.assert_allclose(model.emissions[0].means[0], X.mean(axis=0), atol=1e-10)
    np.testing.assert_allclose(model.emissions[0].covs[0], np.cov(X, rowvar=False, bias=True), atol=1e-6)


def test_student_t_log_density_and_dof_recovery():
    rng = np.random.default_rng(2)
    x = stats.t.rvs(5, loc=0.5, scale=2.0, size=20000, random_state=rng)
    model = HHMM(StateNode("root", [StateNode.leaf("only")]), [StudentTEmission([0])], n_init=1, max_iter=200)
    model.fit(x[:, None])
    e = model.emissions[0]
    assert 4.0 < e.dofs[0] < 6.5
    assert e.means[0, 0] == pytest.approx(0.5, abs=0.06)
    assert np.sqrt(e.covs[0, 0]) == pytest.approx(2.0, abs=0.08)
    expected = stats.t.logpdf(x[:5], e.dofs[0], loc=e.means[0, 0], scale=np.sqrt(e.covs[0, 0]))
    np.testing.assert_allclose(e.log_density(x[:5, None])[:, 0], expected, atol=1e-10)


def test_full_student_t_log_density_matches_scipy():
    rng = np.random.default_rng(3)
    model = HHMM(StateNode("root", [StateNode.leaf("only")]),
                 [StudentTEmission([0, 1], covariance_type=CovarianceType.FULL, learn_dof=False, dof_init=7.0)],
                 n_init=1, max_iter=3)
    X = rng.normal(size=(300, 2)) @ np.array([[1.0, 0.0], [0.5, 1.2]])
    model.fit(X)
    e = model.emissions[0]
    expected = stats.multivariate_t.logpdf(X[:10], loc=e.means[0], shape=e.covs[0], df=7.0)
    np.testing.assert_allclose(e.log_density(X[:10])[:, 0], expected, atol=1e-10)


def test_warm_start_does_not_lose_likelihood():
    truth = _planted_two_level(seed=2)
    _, X = truth.sample(3000, seed=5)
    first = gaussian_model(two_level(), n_init=1, max_iter=40, random_state=1).fit(X)
    second = gaussian_model(two_level(), n_init=0, max_iter=40).fit(X, warm_start=first)
    assert second.log_likelihood(X) >= first.log_likelihood(X) - 1e-9


def test_embedded_fit_never_learns_forbidden_exits():
    day = lambda n: StateNode(n, [StateNode.leaf(f"{n}.f0"), StateNode.leaf(f"{n}.f1")], exit_mask=[False, False])
    root = StateNode("root", [day("calm"), day("busy")])
    truth = gaussian_model(root)
    rng = np.random.default_rng(0)
    set_random_params(truth, rng, rng.normal(size=(20, 1)))
    truth.set_node_params("root", horizontal=[[0.8, 0.2], [0.3, 0.7]])
    set_leaf_means(truth, [-0.5, 0.5, -2.0, 2.0], var=0.3)
    block = np.repeat(np.arange(300), 20)
    sched = ExitSchedule.blocks(block)
    _, X = truth.sample(len(block), seed=1, schedule=sched)
    model = gaussian_model(root, n_init=3, max_iter=200, random_state=2).fit(X, schedule=sched)
    assert model.learned_depths_ == frozenset()
    assert np.all(np.diff(model.fit_report.history) > -1e-7 * len(X))
    assert model.log_likelihood(X, schedule=sched) >= truth.log_likelihood(X, schedule=sched) - 1e-4 * len(X)


def test_decoding_with_exits_the_fit_never_learned_warns():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 1))
    sched = ExitSchedule.blocks(np.arange(600) // 30)
    model = gaussian_model(two_level(), n_init=1, max_iter=5).fit(X, schedule=sched)
    assert model.learned_depths_ == frozenset()
    free_blocked = model.n_free_params()
    with pytest.warns(UserWarning, match="never learned"):
        model.filter(X)
    model.fit(X)
    assert model.learned_depths_ == frozenset({1})
    assert model.n_free_params() == free_blocked + 4      # the four depth-1 exits become free parameters
