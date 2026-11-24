"""Regressions for the 2026-11-24 code review of the HHMM engine.

- log-space recursions: a state ruled out for hundreds of bars is recovered; a tail observation of the only reachable
  leaf keeps its finite density;
- the model owns its emission blocks; overlapping columns are rejected;
- warm starts must match the tree, masks and emission configuration, and ``warm_start=self`` works;
- the objective includes the categorical pseudo-count prior (monotone MAP-EM), and the raw likelihood is reported;
- a schedule mode used only on a sequence's first bar cannot reject the sequence.
"""
import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import norm

from okmich_quant_ml.hmm.hierarchical.engine import (CategoricalEmission, ExitMode, ExitSchedule, GaussianEmission,
                                                     HHMM, StateNode, StudentTEmission)

from ._trees import set_random_params, two_level


def _flat(n: int, mask=None) -> StateNode:
    return StateNode("root", [StateNode.leaf(f"s{k}") for k in range(n)], transition_mask=mask)


def _two_state(entry, horizontal) -> HHMM:
    model = HHMM(_flat(2), [GaussianEmission([0])])
    rng = np.random.default_rng(0)
    model._init_params(rng.normal(size=(10, 1)), rng, 0)
    model.set_node_params("root", entry=entry, horizontal=horizontal)
    model.emissions[0].set_params({"means": np.array([[1.0], [-1.0]]), "covs": np.array([[1.0], [1.0]])})
    return model


def test_a_state_ruled_out_for_hundreds_of_bars_is_recovered():
    """Identity transitions: the posterior log-odds are exactly the summed log-likelihood ratios."""
    model = _two_state([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]])
    X = np.r_[np.full(400, 1.0), np.full(500, -1.0)][:, None]
    log_b = model._log_emissions(X)
    exact = np.cumsum(log_b[:, 1] - log_b[:, 0])            # log P(B)/P(A) after each bar
    post = model.filter(X)[:, 1]
    assert exact[-1] == pytest.approx(200.0)
    assert post[-1] == pytest.approx(1.0)
    middle = np.abs(exact) < 30                              # where the probability is not saturated
    np.testing.assert_allclose(post[middle], expit(exact[middle]), atol=1e-12)
    assert np.isfinite(model.smooth(X)).all() and np.allclose(model.smooth(X).sum(axis=1), 1.0)
    assert np.allclose(model.predict_proba_fixed_lag(X, 50).sum(axis=1), 1.0)


def test_a_tail_observation_of_the_only_reachable_leaf_keeps_its_density():
    model = _two_state([1.0, 0.0], [[0.5, 0.5], [0.5, 0.5]])
    x = -39.0                                                 # far in s0's tail, close to s1's mean
    expected = norm.logpdf(x, loc=1.0, scale=1.0)             # s1 has zero entry probability
    assert model.log_likelihood(np.array([[x]])) == pytest.approx(expected, rel=1e-12)
    assert model.filter(np.array([[x]]))[0, 0] == pytest.approx(1.0)


def test_the_model_owns_its_emission_blocks():
    rng = np.random.default_rng(1)
    block = GaussianEmission([0])
    a = HHMM(_flat(2), [block], n_init=1, max_iter=5).fit(rng.normal(size=(200, 1)))
    x0 = np.array([[0.0]])
    before = a.log_likelihood(x0)
    HHMM(_flat(2), [block], n_init=1, max_iter=5).fit(rng.normal(5, 1, size=(200, 1)))
    assert a.log_likelihood(x0) == before
    assert block.means is None                                # the caller's object was never bound or fitted


def test_overlapping_columns_are_rejected():
    with pytest.raises(ValueError, match="Column 0 is read by emission blocks 0 and 1"):
        HHMM(_flat(1), [CategoricalEmission(0, 2), CategoricalEmission(0, 2)])
    with pytest.raises(ValueError, match="Column 1"):
        HHMM(_flat(1), [GaussianEmission([0, 1]), GaussianEmission([1, 2])])


def test_warm_start_must_match_tree_masks_and_emissions():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 1))
    src = HHMM(_flat(2), [GaussianEmission([0])], n_init=1, max_iter=5).fit(X)
    no_self = _flat(2, mask=[[False, True], [True, False]])
    with pytest.raises(ValueError, match="masks"):
        HHMM(no_self, [GaussianEmission([0])], n_init=0).fit(X, warm_start=src)
    cat = HHMM(_flat(2), [CategoricalEmission(0, 3)], n_init=1, max_iter=3).fit(rng.integers(0, 3, (50, 1)))
    with pytest.raises(ValueError, match="emission blocks"):
        HHMM(_flat(2), [CategoricalEmission(0, 4)], n_init=0).fit(rng.integers(0, 4, (50, 1)), warm_start=cat)
    with pytest.raises(ValueError, match="emission blocks"):
        from okmich_quant_ml.hmm.hierarchical.engine import CovarianceType
        HHMM(_flat(2), [GaussianEmission([0], covariance_type=CovarianceType.FULL)], n_init=0).fit(X, warm_start=src)


def test_warm_start_from_itself():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 1))
    model = HHMM(_flat(2), [GaussianEmission([0])], n_init=1, max_iter=5, random_state=1).fit(X)
    first = model.fit_report.objective
    model.n_init = 0
    model.fit(X, warm_start=model)
    assert model.fit_report.restart == -1
    assert model.fit_report.objective >= first - 1e-9


def test_objective_includes_the_categorical_prior():
    rng = np.random.default_rng(4)
    X = rng.integers(0, 3, (40, 1)).astype(float)
    model = HHMM(_flat(1), [CategoricalEmission(0, 5, pseudo_count=2.0)], n_init=1, max_iter=20,
                 transition_prior=0.0).fit(X)
    assert np.all(np.diff(model.fit_report.history) >= -1e-10)
    assert model.fit_report.loglik == pytest.approx(model.log_likelihood(X))
    prior = 2.0 * np.log(model.emissions[0].probs[0]).sum()
    assert model.fit_report.objective == pytest.approx(model.fit_report.loglik + prior)


def test_a_mode_used_only_on_first_bars_cannot_reject_the_sequence():
    a = StateNode("A", [StateNode.leaf("a1"), StateNode.leaf("a2")], transition_mask=[[False, True], [False, False]],
                  exit_mask=[False, True])
    model = HHMM(StateNode("root", [a, StateNode("B", [StateNode.leaf("b1")])]), [GaussianEmission([0])])
    rng = np.random.default_rng(5)
    set_random_params(model, rng, rng.normal(size=(5, 1)))
    forbidden, learned = int(ExitMode.FORBIDDEN), int(ExitMode.LEARNED)
    first_only = ExitSchedule({1: np.array([forbidden, learned, forbidden, learned])})
    X = rng.normal(size=(4, 1))
    ll = model.log_likelihood(X, lengths=[2, 2], schedule=first_only)         # FORBIDDEN only on start bars
    assert ll == pytest.approx(model.log_likelihood(X, lengths=[2, 2]))
    model.filter(X[:1], schedule=ExitSchedule({1: np.array([forbidden])}))     # one bar: nothing transitions
    with pytest.raises(ValueError, match="no way to continue"):
        model.filter(X, schedule=ExitSchedule({1: np.array([learned, forbidden, learned, learned])}))


def test_a_rejected_refit_leaves_the_model_unchanged():
    rng = np.random.default_rng(7)
    X = rng.integers(0, 3, (200, 1)).astype(float)
    model = HHMM(_flat(2), [CategoricalEmission(0, 3)], n_init=2, max_iter=20).fit(X)
    report, before = model.fit_report, model.filter(X)
    with pytest.raises(ValueError, match="integer symbols"):
        model.fit(np.r_[X, [[5.0]]])                                   # out of the alphabet
    assert model.fit_report is report
    np.testing.assert_array_equal(model.filter(X), before)


def test_a_warm_start_cannot_override_a_fixed_dof():
    rng = np.random.default_rng(8)
    X = rng.standard_t(3, size=(400, 1))
    src = HHMM(_flat(1), [StudentTEmission([0], dof_init=5.0)], n_init=1, max_iter=30).fit(X)
    fixed = HHMM(_flat(1), [StudentTEmission([0], learn_dof=False, dof_init=20.0)], n_init=0)
    with pytest.raises(ValueError, match="emission blocks"):
        fixed.fit(X, warm_start=src)
    same = HHMM(_flat(1), [StudentTEmission([0], learn_dof=False, dof_init=20.0)], n_init=1, max_iter=5).fit(X)
    again = HHMM(_flat(1), [StudentTEmission([0], learn_dof=False, dof_init=20.0)], n_init=0, max_iter=5)
    again.fit(X, warm_start=same)
    assert again.emissions[0].dofs[0] == 20.0


def test_a_rejected_parameter_update_leaves_the_model_unchanged():
    model = HHMM(two_level(), [GaussianEmission([0])])
    rng = np.random.default_rng(9)
    set_random_params(model, rng, rng.normal(size=(10, 1)))
    X = rng.normal(size=(30, 1))
    before = model.filter(X)
    with pytest.raises(ValueError, match="horizontal needs shape"):
        model.set_node_params("A", entry=[0.1, 0.9], horizontal=[[1.0]])         # valid entry, invalid horizontal
    np.testing.assert_array_equal(model.filter(X), before)
    with pytest.raises(ValueError, match="probability vector"):
        model.set_node_params("A", entry=[np.nan, 1.0])
    with pytest.raises(ValueError, match="finite"):
        model.set_node_params("A", exit=[np.nan, 0.1])
    np.testing.assert_array_equal(model.filter(X), before)


def test_fitted_model_two_level_still_recovers_after_log_space_rewrite():
    model = HHMM(two_level(), [GaussianEmission([0])], n_init=2, max_iter=50, random_state=0)
    rng = np.random.default_rng(6)
    model.fit(rng.normal(size=(500, 1)))
    assert np.all(np.diff(model.fit_report.history) > -1e-7 * 500)
