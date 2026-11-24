"""Exact inference of the HHMM engine against independent oracles.

- the flat transition matrix equals the recursive HHMM semantics (Fine et al. 1998), under every exit mode;
- likelihood, filter, smoother, Viterbi and causal Viterbi equal brute-force path enumeration;
- a depth-1 tree is a plain HMM;
- the embedded schedule equals the block-by-block likelihood of Adam et al. (2019), eq. 5;
- every causal output is prefix-invariant.
"""
import numpy as np
import pytest
from scipy.special import logsumexp

from okmich_quant_ml.hmm.hierarchical.engine import (CategoricalEmission, ExitMode, ExitSchedule, GaussianEmission,
                                                     HHMM, StateNode)

from ._trees import (brute_force, categorical_pinned, flat_tree, gaussian_model, semantic_initial, semantic_next_leaf,
                     set_leaf_means, set_random_params, tayal_tree, three_level, two_level)

TREES = {"two_level": two_level, "three_level": three_level, "tayal": tayal_tree}
MODES = [{}, {1: ExitMode.FORBIDDEN}, {1: ExitMode.FORCED}, {2: ExitMode.FORCED}, {2: ExitMode.FORBIDDEN}]


def _random_model(tree: str, seed: int = 0) -> HHMM:
    rng = np.random.default_rng(seed)
    model = gaussian_model(TREES[tree]())
    set_random_params(model, rng, rng.normal(size=(50, 1)))
    set_leaf_means(model, rng.normal(0, 1.5, model.topology.n_leaves))
    return model


def _schedule_for(model: HHMM, modes: dict, n_bars: int) -> ExitSchedule | None:
    modes = {d: m for d, m in modes.items() if d <= model.topology.max_internal_depth}
    if not modes:
        return None
    return ExitSchedule({d: np.full(n_bars, int(m)) for d, m in modes.items()})


@pytest.mark.parametrize("tree", sorted(TREES))
@pytest.mark.parametrize("modes", MODES, ids=lambda m: str({d: v.name for d, v in m.items()}))
def test_flat_transition_equals_hhmm_semantics(tree, modes):
    model = _random_model(tree, seed=3)
    schedule = _schedule_for(model, modes, 2)
    effective = {d: m for d, m in modes.items() if d <= model.topology.max_internal_depth}
    if schedule is None:
        trans = model.transition_prob()
    else:
        try:
            stack, index = model.flat_transitions(schedule)
        except ValueError:
            pytest.skip("this tree has a child that can only exit, which a FORBIDDEN schedule rules out")
        trans = stack[index[1]]
    oracle = np.array([semantic_next_leaf(model, i, effective) for i in range(model.topology.n_leaves)])
    np.testing.assert_allclose(trans, oracle, atol=1e-12)
    np.testing.assert_allclose(trans.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(model.initial_distribution(), semantic_initial(model), atol=1e-12)


@pytest.mark.parametrize("tree", sorted(TREES))
def test_inference_matches_path_enumeration(tree):
    model = _random_model(tree, seed=11)
    _, X = model.sample(6, seed=2)
    total, filtered, smoothed, map_path, causal_ends = brute_force(model, X)
    assert model.log_likelihood(X) == pytest.approx(total, abs=1e-9)
    np.testing.assert_allclose(model.filter(X), filtered, atol=1e-9)
    np.testing.assert_allclose(model.smooth(X), smoothed, atol=1e-9)
    np.testing.assert_array_equal(model.predict_viterbi(X), map_path)
    np.testing.assert_array_equal(model.predict_causal_viterbi(X), causal_ends)


def test_inference_with_a_time_varying_schedule_matches_enumeration():
    model = _random_model("three_level", seed=5)
    modes = [{}, {1: ExitMode.FORCED}, {}, {2: ExitMode.FORCED}, {1: ExitMode.FORBIDDEN}]
    schedule = ExitSchedule({1: [int(m.get(1, ExitMode.LEARNED)) for m in modes],
                             2: [int(m.get(2, ExitMode.LEARNED)) for m in modes]})
    _, X = model.sample(5, seed=8, schedule=schedule)
    total, filtered, smoothed, map_path, _ = brute_force(model, X, modes)
    assert model.log_likelihood(X, schedule=schedule) == pytest.approx(total, abs=1e-9)
    np.testing.assert_allclose(model.filter(X, schedule=schedule), filtered, atol=1e-9)
    np.testing.assert_allclose(model.smooth(X, schedule=schedule), smoothed, atol=1e-9)
    np.testing.assert_array_equal(model.predict_viterbi(X, schedule=schedule), map_path)


def test_categorical_structural_zeros_match_enumeration():
    """Tayal tree with pinned categorical symbols: impossible paths get probability 0, not NaN."""
    rng = np.random.default_rng(4)
    model = categorical_pinned(tayal_tree())
    X0 = rng.integers(0, 8, size=(40, 1)).astype(float)
    set_random_params(model, rng, X0)
    _, X = model.sample(6, seed=1)
    total, filtered, smoothed, map_path, _ = brute_force(model, X)
    assert model.log_likelihood(X) == pytest.approx(total, abs=1e-9)
    np.testing.assert_allclose(model.filter(X), filtered, atol=1e-9)
    np.testing.assert_array_equal(model.predict_viterbi(X), map_path)


def test_missing_observation_contributes_nothing():
    model = _random_model("two_level", seed=7)
    _, X = model.sample(6, seed=3)
    X[2, 0] = np.nan
    total, filtered, _, _, _ = brute_force(model, X)
    assert model.log_likelihood(X) == pytest.approx(total, abs=1e-9)
    np.testing.assert_allclose(model.filter(X), filtered, atol=1e-9)


def test_depth_one_tree_is_a_plain_hmm():
    rng = np.random.default_rng(0)
    model = gaussian_model(flat_tree(3))
    set_random_params(model, rng, rng.normal(size=(30, 1)))
    set_leaf_means(model, [-2.0, 0.0, 2.0])
    A = model.node_params("root")["horizontal"]
    pi = model.node_params("root")["entry"]
    np.testing.assert_allclose(model.transition_prob(), A, atol=1e-14)
    _, X = model.sample(300, seed=1)
    log_b = model._log_emissions(X)
    log_alpha = np.log(pi) + log_b[0]
    for t in range(1, len(X)):
        log_alpha = logsumexp(log_alpha[:, None] + np.log(A), axis=0) + log_b[t]
    assert model.log_likelihood(X) == pytest.approx(logsumexp(log_alpha), abs=1e-8)


def _embedded_model(seed: int = 0) -> tuple[HHMM, np.ndarray, ExitSchedule]:
    rng = np.random.default_rng(seed)
    day = lambda n: StateNode(n, [StateNode.leaf(f"{n}.f{k}") for k in range(3)], exit_mask=[False] * 3)
    model = gaussian_model(StateNode("root", [day("calm"), day("busy")]))
    set_random_params(model, rng, rng.normal(size=(40, 1)))
    set_leaf_means(model, rng.normal(0, 1.0, 6), var=0.5)
    block = np.repeat(np.arange(7), 9)
    schedule = ExitSchedule.blocks(block)
    return model, block, schedule


def test_embedded_schedule_equals_block_likelihood():
    """Adam et al. (2019) eq. 5: a coarse HMM over blocks whose emission is the fine HMM's likelihood of the block."""
    model, block, schedule = _embedded_model(seed=2)
    _, X = model.sample(len(block), seed=5, schedule=schedule)
    log_b = model._log_emissions(X)
    root = model.node_params("root")
    coarse = ["calm", "busy"]
    fine = [model.node_params(n) for n in coarse]
    leaves = [model.topology.descendants[model.topology.index[n]] for n in coarse]

    def block_loglik(s: int, rows: np.ndarray) -> float:
        lb = log_b[rows][:, leaves[s]]
        log_alpha = np.log(fine[s]["entry"]) + lb[0]
        for t in range(1, len(rows)):
            log_alpha = logsumexp(log_alpha[:, None] + np.log(fine[s]["horizontal"]), axis=0) + lb[t]
        return float(logsumexp(log_alpha))

    blocks = [np.flatnonzero(block == b) for b in np.unique(block)]
    log_a = np.log(root["entry"]) + np.array([block_loglik(s, blocks[0]) for s in range(2)])
    for rows in blocks[1:]:
        emis = np.array([block_loglik(s, rows) for s in range(2)])
        log_a = logsumexp(log_a[:, None] + np.log(root["horizontal"]), axis=0) + emis
    assert model.log_likelihood(X, schedule=schedule) == pytest.approx(logsumexp(log_a), abs=1e-8)


def test_embedded_top_state_only_changes_at_block_starts():
    model, block, schedule = _embedded_model(seed=4)
    leaves, _ = model.sample(len(block), seed=9, schedule=schedule)
    top = model.leaf_to_level(leaves, 1)
    changes = np.flatnonzero(np.diff(top) != 0) + 1
    starts = set(np.flatnonzero(np.r_[True, np.diff(block) != 0]))
    assert set(changes) <= starts


def test_sequences_are_independent():
    model = _random_model("two_level", seed=1)
    _, X1 = model.sample(40, seed=1)
    _, X2 = model.sample(30, seed=2)
    joint = model.log_likelihood(np.vstack([X1, X2]), lengths=[40, 30])
    assert joint == pytest.approx(model.log_likelihood(X1) + model.log_likelihood(X2), abs=1e-9)


@pytest.mark.parametrize("tree", sorted(TREES))
def test_causal_outputs_are_prefix_invariant(tree):
    model = _random_model(tree, seed=13)
    _, X = model.sample(200, seed=4)
    full_f, full_cv, full_fl = model.filter(X), model.predict_causal_viterbi(X), model.predict_proba_fixed_lag(X, 5)
    for cut in (37, 120, 199):
        np.testing.assert_allclose(model.filter(X[:cut]), full_f[:cut], atol=1e-12)
        np.testing.assert_array_equal(model.predict_causal_viterbi(X[:cut]), full_cv[:cut])
        np.testing.assert_allclose(model.predict_proba_fixed_lag(X[:cut], 5)[:cut - 5], full_fl[:cut - 5], atol=1e-12)


def test_fixed_lag_limits():
    model = _random_model("three_level", seed=6)
    _, X = model.sample(80, seed=2)
    np.testing.assert_allclose(model.predict_proba_fixed_lag(X, 0), model.filter(X), atol=1e-12)
    np.testing.assert_allclose(model.predict_proba_fixed_lag(X, 80), model.smooth(X), atol=1e-10)


def test_forbidden_exit_on_an_exit_only_child_is_rejected():
    node = StateNode("A", [StateNode.leaf("a1"), StateNode.leaf("a2")], transition_mask=[[False, True], [False, False]],
                     exit_mask=[False, True])
    model = gaussian_model(StateNode("root", [node, StateNode("B", [StateNode.leaf("b1")])]))
    rng = np.random.default_rng(0)
    set_random_params(model, rng, rng.normal(size=(20, 1)))
    schedule = ExitSchedule({1: np.full(10, int(ExitMode.FORBIDDEN))})
    with pytest.raises(ValueError, match="no way to continue"):
        model.filter(rng.normal(size=(10, 1)), schedule=schedule)


def test_observation_no_leaf_can_emit_is_rejected():
    rng = np.random.default_rng(0)
    model = HHMM(two_level(), [CategoricalEmission(0, 4, support={n: [0, 1] for n in ["a1", "a2", "b1", "b2"]})])
    set_random_params(model, rng, np.zeros((10, 1)))
    with pytest.raises(ValueError, match="zero probability"):
        model.filter(np.array([[0.0], [3.0]]))


def test_level_marginals_sum_leaf_posteriors():
    model = _random_model("three_level", seed=2)
    _, X = model.sample(50, seed=1)
    leaf = model.filter(X)
    assert model.level_names(1) == ["X", "z1"]
    assert model.level_names(2) == ["Y", "x1", "z1"]
    assert model.level_names(3) == model.topology.leaf_names
    np.testing.assert_allclose(model.predict_proba_level(X, level=1)[:, 0], leaf[:, :3].sum(axis=1), atol=1e-12)
    np.testing.assert_allclose(model.predict_proba_level(X, level=0), 1.0, atol=1e-12)
