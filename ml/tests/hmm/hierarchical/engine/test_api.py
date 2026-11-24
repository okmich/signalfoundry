"""Tree validation, emission validation, parameter setters, decoding modes, sampling and persistence."""
import numpy as np
import pytest

from okmich_quant_ml.hmm import InferenceMode
from okmich_quant_ml.hmm.hierarchical.engine import (CategoricalEmission, ExitMode, ExitSchedule, GaussianEmission,
                                                     HHMM, StateNode, Topology)

from ._trees import gaussian_model, set_leaf_means, set_random_params, tayal_tree, three_level, two_level


# ---------------------------------------------------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------------------------------------------------
def test_tree_validation():
    with pytest.raises(ValueError, match="unique"):
        Topology(StateNode("root", [StateNode.leaf("a"), StateNode("X", [StateNode.leaf("a")])]))
    with pytest.raises(ValueError, match="at least one child"):
        Topology(StateNode.leaf("root"))
    with pytest.raises(ValueError, match="cannot carry"):
        StateNode("a", entry_mask=[True])
    with pytest.raises(ValueError, match="shape"):
        StateNode("X", [StateNode.leaf("a"), StateNode.leaf("b")], transition_mask=[[True, True]])
    with pytest.raises(ValueError, match="dead end"):
        Topology(StateNode("root", [StateNode("X", [StateNode.leaf("a")], transition_mask=[[False]],
                                              exit_mask=[False])]))
    with pytest.raises(ValueError, match="dead end"):          # the root never ends
        Topology(StateNode("root", [StateNode.leaf("a"), StateNode.leaf("b")],
                           transition_mask=[[False, True], [False, False]]))


def test_topology_layout_of_an_unbalanced_tree():
    top = Topology(three_level())
    assert top.leaf_names == ["y1", "y2", "x1", "z1"]
    assert list(top.leaf_depth) == [3, 3, 2, 1]
    assert top.max_internal_depth == 2
    assert [top.nodes[n].name for n in top.level_nodes(2)] == ["Y", "x1", "z1"]
    np.testing.assert_array_equal(top.group_of_leaves(1), [0, 0, 0, 1])
    np.testing.assert_array_equal(top.group_of_leaves(None), [0, 1, 2, 3])
    np.testing.assert_array_equal(top.group_of_leaves(0), [0, 0, 0, 0])
    assert bool(top.forced_exit(top.index["X"])[1]) is False


# ---------------------------------------------------------------------------------------------------------------------
# Emissions
# ---------------------------------------------------------------------------------------------------------------------
def test_categorical_support_validation():
    with pytest.raises(ValueError, match="not a leaf"):
        HHMM(two_level(), [CategoricalEmission(0, 3, support={"nope": [0]})])
    with pytest.raises(ValueError, match="different support"):
        HHMM(two_level(), [CategoricalEmission(0, 3, support={"a1": [0], "a2": [1]}, tie_level=1)])
    with pytest.raises(ValueError, match="outside"):
        HHMM(two_level(), [CategoricalEmission(0, 3, support={"a1": [5]})])
    model = HHMM(two_level(), [CategoricalEmission(0, 3)])
    rng = np.random.default_rng(0)
    set_random_params(model, rng, np.zeros((5, 1)))
    with pytest.raises(ValueError, match="integer symbols"):
        model.filter(np.array([[0.5]]))
    with pytest.raises(ValueError, match="integer symbols"):
        model.filter(np.array([[3.0]]))


def test_emission_constructor_validation():
    with pytest.raises(ValueError, match="distinct"):
        GaussianEmission([0, 0])
    with pytest.raises(ValueError, match="tie_level"):
        GaussianEmission([0], tie_level=-1)
    with pytest.raises(ValueError, match="columns"):
        HHMM(two_level(), [GaussianEmission([3])]).fit(np.zeros((10, 2)))


# ---------------------------------------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------------------------------------
def test_set_node_params_respects_structure():
    model = gaussian_model(tayal_tree())
    rng = np.random.default_rng(0)
    set_random_params(model, rng, rng.normal(size=(10, 1)))
    with pytest.raises(ValueError, match="masked"):
        model.set_node_params("R1", entry=[0.5, 0.5])
    with pytest.raises(ValueError, match="probability vector"):
        model.set_node_params("R1", horizontal=[[0.0, 0.7], [1.0, 0.0]])
    with pytest.raises(ValueError, match="exit must be 1 where"):
        model.set_node_params("R1", exit=[0.2, 0.3])
    with pytest.raises(ValueError, match="never ends"):
        model.set_node_params("root", exit=[0.1, 0.0])
    with pytest.raises(ValueError, match="is a leaf"):
        model.node_params("R1.D")
    with pytest.raises(KeyError):
        model.node_params("nope")
    model.set_node_params("R1", exit=[0.25, 0.0])
    assert model.node_params("R1")["exit"][0] == 0.25


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError, match="fit"):
        gaussian_model(two_level()).filter(np.zeros((3, 1)))


def test_constructor_validation():
    with pytest.raises(ValueError, match="sticky_init"):
        gaussian_model(two_level(), sticky_init=(1.0,))
    with pytest.raises(ValueError, match="n_init"):
        gaussian_model(two_level(), n_init=-1)
    with pytest.raises(TypeError):
        HHMM(two_level(), [object()])


# ---------------------------------------------------------------------------------------------------------------------
# Decoding, sampling, persistence
# ---------------------------------------------------------------------------------------------------------------------
def _fitted() -> tuple[HHMM, np.ndarray]:
    rng = np.random.default_rng(3)
    model = gaussian_model(two_level())
    set_random_params(model, rng, rng.normal(size=(10, 1)))
    set_leaf_means(model, [-2.0, -1.0, 1.0, 2.0])
    _, X = model.sample(300, seed=2)
    return model, X


def test_predict_follows_inference_mode():
    model, X = _fitted()
    np.testing.assert_array_equal(model.predict(X), model.predict_proba(X).argmax(axis=1))          # leaves
    np.testing.assert_array_equal(model.predict_level(X, level=1), model.predict_proba_level(X, level=1).argmax(axis=1))
    model.inference_mode = InferenceMode.CAUSAL_VITERBI
    np.testing.assert_array_equal(model.predict(X), model.predict_causal_viterbi(X))
    np.testing.assert_array_equal(model.predict_level(X, level=1),
                                  model.leaf_to_level(model.predict_causal_viterbi(X), 1))
    with pytest.raises(ValueError, match="path, not posteriors"):
        model.predict_proba(X)
    model.inference_mode = InferenceMode.SMOOTHING
    np.testing.assert_allclose(model.predict_proba_level(X, level=2), model.smooth(X))
    np.testing.assert_allclose(model.predict_proba(X), model.smooth(X))


def test_sample_shapes_lengths_and_unused_columns():
    model = HHMM(two_level(), [GaussianEmission([2])])
    rng = np.random.default_rng(0)
    set_random_params(model, rng, rng.normal(size=(10, 3)))
    leaves, X = model.sample(50, seed=1, lengths=[20, 30])
    assert leaves.shape == (50,) and X.shape == (50, 3)
    assert np.isnan(X[:, :2]).all() and np.isfinite(X[:, 2]).all()
    with pytest.raises(ValueError, match="lengths"):
        model.sample(50, lengths=[20, 20])


def test_schedule_validation():
    with pytest.raises(ValueError, match="depths >= 1"):
        ExitSchedule({0: np.zeros(3)})
    with pytest.raises(ValueError, match="same length"):
        ExitSchedule({1: np.zeros(3), 2: np.zeros(4)})
    with pytest.raises(ValueError, match="ExitMode"):
        ExitSchedule({1: np.full(3, 7)})
    model, X = _fitted()
    with pytest.raises(ValueError, match="covers"):
        model.filter(X, schedule=ExitSchedule({1: np.zeros(10)}))
    with pytest.raises(ValueError, match="no nodes that can end"):
        model.filter(X, schedule=ExitSchedule({2: np.zeros(len(X))}))
    sched = ExitSchedule.boundaries(np.r_[True, np.zeros(len(X) - 1, bool)], inside=ExitMode.LEARNED)
    np.testing.assert_allclose(model.filter(X, schedule=sched), model.filter(X), atol=1e-12)


def test_save_load_roundtrip(tmp_path):
    model, X = _fitted()
    path = tmp_path / "hhmm.joblib"
    model.save(str(path))
    loaded = HHMM.load(str(path))
    assert loaded.log_likelihood(X) == pytest.approx(model.log_likelihood(X), abs=1e-12)
    params = loaded.get_params()
    assert set(params["entry"]) == {"root", "A", "B"}


def test_aic_bic_and_free_parameters():
    model, X = _fitted()
    # root: entry 1 + 2 rows x 1; A, B: entry 1 + 2 rows x 1 + 2 exits each; Gaussian 4 leaves x 2
    assert model.n_free_params() == 3 + 2 * 5 + 8
    aic, bic = model.get_aic_bic(X)
    ll = model.log_likelihood(X)
    assert aic == pytest.approx(2 * 21 - 2 * ll)
    assert bic == pytest.approx(21 * np.log(len(X)) - 2 * ll)
