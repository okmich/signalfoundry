"""The legacy build on the engine (depth-1 tree, loaders for saved models) and the two bar flavours."""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.hmm.hierarchical import (ExitMode, block_ids, block_schedule, embedded_hhmm, joint_multiscale,
                                              legacy_block_posterior, legacy_model)
from okmich_quant_ml.hmm.hierarchical.engine import ExitSchedule


def _alternating_symbols(n: int = 3000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(np.arange(n) % 2 == 0, rng.integers(9, 18, n), rng.integers(0, 9, n))


def test_legacy_model_is_a_depth_one_tree_and_fits():
    X = _alternating_symbols(seed=2)[:, None].astype(float)
    engine = legacy_model(n_init=2, max_iter=100, transition_prior=0.0, random_state=1).fit(X)
    assert engine.topology.max_depth == 1 and engine.n_states == 4
    assert np.all(np.diff(engine.fit_report.history) > -1e-7 * len(X))
    np.testing.assert_allclose(legacy_block_posterior(engine.filter(X)).sum(axis=1), 1.0)


def test_default_legacy_model_survives_an_unseen_allowed_symbol():
    train = _alternating_symbols(seed=3)
    train = np.where(train == 17, 16, train)[:, None].astype(float)          # up symbol 17 never appears
    model = legacy_model(n_init=1, max_iter=30).fit(train)
    post = model.filter(np.array([[10.0], [4.0], [17.0], [2.0]]))
    assert np.isfinite(post).all() and np.allclose(post.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------------------------------------------------
# Joint multi-scale
# ---------------------------------------------------------------------------------------------------------------------
def test_joint_multiscale_structure_and_recovery():
    model = joint_multiscale([0], [1], n_top=2, n_fast=2, n_init=3, max_iter=200, random_state=0)
    assert model.emissions[0].n_groups == 2 and model.emissions[1].n_groups == 4
    root = model.topology.root
    assert not root._transition.diagonal().any()
    rng = np.random.default_rng(0)
    model._init_params(rng.normal(size=(20, 2)), rng, 0)
    model.set_node_params("root", horizontal=[[0.0, 1.0], [1.0, 0.0]])
    model.set_node_params("backdrop0", horizontal=[[0.8, 0.2], [0.3, 0.7]], exit=[0.01, 0.01])
    model.set_node_params("backdrop1", horizontal=[[0.7, 0.3], [0.2, 0.8]], exit=[0.01, 0.01])
    model.emissions[0].set_params({"means": np.array([[-1.5], [1.5]]), "covs": np.array([[0.5], [0.5]])})
    model.emissions[1].set_params({"means": np.array([[-1.0], [1.0], [-1.0], [1.0]]), "covs": np.full((4, 1), 0.3)})
    leaves, X = model.sample(8000, seed=2)
    truth = model.leaf_to_level(leaves, 1)
    fitted = joint_multiscale([0], [1], n_top=2, n_fast=2, n_init=3, max_iter=200, random_state=4).fit(X)
    read = fitted.predict_level(X, level=1)
    accuracy = max(np.mean(read == truth), np.mean(read != truth))
    assert accuracy > 0.9
    with pytest.raises(ValueError, match="overlap"):
        joint_multiscale([0, 1], [1])


# ---------------------------------------------------------------------------------------------------------------------
# Embedded multi-timeframe
# ---------------------------------------------------------------------------------------------------------------------
def test_block_ids_follow_the_fx_day():
    idx = pd.DatetimeIndex(["2026-03-02 21:50", "2026-03-02 21:55", "2026-03-02 22:00", "2026-03-03 21:55"],
                           tz="UTC")                       # NY is UTC-5 in early March: 17:00 NY = 22:00 UTC
    ids = block_ids(idx, tz="America/New_York", roll_hour=17)
    assert ids[0] == ids[1] != ids[2] == ids[3]
    weeks = block_ids(pd.date_range("2026-03-01", periods=20, freq="D", tz="UTC"), unit="week")
    assert len(np.unique(weeks)) == 4
    sched = block_schedule(idx, tz="America/New_York", roll_hour=17)
    assert sched.modes[1].tolist()[1:] == [int(ExitMode.FORBIDDEN), int(ExitMode.FORCED), int(ExitMode.FORBIDDEN)]


def test_embedded_recovers_the_day_state():
    model = embedded_hhmm([0], n_coarse=2, n_fine=2, n_init=3, max_iter=200, random_state=0)
    rng = np.random.default_rng(1)
    model._init_params(rng.normal(size=(20, 1)), rng, 0)
    model.set_node_params("root", horizontal=[[0.85, 0.15], [0.2, 0.8]])
    model.emissions[0].set_params({"means": np.array([[-0.4], [0.4], [-1.6], [1.6]]), "covs": np.full((4, 1), 0.4)})
    block = np.repeat(np.arange(250), 48)
    sched = ExitSchedule.blocks(block)
    leaves, X = model.sample(len(block), seed=3, schedule=sched)
    truth = model.leaf_to_level(leaves, 1)
    fitted = embedded_hhmm([0], n_coarse=2, n_fine=2, n_init=3, max_iter=200, random_state=2).fit(X, schedule=sched)
    end_of_day = np.r_[np.flatnonzero(np.diff(block) != 0), len(block) - 1]
    read = fitted.predict_level(X, level=1, schedule=sched)[end_of_day]
    accuracy = max(np.mean(read == truth[end_of_day]), np.mean(read != truth[end_of_day]))
    assert accuracy > 0.85
    coarse = embedded_hhmm([0], coarse_columns=[1])
    assert coarse.emissions[1].n_groups == 2
