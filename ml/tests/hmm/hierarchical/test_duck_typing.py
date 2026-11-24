"""Duck typing: every hierarchical model answers BasePomegranateHMM's public methods like PomegranateHMM does, and runs
through the live ``HmmModelWrapper`` (three-file export) and ``compute_trajectories`` (adaptive lag).

Models: PomegranateHMM (reference), a two-level engine HHMM, joint_multiscale, embedded_hhmm with a block-id column,
legacy_model, and TayalHHMM through its bar-level adapter.
"""
import itertools
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from okmich_quant_ml.hmm import DistType, PomegranateHMM
from okmich_quant_ml.hmm.base_pomegranate import BasePomegranateHMM
from okmich_quant_ml.hmm.hierarchical import (HHMM, BarLayout, ExitSchedule, GaussianEmission, LegThreshold, MarketData,
                                              StateNode, TayalHHMM, block_ids, embedded_hhmm, joint_multiscale,
                                              legacy_model)
from okmich_quant_ml.hmm.util import SelectionScores
from okmich_quant_ml.inference_model_wrappers import HmmModelWrapper
from okmich_quant_ml.posterior_inference import compute_trajectories

warnings.filterwarnings("ignore", category=DeprecationWarning)
PUBLIC = sorted(n for n in dir(BasePomegranateHMM) if not n.startswith("_"))
T_BARS = 600


def _gauss(n: int = T_BARS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.integers(0, 2, n // 50 + 1), 50)[:n]
    return np.column_stack([rng.normal(np.where(regime == 0, -1.0, 1.0), 1.0), rng.normal(size=n)])


def _bars(n: int = 3000, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    return np.column_stack([close, rng.poisson(50, n).astype(float)])


def _two_level() -> HHMM:
    regime = lambda n: StateNode(n, [StateNode.leaf(f"{n}.a"), StateNode.leaf(f"{n}.b")])
    root = StateNode("root", [regime("bull"), regime("bear")], transition_mask=[[False, True], [True, False]])
    return HHMM(root, [GaussianEmission([0, 1])], n_init=1, max_iter=20, random_state=0)


def _with_days(X: np.ndarray, bars_per_day: int = 60) -> np.ndarray:
    return np.column_stack([X, np.arange(len(X)) // bars_per_day])


def _symbols(n: int = T_BARS, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(np.arange(n) % 2 == 0, rng.integers(9, 18, n), rng.integers(0, 9, n))[:, None].astype(float)


CASES = {
    "pomegranate": (lambda: PomegranateHMM(DistType.NORMAL, n_states=2, max_iter=20), _gauss),
    "hhmm": (_two_level, _gauss),
    "joint_multiscale": (lambda: joint_multiscale([0], [1], n_init=1, max_iter=20), _gauss),
    "embedded": (lambda: embedded_hhmm([0, 1], n_coarse=2, n_fine=2, block_column=2, n_init=1, max_iter=20),
                 lambda: _with_days(_gauss())),
    "legacy": (lambda: legacy_model(n_init=1, max_iter=20), _symbols),
    "tayal": (lambda: TayalHHMM(LegThreshold.legs_per_hour(1.0), n_init=1, max_iter=30,
                                bar_layout=BarLayout(close=0, volume=1)), _bars),
}


@pytest.fixture(scope="module", params=sorted(CASES))
def fitted(request):
    make, data = CASES[request.param]
    X = data()
    return request.param, make().fit(X), X


def test_every_public_base_method_exists(fitted):
    name, model, _ = fitted
    missing = [m for m in PUBLIC if not hasattr(model, m)]
    assert not missing, f"{name} lacks {missing}"


def test_shapes_and_simplices(fitted):
    name, model, X = fitted
    T, K = len(X), model.n_states
    simplex = lambda P: P.shape == (T, K) and np.all(P >= -1e-12) and np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert simplex(np.asarray(model.predict_proba(X)))
    assert simplex(np.asarray(model.state_posteriors(X)))
    assert simplex(np.asarray(model.predict_proba_fixed_lag(X, lag=3)))
    sweep = model.predict_proba_fixed_lag_sweep(X, lags=[0, 3])
    assert set(sweep) == {0, 3} and all(simplex(np.asarray(v)) for v in sweep.values())
    np.testing.assert_allclose(sweep[0], model.predict_proba(X), atol=1e-6)
    for labels in (model.predict(X), model.predict_viterbi(X), model.predict_causal_viterbi(X),
                   model.predict_fixed_lag(X, 3)):
        labels = np.asarray(labels)
        assert labels.shape == (T,) and labels.min() >= 0 and labels.max() < K
    assert np.asarray(model.causal_viterbi_scores(X)).shape == (T, K)
    diag = model.fixed_lag_diagnostics(X, 3)
    assert set(diag) == {"posteriors", "map_labels", "max_posterior", "entropy", "posterior_delta"}
    loglik = np.asarray(model.compute_per_bar_predictive_loglik(X))
    assert loglik.shape == (T,) and np.isfinite(loglik).all()
    assert np.isfinite(model.log_likelihood(X))
    A = model.transition_prob()
    assert A.shape == (K, K) and np.allclose(A.sum(axis=1), 1.0, atol=1e-6)
    pi = model.get_stationary_distribution()
    assert pi.shape == (K,) and np.isclose(pi.sum(), 1.0)
    assert model.forecast(np.asarray(model.predict_proba(X))[-1], 3).shape == (K,)
    assert model.expected_duration(0) >= 1.0
    scores = model.selection_scores(X)
    assert isinstance(scores, SelectionScores) and np.isfinite([scores.aic, scores.bic, scores.icl]).all()
    assert model.path_entropy(X) >= -1e-9
    assert len(model.get_aic_bic(X)) == 2


def test_save_load_round_trip(fitted, tmp_path):
    name, model, X = fitted
    path = tmp_path / f"{name}.joblib"
    model.save(str(path))
    loaded = type(model).load(str(path))
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X), atol=1e-10)


def _export(tmp_path, model, pipeline, X_fit) -> dict:
    pipeline.fit(X_fit)
    joblib.dump(model, tmp_path / "hmm_model.joblib")
    joblib.dump(pipeline, tmp_path / "transform_pipeline.joblib")
    (tmp_path / "metadata.json").write_text(json.dumps({"state_mapping": {}, "target_lag": 3}))
    return {"type": "hmm", "model_path": str(tmp_path)}


@pytest.mark.parametrize("name", ["hhmm", "joint_multiscale", "embedded", "legacy", "tayal"])
def test_hmm_model_wrapper_runs_every_hierarchical_model(name, tmp_path):
    make, data = CASES[name]
    X_raw = data()
    identity = name in ("legacy", "tayal")                 # symbols / prices must not be rescaled
    pipeline = FunctionTransformer() if identity else StandardScaler()
    pipeline.fit(X_raw)
    model = make().fit(pipeline.transform(X_raw))
    spec = _export(tmp_path, model, pipeline, X_raw)
    probs, labels = HmmModelWrapper(spec).predict(X_raw)
    np.testing.assert_allclose(probs, model.predict_proba(pipeline.transform(X_raw)), atol=1e-10)
    lag_probs, _ = HmmModelWrapper(spec, use_fixed_lag_posterior=True).predict(X_raw)
    assert lag_probs.shape == (1, model.n_states)
    loglik = HmmModelWrapper(spec).compute_per_bar_predictive_loglik(X_raw)
    assert loglik.shape == (len(X_raw),)


def test_embedded_block_column_survives_the_scaler():
    X_raw = _with_days(_gauss())
    scaler = StandardScaler().fit(X_raw)
    Xs = scaler.transform(X_raw)
    model = embedded_hhmm([0, 1], n_coarse=2, n_fine=2, block_column=2, n_init=1, max_iter=20).fit(Xs)
    explicit = ExitSchedule.blocks(X_raw[:, 2])
    np.testing.assert_allclose(model.predict_proba(Xs), model.predict_proba(Xs, schedule=explicit), atol=1e-12)
    no_column = embedded_hhmm([0, 1], n_coarse=2, n_fine=2, n_init=0)
    with pytest.raises(ValueError, match="schedule_column"):
        HHMM(no_column.topology.root, [GaussianEmission([0, 1])], schedule_column=1)


@pytest.mark.parametrize("name, window", [("hhmm", 24), ("embedded", 24), ("legacy", 24), ("hhmm", None),
                                          ("tayal", None)])
def test_adaptive_lag_trajectories(name, window):
    """compute_trajectories stacks (n_lags, T, K); bounded history (the production default) needs window_size."""
    make, data = CASES[name]
    X = data()
    model = make().fit(X)
    traj = compute_trajectories(model, X, k_max=4, window_size=window)
    assert traj.shape == (5, len(X), model.n_states)
    assert np.allclose(traj.sum(axis=2), 1.0, atol=1e-6)


def test_tayal_rejects_a_bounded_window():
    make, data = CASES["tayal"]
    X = data()
    with pytest.raises(NotImplementedError, match="window_size"):
        make().fit(X).predict_proba_fixed_lag_sweep(X, [0, 2], window_size=24)


# ---------------------------------------------------------------------------------------------------------------------
# Correctness of the added methods
# ---------------------------------------------------------------------------------------------------------------------
def _small_model(seed: int = 3) -> tuple[HHMM, np.ndarray]:
    model = _two_level()
    rng = np.random.default_rng(seed)
    model.fit(_gauss(300, seed))
    _, X = model.sample(6, seed=seed)
    return model, X


def _paths(model: HHMM, X: np.ndarray):
    n, T = model.n_states, len(X)
    log_b = model._log_emissions(X)
    with np.errstate(divide="ignore"):
        log_t, log_pi = np.log(model.transition_prob()), np.log(model.initial_distribution())
    paths = np.array(list(itertools.product(range(n), repeat=T)))
    prefix = [log_pi[paths[:, 0]] + log_b[0, paths[:, 0]]]
    for t in range(1, T):
        prefix.append(prefix[-1] + log_t[paths[:, t - 1], paths[:, t]] + log_b[t, paths[:, t]])
    return paths, prefix


def test_path_entropy_and_viterbi_scores_match_enumeration():
    model, X = _small_model()
    paths, prefix = _paths(model, X)
    score = prefix[-1]
    post = np.exp(score - logsumexp(score))
    exact = -np.sum(post[post > 0] * np.log(post[post > 0]))
    assert model.path_entropy(X) == pytest.approx(exact, abs=1e-9)
    scores = model.causal_viterbi_scores(X)
    for t in range(len(X)):
        best = [np.max(prefix[t][paths[:, t] == j]) for j in range(model.n_states)]
        np.testing.assert_allclose(scores[t], best, atol=1e-9)
    np.testing.assert_array_equal(scores.argmax(axis=1), model.predict_causal_viterbi(X))


def test_windowed_fixed_lag_matches_the_base_algorithm():
    model, _ = _small_model(4)
    _, X = model.sample(40, seed=5)
    log_b = model._log_emissions(X)
    with np.errstate(divide="ignore"):
        log_A, log_pi = np.log(model.transition_prob()), np.log(model.initial_distribution())
    lag, W = 3, 10
    expected = np.empty((len(X), model.n_states))
    for t in range(len(X)):                       # BasePomegranateHMM's bounded-history recomputation
        end = min(t + lag, len(X) - 1)
        start = max(0, end - W + 1)
        la = log_pi + log_b[start]
        for j in range(start + 1, t + 1):
            la = logsumexp(la[:, None] + log_A, axis=0) + log_b[j]
        lb = np.zeros(model.n_states)
        for s in range(end, t, -1):
            lb = logsumexp(log_A + log_b[s] + lb, axis=1)
        p = la + lb
        expected[t] = np.exp(p - logsumexp(p))
    got = model.predict_proba_fixed_lag_sweep(X, [lag], window_size=W)[lag]
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_expected_level_duration_matches_simulation():
    model, _ = _small_model(6)
    leaves, _ = model.sample(200000, seed=7)
    top = model.leaf_to_level(leaves, 1)
    change = np.flatnonzero(np.diff(top) != 0) + 1
    runs = np.diff(np.r_[0, change, len(top)])[1:-1]                # complete runs only
    first = top[np.r_[0, change][1:-1]]
    for k in range(2):
        assert model.expected_level_duration(k) == pytest.approx(runs[first == k].mean(), rel=0.05)
    assert model.expected_level_duration("bull") == model.expected_level_duration(0)
    for leaf in range(model.n_states):
        assert model.expected_level_duration(leaf, level=2) == pytest.approx(model.expected_duration(leaf))


def test_level_functions_marginalise_the_leaf_api():
    model, X = _small_model(8)
    np.testing.assert_allclose(model.predict_proba_level(X, 1), model.level_marginal(model.predict_proba(X), 1))
    np.testing.assert_allclose(model.predict_proba_fixed_lag_level(X, 2, 1),
                               model.level_marginal(model.predict_proba_fixed_lag(X, 2), 1))
    np.testing.assert_allclose(model.stationary_level_distribution(1).sum(), 1.0)


# ---------------------------------------------------------------------------------------------------------------------
# Review round 4
# ---------------------------------------------------------------------------------------------------------------------
def test_tayal_viterbi_scores_follow_the_legs():
    """-inf is a valid score (an unreachable leaf): the held rows must still update with every confirmed leg."""
    X = _bars()
    model = TayalHHMM(LegThreshold.legs_per_hour(1.0), n_init=1, max_iter=30).fit(X)
    scores = model.causal_viterbi_scores(X)
    assert np.isneginf(scores).any()                                             # the case that broke it
    np.testing.assert_array_equal(scores.argmax(axis=1), model.predict_causal_viterbi(X))
    assert len(np.unique(scores[np.isfinite(scores).any(axis=1)], axis=0)) > 10


def _gappy_frame(n: int = 3000, seed: int = 4) -> pd.DataFrame:
    X = _bars(n, seed)
    index = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    index = index + pd.to_timedelta((np.arange(n) // 288) * 2, unit="h")      # a 2-hour gap after every day
    return pd.DataFrame(X, index=index, columns=["close", "volume"])


def test_tayal_refuses_numpy_when_it_reads_the_clock():
    frame = _gappy_frame()
    from okmich_quant_ml.hmm.hierarchical.inputs import VolumeRateUnit
    from okmich_quant_ml.hmm.hierarchical.flavours import FlowInput
    clocked = TayalHHMM(LegThreshold.legs_per_hour(1.0), flow=FlowInput.VOLUME_RATE, n_init=1, max_iter=20,
                        bar_layout=BarLayout(close=0, volume=1)).fit(frame)
    assert clocked.reads_the_clock
    with pytest.raises(ValueError, match="reads the clock"):
        clocked.predict_proba(frame.to_numpy())
    clock_free = TayalHHMM(LegThreshold.legs_per_hour(1.0), flow=FlowInput.VOLUME_RATE, volume_unit=VolumeRateUnit.BAR,
                           n_init=1, max_iter=20, bar_layout=BarLayout(close=0, volume=1)).fit(frame)
    assert not clock_free.reads_the_clock
    np.testing.assert_allclose(clock_free.predict_proba(frame.to_numpy()), clock_free.predict_proba(frame))


def test_models_saved_before_later_attributes_still_load(tmp_path):
    model = _two_level().fit(_gauss())
    X = _gauss(100, 9)
    expected = model.predict_proba(X)
    for attr in ("schedule_column", "schedule_depth", "inference_mode"):
        del model.__dict__[attr]
    joblib.dump(model, tmp_path / "old.joblib")
    old = HHMM.load(str(tmp_path / "old.joblib"))
    np.testing.assert_allclose(old.predict_proba(X), expected)
    tayal = TayalHHMM(LegThreshold.legs_per_hour(1.0), n_init=1, max_iter=20).fit(_bars())
    bars_expected = tayal.predict_proba(_bars())
    for attr in ("bar_layout", "bar_interval", "inference_mode"):
        del tayal.__dict__[attr]
    joblib.dump(tayal, tmp_path / "old_tayal.joblib")
    np.testing.assert_allclose(TayalHHMM.load(str(tmp_path / "old_tayal.joblib")).predict_proba(_bars()), bars_expected)


def test_lengths_must_be_whole_numbers():
    model = _two_level().fit(_gauss())
    with pytest.raises(ValueError, match="whole numbers"):
        model.filter(_gauss(5), lengths=[2.5, 2.5])
    with pytest.raises(ValueError, match="whole numbers"):
        PomegranateHMM(DistType.NORMAL, n_states=2, max_iter=5).fit(_gauss(), lengths=[300.5, 299.5])


def test_an_embedded_model_scores_its_own_sample():
    model = embedded_hhmm([0, 1], n_coarse=2, n_fine=2, block_column=2, n_init=1, max_iter=10).fit(_with_days(_gauss()))
    sched = ExitSchedule.blocks(np.arange(300) // 50)
    with pytest.raises(ValueError, match="pass the schedule"):
        model.sample(300)
    _, X = model.sample(300, seed=1, schedule=sched)
    assert X.shape[1] == 3
    np.testing.assert_array_equal(X[:, 2], np.arange(300) // 50)
    np.testing.assert_allclose(model.predict_proba(X), model.predict_proba(X, schedule=sched))


def test_tayal_bar_adapter_semantics():
    X = _bars()
    model = TayalHHMM(LegThreshold.legs_per_hour(1.0), n_init=1, max_iter=30,
                      bar_layout=BarLayout(close=0, volume=1)).fit(X)
    md, legs, S = model._prepare(X)
    first = legs[0].confirm_bar
    probs = model.predict_proba(X)
    np.testing.assert_allclose(probs[:first], np.tile(model.model_.initial_distribution(), (first, 1)))
    leg_post = model.model_.filter(S)
    np.testing.assert_allclose(probs[legs[5].confirm_bar], leg_post[5])
    np.testing.assert_allclose(probs[legs[6].confirm_bar - 1], leg_post[5])          # held until the next leg
    loglik = model.compute_per_bar_predictive_loglik(X)
    assert loglik.sum() == pytest.approx(model.log_likelihood(X))
    np.testing.assert_allclose(model.predict_proba_fixed_lag(X, 0), probs)
    huge = model.predict_proba_fixed_lag(X, len(X))                                  # every leg known
    np.testing.assert_allclose(huge[legs[5].confirm_bar], model.model_.smooth(S)[5], atol=1e-10)
    regular = pd.date_range("2000-01-03", periods=len(X), freq="5min", tz="UTC")
    same = model.predict_proba(MarketData(close=pd.Series(X[:, 0], index=regular),
                                          volume=pd.Series(X[:, 1], index=regular)))
    np.testing.assert_allclose(same, probs)
    assert model.predict_proba_level(X).shape == (len(X), 2)
    with pytest.raises(NotImplementedError):
        model.fit(X, lengths=[len(X)])
