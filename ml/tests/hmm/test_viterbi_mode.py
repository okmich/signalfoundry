"""``InferenceMode.VITERBI`` must return the TRUE Viterbi path: max-product forward, then backward traceback.

History: until 2026-09-27 this mode called pomegranate 1.x ``DenseHMM.predict``. That is the argmax of the
forward-backward posterior, and it returns exactly the SMOOTHING labels. On the 3000-bar, 3-state fixture below
the two decoders disagree on about 1% of bars. Every test here except the save/load and ``predict_proba`` ones
fails against that implementation.

Run with: pytest tests/hmm/test_viterbi_mode.py -v
"""

import numpy as np
import pytest

from okmich_quant_ml.hmm import DistType, InferenceMode, PomegranateHMM, PomegranateMixtureHMM


@pytest.fixture(scope="module")
def noisy():
    """Sticky 3-state chain with overlapping 2-D emissions, so the per-bar MAP and the MAP path disagree."""
    rng = np.random.default_rng(7)
    T, K = 3000, 3
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        s[t] = s[t - 1] if rng.random() < 0.95 else rng.integers(0, K)
    means = np.array([[-1.0, 0.5], [0.0, -0.5], [1.0, 0.8]])
    return means[s] + 0.9 * rng.standard_normal((T, 2))


@pytest.fixture(scope="module")
def fitted(noisy):
    model = PomegranateHMM(DistType.NORMAL, n_states=3, random_state=3, max_iter=50,
                           inference_mode=InferenceMode.VITERBI)
    model.fit(noisy)
    return model


def _reference_viterbi(log_pi, log_A, log_B):
    """Textbook max-product with backpointers and a full backward traceback."""
    T, K = log_B.shape
    delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = log_pi + log_B[0]
    for t in range(1, T):
        scores = delta[t - 1, :, None] + log_A
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = np.max(scores, axis=0) + log_B[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def _path_log_prob(log_pi, log_A, log_B, path):
    """Joint log-probability of one state path with the observations."""
    return (log_pi[path[0]] + log_B[0, path[0]] + log_A[path[:-1], path[1:]].sum()
            + log_B[np.arange(1, len(path)), path[1:]].sum())


def _params(model, X):
    return model._extract_hmm_parameters(model._preprocess_input(X))


def _labels(model, X, mode):
    model.inference_mode = mode
    try:
        return np.asarray(model.predict(X))
    finally:
        model.inference_mode = InferenceMode.VITERBI


# --------------------------------------------------------------------------- correctness
def test_viterbi_matches_the_reference_implementation(fitted, noisy):
    np.testing.assert_array_equal(fitted.predict(noisy), _reference_viterbi(*_params(fitted, noisy)))


def test_viterbi_path_maximises_the_joint_probability(fitted, noisy):
    """What defines Viterbi: no other state path scores higher, including the smoothed per-bar argmax."""
    log_pi, log_A, log_B = _params(fitted, noisy)
    best = fitted.predict(noisy)
    best_lp = _path_log_prob(log_pi, log_A, log_B, best)
    assert best_lp >= _path_log_prob(log_pi, log_A, log_B, _labels(fitted, noisy, InferenceMode.SMOOTHING)) - 1e-9
    rng = np.random.default_rng(1)
    for _ in range(50):
        other = best.copy()
        idx = rng.integers(0, len(other), 5)
        other[idx] = rng.integers(0, fitted.n_states, 5)
        assert best_lp >= _path_log_prob(log_pi, log_A, log_B, other) - 1e-9


def test_viterbi_is_not_the_smoothed_argmax(fitted, noisy):
    """Regression guard for the original defect: VITERBI silently returned SMOOTHING."""
    viterbi = fitted.predict(noisy)
    smoothed = _labels(fitted, noisy, InferenceMode.SMOOTHING)
    assert (viterbi != smoothed).any(), "VITERBI equals SMOOTHING again -- the mode is not running Viterbi"


def test_termination_step_equals_causal_viterbi(fitted, noisy):
    """Viterbi's last label is argmax of the max-product frontier, i.e. CAUSAL_VITERBI's label at that bar."""
    causal = fitted.predict_causal_viterbi(noisy)
    for t in (50, 777, 1500, len(noisy) - 1):
        assert fitted.predict_viterbi(noisy[: t + 1])[-1] == causal[t]


def test_viterbi_rewrites_history_as_data_arrives(fitted, noisy):
    """It is a smoother: some earlier label changes once later bars arrive (unlike CAUSAL_VITERBI)."""
    full = fitted.predict(noisy)
    assert any(not np.array_equal(fitted.predict(noisy[: t + 1]), full[: t + 1]) for t in range(200, 3000, 200))


# --------------------------------------------------------------------------- dispatch and API
def test_predict_dispatches_on_viterbi_mode(fitted, noisy):
    np.testing.assert_array_equal(fitted.predict(noisy), fitted.predict_viterbi(noisy))
    assert fitted.predict(noisy).dtype == np.int64


def test_predict_proba_rejects_viterbi(fitted, noisy):
    with pytest.raises(ValueError, match="VITERBI"):
        fitted.predict_proba(noisy)


def test_mode_survives_a_save_load_round_trip(fitted, noisy, tmp_path):
    path = str(tmp_path / "m.pkl")
    fitted.save(path)
    reloaded = PomegranateHMM.load(path)
    assert reloaded.inference_mode == InferenceMode.VITERBI
    np.testing.assert_array_equal(reloaded.predict(noisy), fitted.predict(noisy))


def test_mixture_model_runs_true_viterbi(noisy):
    model = PomegranateMixtureHMM(DistType.NORMAL, n_states=2, n_components=2, random_state=5, max_iter=30,
                                  inference_mode=InferenceMode.VITERBI)
    X = noisy[:800]
    model.fit(X)
    np.testing.assert_array_equal(model.predict(X), _reference_viterbi(*_params(model, X)))


# --------------------------------------------------------------------------- failure modes
def test_infeasible_path_raises_instead_of_tying_on_index_zero(fitted, noisy, monkeypatch):
    log_pi, log_A, log_B = _params(fitted, noisy)
    log_B = log_B.copy()
    log_B[30] = -np.inf                                   # no state can emit this bar
    monkeypatch.setattr(fitted, "_extract_hmm_parameters", lambda _X: (log_pi, log_A, log_B))
    with pytest.raises(ValueError, match="no feasible state path exists at bar 30"):
        fitted.predict_viterbi(noisy)


def test_rejects_unfitted_and_dirty_input(noisy):
    unfitted = PomegranateHMM(DistType.NORMAL, n_states=2)
    with pytest.raises(RuntimeError, match="not been fitted"):
        unfitted.predict_viterbi(noisy)
    model = PomegranateHMM(DistType.NORMAL, n_states=2, max_iter=20)
    model.fit(noisy[:500])
    dirty = noisy[:500].copy()
    dirty[5] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.predict_viterbi(dirty)
