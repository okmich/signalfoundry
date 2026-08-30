"""Invariants for CAUSAL_VITERBI decoding on :class:`BasePomegranateHMM`.

The failure modes pinned here are all SILENT. A decoder that leaks future information, a max-product
recursion that has quietly become sum-product, or an infeasible path that ties ``argmax`` on index 0
each return a plausible label vector rather than raising. Only a direct invariant catches them.

Run with: pytest tests/hmm/test_causal_viterbi.py -v
"""

import numpy as np
import pytest

from okmich_quant_ml.hmm import DistType, InferenceMode, PomegranateHMM
from okmich_quant_ml.hmm.base_pomegranate import BasePomegranateHMM


@pytest.fixture
def regime_data():
    """Two well-separated regimes with several genuine switches, so a path decoder has work to do."""
    rng = np.random.default_rng(11)
    blocks = [(0.0, 40), (3.0, 35), (0.0, 30), (3.0, 45)]
    return np.concatenate([rng.normal(mu, 0.6, size=(n, 1)) for mu, n in blocks])


@pytest.fixture
def fitted(regime_data):
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2,
                           inference_mode=InferenceMode.CAUSAL_VITERBI)
    model.fit(regime_data)
    return model


# --------------------------------------------------------------------------- causality
def test_causal_viterbi_is_prefix_consistent(fitted, regime_data):
    """The label at ``t`` must not move when later bars are appended - the whole point of the operator."""
    full = fitted.predict_causal_viterbi(regime_data)
    for t in (17, 44, 75, 119, len(regime_data) - 1):
        prefix = fitted.predict_causal_viterbi(regime_data[: t + 1])
        assert prefix[-1] == full[t], f"label at t={t} changed once future bars were appended"


def test_causal_viterbi_prefix_agrees_over_the_whole_prefix(fitted, regime_data):
    """Stronger form: the entire earlier label vector is stable, not merely its last element."""
    full = fitted.predict_causal_viterbi(regime_data)
    cut = 90
    np.testing.assert_array_equal(fitted.predict_causal_viterbi(regime_data[:cut]), full[:cut])


def test_standard_viterbi_is_not_prefix_consistent(regime_data):
    """Control: the property under test is a real distinction, not one every decoder happens to satisfy.

    Standard Viterbi's backward traceback rewrites earlier labels, so at least one bar must disagree
    between the live view and the completed-sequence view. If this ever passes, the fixture stopped
    exercising a path switch and the causal tests above are no longer proving anything.
    """
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2,
                           inference_mode=InferenceMode.VITERBI)
    model.fit(regime_data)
    full = model.predict(regime_data)
    rewrote = any(not np.array_equal(model.predict(regime_data[: t + 1]), full[: t + 1])
                  for t in range(20, len(regime_data), 10))
    assert rewrote, "expected standard Viterbi to rewrite history; fixture no longer exercises a switch"


# --------------------------------------------------------------------------- equivalence
def test_equals_expanding_window_viterbi_terminal_states(fitted, regime_data):
    """The identity that makes the operator cheap: full Viterbi's TERMINATION step is this decoder.

    Re-running full Viterbi on every prefix and keeping only its last element is ``O(T^2 K^2)`` and
    must return exactly what the ``O(T K^2)`` forward-only pass returns.
    """
    X = fitted._preprocess_input(regime_data[:60])
    log_pi, log_A, log_B = fitted._extract_hmm_parameters(X)

    expanding = np.array([_full_viterbi(log_pi, log_A, log_B[: t + 1])[-1] for t in range(len(X))])
    np.testing.assert_array_equal(fitted.predict_causal_viterbi(regime_data[:60]), expanding)


def _full_viterbi(log_pi, log_A, log_B):
    """Reference implementation: max-product forward WITH backpointers and a full backward traceback."""
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


def test_max_product_differs_from_sum_product(fitted, regime_data):
    """Guards against the recursion silently reverting to ``logaddexp`` - which would still run clean.

    Both aggregate the same prefix over the same observations; only ``max`` versus ``sum`` differs, and
    that difference is the entire behavioural claim. Identical output means the max is not happening.
    """
    X = fitted._preprocess_input(regime_data)
    log_pi, log_A, log_B = fitted._extract_hmm_parameters(X)
    delta = BasePomegranateHMM._max_product_forward_pass(log_pi, log_A, log_B)
    alpha = BasePomegranateHMM._forward_pass(log_pi, log_A, log_B)

    assert not np.allclose(delta, alpha), "max-product frontier equals sum-product; the max was lost"
    # max over paths can never exceed the sum over the same paths
    assert np.all(delta <= alpha + 1e-9), "a single best path scored above the sum over all paths"


def test_max_product_frontier_is_a_path_score_not_a_distribution(fitted, regime_data):
    scores = fitted.causal_viterbi_scores(regime_data)
    assert scores.shape == (len(regime_data), fitted.n_states)
    assert not np.allclose(np.exp(scores).sum(axis=1), 1.0)


# --------------------------------------------------------------------------- behaviour
def test_causal_viterbi_flips_less_than_filtered_argmax(fitted, regime_data):
    """The transition penalty must actually buy persistence, else the decoder has no reason to exist."""
    causal = fitted.predict_causal_viterbi(regime_data)
    filtered = np.argmax(fitted._predict_proba_filtered(fitted._preprocess_input(regime_data)), axis=1)
    assert np.mean(causal[1:] != causal[:-1]) <= np.mean(filtered[1:] != filtered[:-1])


def test_causal_viterbi_recovers_the_regimes(fitted, regime_data):
    """Sanity: persistence must not have been bought by collapsing to one state."""
    labels = fitted.predict_causal_viterbi(regime_data)
    assert set(np.unique(labels)) == {0, 1}
    assert 1 < np.sum(labels[1:] != labels[:-1]) < 20


# --------------------------------------------------------------------------- dispatch & guards
def test_predict_dispatches_on_causal_viterbi_mode(fitted, regime_data):
    np.testing.assert_array_equal(fitted.predict(regime_data), fitted.predict_causal_viterbi(regime_data))


def test_predict_proba_rejects_causal_viterbi(fitted, regime_data):
    with pytest.raises(ValueError, match="CAUSAL_VITERBI"):
        fitted.predict_proba(regime_data)


def test_mode_survives_a_save_load_round_trip(fitted, regime_data, tmp_path):
    path = str(tmp_path / "m.pkl")
    fitted.save(path)
    reloaded = PomegranateHMM.load(path)
    assert reloaded.inference_mode == InferenceMode.CAUSAL_VITERBI
    np.testing.assert_array_equal(reloaded.predict(regime_data), fitted.predict(regime_data))


def test_infeasible_path_raises_instead_of_tying_on_index_zero(fitted, regime_data, monkeypatch):
    """An all -inf frontier row makes ``argmax`` return 0 silently; that must be an error, not a label."""
    X = fitted._preprocess_input(regime_data)
    log_pi, log_A, log_B = fitted._extract_hmm_parameters(X)
    log_B = log_B.copy()
    log_B[30] = -np.inf                                   # no state can emit this bar
    monkeypatch.setattr(fitted, "_extract_hmm_parameters", lambda _X: (log_pi, log_A, log_B))
    with pytest.raises(ValueError, match="no feasible state path exists at bar 30"):
        fitted.predict_causal_viterbi(regime_data)


def test_rejects_unfitted_and_dirty_input(regime_data):
    unfitted = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)
    with pytest.raises(RuntimeError, match="not been fitted"):
        unfitted.predict_causal_viterbi(regime_data)

    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)
    model.fit(regime_data)
    dirty = regime_data.copy()
    dirty[5] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.predict_causal_viterbi(dirty)
