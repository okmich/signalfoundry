"""Tests for the four operators migrated from regime_filters into posterior_inference."""
import numpy as np
import pytest

from okmich_quant_ml.posterior_inference import (
    AbstainMode,
    ConfidenceHysteresisInferer,
    ConfidenceWeightedModeInferer,
    KalmanPosteriorTransformer,
    ViterbiInferer,
)


def _confident_argmax_series(label_sequence: list[int], K: int, confidence: float = 0.9) -> np.ndarray:
    T = len(label_sequence)
    p = np.full((T, K), (1.0 - confidence) / (K - 1), dtype=float)
    for t, label in enumerate(label_sequence):
        p[t, label] = confidence
    return p


# --- ViterbiInferer -----------------------------------------------------------

def test_viterbi_inferer_zero_transition_cost_matches_argmax() -> None:
    probs = _confident_argmax_series([0, 1, 2, 0, 1], K=3)
    inferer = ViterbiInferer(transition_cost=np.zeros((3, 3)))

    labels = inferer.infer(probs)

    np.testing.assert_array_equal(labels, np.argmax(probs, axis=1))


def test_viterbi_inferer_high_transition_cost_smooths_single_bar_flip() -> None:
    # Run of 0s with a single noisy bar of 1 in the middle. Strong transition penalty should suppress it.
    seq = [0, 0, 0, 1, 0, 0, 0]
    probs = _confident_argmax_series(seq, K=2, confidence=0.55)  # weak observation
    inferer = ViterbiInferer.with_uniform_smoothing(n_states=2, off_diagonal_cost=5.0)

    labels = inferer.infer(probs)

    assert labels.tolist() == [0] * 7


def test_viterbi_inferer_low_transition_cost_follows_strong_observation() -> None:
    # Same flip but very confident observation — Viterbi should accept it.
    seq = [0, 0, 0, 1, 0, 0, 0]
    probs = _confident_argmax_series(seq, K=2, confidence=0.99)
    inferer = ViterbiInferer.with_uniform_smoothing(n_states=2, off_diagonal_cost=0.5)

    labels = inferer.infer(probs)

    assert labels.tolist() == seq


def test_viterbi_inferer_from_transition_probabilities_round_trip() -> None:
    transmat = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)
    inferer = ViterbiInferer.from_transition_probabilities(transmat)

    # cost = -log(transmat + eps). Self-loop cost ~ -log(0.9), off-diagonal ~ -log(0.1).
    assert inferer.transition_cost[0, 0] == pytest.approx(-np.log(0.9), abs=1e-6)
    assert inferer.transition_cost[0, 1] == pytest.approx(-np.log(0.1), abs=1e-6)


def test_viterbi_inferer_metadata_reports_path_cost_decomposition() -> None:
    probs = _confident_argmax_series([0, 1, 0, 1, 0], K=2)
    inferer = ViterbiInferer.with_uniform_smoothing(n_states=2, off_diagonal_cost=1.0)
    inferer.infer(probs)

    meta = inferer.get_metadata()
    assert meta["inferer"] == "ViterbiInferer"
    assert meta["n_bars"] == 5
    assert meta["n_states"] == 2
    assert meta["total_path_cost"] == pytest.approx(meta["total_unary_cost"] + meta["total_transition_cost"])


def test_viterbi_inferer_rejects_k_mismatch() -> None:
    probs = _confident_argmax_series([0, 1], K=3)
    inferer = ViterbiInferer(transition_cost=np.zeros((2, 2)))

    with pytest.raises(ValueError, match="does not match transition_cost"):
        inferer.infer(probs)


def test_viterbi_inferer_rejects_bad_observation_weight() -> None:
    with pytest.raises(ValueError, match="observation_weight"):
        ViterbiInferer(transition_cost=np.zeros((3, 3)), observation_weight=0.0)


def test_viterbi_inferer_rejects_nan_observation_weight_and_eps() -> None:
    with pytest.raises(ValueError, match="observation_weight must be finite and > 0"):
        ViterbiInferer(transition_cost=np.zeros((3, 3)), observation_weight=float("nan"))
    with pytest.raises(ValueError, match="eps must be finite and > 0"):
        ViterbiInferer(transition_cost=np.zeros((3, 3)), eps=float("nan"))


def test_viterbi_from_transition_probabilities_rejects_non_stochastic_rows() -> None:
    # Row sums are 0.8 and 0.9 — not stochastic. Method name promises probabilities, so it must reject.
    bad_transmat = np.array([[0.7, 0.1], [0.3, 0.6]], dtype=float)

    with pytest.raises(ValueError, match="rows must sum to 1"):
        ViterbiInferer.from_transition_probabilities(bad_transmat)


def test_viterbi_from_transition_probabilities_rejects_nan() -> None:
    bad_transmat = np.array([[np.nan, 1.0], [0.5, 0.5]], dtype=float)

    with pytest.raises(ValueError, match="NaN or Inf"):
        ViterbiInferer.from_transition_probabilities(bad_transmat)


def test_viterbi_from_transition_probabilities_accepts_valid_stochastic_matrix() -> None:
    transmat = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    inferer = ViterbiInferer.from_transition_probabilities(transmat)

    # cost = -log(transmat); confirm round-trip.
    np.testing.assert_allclose(inferer.transition_cost, -np.log(transmat), atol=1e-6)


def test_viterbi_from_transition_probabilities_maps_zero_to_inf_cost() -> None:
    transmat = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=float)
    inferer = ViterbiInferer.from_transition_probabilities(transmat)

    assert inferer.transition_cost[0, 1] == np.inf
    assert inferer.transition_cost[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_viterbi_inferer_never_crosses_a_forbidden_transition() -> None:
    # State 0 -> 1 has zero model probability (transmat[0, 1] == 0): that specific transition must
    # never appear in the decoded path, no matter how the unary evidence swings bar-to-bar. An
    # eps-clipped finite cost for the zero transition would let strong-enough evidence buy a
    # crossing anyway; +inf cannot be outbid regardless of the evidence sequence, so this checks
    # the invariant directly rather than hand-predicting one specific globally-optimal path.
    transmat = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=float)
    inferer = ViterbiInferer.from_transition_probabilities(transmat)
    rng = np.random.default_rng(7)
    probs = rng.dirichlet(alpha=[1.0, 1.0], size=30)
    probs[5] = [0.99, 0.01]   # strong state-0 evidence immediately followed by...
    probs[6] = [0.01, 0.99]   # ...strong state-1 evidence: exactly the pattern that would buy a
    #                           crossing under a merely-expensive (eps-clipped) forbidden transition.

    labels = inferer.infer(probs)

    forbidden = (labels[:-1] == 0) & (labels[1:] == 1)
    assert not forbidden.any()


def test_viterbi_inferer_constructor_accepts_inf_transition_cost() -> None:
    cost = np.array([[0.0, np.inf], [1.0, 0.0]], dtype=float)
    inferer = ViterbiInferer(transition_cost=cost)

    assert inferer.transition_cost[0, 1] == np.inf


def test_viterbi_inferer_constructor_rejects_nan_and_neg_inf_transition_cost() -> None:
    with pytest.raises(ValueError, match="NaN or -Inf"):
        ViterbiInferer(transition_cost=np.array([[0.0, np.nan], [1.0, 0.0]], dtype=float))
    with pytest.raises(ValueError, match="NaN or -Inf"):
        ViterbiInferer(transition_cost=np.array([[0.0, -np.inf], [1.0, 0.0]], dtype=float))


def test_viterbi_inferer_raises_when_no_finite_path_exists() -> None:
    # Every transition is forbidden — for T >= 2 there is no finite-cost label sequence at all.
    # The all-inf DP frontier must not silently resolve via np.argmin's index-0 tie-break; the
    # decoder must signal infeasibility instead of returning a bogus, infinite-cost path.
    cost = np.full((2, 2), np.inf, dtype=float)
    inferer = ViterbiInferer(transition_cost=cost)
    probs = np.array([[0.6, 0.4], [0.5, 0.5]], dtype=float)

    with pytest.raises(ValueError, match="no finite-cost"):
        inferer.infer(probs)


# --- KalmanPosteriorTransformer -----------------------------------------------

def test_kalman_posterior_transformer_preserves_shape_and_simplex() -> None:
    probs = _confident_argmax_series([0, 1, 2, 0, 1, 2, 0], K=3, confidence=0.8)
    transformer = KalmanPosteriorTransformer(process_noise=0.1, measurement_noise=0.2)

    out = transformer.transform(probs)

    assert out.shape == probs.shape
    np.testing.assert_allclose(out.sum(axis=1), np.ones(len(probs)), atol=1e-9)
    assert (out >= 0.0).all()


def test_kalman_posterior_transformer_high_process_noise_pulls_toward_uniform() -> None:
    # process_noise=1 fully resets belief to uniform each bar, before the update applies.
    probs = _confident_argmax_series([0, 0, 0, 0, 0], K=3, confidence=0.50)
    transformer = KalmanPosteriorTransformer(process_noise=1.0, measurement_noise=10.0)

    out = transformer.transform(probs)

    # With strong process noise diffusing toward uniform and a heavily-tempered update,
    # the transformed posterior is closer to uniform than the input was.
    input_max = float(probs.max(axis=1).mean())
    output_max = float(out.max(axis=1).mean())
    assert output_max < input_max


def test_kalman_posterior_transformer_low_noise_tracks_observation() -> None:
    # Very low process_noise + very low measurement_noise → output should closely match input.
    probs = _confident_argmax_series([0, 1, 2, 0, 1, 2], K=3, confidence=0.85)
    transformer = KalmanPosteriorTransformer(process_noise=0.01, measurement_noise=0.01)

    out = transformer.transform(probs)

    np.testing.assert_array_equal(np.argmax(out, axis=1), np.argmax(probs, axis=1))


def test_kalman_posterior_transformer_extreme_low_noise_does_not_underflow_to_uniform() -> None:
    # measurement_noise = 1e-6 ⇒ measurement_weight ≈ 1e6. The naive linear-space update
    # ``probs ** 1e6`` underflows for every class and would collapse the belief to uniform;
    # the log-space update tracks the observation argmax cleanly.
    probs = _confident_argmax_series([0, 1, 2, 0, 1, 2, 0, 1], K=3, confidence=0.7)
    transformer = KalmanPosteriorTransformer(process_noise=0.05, measurement_noise=1e-6)

    out = transformer.transform(probs)

    np.testing.assert_array_equal(np.argmax(out, axis=1), np.argmax(probs, axis=1))
    # Output must not be uniform — that would indicate underflow collapse.
    K = probs.shape[1]
    assert (out.max(axis=1) > 1.0 / K + 0.05).all()


def test_kalman_posterior_transformer_handles_empty_input() -> None:
    transformer = KalmanPosteriorTransformer()
    out = transformer.transform(np.zeros((0, 3), dtype=float))

    assert out.shape == (0, 3)


def test_kalman_posterior_transformer_rejects_invalid_hyperparams() -> None:
    with pytest.raises(ValueError, match="process_noise"):
        KalmanPosteriorTransformer(process_noise=1.5)
    with pytest.raises(ValueError, match="measurement_noise"):
        KalmanPosteriorTransformer(measurement_noise=0.0)
    with pytest.raises(ValueError, match="adaptation_rate"):
        KalmanPosteriorTransformer(adaptation_rate=1.0)
    with pytest.raises(ValueError, match="error_window"):
        KalmanPosteriorTransformer(error_window=0)


def test_kalman_posterior_transformer_rejects_bool_and_non_integral_error_window() -> None:
    with pytest.raises(ValueError, match="must be an int, not bool"):
        KalmanPosteriorTransformer(error_window=True)
    with pytest.raises(ValueError, match="must be an integer value"):
        KalmanPosteriorTransformer(error_window=5.5)


def test_kalman_posterior_transformer_rejects_nan_measurement_noise_and_eps() -> None:
    with pytest.raises(ValueError, match="measurement_noise must be finite and > 0"):
        KalmanPosteriorTransformer(measurement_noise=float("nan"))
    with pytest.raises(ValueError, match="eps must be finite and > 0"):
        KalmanPosteriorTransformer(eps=float("nan"))


# --- ConfidenceWeightedModeInferer --------------------------------------------

def test_confidence_weighted_mode_inferer_returns_int64_labels() -> None:
    probs = _confident_argmax_series([0, 0, 0, 1, 1, 1, 2], K=3)
    inferer = ConfidenceWeightedModeInferer(window=3)

    labels = inferer.infer(probs)

    assert labels.dtype == np.int64
    assert labels.shape == (7,)


def test_confidence_weighted_mode_inferer_smooths_single_outlier() -> None:
    # Run of 0s with a single bar of 1 — the window-mode should keep 0 dominant.
    seq = [0, 0, 0, 1, 0, 0, 0]
    probs = _confident_argmax_series(seq, K=2, confidence=0.7)
    inferer = ConfidenceWeightedModeInferer(window=5, confidence_weight=1.0)

    labels = inferer.infer(probs)

    # At t=3, the trailing window [0..3] has three 0s and one 1, so the mode is 0.
    assert labels[3] == 0


def test_confidence_weighted_mode_inferer_threshold_triggers_abstain() -> None:
    probs = _confident_argmax_series([0, 0, 1, 1, 0], K=2, confidence=0.6)
    inferer = ConfidenceWeightedModeInferer(window=3, min_score_threshold=10.0,
                                            abstain_mode=AbstainMode.FLAT, abstain_label=-1)

    labels = inferer.infer(probs)

    assert (labels == -1).all()


def test_confidence_weighted_mode_inferer_metadata_reports_gate_rate() -> None:
    probs = _confident_argmax_series([0, 1, 0, 1, 0], K=2)
    inferer = ConfidenceWeightedModeInferer(window=3, min_score_threshold=0.0)
    inferer.infer(probs)

    meta = inferer.get_metadata()
    assert meta["inferer"] == "ConfidenceWeightedModeInferer"
    assert "gate_open_rate" in meta
    assert meta["window"] == 3


def test_confidence_weighted_mode_inferer_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="window"):
        ConfidenceWeightedModeInferer(window=0)


def test_confidence_weighted_mode_inferer_rejects_bool_and_non_integral_window() -> None:
    with pytest.raises(ValueError, match="must be an int, not bool"):
        ConfidenceWeightedModeInferer(window=True)
    with pytest.raises(ValueError, match="must be an integer value"):
        ConfidenceWeightedModeInferer(window=3.5)


def test_confidence_weighted_mode_inferer_rejects_nan_confidence_weight_and_threshold() -> None:
    with pytest.raises(ValueError, match="confidence_weight must be finite and >= 0"):
        ConfidenceWeightedModeInferer(confidence_weight=float("nan"))
    with pytest.raises(ValueError, match="min_score_threshold must be finite and >= 0"):
        ConfidenceWeightedModeInferer(min_score_threshold=float("nan"))


def test_confidence_weighted_mode_inferer_matches_brute_force_reference() -> None:
    """Cross-checks the O(T*K) rolling-window scorer (running per-state count/prob-sum, added and
    removed once per bar) against a brute-force O(T*K*window) reference computed directly from the
    definition, across several window/K/confidence_weight combinations — catches any off-by-one in
    the incremental add/remove bookkeeping that unit tests on hand-built sequences might miss."""
    rng = np.random.default_rng(99)
    T, K = 40, 4
    probs = rng.dirichlet(np.ones(K), size=T)
    argmax = np.argmax(probs, axis=1)

    for window in (1, 2, 5, 7, 15, 100):
        for confidence_weight in (0.0, 1.0, 2.5):
            inferer = ConfidenceWeightedModeInferer(window=window, confidence_weight=confidence_weight,
                                                    min_score_threshold=0.0)
            labels = inferer.infer(probs)

            expected = np.zeros(T, dtype=np.int64)
            for t in range(T):
                start = max(0, t - window + 1)
                best_score, best_label = -1.0, 0
                for k in range(K):
                    mask = argmax[start:t + 1] == k
                    count = int(mask.sum())
                    if count == 0:
                        continue
                    if confidence_weight == 0.0:
                        score = float(count)
                    else:
                        mean_prob = probs[start:t + 1, k][mask].mean()
                        score = float(count) * (mean_prob ** confidence_weight)
                    if score > best_score:
                        best_score, best_label = score, k
                expected[t] = best_label

            np.testing.assert_array_equal(
                labels, expected, err_msg=f"mismatch at window={window}, confidence_weight={confidence_weight}"
            )


# --- ConfidenceHysteresisInferer ----------------------------------------------

def test_confidence_hysteresis_inferer_returns_int64_labels() -> None:
    probs = _confident_argmax_series([0, 0, 1, 1, 0], K=2)
    inferer = ConfidenceHysteresisInferer(entry_threshold=1.0, exit_threshold=1.0)

    labels = inferer.infer(probs)

    assert labels.dtype == np.int64
    assert labels.shape == (5,)


def test_confidence_hysteresis_inferer_high_exit_threshold_locks_initial_regime() -> None:
    # Long run of 0 with brief 1s — high exit threshold should never allow exit.
    probs = _confident_argmax_series([0, 0, 0, 1, 1, 0, 1, 0], K=2, confidence=0.7)
    inferer = ConfidenceHysteresisInferer(entry_threshold=1.0, exit_threshold=100.0)

    labels = inferer.infer(probs)

    assert (labels == 0).all()


def test_confidence_hysteresis_inferer_low_thresholds_track_argmax() -> None:
    seq = [0, 0, 1, 1, 1, 0, 0]
    probs = _confident_argmax_series(seq, K=2, confidence=0.95)
    inferer = ConfidenceHysteresisInferer(entry_threshold=0.1, exit_threshold=0.1)

    labels = inferer.infer(probs)

    np.testing.assert_array_equal(labels, np.argmax(probs, axis=1))


def test_confidence_hysteresis_inferer_metadata_reports_smoothing_ratio() -> None:
    seq = [0, 0, 1, 0, 1, 1, 1, 0, 1, 0]
    probs = _confident_argmax_series(seq, K=2, confidence=0.7)
    inferer = ConfidenceHysteresisInferer(entry_threshold=2.0, exit_threshold=2.0)
    inferer.infer(probs)

    meta = inferer.get_metadata()
    assert meta["inferer"] == "ConfidenceHysteresisInferer"
    assert meta["n_argmax_transitions"] > meta["n_label_transitions"]
    assert 0.0 < meta["smoothing_ratio"] <= 1.0


def test_confidence_hysteresis_inferer_rejects_non_positive_thresholds() -> None:
    with pytest.raises(ValueError, match="entry_threshold"):
        ConfidenceHysteresisInferer(entry_threshold=0.0, exit_threshold=1.0)
    with pytest.raises(ValueError, match="exit_threshold"):
        ConfidenceHysteresisInferer(entry_threshold=1.0, exit_threshold=-0.5)


def test_confidence_hysteresis_inferer_rejects_nan_thresholds() -> None:
    with pytest.raises(ValueError, match="entry_threshold must be finite and > 0"):
        ConfidenceHysteresisInferer(entry_threshold=float("nan"), exit_threshold=1.0)
    with pytest.raises(ValueError, match="exit_threshold must be finite and > 0"):
        ConfidenceHysteresisInferer(entry_threshold=1.0, exit_threshold=float("nan"))
