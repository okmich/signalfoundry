import numpy as np

from okmich_quant_neural_net.keras.metrics import CausalRegimeAccuracy, RegimeTransitionRecall, \
    RegimeTransitionPrecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_onehot(labels_2d: np.ndarray, num_classes: int) -> np.ndarray:
    """(batch, seq) int → (batch, seq, num_classes) float32 one-hot."""
    b, s = labels_2d.shape
    oh = np.zeros((b, s, num_classes), dtype=np.float32)
    for i in range(b):
        for j in range(s):
            oh[i, j, labels_2d[i, j]] = 1.0
    return oh


def _classes_to_logits(labels_2d: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer class labels to near-perfect softmax logits."""
    b, s = labels_2d.shape
    logits = np.zeros((b, s, num_classes), dtype=np.float32)
    for i in range(b):
        for j in range(s):
            logits[i, j, labels_2d[i, j]] = 10.0  # dominant class
    # softmax — class with 10.0 gets ~1.0
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def _reset_and_call(metric, y_true, y_pred):
    metric.reset_state()
    metric.update_state(y_true, y_pred)
    return float(metric.result().numpy())


# ---------------------------------------------------------------------------
# CausalRegimeAccuracy
# ---------------------------------------------------------------------------

class TestCausalRegimeAccuracy:

    def test_perfect_predictions_2d(self):
        """Matching every timestep → score = 1.0."""
        labels = np.array([[0, 1, 2, 1, 0]], dtype=np.int32)  # (1, 5)
        y_pred = _classes_to_logits(labels, num_classes=3)
        metric = CausalRegimeAccuracy(window_size=3)
        score = _reset_and_call(metric, labels, y_pred)
        assert abs(score - 1.0) < 1e-4

    def test_all_wrong_predictions_2d(self):
        """Never matching → score = 0.0."""
        labels = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        # Predict class 1 everywhere
        wrong = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)
        y_pred = _classes_to_logits(wrong, num_classes=3)
        metric = CausalRegimeAccuracy(window_size=3)
        score = _reset_and_call(metric, labels, y_pred)
        assert abs(score - 0.0) < 1e-4

    def test_flickering_lower_than_consistent(self):
        """
        Two models both have 50% point-accuracy but different consistency.
        Consistent model should score higher.
        """
        seq_len = 10
        # Consistent: always predicts class 0; true alternates 0/1
        true_alt = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.int32)

        # Consistent: always class 0 → correct on even positions only
        pred_consistent = np.zeros((1, seq_len), dtype=np.int32)
        # Flickering: alternates 0/1/0/1 but shifted → also 50% correct
        pred_flickering = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.int32)

        metric = CausalRegimeAccuracy(window_size=5)

        y_pred_c = _classes_to_logits(pred_consistent, num_classes=3)
        y_pred_f = _classes_to_logits(pred_flickering, num_classes=3)

        score_c = _reset_and_call(metric, true_alt, y_pred_c)
        score_f = _reset_and_call(metric, true_alt, y_pred_f)

        # Consistent model has the same correct positions bunched together
        # while the flickering model never has two consecutive correct answers,
        # so its windowed average must be lower or equal.
        assert score_c >= score_f

    def test_early_timestep_normalisation(self):
        """
        At position t=0 there is only 1 bar of history; the score should
        not be penalised relative to having a full window.
        Window=5, all-correct sequence → still 1.0.
        """
        labels = np.array([[2, 2, 2]], dtype=np.int32)  # very short seq
        y_pred = _classes_to_logits(labels, num_classes=3)
        metric = CausalRegimeAccuracy(window_size=5)  # window > seq_len
        score = _reset_and_call(metric, labels, y_pred)
        assert abs(score - 1.0) < 1e-4

    def test_accepts_1d_y_true(self):
        """Scalar (batch,) y_true should not raise — batch treated as sequence."""
        labels_1d = np.array([0, 1, 2, 1, 0], dtype=np.int32)  # (5,)
        # Matching predictions
        pred_2d = np.array([[0, 1, 2, 1, 0]], dtype=np.int32)  # used for logits
        y_pred = _classes_to_logits(pred_2d, num_classes=3).squeeze(0)  # (5, 3)
        metric = CausalRegimeAccuracy(window_size=3)
        score = _reset_and_call(metric, labels_1d, y_pred)
        assert 0.0 <= score <= 1.0

    def test_accepts_3d_onehot_y_true(self):
        """One-hot (batch, seq, classes) y_true should work."""
        labels = np.array([[0, 1, 2, 0, 1]], dtype=np.int32)
        y_pred = _classes_to_logits(labels, num_classes=3)
        y_true_oh = _to_onehot(labels, num_classes=3)
        metric = CausalRegimeAccuracy(window_size=3)
        score_int = _reset_and_call(metric, labels, y_pred)
        score_oh = _reset_and_call(metric, y_true_oh, y_pred)
        assert abs(score_int - score_oh) < 1e-4

    def test_reset_state_clears_accumulators(self):
        labels = np.array([[0, 1, 2]], dtype=np.int32)
        y_pred = _classes_to_logits(labels, num_classes=3)
        metric = CausalRegimeAccuracy(window_size=3)
        metric.update_state(labels, y_pred)
        metric.reset_state()
        assert float(metric.total_windowed_accuracy.numpy()) == 0.0
        assert float(metric.count.numpy()) == 0.0


# ---------------------------------------------------------------------------
# RegimeTransitionRecall
# ---------------------------------------------------------------------------

class TestRegimeTransitionRecall:

    def test_no_true_transitions_returns_zero(self):
        """Flat true labels → no transitions to recall → 0.0."""
        labels = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)
        pred = np.array([[0, 1, 2, 1, 0]], dtype=np.int32)  # lots of pred transitions
        y_pred = _classes_to_logits(pred, num_classes=3)
        metric = RegimeTransitionRecall(lag_tolerance=2)
        score = _reset_and_call(metric, labels, y_pred)
        assert abs(score - 0.0) < 1e-4

    def test_all_transitions_caught_exact(self):
        """Predicted transitions at exact same bars as true → recall = 1.0."""
        labels = np.array([[0, 0, 1, 1, 2, 2]], dtype=np.int32)
        # Same transitions in prediction
        pred = np.array([[0, 0, 1, 1, 2, 2]], dtype=np.int32)
        y_pred = _classes_to_logits(pred, num_classes=3)
        metric = RegimeTransitionRecall(lag_tolerance=0)
        score = _reset_and_call(metric, labels, y_pred)
        assert abs(score - 1.0) < 1e-4

    def test_lag_tolerance_credits_early_detection(self):
        """
        True transition at t=3; prediction at t=2 (1 bar early).
        Recall uses a left-padded backward window: dilated_pred[t] = max(pred[t-lag..t]).
        So a true transition at t is caught if any prediction falls in [t-lag, t].
        With lag_tolerance=2 → early prediction at t=2 should be credited.
        With lag_tolerance=0 → should NOT be credited (only exact match).
        """
        labels = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int32)  # transition at t=3
        pred = np.array([[0, 0, 1, 1, 1, 1]], dtype=np.int32)  # transition at t=2 (1 bar early)

        y_pred = _classes_to_logits(pred, num_classes=3)

        metric_with_lag = RegimeTransitionRecall(lag_tolerance=2)
        metric_no_lag = RegimeTransitionRecall(lag_tolerance=0)

        score_with = _reset_and_call(metric_with_lag, labels, y_pred)
        score_none = _reset_and_call(metric_no_lag, labels, y_pred)

        assert abs(score_with - 1.0) < 1e-4, "should be credited within lag"
        assert abs(score_none - 0.0) < 1e-4, "should not be credited without lag"

    def test_early_detection_too_far_not_credited(self):
        """
        True transition at t=6; prediction at t=2 (4 bars early).
        lag_tolerance=2 → NOT credited (too far in advance).
        """
        labels = np.array([[0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.int32)  # transition at t=6
        pred = np.array([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=np.int32)  # transition at t=2
        y_pred = _classes_to_logits(pred, num_classes=3)
        metric = RegimeTransitionRecall(lag_tolerance=2)
        score = _reset_and_call(metric, labels, y_pred)
        assert abs(score - 0.0) < 1e-4

    def test_accepts_1d_y_true(self):
        labels_1d = np.array([0, 0, 1, 1, 2], dtype=np.int32)
        pred_2d = np.array([[0, 0, 1, 1, 2]], dtype=np.int32)
        y_pred = _classes_to_logits(pred_2d, num_classes=3).squeeze(0)  # (5, 3)
        metric = RegimeTransitionRecall(lag_tolerance=1)
        score = _reset_and_call(metric, labels_1d, y_pred)
        assert 0.0 <= score <= 1.0

    def test_accepts_3d_onehot_y_true(self):
        labels = np.array([[0, 0, 1, 1, 2]], dtype=np.int32)
        y_pred = _classes_to_logits(labels, num_classes=3)
        y_true_oh = _to_onehot(labels, num_classes=3)
        metric = RegimeTransitionRecall(lag_tolerance=1)
        score_int = _reset_and_call(metric, labels, y_pred)
        score_oh = _reset_and_call(metric, y_true_oh, y_pred)
        assert abs(score_int - score_oh) < 1e-4

    def test_reset_clears_state(self):
        labels = np.array([[0, 1, 0]], dtype=np.int32)
        y_pred = _classes_to_logits(labels, num_classes=3)
        metric = RegimeTransitionRecall(lag_tolerance=1)
        metric.update_state(labels, y_pred)
        metric.reset_state()
        assert float(metric.transitions_caught.numpy()) == 0.0
        assert float(metric.total_true_transitions.numpy()) == 0.0


# ---------------------------------------------------------------------------
# RegimeTransitionPrecision
# ---------------------------------------------------------------------------

class TestRegimeTransitionPrecision:

    def test_no_predicted_transitions_returns_zero(self):
        """Flat predictions → no transitions predicted → precision = 0.0."""
        labels = np.array([[0, 0, 1, 1, 2, 2]], dtype=np.int32)  # real transitions
        pred = np.array([[1, 1, 1, 1, 1, 1]], dtype=np.int32)  # never transitions
        y_pred = _classes_to_logits(pred, num_classes=3)
        metric = RegimeTransitionPrecision(lag_tolerance=2)
        score = _reset_and_call(metric, labels, y_pred)
        assert abs(score - 0.0) < 1e-4

    def test_all_predicted_transitions_correct(self):
        """Predicted transitions exactly match true transitions → precision = 1.0."""
        labels = np.array([[0, 0, 1, 1, 2, 2]], dtype=np.int32)
        pred = np.array([[0, 0, 1, 1, 2, 2]], dtype=np.int32)
        y_pred = _classes_to_logits(pred, num_classes=3)
        metric = RegimeTransitionPrecision(lag_tolerance=0)
        score = _reset_and_call(metric, labels, y_pred)
        assert abs(score - 1.0) < 1e-4

    def test_lag_tolerance_credits_predictions(self):
        """
        True transition at t=2; prediction at t=3 (1 bar late).
        Precision uses a left-padded backward window: dilated_true[t] = max(true[t-lag..t]).
        So a predicted transition at t is credited if a true transition falls in [t-lag, t].
        lag_tolerance=2 → late prediction at t=3 should be credited (true t=2 within [1,3]).
        lag_tolerance=0 → NOT credited (true transition at t=2 is not at t=3).
        """
        labels = np.array([[0, 0, 1, 1, 1, 1]], dtype=np.int32)  # true t=2
        pred = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int32)  # pred t=3 (1 bar late)

        y_pred = _classes_to_logits(pred, num_classes=3)

        metric_with_lag = RegimeTransitionPrecision(lag_tolerance=2)
        metric_no_lag = RegimeTransitionPrecision(lag_tolerance=0)

        score_with = _reset_and_call(metric_with_lag, labels, y_pred)
        score_none = _reset_and_call(metric_no_lag, labels, y_pred)

        assert abs(score_with - 1.0) < 1e-4, "should be credited within lag"
        assert abs(score_none - 0.0) < 1e-4, "should not be credited without lag"

    def test_spurious_predictions_reduce_precision(self):
        """
        Extra predicted transitions with no nearby true transition → precision < 1.0.
        One real transition at t=2; prediction at t=2 (correct) and spurious at t=4.
        pred=[0,0,1,1,0,0] has edges at t=2 (0→1) and t=4 (1→0) — exactly 2.
        """
        labels = np.array([[0, 0, 1, 1, 1, 1]], dtype=np.int32)
        pred = np.array([[0, 0, 1, 1, 0, 0]], dtype=np.int32)  # transitions t=2 and t=4
        y_pred = _classes_to_logits(pred, num_classes=3)
        metric = RegimeTransitionPrecision(lag_tolerance=0)
        score = _reset_and_call(metric, labels, y_pred)
        # t=2: pred_edge=1, dilated_true[2]=true_edge[2]=1 → correct
        # t=4: pred_edge=1, dilated_true[4]=true_edge[4]=0 → spurious
        # 1 correct out of 2 predicted → 0.5
        assert abs(score - 0.5) < 1e-4

    def test_accepts_3d_onehot_y_true(self):
        labels = np.array([[0, 0, 1, 1, 2]], dtype=np.int32)
        y_pred = _classes_to_logits(labels, num_classes=3)
        y_true_oh = _to_onehot(labels, num_classes=3)
        metric = RegimeTransitionPrecision(lag_tolerance=1)
        score_int = _reset_and_call(metric, labels, y_pred)
        score_oh = _reset_and_call(metric, y_true_oh, y_pred)
        assert abs(score_int - score_oh) < 1e-4

    def test_reset_clears_state(self):
        labels = np.array([[0, 1, 0]], dtype=np.int32)
        y_pred = _classes_to_logits(labels, num_classes=3)
        metric = RegimeTransitionPrecision(lag_tolerance=1)
        metric.update_state(labels, y_pred)
        metric.reset_state()
        assert float(metric.correct_predictions.numpy()) == 0.0
        assert float(metric.total_predictions.numpy()) == 0.0
