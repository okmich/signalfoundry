import numpy as np
import pytest
import tensorflow as tf

from okmich_quant_neural_net.keras.losses.persistence import conditional_persistence_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def _one_hot_soft(labels: np.ndarray, num_classes: int, confidence: float = 0.95) -> np.ndarray:
    """Convert integer labels to near-one-hot softmax vectors with given confidence."""
    n = len(labels)
    probs = np.full((n, num_classes), (1 - confidence) / (num_classes - 1), dtype=np.float32)
    for i, c in enumerate(labels):
        probs[i, c] = confidence
    return probs


def _compute_loss(loss_fn, y_true, y_pred):
    """Run loss function and return numpy array."""
    return loss_fn(tf.constant(y_true), tf.constant(y_pred)).numpy()


# ---------------------------------------------------------------------------
# Test: lambda=0 is pure CE
# ---------------------------------------------------------------------------

class TestLambdaZero:

    def test_matches_keras_ce(self):
        """With lambda=0, output should exactly match sparse_categorical_crossentropy."""
        y_true = np.array([0, 1, 2, 0, 1], dtype=np.int32)
        y_pred = _one_hot_soft(y_true, num_classes=3, confidence=0.9)

        loss_fn = conditional_persistence_loss(0.0)
        our_loss = _compute_loss(loss_fn, y_true, y_pred)

        keras_ce = tf.keras.losses.sparse_categorical_crossentropy(
            tf.constant(y_true), tf.constant(y_pred)
        ).numpy()

        np.testing.assert_allclose(our_loss, keras_ce, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: penalty fires on false transitions only
# ---------------------------------------------------------------------------

class TestConditionalPenalty:

    def test_no_penalty_when_all_predictions_agree_with_oracle_stay(self):
        """
        Oracle stays in same class, model predicts same class for all bars.
        Penalty should be ~0 (predictions agree, oracle says stay).
        Note: at confidence=0.999, dot(p, p) ≈ 0.998 so residual is negligible.
        """
        y_true = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        y_pred = _one_hot_soft(np.array([0, 0, 0, 0, 0]), num_classes=3, confidence=0.999)

        loss_high = _compute_loss(conditional_persistence_loss(10.0), y_true, y_pred)
        loss_zero = _compute_loss(conditional_persistence_loss(0.0), y_true, y_pred)

        # With near-perfect stay predictions, penalty ≈ 0
        np.testing.assert_allclose(loss_high, loss_zero, atol=0.05)

    def test_penalty_fires_on_false_transition(self):
        """
        Oracle stays in class 0, but model flips to class 1 mid-sequence.
        Higher lambda should increase loss.
        """
        y_true = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        # Model predicts: 0, 0, 1, 1, 0 — two false transitions
        y_pred = _one_hot_soft(np.array([0, 0, 1, 1, 0]), num_classes=3, confidence=0.95)

        loss_low = _compute_loss(conditional_persistence_loss(0.1), y_true, y_pred).mean()
        loss_high = _compute_loss(conditional_persistence_loss(1.0), y_true, y_pred).mean()

        assert loss_high > loss_low, (
            f"Higher lambda should increase loss on false transitions: "
            f"lambda=1.0 gave {loss_high:.4f} vs lambda=0.1 gave {loss_low:.4f}"
        )

    def test_no_penalty_on_true_transitions(self):
        """
        Oracle transitions from class 0 to class 1. Model correctly follows.
        The penalty should NOT fire at the transition point.
        """
        y_true = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        # Model correctly follows: predictions match oracle
        y_pred = _one_hot_soft(y_true, num_classes=3, confidence=0.999)

        loss_high = _compute_loss(conditional_persistence_loss(10.0), y_true, y_pred)
        loss_zero = _compute_loss(conditional_persistence_loss(0.0), y_true, y_pred)

        # True transition is excluded from penalty, stay-pairs have near-perfect dot
        np.testing.assert_allclose(loss_high, loss_zero, atol=0.05)

    def test_penalty_ignores_true_transition_penalizes_false(self):
        """
        Oracle: [0, 0, 1, 1, 1] — one true transition at index 2.
        Model:  [0, 1, 1, 0, 1] — false transitions at index 1 (oracle stayed 0)
                                   and index 3 (oracle stayed 1).

        The penalty should fire on the false transitions but NOT the true one.
        """
        y_true = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        y_pred_wrong = _one_hot_soft(np.array([0, 1, 1, 0, 1]), num_classes=3, confidence=0.95)
        y_pred_right = _one_hot_soft(y_true, num_classes=3, confidence=0.95)

        lam = 5.0
        loss_fn = conditional_persistence_loss(lam)

        loss_wrong = _compute_loss(loss_fn, y_true, y_pred_wrong).mean()
        loss_right = _compute_loss(loss_fn, y_true, y_pred_right).mean()

        assert loss_wrong > loss_right, (
            f"False transitions should be penalized more than correct predictions: "
            f"wrong={loss_wrong:.4f} vs right={loss_right:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: penalty scales with lambda
# ---------------------------------------------------------------------------

class TestLambdaScaling:

    def test_monotonically_increasing_with_lambda(self):
        """Loss should increase monotonically with lambda when false transitions exist."""
        y_true = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        # Model flips: 0 → 1 → 0 → 1 → 0 (maximum false transitions)
        y_pred = _one_hot_soft(np.array([0, 1, 0, 1, 0]), num_classes=3, confidence=0.95)

        lambdas = [0.0, 0.1, 0.5, 1.0, 5.0]
        losses = [_compute_loss(conditional_persistence_loss(lam), y_true, y_pred).mean() for lam in lambdas]

        for i in range(len(losses) - 1):
            assert losses[i] < losses[i + 1], (
                f"Loss should increase with lambda: "
                f"lambda={lambdas[i]} gave {losses[i]:.4f}, "
                f"lambda={lambdas[i+1]} gave {losses[i+1]:.4f}"
            )


# ---------------------------------------------------------------------------
# Test: gradient flows correctly
# ---------------------------------------------------------------------------

class TestGradient:

    def test_gradient_nonzero_on_false_transition(self):
        """Gradient w.r.t. y_pred should be nonzero when false transitions exist."""
        y_true = tf.constant([0, 0, 0, 0, 0], dtype=tf.int32)
        y_pred = tf.Variable(_one_hot_soft(np.array([0, 0, 1, 1, 0]), num_classes=3, confidence=0.95))

        loss_fn = conditional_persistence_loss(1.0)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(loss_fn(y_true, y_pred))
        grad = tape.gradient(loss, y_pred)

        assert grad is not None, "Gradient should not be None"
        assert tf.reduce_any(tf.not_equal(grad, 0.0)).numpy(), "Gradient should have non-zero entries"

    def test_gradient_exists_at_lambda_zero(self):
        """Even at lambda=0, CE gradient should still flow."""
        y_true = tf.constant([0, 1, 2], dtype=tf.int32)
        y_pred = tf.Variable(_one_hot_soft(np.array([0, 1, 2]), num_classes=3, confidence=0.9))

        loss_fn = conditional_persistence_loss(0.0)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(loss_fn(y_true, y_pred))
        grad = tape.gradient(loss, y_pred)

        assert grad is not None
        assert tf.reduce_any(tf.not_equal(grad, 0.0)).numpy()


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_sample_batch(self):
        """Batch of size 1 — no consecutive pairs, penalty should be 0."""
        y_true = np.array([1], dtype=np.int32)
        y_pred = _one_hot_soft(np.array([1]), num_classes=3, confidence=0.95)

        loss_high = _compute_loss(conditional_persistence_loss(10.0), y_true, y_pred)
        loss_zero = _compute_loss(conditional_persistence_loss(0.0), y_true, y_pred)

        np.testing.assert_allclose(loss_high, loss_zero, atol=1e-6)

    def test_all_true_transitions(self):
        """Every consecutive pair is a true transition — penalty should be ~0."""
        y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
        y_pred = _one_hot_soft(y_true, num_classes=3, confidence=0.95)

        loss_high = _compute_loss(conditional_persistence_loss(10.0), y_true, y_pred)
        loss_zero = _compute_loss(conditional_persistence_loss(0.0), y_true, y_pred)

        # No stay-pairs exist, so penalty = 0/0 → 0 via epsilon
        np.testing.assert_allclose(loss_high, loss_zero, atol=0.05)

    def test_negative_lambda_raises(self):
        """Negative lambda should raise ValueError."""
        with pytest.raises(ValueError, match="lambda_val must be >= 0"):
            conditional_persistence_loss(-0.1)

    def test_two_sample_batch(self):
        """Batch of size 2 — one pair to evaluate."""
        y_true = np.array([0, 0], dtype=np.int32)
        # Model agrees on both — no false transition
        y_pred = _one_hot_soft(np.array([0, 0]), num_classes=3, confidence=0.999)

        loss_high = _compute_loss(conditional_persistence_loss(10.0), y_true, y_pred)
        loss_zero = _compute_loss(conditional_persistence_loss(0.0), y_true, y_pred)

        np.testing.assert_allclose(loss_high, loss_zero, atol=0.05)


# ---------------------------------------------------------------------------
# Test: rank-3 (batch, time, classes) — per-sequence, no cross-sample bleed
# ---------------------------------------------------------------------------

class TestRank3:

    def test_no_cross_batch_penalty(self):
        """
        Two independent sequences in one batch (batch=2, time=3, classes=3).
        Seq 0 ends with class 0, seq 1 starts with class 1.
        Penalty must NOT connect seq 0's last step to seq 1's first step.
        """
        # Seq 0: [0,0,0] — all stay, no false transitions
        # Seq 1: [1,1,1] — all stay, no false transitions
        y_true = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
        y_pred = np.stack([
            _one_hot_soft(np.array([0, 0, 0]), num_classes=3, confidence=0.999),
            _one_hot_soft(np.array([1, 1, 1]), num_classes=3, confidence=0.999),
        ])  # (2, 3, 3)

        loss_high = _compute_loss(conditional_persistence_loss(10.0), y_true, y_pred)
        loss_zero = _compute_loss(conditional_persistence_loss(0.0), y_true, y_pred)

        # No false transitions within either sequence — penalty should be ~0
        np.testing.assert_allclose(loss_high, loss_zero, atol=0.05)

    def test_rank3_penalty_fires_on_false_transition(self):
        """Rank-3 input with false transitions should be penalized."""
        # Seq 0: oracle stays 0, model flips to 1
        # Seq 1: oracle stays 1, model stays 1 (no penalty)
        y_true = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
        y_pred = np.stack([
            _one_hot_soft(np.array([0, 1, 0]), num_classes=3, confidence=0.95),
            _one_hot_soft(np.array([1, 1, 1]), num_classes=3, confidence=0.95),
        ])

        loss_high = _compute_loss(conditional_persistence_loss(5.0), y_true, y_pred)
        loss_zero = _compute_loss(conditional_persistence_loss(0.0), y_true, y_pred)

        # Seq 0 has false transitions, so penalty > 0
        assert loss_high.mean() > loss_zero.mean()

    def test_rank3_matches_rank2_for_single_sequence(self):
        """For a single sequence, rank-3 (1,T,C) should match rank-2 (T,C) penalty."""
        y_true_flat = np.array([0, 0, 1, 1, 0], dtype=np.int32)
        y_pred_flat = _one_hot_soft(np.array([0, 1, 1, 0, 0]), num_classes=3, confidence=0.95)

        y_true_3d = y_true_flat[np.newaxis, :]  # (1, 5)
        y_pred_3d = y_pred_flat[np.newaxis, :]  # (1, 5, 3)

        lam = 2.0
        loss_2d = _compute_loss(conditional_persistence_loss(lam), y_true_flat, y_pred_flat).mean()
        loss_3d = _compute_loss(conditional_persistence_loss(lam), y_true_3d, y_pred_3d).mean()

        np.testing.assert_allclose(loss_2d, loss_3d, atol=1e-5)

    def test_rank3_gradient_flows(self):
        """Gradient should flow through the rank-3 path."""
        y_true = tf.constant([[0, 0, 0], [1, 1, 1]], dtype=tf.int32)
        y_pred = tf.Variable(np.stack([
            _one_hot_soft(np.array([0, 1, 0]), num_classes=3, confidence=0.95),
            _one_hot_soft(np.array([1, 1, 1]), num_classes=3, confidence=0.95),
        ]))

        loss_fn = conditional_persistence_loss(1.0)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(loss_fn(y_true, y_pred))
        grad = tape.gradient(loss, y_pred)

        assert grad is not None
        assert tf.reduce_any(tf.not_equal(grad, 0.0)).numpy()

    def test_rank3_all_stay_no_false_transitions(self):
        """When all predictions match oracle stays, penalty ≈ 0 in rank-3."""
        y_true = np.array([[2, 2, 2, 2], [0, 0, 0, 0]], dtype=np.int32)
        y_pred = np.stack([
            _one_hot_soft(np.array([2, 2, 2, 2]), num_classes=3, confidence=0.999),
            _one_hot_soft(np.array([0, 0, 0, 0]), num_classes=3, confidence=0.999),
        ])

        loss_high = _compute_loss(conditional_persistence_loss(10.0), y_true, y_pred)
        loss_zero = _compute_loss(conditional_persistence_loss(0.0), y_true, y_pred)

        np.testing.assert_allclose(loss_high, loss_zero, atol=0.05)


# ---------------------------------------------------------------------------
# Test: from_logits flag
# ---------------------------------------------------------------------------

class TestFromLogits:

    def test_from_logits_matches_softmaxed(self):
        """from_logits=True on raw logits should match from_logits=False on softmaxed probs."""
        y_true = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        logits = np.random.RandomState(42).randn(5, 3).astype(np.float32)
        probs = _softmax(logits)

        lam = 1.0
        loss_logits = _compute_loss(conditional_persistence_loss(lam, from_logits=True), y_true, logits)
        loss_probs = _compute_loss(conditional_persistence_loss(lam, from_logits=False), y_true, probs)

        np.testing.assert_allclose(loss_logits, loss_probs, atol=1e-5)

    def test_from_logits_rank3_matches_softmaxed(self):
        """from_logits=True on rank-3 logits should match from_logits=False on softmaxed probs."""
        y_true = np.array([[0, 0, 1, 1], [2, 2, 2, 0]], dtype=np.int32)
        logits = np.random.RandomState(99).randn(2, 4, 3).astype(np.float32)
        probs = _softmax(logits)

        lam = 2.0
        loss_logits = _compute_loss(conditional_persistence_loss(lam, from_logits=True), y_true, logits)
        loss_probs = _compute_loss(conditional_persistence_loss(lam, from_logits=False), y_true, probs)

        np.testing.assert_allclose(loss_logits, loss_probs, atol=1e-5)

    def test_from_logits_gradient_flows(self):
        """Gradient should flow when from_logits=True."""
        y_true = tf.constant([0, 0, 1, 1], dtype=tf.int32)
        logits = tf.Variable(np.random.RandomState(42).randn(4, 3).astype(np.float32))

        loss_fn = conditional_persistence_loss(1.0, from_logits=True)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(loss_fn(y_true, logits))
        grad = tape.gradient(loss, logits)

        assert grad is not None
        assert tf.reduce_any(tf.not_equal(grad, 0.0)).numpy()