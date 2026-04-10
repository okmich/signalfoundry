"""
Conditional Persistence Loss for Regime Classification.

Adds a differentiable transition penalty to cross-entropy that ONLY penalizes
false transitions — cases where the model switches regime but the oracle didn't.

True transitions (where the oracle label changes) are left unpenalized so the
CE gradient can drive the model to detect them without interference.

Key design decisions:
    - The penalty is conditional on y_true: only pairs where y_true[t] == y_true[t+1]
      contribute to the penalty term.
    - The penalty uses 1 - dot(p_t, p_{t+1}) as a differentiable proxy for regime flips.
    - The penalty is averaged only over stay-pairs (not all pairs), so the gradient
      is not diluted by the ~97% of pairs that are already correct.

Rank-2 (batch,classes) — legacy flat-sequence mode:
    Requires shuffle=False in model.fit() so consecutive batch items are temporally
    adjacent. Transitions are computed across the single flattened axis.

Rank-3 (batch,time,classes) — per-sequence mode:
    Transitions are computed along the time axis independently for each sequence.
    No cross-sample coupling: the last timestep of sequence i is never linked to the
    first timestep of sequence i+1.

Usage:
    >>> from okmich_quant_neural_net.keras.losses import conditional_persistence_loss
    >>> loss_fn = conditional_persistence_loss(lambda_val=0.5)
    >>> model.compile(optimizer='adam', loss=loss_fn)
    >>> model.fit(X, y, shuffle=False, ...)

    # With logits (no softmax output layer):
    >>> loss_fn = conditional_persistence_loss(lambda_val=0.5, from_logits=True)
"""

import tensorflow as tf


def conditional_persistence_loss(lambda_val: float, from_logits: bool = False):
    """
    Create a Keras-compatible loss function with a conditional persistence penalty.

    loss = CE(y_true, y_pred) + lambda_val * penalty

    where penalty = mean(1 - dot(p_t, p_{t+1})) averaged ONLY over consecutive
    pairs where y_true[t] == y_true[t+1] (i.e., the oracle says "stay").

    Parameters
    ----------
    lambda_val : float
        Weight of the persistence penalty. 0.0 = pure CE. Higher values
        penalize false transitions more aggressively.
    from_logits : bool
        If True, y_pred contains raw logits. Softmax is applied internally for the
        penalty term, and from_logits=True is passed to sparse_categorical_crossentropy.

    Returns
    -------
    loss_fn : callable
        A Keras loss function accepting (y_true, y_pred).
    """
    if lambda_val < 0:
        raise ValueError(f"lambda_val must be >= 0, got {lambda_val}")

    def _loss(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)

        rank = len(y_pred.shape)
        if rank == 3:
            return _loss_rank3(y_true, y_pred)
        elif rank == 2:
            return _loss_rank2(y_true, y_pred)
        else:
            raise ValueError(f"y_pred must be rank 2 or 3, got rank {rank}")

    def _loss_rank2(y_true, y_pred):
        """Flat-sequence mode: (batch, classes). Legacy behavior."""
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

        if lambda_val == 0.0:
            return ce

        y_true_flat = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        # Probabilities for the penalty term
        probs = tf.nn.softmax(y_pred, axis=-1) if from_logits else y_pred

        oracle_stayed = tf.cast(tf.equal(y_true_flat[:-1], y_true_flat[1:]), tf.float32)
        stay_prob = tf.reduce_sum(probs[:-1] * probs[1:], axis=-1)
        false_switch_cost = (1.0 - stay_prob) * oracle_stayed

        n_stay = tf.reduce_sum(oracle_stayed)
        penalty = tf.reduce_sum(false_switch_cost) / (n_stay + 1e-8)

        return ce + lambda_val * penalty

    def _loss_rank3(y_true, y_pred):
        """Per-sequence mode: (batch, time, classes). No cross-sample bleed."""
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

        if lambda_val == 0.0:
            return ce

        # y_true: (batch, time) or (batch, time, 1)
        y_true_int = tf.cast(tf.squeeze(y_true, axis=-1) if len(y_true.shape) == 3 else y_true, tf.int32)

        # Probabilities for the penalty term
        probs = tf.nn.softmax(y_pred, axis=-1) if from_logits else y_pred

        # Consecutive pairs along time axis — no cross-sample coupling
        oracle_stayed = tf.cast(tf.equal(y_true_int[:, :-1], y_true_int[:, 1:]), tf.float32)  # (batch, time-1)
        stay_prob = tf.reduce_sum(probs[:, :-1] * probs[:, 1:], axis=-1)  # (batch, time-1)
        false_switch_cost = (1.0 - stay_prob) * oracle_stayed  # (batch, time-1)

        # Average over stay-pairs per sequence, then mean across batch
        n_stay = tf.reduce_sum(oracle_stayed, axis=-1)  # (batch,)
        penalty_per_seq = tf.reduce_sum(false_switch_cost, axis=-1) / (n_stay + 1e-8)  # (batch,)
        penalty = tf.reduce_mean(penalty_per_seq)

        # ce is (batch, time) — return (batch, time) with penalty broadcast
        return ce + lambda_val * penalty

    _loss.__name__ = f"conditional_persistence_loss_lambda_{lambda_val}"
    return _loss
