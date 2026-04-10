"""
Conditional Random Field (CRF) layer for sequence labelling.

Adds a learnable transition matrix on top of per-timestep emission scores
so that the decoder selects the most probable *path* (Viterbi) rather than
the most probable *label at each step* (argmax). This eliminates the
flickering problem where an upstream model oscillates between classes at
decision boundaries.

Architecture:
    emissions (B, T, C) ──► CRF ──► decoded labels (B, T, C) one-hot
                               │
                        transitions (C, C)

    Training:  returns raw emissions (loss is CRF negative log-likelihood).
    Inference: returns Viterbi-decoded one-hot labels.

The layer exposes a `.loss` property that returns a Keras-compatible loss
function computing the CRF negative log-likelihood.  The loss uses the
forward algorithm to compute the log partition function and subtracts
the score of the gold path.

Usage:
    >>> crf = CRF(num_classes=3)
    >>> emissions = Dense(3)(backbone_output)   # (B, T, 3)
    >>> out = crf(emissions)                    # training: emissions, inference: one-hot Viterbi
    >>> model.compile(optimizer="adam", loss=crf.loss)

    # Constrain transitions (e.g. forbid 0→2):
    >>> crf = CRF(num_classes=3, transition_mask=[[1,1,0],[1,1,1],[0,1,1]])

References:
    Lafferty, McCallum, Pereira (2001) — Conditional Random Fields.
    Forney (1973) — The Viterbi Algorithm.
"""

import tensorflow as tf
import numpy as np
from keras import layers
from keras.src.saving import register_keras_serializable


@register_keras_serializable()
class CRF(layers.Layer):
    """Linear-chain CRF layer for sequence labelling.

    Learns a (C, C) transition matrix where entry [i, j] is the score for
    transitioning from class i to class j, plus start/end transition vectors.

    Args:
        num_classes: Number of output classes (C).
        transition_mask: Optional (C, C) binary matrix. Zeros forbid transitions.
            Use this to encode domain constraints (e.g. regime A cannot jump
            directly to regime C). Default: all transitions allowed.
        transition_regularizer: Optional regularizer applied to the transition
            matrix (e.g. keras.regularizers.l2(1e-3) to encourage persistence).
        **kwargs: Passed to the base Layer.

    Input shape:  (B, T, C) — emission scores (logits, not probabilities).
    Output shape: (B, T, C) — one-hot Viterbi path (inference) or raw emissions (training).
    """

    def __init__(self, num_classes: int, transition_mask=None, transition_regularizer=None, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.num_classes = num_classes
        self._transition_mask_init = transition_mask
        self.transition_regularizer = tf.keras.regularizers.get(transition_regularizer)

    def build(self, input_shape):
        C = self.num_classes

        self.transitions = self.add_weight(
            shape=(C, C), name="transitions",
            initializer="glorot_uniform",
            regularizer=self.transition_regularizer,
        )
        self.start_transitions = self.add_weight(
            shape=(C,), name="start_transitions", initializer="zeros",
        )
        self.end_transitions = self.add_weight(
            shape=(C,), name="end_transitions", initializer="zeros",
        )

        if self._transition_mask_init is not None:
            self._transition_mask = tf.constant(
                np.asarray(self._transition_mask_init, dtype=np.float32), name="transition_mask",
            )
        else:
            self._transition_mask = None

        super().build(input_shape)

    def _get_transitions(self):
        """Return the transition matrix, applying mask if set."""
        trans = self.transitions
        if self._transition_mask is not None:
            # Large negative score for forbidden transitions
            trans = trans + (1.0 - self._transition_mask) * -1e9
        return trans

    # ------------------------------------------------------------------
    # Forward algorithm (training) — computes log Z (partition function)
    # ------------------------------------------------------------------

    def _forward_algorithm(self, emissions):
        """Compute log partition function for a batch of sequences.

        Args:
            emissions: (B, T, C) emission scores.

        Returns:
            log_Z: (B,) log partition function per sequence.
        """
        trans = self._get_transitions()  # (C, C)

        # alpha_0 = start_transitions + emissions[:, 0]
        alpha = self.start_transitions + emissions[:, 0]  # (B, C)

        T = tf.shape(emissions)[1]

        def _step(t, alpha):
            # alpha: (B, C) — broadcast to (B, C, 1) + trans (C, C) → (B, C, C)
            # then logsumexp over the "from" axis (axis=1)
            emit = emissions[:, t]  # (B, C)
            scores = tf.expand_dims(alpha, 2) + trans  # (B, C_from, C_to)
            alpha = tf.reduce_logsumexp(scores, axis=1) + emit  # (B, C)
            return t + 1, alpha

        _, alpha = tf.while_loop(
            cond=lambda t, _: t < T,
            body=_step,
            loop_vars=[tf.constant(1), alpha],
        )

        # Add end transitions and final logsumexp
        log_Z = tf.reduce_logsumexp(alpha + self.end_transitions, axis=-1)  # (B,)
        return log_Z

    # ------------------------------------------------------------------
    # Gold score — score of the true label path
    # ------------------------------------------------------------------

    def _score_sequence(self, emissions, tags):
        """Score the gold tag sequence.

        Args:
            emissions: (B, T, C) emission scores.
            tags: (B, T) integer class indices.

        Returns:
            score: (B,) score of the gold path.
        """
        trans = self._get_transitions()  # (C, C)
        B = tf.shape(emissions)[0]
        T = tf.shape(emissions)[1]

        # Gather emission scores at gold tags
        # tags_one_hot: (B, T, C)
        tags_oh = tf.one_hot(tags, self.num_classes)
        emit_scores = tf.reduce_sum(emissions * tags_oh, axis=-1)  # (B, T)
        emit_total = tf.reduce_sum(emit_scores, axis=-1)  # (B,)

        # Start transition scores
        start_tag = tags[:, 0]  # (B,)
        score = tf.gather(self.start_transitions, start_tag) + emit_total  # (B,)

        # Transition scores: sum of trans[tag_t, tag_{t+1}] for t = 0..T-2
        tags_from = tags[:, :-1]  # (B, T-1)
        tags_to = tags[:, 1:]  # (B, T-1)

        # Flatten to gather from the (C, C) matrix
        flat_idx = tags_from * self.num_classes + tags_to  # (B, T-1)
        trans_flat = tf.reshape(trans, [-1])  # (C*C,)
        trans_scores = tf.gather(trans_flat, flat_idx)  # (B, T-1)
        score = score + tf.reduce_sum(trans_scores, axis=-1)  # (B,)

        # End transition scores
        end_tag = tags[:, -1]  # (B,)
        score = score + tf.gather(self.end_transitions, end_tag)  # (B,)

        return score

    # ------------------------------------------------------------------
    # Viterbi decoding (inference)
    # ------------------------------------------------------------------

    def viterbi_decode(self, emissions):
        """Decode the best path for each sequence in the batch.

        Args:
            emissions: (B, T, C) emission scores.

        Returns:
            best_paths: (B, T) integer class indices.
        """
        trans = self._get_transitions()
        T = tf.shape(emissions)[1]

        # Viterbi forward pass
        viterbi = self.start_transitions + emissions[:, 0]  # (B, C)
        # Store backpointers as a TensorArray
        backpointers = tf.TensorArray(dtype=tf.int32, size=T - 1, dynamic_size=False)

        def _step(t, viterbi, backpointers):
            # (B, C_from, 1) + (C_from, C_to) → (B, C_from, C_to)
            scores = tf.expand_dims(viterbi, 2) + trans
            best_from = tf.cast(tf.argmax(scores, axis=1), tf.int32)  # (B, C_to)
            viterbi = tf.reduce_max(scores, axis=1) + emissions[:, t]  # (B, C_to)
            backpointers = backpointers.write(t - 1, best_from)  # index t-1
            return t + 1, viterbi, backpointers

        _, viterbi, backpointers = tf.while_loop(
            cond=lambda t, *_: t < T,
            body=_step,
            loop_vars=[tf.constant(1), viterbi, backpointers],
        )

        # Add end transitions, pick best final state
        viterbi = viterbi + self.end_transitions  # (B, C)
        best_last = tf.cast(tf.argmax(viterbi, axis=-1), tf.int32)  # (B,)

        # Backtrace
        bp_stack = backpointers.stack()  # (T-1, B, C)

        # Reverse backtrace
        path = tf.TensorArray(dtype=tf.int32, size=T, dynamic_size=False)
        path = path.write(T - 1, best_last)

        def _backtrace(t, best, path):
            # bp_stack[t] is (B, C) — gather the predecessor for each batch item
            bp_t = bp_stack[t]  # (B, C)
            best = tf.gather(bp_t, best, batch_dims=1)  # (B,)
            path = path.write(t, best)
            return t - 1, best, path

        _, _, path = tf.while_loop(
            cond=lambda t, *_: t >= 0,
            body=_backtrace,
            loop_vars=[T - 2, best_last, path],
        )

        best_paths = tf.transpose(path.stack())  # (T, B) → (B, T)
        return best_paths

    # ------------------------------------------------------------------
    # Keras call
    # ------------------------------------------------------------------

    def call(self, emissions, training=None):
        """Forward pass.

        Args:
            emissions: (B, T, C) emission scores (logits).
            training: If True, returns raw emissions. If False, returns
                Viterbi-decoded one-hot labels.

        Returns:
            (B, T, C) tensor.
        """
        if training:
            return emissions

        best_paths = self.viterbi_decode(emissions)  # (B, T)
        return tf.one_hot(best_paths, self.num_classes)  # (B, T, C)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @property
    def loss(self):
        """Return a Keras-compatible CRF negative log-likelihood loss function.

        The returned callable expects:
            y_true: (B, T) or (B, T, 1) integer class indices.
            y_pred: (B, T, C) emission scores (logits).

        Returns scalar mean NLL over the batch.
        """
        def crf_nll(y_true, y_pred):
            y_pred = tf.cast(y_pred, tf.float32)
            tags = tf.cast(tf.squeeze(y_true, axis=-1) if len(y_true.shape) == 3 else y_true, tf.int32)

            log_Z = self._forward_algorithm(y_pred)  # (B,)
            gold_score = self._score_sequence(y_pred, tags)  # (B,)

            nll = log_Z - gold_score  # (B,)
            return tf.reduce_mean(nll)

        crf_nll.__name__ = "crf_nll"
        return crf_nll

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "transition_mask": self._transition_mask_init if self._transition_mask_init is not None else None,
            "transition_regularizer": tf.keras.regularizers.serialize(self.transition_regularizer),
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape