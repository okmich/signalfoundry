import numpy as np
import tensorflow as tf
from keras import layers
from keras.src.saving import register_keras_serializable


@register_keras_serializable()
class LightweightAttention(layers.Layer):
    """
    Lightweight attention for temporal pooling: (B, T, units) → (B, units).

    Supports three attention types:
      - "bahdanau": additive (Bahdanau) attention
      - "dot":      multi-head scaled dot-product attention
      - "decay":    learnable exponential decay weighting (no softmax, fully causal)

    When to use this vs BahdanauAttention
    --------------------------------------
    Use THIS layer when you have an encoder-only model and want to pool a
    single sequence (e.g. GRU outputs) down to one context vector.  Query and
    keys are both derived from the same input tensor (self-attention style).
    Supports causal masking and padding masks out of the box.

    Use BahdanauAttention when you have a true encoder-decoder (seq2seq)
    architecture where a separate decoder state (query) attends over a
    different encoder output sequence (values).

    Causality
    ---------
    causal=True (default): the context vector is computed so that position t only attends to positions 0..t.  This is
    the correct default for live trading models.
    Set causal=False only when operating on a fixed historical window where all bars are already in the past and
    full-window normalisation is intentional.

    Input:  (batch, timesteps, units)  — LSTM/GRU hidden states
    Output: (batch, units)            — context vector

    Args:
        attn_type: "bahdanau", "dot", or "decay". Default "bahdanau".
        heads: Number of attention heads (dot only). Default 4.
        causal: Enforce causal masking. Default True.
        decay_init: Initial value for the decay rate λ (decay only). Default 0.9.
        return_attention_scores: Also return attention weights. Default True.

    Examples:
        # Causal Bahdanau (default)
        attn = LightweightAttention(attn_type="bahdanau")
        context, weights = attn(gru_outputs)

        # Causal multi-head dot-product
        attn = LightweightAttention(attn_type="dot", heads=4)
        context, weights = attn(gru_outputs)

        # Exponential decay (always causal)
        attn = LightweightAttention(attn_type="decay")
        context, weights = attn(gru_outputs)

        # Explicit non-causal (opt-in)
        attn = LightweightAttention(attn_type="dot", causal=False)
        context = attn(lstm_outputs, return_attention_scores=False)
    """

    def __init__(self, attn_type="bahdanau", heads=4, causal=True, decay_init=0.9, return_attention_scores=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        self.attn_type = attn_type.lower()
        self.heads = heads
        self.causal = causal
        self.decay_init = decay_init
        self.return_attention_scores = return_attention_scores

        # Built in build()
        self.units = None
        self.scale = None

        # Bahdanau
        self.W = None
        self.U = None
        self.V = None

        # Dot-product
        self.Q_proj = None
        self.K_proj = None
        self.V_proj = None

        # Decay
        self.lambda_ = None

    def build(self, input_shape):
        _, T, self.units = input_shape

        if self.attn_type == "bahdanau":
            self.W = layers.Dense(self.units, use_bias=False, name="query")
            self.U = layers.Dense(self.units, use_bias=False, name="key")
            self.V = layers.Dense(1, use_bias=False, name="score")
            self.W.build(input_shape)
            self.U.build(input_shape)
            self.V.build((*input_shape[:-1], self.units))

        elif self.attn_type == "dot":
            if self.units % self.heads != 0:
                raise ValueError(f"units ({self.units}) must be divisible by heads ({self.heads})")
            self.head_dim = self.units // self.heads
            self.scale = float(np.sqrt(self.head_dim))

            self.Q_proj = layers.Dense(self.units, use_bias=False, name="Q")
            self.K_proj = layers.Dense(self.units, use_bias=False, name="K")
            self.V_proj = layers.Dense(self.units, use_bias=False, name="V")
            self.Q_proj.build(input_shape)
            self.K_proj.build(input_shape)
            self.V_proj.build(input_shape)

        elif self.attn_type == "decay":
            # Learnable scalar decay rate; sigmoid-constrained to (0, 1) at call time
            self.lambda_ = self.add_weight(
                name="lambda",
                shape=(),
                initializer=tf.keras.initializers.Constant(
                    np.log(self.decay_init / (1.0 - self.decay_init))  # logit of decay_init
                ),
                trainable=True,
            )

        else:
            raise ValueError(
                f"Unknown attn_type '{self.attn_type}'. Choose 'bahdanau', 'dot', or 'decay'."
            )

        super().build(input_shape)

    def call(self, inputs, mask=None):
        """
        Args:
            inputs: (B, T, units)
            mask:   (B, T) boolean — True for valid positions, False for padding

        Returns:
            context:           (B, units)
            attention_weights: (B, T) for bahdanau/decay, (B, H, T, T) for dot
                               (only when return_attention_scores=True)
        """
        if self.attn_type == "bahdanau":
            return self._call_bahdanau(inputs, mask)
        elif self.attn_type == "dot":
            return self._call_dot(inputs, mask)
        else:
            return self._call_decay(inputs)

    def _call_bahdanau(self, inputs, mask):
        """Bahdanau (additive) attention.

        Causal mode: expand scores to (B, T, T) with a lower-triangular mask so that position i only considers
        positions 0..i, then return the last position's context and weights.

        Non-causal mode: original full-window softmax behaviour.
        """
        query = self.W(inputs)  # (B, T, units)
        keys = self.U(inputs)  # (B, T, units)
        scores = self.V(tf.nn.tanh(query + keys))  # (B, T, 1)
        scores = tf.squeeze(scores, axis=-1)  # (B, T)

        if self.causal:
            T = tf.shape(inputs)[1]
            # Tile scores into (B, T_query, T_key) and apply lower-triangular mask
            scores_tiled = tf.tile(tf.expand_dims(scores, 1), [1, T, 1])  # (B, T, T)
            causal_mask = tf.linalg.band_part(tf.ones((T, T), dtype=scores.dtype), -1, 0)  # lower-tri = 1
            future_mask = (1.0 - causal_mask) * -1e9  # upper-tri = -inf
            scores_masked = scores_tiled + future_mask[tf.newaxis, :, :]  # (B, T, T)

            # Apply padding mask if provided (mask out key positions)
            if mask is not None:
                mask_float = tf.cast(mask, dtype=scores.dtype)
                scores_masked = scores_masked + (1.0 - mask_float[:, tf.newaxis, :]) * -1e9

            attn_weights_full = tf.nn.softmax(scores_masked, axis=-1)  # (B, T, T)

            # Context sequence: (B, T, units)
            context_seq = tf.matmul(attn_weights_full, inputs)  # (B, T, units)

            # Return the last timestep's context and weights
            context = context_seq[:, -1, :]  # (B, units)
            attn_weights = attn_weights_full[:, -1, :]  # (B, T)
        else:
            # Original non-causal behaviour
            if mask is not None:
                mask_float = tf.cast(mask, dtype=scores.dtype)
                scores = scores + (1.0 - mask_float) * -1e9
            attn_weights = tf.nn.softmax(scores, axis=-1)  # (B, T)
            context = tf.reduce_sum(
                tf.expand_dims(attn_weights, -1) * inputs, axis=1  # (B, units)
            )

        if self.return_attention_scores:
            return context, attn_weights
        return context

    def _call_dot(self, inputs, mask):
        """Multi-head scaled dot-product attention.

        Causal mode: apply GPT-style upper-triangular mask; return the last
        timestep's attended representation as the context vector.

        Non-causal mode: original mean-pooling behaviour.
        """
        B = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]

        Q = self.Q_proj(inputs)  # (B, T, units)
        K = self.K_proj(inputs)
        V = self.V_proj(inputs)

        # Reshape to multi-head: (B, H, T, D)
        Q = tf.transpose(tf.reshape(Q, [B, T, self.heads, self.head_dim]), [0, 2, 1, 3])
        K = tf.transpose(tf.reshape(K, [B, T, self.heads, self.head_dim]), [0, 2, 1, 3])
        V = tf.transpose(tf.reshape(V, [B, T, self.heads, self.head_dim]), [0, 2, 1, 3])

        attn_logits = tf.matmul(Q, K, transpose_b=True) / self.scale  # (B, H, T, T)

        if self.causal:
            # Upper-triangular mask: position i must not attend to j > i
            causal_mask = tf.linalg.band_part(
                tf.ones((T, T), dtype=attn_logits.dtype), -1, 0
            )  # lower-tri = 1
            attn_logits = attn_logits + (1.0 - causal_mask)[tf.newaxis, tf.newaxis, :, :] * -1e9

        # Padding mask (mask out key positions)
        if mask is not None:
            mask_float = tf.cast(mask, dtype=attn_logits.dtype)
            attn_logits = attn_logits + (1.0 - mask_float[:, tf.newaxis, tf.newaxis, :]) * -1e9

        attn_weights = tf.nn.softmax(attn_logits, axis=-1)  # (B, H, T, T)
        attended = tf.matmul(attn_weights, V)  # (B, H, T, D)

        # Merge heads: (B, T, units)
        attended = tf.reshape(tf.transpose(attended, [0, 2, 1, 3]), [B, T, self.units])
        if self.causal:
            # Last timestep has attended to all past positions — use it as context
            context = attended[:, -1, :]  # (B, units)
        else:
            # Original: mean-pool attention weights across heads and query positions,
            # re-normalise, then weighted-sum the attended sequence
            pooling_weights = tf.nn.softmax(tf.reduce_mean(attn_weights, axis=[1, 2]), axis=-1)  # (B, T)
            context = tf.reduce_sum(tf.expand_dims(pooling_weights, -1) * attended, axis=1)  # (B, units)

        if self.return_attention_scores:
            return context, attn_weights
        return context

    def _call_decay(self, inputs):
        """Exponential decay temporal weighting — inherently causal.

        weights[t] = λ^(T-1-t)  (most recent bar gets weight λ^0 = 1)
        Weights are normalised to sum to 1, then used to compute a weighted
        sum of hidden states.  λ is learned via a logit reparameterisation so
        it always stays in (0, 1).
        """
        T = tf.shape(inputs)[1]
        lam = tf.sigmoid(self.lambda_)  # constrain to (0, 1)

        # positions = [T-1, T-2, ..., 1, 0] → exponents for oldest→newest
        positions = tf.cast(tf.range(T - 1, -1, -1), dtype=inputs.dtype)  # (T,)
        weights = lam ** positions  # (T,)
        weights = weights / tf.reduce_sum(weights)  # normalise

        context = tf.reduce_sum(inputs * weights[tf.newaxis, :, tf.newaxis], axis=1)  # (B, units)
        if self.return_attention_scores:
            return context, tf.tile(weights[tf.newaxis, :], [tf.shape(inputs)[0], 1])
        return context

    def compute_mask(self, inputs, mask=None):
        """Output has no time dimension, so no mask propagates."""
        return None

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "attn_type": self.attn_type,
                "heads": self.heads,
                "causal": self.causal,
                "decay_init": self.decay_init,
                "return_attention_scores": self.return_attention_scores,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        batch_size, seq_len, units = input_shape

        if self.return_attention_scores:
            if self.attn_type in ("bahdanau", "decay"):
                return ((batch_size, units), (batch_size, seq_len))
            else:  # dot
                return ((batch_size, units), (batch_size, self.heads, seq_len, seq_len))
        return (batch_size, units)
