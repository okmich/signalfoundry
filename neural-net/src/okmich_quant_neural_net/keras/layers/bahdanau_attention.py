"""
Bahdanau Attention Layer Implementation
========================================

Reference:
Bahdanau, D., Cho, K., & Bengio, Y. (2014).
"Neural Machine Translation by Jointly Learning to Align and Translate"
arXiv:1409.0473

Bahdanau attention (also called additive attention) is a widely used attention mechanism for sequence-to-sequence models.
It computes attention weights by using a learned alignment function that combines encoder outputs with the decoder state.

When to use this vs LightweightAttention(attn_type="bahdanau")
--------------------------------------------------------------
Use THIS layer when you have an encoder-decoder (seq2seq) architecture where a separate decoder state (query) attends over
encoder outputs (values).  The two inputs are *different* tensors.

    query  : (B, D)    — decoder hidden state at current step
    values : (B, T, D) — full encoder output sequence

Use LightweightAttention(attn_type="bahdanau") when you have an encoder-only model and want to pool a single sequence down
to one context vector (temporal pooling via self-attention).  It applies the same additive scoring function but query and
keys are both projections of the *same* input sequence, supports causal masking, and handles padding masks automatically.
"""

import tensorflow as tf
from keras import layers
from keras.src.saving import register_keras_serializable


@register_keras_serializable()
class BahdanauAttention(layers.Layer):
    """
    Bahdanau Attention mechanism for sequence-to-sequence models.

    Computes attention weights over encoder sequence outputs and returns a context vector as a weighted sum of encoder outputs.
    """

    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units, name="attention_W1")
        self.W2 = layers.Dense(units, name="attention_W2")
        self.V = layers.Dense(1, name="attention_V")

    def call(self, query, values):
        """
        Compute attention context vector.

        Parameters
        ----------
        query : tensor
            Decoder state (batch_size, decoder_units)
        values : tensor
            Encoder outputs (batch_size, sequence_length, encoder_units)

        Returns
        -------
        context_vector : tensor
            Attention-weighted context (batch_size, encoder_units)
        attention_weights : tensor
            Attention weights (batch_size, sequence_length, 1)
        """
        # Expand query to (batch_size, 1, decoder_units)
        query_with_time_axis = tf.expand_dims(query, 1)

        # Score shape: (batch_size, sequence_length, units)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # Attention weights shape: (batch_size, sequence_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # Context vector shape: (batch_size, encoder_units)
        context_vector = tf.reduce_sum(attention_weights * values, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({"units": self.units})
        return config
