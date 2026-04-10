import tensorflow as tf
from keras import layers
from keras.src.saving import register_keras_serializable


@register_keras_serializable()
class FeatureAttention(layers.Layer):
    """Pointwise feature attention: (B, T, F) → (B, T, F).

    Computes independent attention weights over the F features at each
    timestep.  There is no cross-time interaction, so this layer is
    strictly causal by construction.

    Formula:
        e     = tanh(W_e · x)           W_e ∈ R^{F×F}
        alpha = softmax(e, axis=-1)     softmax over feature dim
        out   = x * alpha

    Args:
        **kwargs: Passed to the base Layer.

    Examples:
        fa = FeatureAttention(name="feature_attention")
        out = fa(inputs)  # (B, T, F) → (B, T, F)
    """

    def build(self, input_shape):
        F = input_shape[-1]
        self.W_e = layers.Dense(F, use_bias=False, name="feat_attn_W")
        self.W_e.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(self.W_e(x))  # (B, T, F)
        alpha = tf.nn.softmax(e, axis=-1)  # softmax over feature dim
        return x * alpha  # (B, T, F)

    def get_config(self):
        return super().get_config()

    def compute_output_shape(self, input_shape):
        return input_shape
