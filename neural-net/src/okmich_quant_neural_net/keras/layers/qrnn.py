"""
Quasi-Recurrent Neural Network (QRNN) Layer
============================================

QRNN is a hybrid between CNNs and RNNs:
- Uses convolutions for parallel feature extraction (fast)
- Uses recurrent pooling for temporal modeling (simple, efficient)

Key advantages:
- 2-20x faster than LSTM during training and inference
- Better parallelization than LSTM
- Less parameters than LSTM
- Maintains temporal modeling capability

Reference:
Bradbury et al. (2016) "Quasi-Recurrent Neural Networks" (https://arxiv.org/abs/1611.01576)
"""

import tensorflow as tf
from keras import layers
from keras.src.saving import register_keras_serializable


@register_keras_serializable()
class QRNN(layers.Layer):
    """
    Quasi-Recurrent Neural Network layer.
    QRNN applies convolutions across time for feature extraction, then uses element-wise gated recurrence for temporal pooling.

    Architecture:
        Z = tanh(Conv1D(X))      # Candidate values
        F = sigmoid(Conv1D(X))   # Forget gate
        O = sigmoid(Conv1D(X))   # Output gate (optional)
        H = f_pooling(Z, F, O)   # Recurrent pooling

    Args:
        units: Dimensionality of output space
        window_size: Size of convolutional window (default: 2)
        pooling: Type of pooling ('f' or 'fo') (default: 'fo')
            - 'f': Forget pooling (like GRU)
            - 'fo': Forget + Output pooling (like LSTM)
        return_sequences: Return full sequence or last timestep (default: True)
        kernel_initializer: Initializer for convolution kernels (default: 'glorot_uniform')
        **kwargs: Additional layer arguments

    Input shape:
        (batch_size, timesteps, input_dim)

    Output shape:
        If return_sequences=True: (batch_size, timesteps, units)
        If return_sequences=False: (batch_size, units)

    Example:
        >>> qrnn = QRNN(units=128, window_size=2, pooling='fo')
        >>> output = qrnn(inputs)  # (batch, timesteps, 128)
    """

    def __init__(
            self,
            units,
            window_size=2,
            pooling="fo",
            return_sequences=True,
            kernel_initializer="glorot_uniform",
            **kwargs,
    ):
        super(QRNN, self).__init__(**kwargs)
        self.units = units
        self.window_size = window_size
        self.pooling = pooling.lower()
        self.return_sequences = return_sequences
        self.kernel_initializer = kernel_initializer

        if self.pooling not in ["f", "fo"]:
            raise ValueError(f"pooling must be 'f' or 'fo', got '{pooling}'")

        # Will be built in build()
        self.conv_z = None
        self.conv_f = None
        self.conv_o = None

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Candidate values (Z) - uses tanh activation
        # padding="causal" ensures z[t] depends only on x[0..t] (no look-ahead).
        self.conv_z = layers.Conv1D(
            filters=self.units,
            kernel_size=self.window_size,
            padding="causal",
            activation="tanh",
            kernel_initializer=self.kernel_initializer,
            name="conv_z",
        )

        # Forget gate (F) - uses sigmoid activation
        self.conv_f = layers.Conv1D(
            filters=self.units,
            kernel_size=self.window_size,
            padding="causal",
            activation="sigmoid",
            kernel_initializer=self.kernel_initializer,
            name="conv_f",
        )

        # Output gate (O) - only for 'fo' pooling
        if self.pooling == "fo":
            self.conv_o = layers.Conv1D(
                filters=self.units,
                kernel_size=self.window_size,
                padding="causal",
                activation="sigmoid",
                kernel_initializer=self.kernel_initializer,
                name="conv_o",
            )
            self.conv_o.build(input_shape)

        self.conv_z.build(input_shape)
        self.conv_f.build(input_shape)
        super(QRNN, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        """
        Forward pass through QRNN layer.

        Args:
            inputs: Input tensor (batch, timesteps, features)
            mask: Optional mask tensor
            training: Whether in training mode

        Returns:
            Output tensor (batch, timesteps, units) or (batch, units)
        """
        # Compute gates through convolution (parallel across time)
        z = self.conv_z(inputs)  # Candidate values
        f = self.conv_f(inputs)  # Forget gate

        if self.pooling == "fo":
            o = self.conv_o(inputs)  # Output gate
        else:
            o = None

        # Apply recurrent pooling (sequential across time)
        h = self._recurrent_pooling(z, f, o)

        # Return sequences or just last timestep
        if not self.return_sequences:
            h = h[:, -1, :]

        return h

    def _recurrent_pooling(self, z, f, o=None):
        """
        Apply recurrent pooling across time dimension.

        This is the sequential part of QRNN, but much simpler than LSTM.
        Uses tf.while_loop so that the layer works correctly in graph mode
        and under tf.function — Python range() on a symbolic tf.Tensor fails
        at trace time.

        Args:
            z: Candidate values (batch, time, units)
            f: Forget gate (batch, time, units)
            o: Output gate (batch, time, units) - optional

        Returns:
            Hidden states (batch, time, units)
        """
        batch_size = tf.shape(z)[0]
        timesteps = tf.shape(z)[1]

        h_init = tf.zeros((batch_size, self.units), dtype=z.dtype)
        h_ta = tf.TensorArray(dtype=z.dtype, size=timesteps, dynamic_size=False)

        if o is not None:
            def loop_body(t, h_prev, h_ta):
                h_t = f[:, t, :] * h_prev + (1.0 - f[:, t, :]) * z[:, t, :]
                h_ta = h_ta.write(t, o[:, t, :] * h_t)
                return t + 1, h_t, h_ta
        else:
            def loop_body(t, h_prev, h_ta):
                h_t = f[:, t, :] * h_prev + (1.0 - f[:, t, :]) * z[:, t, :]
                h_ta = h_ta.write(t, h_t)
                return t + 1, h_t, h_ta

        _, _, h_ta = tf.while_loop(
            lambda t, *_: t < timesteps,
            loop_body,
            (tf.constant(0), h_init, h_ta),
        )

        # h_ta.stack() → (timesteps, batch, units)
        return tf.transpose(h_ta.stack(), [1, 0, 2])  # (batch, time, units)

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        else:
            return (input_shape[0], self.units)

    def get_config(self):
        """Get layer configuration."""
        config = super(QRNN, self).get_config()
        config.update(
            {
                "units": self.units,
                "window_size": self.window_size,
                "pooling": self.pooling,
                "return_sequences": self.return_sequences,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        return cls(**config)
