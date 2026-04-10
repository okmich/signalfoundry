"""
Independently Recurrent Neural Network (IndRNN) Layer Implementation
=====================================================================

Reference:
Li, S., Li, W., Cook, C., Zhu, C., & Gao, Y. (2018).
"Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN"
arXiv:1803.04831

Key innovation: Each neuron's recurrent connection is independent (vector, not matrix).
This allows for better gradient flow and enables very deep stacking.
"""

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="CustomLayers")
class IndRNNCell(layers.Layer):
    """
    Independently Recurrent Neural Network (IndRNN) Cell.

    Reference:
    Li, S., Li, W., Cook, C., Zhu, C., & Gao, Y. (2018).
    "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN"
    arXiv:1803.04831

    Key innovation: Each neuron's recurrent connection is independent (vector, not matrix).
    This allows for better gradient flow and enables very deep stacking.
    """

    def __init__(
            self,
            units,
            activation="relu",
            recurrent_clip_min=None,
            recurrent_clip_max=None,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="uniform",
            bias_initializer="zeros",
            **kwargs,
    ):
        super(IndRNNCell, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.recurrent_clip_min = recurrent_clip_min
        self.recurrent_clip_max = recurrent_clip_max
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = keras.initializers.get(recurrent_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.state_size = units
        self.output_size = units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Input-to-hidden weights (matrix)
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            name="kernel",
            initializer=self.kernel_initializer,
        )

        # Recurrent weights (VECTOR - key difference from standard RNN!)
        # Initialize with small values for long sequences
        if isinstance(self.recurrent_initializer, keras.initializers.Constant):
            recurrent_init = self.recurrent_initializer
        else:
            # Default: uniform initialization for recurrent weights
            recurrent_init = keras.initializers.RandomUniform(minval=0, maxval=1)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units,),  # Vector, not matrix!
            name="recurrent_kernel",
            initializer=recurrent_init,
        )

        # Bias
        self.bias = self.add_weight(
            shape=(self.units,), name="bias", initializer=self.bias_initializer
        )

        super(IndRNNCell, self).build(input_shape)

    def call(self, inputs, states):
        prev_output = states[0]

        # h_t = activation(W * x_t + u ⊙ h_{t-1} + b)
        # W * x_t: input transformation
        h = ops.dot(inputs, self.kernel)

        # u ⊙ h_{t-1}: element-wise recurrence (independent neurons).
        # IndRNN stability guarantee (Li et al. 2018): clip the recurrent weight
        # vector u (self.recurrent_kernel) so |u_i| <= clip_max.  Clipping the
        # weight, not the per-step product, is what prevents gradient explosion
        # through the recurrent connection across long sequences.
        recurrent_kernel = self.recurrent_kernel
        if self.recurrent_clip_min is not None and self.recurrent_clip_max is not None:
            recurrent_kernel = ops.clip(
                recurrent_kernel, self.recurrent_clip_min, self.recurrent_clip_max
            )
        recurrent_h = prev_output * recurrent_kernel

        # Add bias
        output = h + recurrent_h + self.bias

        # Apply activation
        output = self.activation(output)

        return output, [output]

    def get_config(self):
        config = {
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "recurrent_clip_min": self.recurrent_clip_min,
            "recurrent_clip_max": self.recurrent_clip_max,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "recurrent_initializer": keras.initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
        }
        base_config = super(IndRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras.saving.register_keras_serializable(package="CustomLayers")
class IndRNN(layers.RNN):
    """
    Independently Recurrent Neural Network (IndRNN) Layer.

    Wrapper around IndRNNCell to provide full RNN functionality.
    """

    def __init__(
            self,
            units,
            activation="relu",
            recurrent_clip_min=None,
            recurrent_clip_max=None,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=False,
            **kwargs,
    ):
        cell = IndRNNCell(
            units,
            activation=activation,
            recurrent_clip_min=recurrent_clip_min,
            recurrent_clip_max=recurrent_clip_max,
        )
        super(IndRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs,
        )
        self._units = units

    @property
    def units(self):
        return self._units

    def get_config(self):
        config = super(IndRNN, self).get_config()
        # Add cell parameters at top level for easier access
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.cell.activation),
            "recurrent_clip_min": self.cell.recurrent_clip_min,
            "recurrent_clip_max": self.cell.recurrent_clip_max,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Extract cell config and our custom parameters
        cell_config = config.pop("cell", None)

        # Remove our custom top-level parameters to avoid duplication
        units = config.pop("units", None)
        activation = config.pop("activation", None)
        recurrent_clip_min = config.pop("recurrent_clip_min", None)
        recurrent_clip_max = config.pop("recurrent_clip_max", None)

        # If we don't have top-level params, try to get from cell config
        if units is None and cell_config:
            cell_dict = cell_config["config"]
            units = cell_dict["units"]
            activation = cell_dict["activation"]
            recurrent_clip_min = cell_dict.get("recurrent_clip_min")
            recurrent_clip_max = cell_dict.get("recurrent_clip_max")

        if units is not None:
            return cls(
                units=units,
                activation=activation,
                recurrent_clip_min=recurrent_clip_min,
                recurrent_clip_max=recurrent_clip_max,
                **config
            )
        return super(IndRNN, cls).from_config(config)
