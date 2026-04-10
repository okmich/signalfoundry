"""
Echo State Network (ESN) Layer - Reservoir Computing
=====================================================

ESN is a type of Recurrent Neural Network using the "reservoir computing" paradigm.
Unlike traditional RNNs/LSTMs, the recurrent layer is randomly initialized and NEVER trained.
Only the output layer is trained using simple linear regression.

Key Advantages:
---------------
- 10-100x faster training than LSTM (no backpropagation through time)
- Real-time adaptability (can retrain output layer in milliseconds)
- Good memory capacity without expensive training
- Excellent for non-stationary data (cryptocurrency markets)
- Computationally efficient for resource-constrained environments
- CPU-friendly (no heavy GPU needed)

Reference Paper:
----------------
"Echo State Networks for Bitcoin Time Series Prediction"
Mansi Sharma, Enrico Sartor, Marc Cavazza, Helmut Prendinger
https://arxiv.org/pdf/2508.05416

Paper Summary:
--------------
The paper investigates Echo State Networks for predicting Bitcoin price movements.
ESNs employ a "reservoir computing" paradigm where a large random recurrent layer processes temporal sequences. Rather
than training all network weights, only the output layer requires training through linear regression. This approach
significantly reduces computational overhead compared to LSTMs or standard RNNs that demand backpropagation through time.

Key findings:
- ESNs achieve competitive or superior performance relative to conventional deep learning baselines on Bitcoin price prediction tasks
- Particular strength in capturing non-linear temporal dependencies inherent in cryptocurrency markets
- Reduced training time and computational resource requirements
- "Memory capacity and nonlinear transformation" properties through randomized reservoir initialization
- Effective temporal pattern recognition without expensive gradient-based optimization

Applications to Cryptocurrency Trading:
----------------------------------------
✓ Real-time cryptocurrency price forecasting with minimal latency
✓ Portfolio optimization and risk management strategies
✓ Algorithmic trading systems requiring quick model updates
✓ Walk-forward analysis (fast retraining across windows)
✓ Resource-constrained production environments
✓ Adaptive trading systems (quick regime adaptation)

Architecture:
-------------
Traditional RNN/LSTM:
    Input → Trained Recurrent Layer → Trained Output Layer
             (slow backprop through time)

Echo State Network:
    Input → Random Fixed Reservoir → Trained Output Layer (linear regression)
             (never trained!)        (fast!)

How It Works:
-------------
1. Reservoir (Random, Fixed):
   - Large sparse random recurrent matrix
   - Initialized once, never trained
   - Acts as a "temporal kernel" transforming input
   - Captures complex temporal dynamics

2. Training:
   - Only output weights trained via linear regression
   - Can use Ridge regression for regularization
   - Training is 10-100x faster than LSTM

3. Key Hyperparameters:
   - reservoir_size: 100-1000 neurons (larger = more capacity)
   - spectral_radius: 0.9-1.5 (controls memory, echo state property)
   - sparsity: 0.1-0.3 (reservoir connectivity, lower = sparser)
   - input_scaling: 0.1-1.0 (scales input to reservoir)
   - leak_rate: 0.1-1.0 (leaky integration, 1.0 = no leak)
"""

import numpy as np
import tensorflow as tf
from keras import layers
from keras.src.saving import register_keras_serializable


@register_keras_serializable()
class EchoStateNetwork(layers.Layer):
    """
    Echo State Network layer using reservoir computing.

    The reservoir (recurrent layer) is randomly initialized and fixed.
    Only the output weights are trained using linear regression.

    Args:
        reservoir_size: Number of neurons in the reservoir (default: 500)
        spectral_radius: Spectral radius of reservoir matrix (default: 1.2)
            - Controls memory/echo state property
            - Values > 1.0 allow longer memory
            - Typical range: 0.9-1.5
        sparsity: Sparsity of reservoir connections (default: 0.2)
            - Fraction of connections to keep
            - Lower = sparser network
            - Typical range: 0.1-0.3
        input_scaling: Scaling factor for input weights (default: 0.5)
            - Controls input signal strength
            - Typical range: 0.1-1.0
        leak_rate: Leaky integration rate (default: 0.3)
            - 1.0 = no leak (standard ESN)
            - Lower values = more smoothing
            - Typical range: 0.1-1.0
        return_sequences: Return full sequence or last state (default: True)
        random_state: Random seed for reproducibility (default: 42)

    Input shape:
        (batch_size, timesteps, input_dim)

    Output shape:
        If return_sequences=True: (batch_size, timesteps, reservoir_size)
        If return_sequences=False: (batch_size, reservoir_size)

    Example:
        >>> esn = EchoStateNetwork(
        ...     reservoir_size=500,
        ...     spectral_radius=1.2,
        ...     sparsity=0.2,
        ...     input_scaling=0.5
        ... )
        >>> output = esn(inputs)  # (batch, timesteps, 500)
    """

    def __init__(
            self,
            reservoir_size=500,
            spectral_radius=1.2,
            sparsity=0.2,
            input_scaling=0.5,
            leak_rate=0.3,
            return_sequences=True,
            random_state=42,
            **kwargs,
    ):
        super(EchoStateNetwork, self).__init__(**kwargs)
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.return_sequences = return_sequences
        self.random_state = random_state

        # Will be initialized in build()
        self.W_in = None  # Input weights
        self.W_res = None  # Reservoir weights (recurrent)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Use numpy for reproducible random initialization
        rng = np.random.RandomState(self.random_state)

        # Initialize input weights (input_dim → reservoir_size)
        # Random uniform in [-input_scaling, input_scaling]
        W_in_np = rng.uniform(
            -self.input_scaling,
            self.input_scaling,
            size=(input_dim, self.reservoir_size),
        )
        self.W_in = tf.constant(W_in_np, dtype=tf.float32)

        # Initialize reservoir weights (recurrent matrix)
        # 1. Create random matrix
        W_res_np = rng.randn(self.reservoir_size, self.reservoir_size)

        # 2. Apply sparsity (make it sparse)
        mask = rng.rand(self.reservoir_size, self.reservoir_size) < self.sparsity
        W_res_np *= mask

        # 3. Scale by spectral radius (echo state property)
        # Compute current spectral radius
        eigenvalues = np.linalg.eigvals(W_res_np)
        current_spectral_radius = np.max(np.abs(eigenvalues))

        # Scale to desired spectral radius
        if current_spectral_radius > 0:
            W_res_np *= self.spectral_radius / current_spectral_radius

        self.W_res = tf.constant(W_res_np, dtype=tf.float32)

        super(EchoStateNetwork, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        """
        Forward pass through ESN layer.

        Args:
            inputs: Input tensor (batch, timesteps, features)
            mask: Optional mask tensor
            training: Whether in training mode (not used, reservoir is fixed)

        Returns:
            Reservoir states (batch, timesteps, reservoir_size) or
            (batch, reservoir_size) if return_sequences=False
        """
        batch_size = tf.shape(inputs)[0]
        timesteps = tf.shape(inputs)[1]

        # Cast fixed weights to match input dtype so the layer works correctly
        # under mixed precision (float16) or float64 models.
        W_in = tf.cast(self.W_in, inputs.dtype)
        W_res = tf.cast(self.W_res, inputs.dtype)
        leak_rate = tf.cast(self.leak_rate, inputs.dtype)

        # Initial reservoir state
        state_init = tf.zeros((batch_size, self.reservoir_size), dtype=inputs.dtype)
        states_ta = tf.TensorArray(dtype=inputs.dtype, size=timesteps, dynamic_size=False)

        # Use tf.while_loop so this works in graph mode / tf.function —
        # Python range() on a symbolic tf.Tensor would fail at trace time.
        def loop_body(t, state, states_ta):
            x_t = inputs[:, t, :]
            new_state = tf.tanh(tf.matmul(x_t, W_in) + tf.matmul(state, W_res))
            state = (1.0 - leak_rate) * state + leak_rate * new_state
            states_ta = states_ta.write(t, state)
            return t + 1, state, states_ta

        _, _, states_ta = tf.while_loop(
            lambda t, *_: t < timesteps,
            loop_body,
            (tf.constant(0), state_init, states_ta),
        )

        # states_ta.stack() → (timesteps, batch, reservoir_size)
        output = tf.transpose(states_ta.stack(), [1, 0, 2])  # (batch, timesteps, reservoir_size)

        if not self.return_sequences:
            output = output[:, -1, :]

        return output

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.reservoir_size)
        else:
            return (input_shape[0], self.reservoir_size)

    def get_config(self):
        """Get layer configuration."""
        config = super(EchoStateNetwork, self).get_config()
        config.update(
            {
                "reservoir_size": self.reservoir_size,
                "spectral_radius": self.spectral_radius,
                "sparsity": self.sparsity,
                "input_scaling": self.input_scaling,
                "leak_rate": self.leak_rate,
                "return_sequences": self.return_sequences,
                "random_state": self.random_state,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        return cls(**config)


@register_keras_serializable()
class DeepEchoStateNetwork(layers.Layer):
    """
    Deep Echo State Network with multiple stacked reservoir layers.

    Stacks multiple ESN reservoirs for hierarchical temporal feature learning.
    Each reservoir layer operates at different temporal scales.

    Args:
        reservoir_sizes: List of reservoir sizes for each layer (e.g., [500, 300, 200])
        spectral_radius: Spectral radius for all reservoirs (default: 1.2)
        sparsity: Sparsity for all reservoirs (default: 0.2)
        input_scaling: Input scaling for first layer (default: 0.5)
        inter_scaling: Scaling between reservoir layers (default: 0.5)
        leak_rate: Leak rate for all layers (default: 0.3)
        return_sequences: Return sequences or last state (default: True)
        random_state: Random seed (default: 42)

    Input shape:
        (batch_size, timesteps, input_dim)

    Output shape:
        If return_sequences=True: (batch_size, timesteps, reservoir_sizes[-1])
        If return_sequences=False: (batch_size, reservoir_sizes[-1])

    Example:
        >>> deep_esn = DeepEchoStateNetwork(
        ...     reservoir_sizes=[500, 300, 200],
        ...     spectral_radius=1.2
        ... )
        >>> output = deep_esn(inputs)
    """

    def __init__(
            self,
            reservoir_sizes=(500, 300, 200),
            spectral_radius=1.2,
            sparsity=0.2,
            input_scaling=0.5,
            inter_scaling=0.5,
            leak_rate=0.3,
            return_sequences=True,
            random_state=42,
            **kwargs,
    ):
        super(DeepEchoStateNetwork, self).__init__(**kwargs)
        self.reservoir_sizes = list(reservoir_sizes)
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.inter_scaling = inter_scaling
        self.leak_rate = leak_rate
        self.return_sequences = return_sequences
        self.random_state = random_state

        # Create ESN layers
        self.esn_layers = []
        for i, size in enumerate(self.reservoir_sizes):
            # First layer uses input_scaling, others use inter_scaling
            scaling = self.input_scaling if i == 0 else self.inter_scaling

            # All layers return sequences except the last (if return_sequences=False)
            return_seq = (
                True if i < len(self.reservoir_sizes) - 1 else self.return_sequences
            )

            esn_layer = EchoStateNetwork(
                reservoir_size=size,
                spectral_radius=self.spectral_radius,
                sparsity=self.sparsity,
                input_scaling=scaling,
                leak_rate=self.leak_rate,
                return_sequences=return_seq,
                random_state=self.random_state + i,
                name=f"esn_layer_{i}",
            )
            self.esn_layers.append(esn_layer)

    def call(self, inputs, mask=None, training=None):
        """Forward pass through stacked ESN layers."""
        x = inputs
        for esn_layer in self.esn_layers:
            x = esn_layer(x, mask=mask, training=training)
        return x

    def get_config(self):
        """Get layer configuration."""
        config = super(DeepEchoStateNetwork, self).get_config()
        config.update(
            {
                "reservoir_sizes": self.reservoir_sizes,
                "spectral_radius": self.spectral_radius,
                "sparsity": self.sparsity,
                "input_scaling": self.input_scaling,
                "inter_scaling": self.inter_scaling,
                "leak_rate": self.leak_rate,
                "return_sequences": self.return_sequences,
                "random_state": self.random_state,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        return cls(**config)
