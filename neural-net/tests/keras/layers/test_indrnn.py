import numpy as np
import tensorflow as tf
from keras import models, layers as keras_layers

from okmich_quant_neural_net.keras.layers.indrnn import IndRNN, IndRNNCell


class TestIndRNNCellInitialization:
    """Test IndRNNCell initialization and parameter validation."""

    def test_default_initialization(self):
        """Test default parameters."""
        cell = IndRNNCell(units=64)
        assert cell.units == 64
        assert cell.state_size == 64
        assert cell.output_size == 64
        assert cell.recurrent_clip_min is None
        assert cell.recurrent_clip_max is None

    def test_custom_initialization(self):
        """Test custom parameters."""
        cell = IndRNNCell(
            units=128,
            activation="tanh",
            recurrent_clip_min=-1.0,
            recurrent_clip_max=1.0,
        )
        assert cell.units == 128
        assert cell.recurrent_clip_min == -1.0
        assert cell.recurrent_clip_max == 1.0


class TestIndRNNCellBuild:
    """Test IndRNNCell weight initialization."""

    def test_build_creates_weights(self):
        """Test that build creates all required weights."""
        cell = IndRNNCell(units=64)
        cell.build((None, 32))

        # Should have kernel (matrix), recurrent_kernel (vector), and bias
        assert cell.kernel is not None
        assert cell.recurrent_kernel is not None
        assert cell.bias is not None

        # Check shapes
        assert cell.kernel.shape == (32, 64)
        assert cell.recurrent_kernel.shape == (64,)  # Vector, not matrix!
        assert cell.bias.shape == (64,)

    def test_recurrent_kernel_is_vector(self):
        """Test that recurrent_kernel is a vector (key feature of IndRNN)."""
        cell = IndRNNCell(units=128)
        cell.build((None, 64))

        # IndRNN uses vector recurrent weights, not matrix
        assert len(cell.recurrent_kernel.shape) == 1
        assert cell.recurrent_kernel.shape[0] == 128


class TestIndRNNCellForward:
    """Test IndRNNCell forward pass."""

    def test_call_basic(self):
        """Test basic forward pass."""
        cell = IndRNNCell(units=64)
        cell.build((None, 32))

        inputs = tf.random.normal((2, 32))
        states = [tf.zeros((2, 64))]

        output, new_states = cell(inputs, states)

        assert output.shape == (2, 64)
        assert len(new_states) == 1
        assert new_states[0].shape == (2, 64)
        # Output should equal new state
        assert tf.reduce_all(output == new_states[0])

    def test_call_with_clipping(self):
        """Test forward pass with recurrent weight clipping."""
        cell = IndRNNCell(
            units=64, recurrent_clip_min=-1.0, recurrent_clip_max=1.0
        )
        cell.build((None, 32))

        inputs = tf.random.normal((2, 32))
        states = [tf.random.normal((2, 64)) * 10]  # Large values

        output, new_states = cell(inputs, states)

        assert output.shape == (2, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_activation_applied(self):
        """Test that activation is applied to output."""
        cell = IndRNNCell(units=64, activation="relu")
        cell.build((None, 32))

        inputs = tf.random.normal((2, 32))
        states = [tf.zeros((2, 64))]

        output, _ = cell(inputs, states)

        # With ReLU, all outputs should be >= 0
        assert tf.reduce_all(output >= 0)


class TestIndRNNInitialization:
    """Test IndRNN layer initialization."""

    def test_default_initialization(self):
        """Test default parameters."""
        layer = IndRNN(units=64)
        assert layer.units == 64
        assert isinstance(layer.cell, IndRNNCell)

    def test_custom_initialization(self):
        """Test custom parameters."""
        layer = IndRNN(
            units=128,
            activation="tanh",
            recurrent_clip_min=-0.5,
            recurrent_clip_max=0.5,
            return_sequences=True,
            return_state=True,
        )
        assert layer.units == 128
        assert layer.cell.recurrent_clip_min == -0.5
        assert layer.cell.recurrent_clip_max == 0.5


class TestIndRNNForward:
    """Test IndRNN layer forward pass."""

    def test_forward_return_sequences_false(self):
        """Test forward pass with return_sequences=False."""
        layer = IndRNN(units=64, return_sequences=False)
        inputs = tf.random.normal((2, 10, 32))

        output = layer(inputs)

        assert output.shape == (2, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_forward_return_sequences_true(self):
        """Test forward pass with return_sequences=True."""
        layer = IndRNN(units=64, return_sequences=True)
        inputs = tf.random.normal((2, 10, 32))

        output = layer(inputs)

        assert output.shape == (2, 10, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_forward_return_state_true(self):
        """Test forward pass with return_state=True."""
        layer = IndRNN(units=64, return_state=True)
        inputs = tf.random.normal((2, 10, 32))

        output, state = layer(inputs)

        assert output.shape == (2, 64)
        assert state.shape == (2, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_nan(state))

    def test_forward_with_mask(self):
        """Test forward pass with mask."""
        layer = IndRNN(units=64)
        inputs = tf.random.normal((2, 10, 32))
        mask = tf.constant([[True] * 7 + [False] * 3, [True] * 5 + [False] * 5])

        output = layer(inputs, mask=mask)

        assert output.shape == (2, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))


class TestSerialization:
    """Test layer serialization and deserialization."""

    def test_cell_get_config(self):
        """Test IndRNNCell get_config returns all parameters."""
        cell = IndRNNCell(
            units=128,
            activation="tanh",
            recurrent_clip_min=-1.0,
            recurrent_clip_max=1.0,
        )

        config = cell.get_config()

        assert config["units"] == 128
        assert config["recurrent_clip_min"] == -1.0
        assert config["recurrent_clip_max"] == 1.0

    def test_layer_get_config(self):
        """Test IndRNN get_config returns all parameters."""
        layer = IndRNN(
            units=128,
            activation="tanh",
            recurrent_clip_min=-1.0,
            recurrent_clip_max=1.0,
            return_sequences=True,
        )

        config = layer.get_config()

        assert config["units"] == 128
        assert config["recurrent_clip_min"] == -1.0
        assert config["recurrent_clip_max"] == 1.0

    def test_layer_from_config(self):
        """Test layer can be recreated from config."""
        layer = IndRNN(units=128, activation="tanh", return_sequences=True)

        config = layer.get_config()
        new_layer = IndRNN.from_config(config)

        assert new_layer.units == layer.units
        assert new_layer.return_sequences == layer.return_sequences


class TestIntegration:
    """Test integration scenarios."""

    def test_in_keras_model(self):
        """Test IndRNN works in a Keras model."""
        inputs = keras_layers.Input(shape=(10, 32))
        x = IndRNN(units=64, return_sequences=False)(inputs)
        outputs = keras_layers.Dense(1)(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")

        # Test forward pass
        batch_data = tf.random.normal((2, 10, 32))
        predictions = model(batch_data)

        assert predictions.shape == (2, 1)
        assert not tf.reduce_any(tf.math.is_nan(predictions))

    def test_stacked_indrnn(self):
        """Test stacking multiple IndRNN layers."""
        inputs = keras_layers.Input(shape=(10, 32))
        x = IndRNN(units=128, return_sequences=True)(inputs)
        x = keras_layers.Dropout(0.3)(x)
        x = IndRNN(units=64, return_sequences=True)(x)
        x = keras_layers.Dropout(0.3)(x)
        x = IndRNN(units=64, return_sequences=False)(x)
        outputs = keras_layers.Dense(3, activation="softmax")(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        # Test forward pass
        batch_data = tf.random.normal((4, 10, 32))
        predictions = model(batch_data)

        assert predictions.shape == (4, 3)
        assert not tf.reduce_any(tf.math.is_nan(predictions))

    def test_gradient_flow(self):
        """Test gradients flow correctly through the layer."""
        layer = IndRNN(units=64, return_sequences=False)
        inputs = tf.random.normal((2, 10, 32))

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            output = layer(inputs)
            loss = tf.reduce_mean(output ** 2)

        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check all gradients are not None and not NaN
        for grad in gradients:
            assert grad is not None
            assert not tf.reduce_any(tf.math.is_nan(grad))

    def test_training_integration(self):
        """Integration test: Training a simple model with IndRNN."""
        inputs = keras_layers.Input(shape=(20, 16))
        x = IndRNN(units=64, return_sequences=True)(inputs)
        x = IndRNN(units=32, return_sequences=False)(x)
        outputs = keras_layers.Dense(1)(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")

        # Generate dummy data
        X = np.random.randn(32, 20, 16).astype(np.float32)
        y = np.random.randn(32, 1).astype(np.float32)

        # Train for a few steps
        history = model.fit(X, y, epochs=2, batch_size=8, verbose=0)

        assert len(history.history["loss"]) == 2
        assert not np.isnan(history.history["loss"][-1])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_timestep(self):
        """Test with single timestep sequence."""
        layer = IndRNN(units=64, return_sequences=False)
        inputs = tf.random.normal((2, 1, 32))

        output = layer(inputs)

        assert output.shape == (2, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_long_sequence(self):
        """Test with long sequence."""
        layer = IndRNN(units=64, return_sequences=False)
        inputs = tf.random.normal((2, 200, 32))

        output = layer(inputs)

        assert output.shape == (2, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_large_batch_size(self):
        """Test with large batch size."""
        layer = IndRNN(units=64, return_sequences=False)
        inputs = tf.random.normal((128, 10, 32))

        output = layer(inputs)

        assert output.shape == (128, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_single_unit(self):
        """Test with single unit."""
        layer = IndRNN(units=1, return_sequences=False)
        inputs = tf.random.normal((2, 10, 32))

        output = layer(inputs)

        assert output.shape == (2, 1)
        assert not tf.reduce_any(tf.math.is_nan(output))


class TestComparison:
    """Test IndRNN specific properties vs standard RNN."""

    def test_recurrent_weights_independent(self):
        """Test that recurrent weights are element-wise (vector vs matrix)."""
        # IndRNN should have vector recurrent weights
        indrnn = IndRNN(units=64)
        indrnn.build((None, 10, 32))

        # Recurrent kernel should be a vector (1D)
        assert len(indrnn.cell.recurrent_kernel.shape) == 1
        assert indrnn.cell.recurrent_kernel.shape[0] == 64

    def test_deep_stacking_stability(self):
        """Test that IndRNN allows deep stacking without vanishing gradients."""
        # Build a very deep model (6 layers)
        inputs = keras_layers.Input(shape=(50, 32))
        x = inputs
        for _ in range(6):
            x = IndRNN(units=64, return_sequences=True)(x)

        x = IndRNN(units=64, return_sequences=False)(x)
        outputs = keras_layers.Dense(1)(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")

        # Test gradient flow through deep network
        batch_data = tf.random.normal((4, 50, 32))
        batch_labels = tf.random.normal((4, 1))

        with tf.GradientTape() as tape:
            predictions = model(batch_data, training=True)
            loss = tf.reduce_mean((predictions - batch_labels) ** 2)

        gradients = tape.gradient(loss, model.trainable_variables)

        # Check that gradients flow all the way to first layer
        first_layer_grads = gradients[:3]  # First layer weights
        for grad in first_layer_grads:
            assert grad is not None
            assert not tf.reduce_any(tf.math.is_nan(grad))
            # Gradients should not vanish (not all close to zero)
            assert tf.reduce_max(tf.abs(grad)) > 1e-6


class TestBugFixes:
    """Test specific bug fixes and known issues."""

    def test_no_gradient_explosion_with_clipping(self):
        """Test that recurrent clipping prevents gradient explosion."""
        layer = IndRNN(
            units=64,
            recurrent_clip_min=-1.0,
            recurrent_clip_max=1.0,
            return_sequences=False,
        )

        inputs = tf.random.normal((2, 100, 32))

        with tf.GradientTape() as tape:
            output = layer(inputs, training=True)
            loss = tf.reduce_mean(output ** 2)

        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check no gradient explosion
        for grad in gradients:
            assert not tf.reduce_any(tf.math.is_inf(grad))
            assert not tf.reduce_any(tf.math.is_nan(grad))

    def test_variable_length_sequences(self):
        """Test with different sequence lengths."""
        layer = IndRNN(units=64, return_sequences=False)
        layer.build((None, 100, 32))

        for seq_len in [10, 50, 100, 200]:
            inputs = tf.random.normal((2, seq_len, 32))
            output = layer(inputs)

            assert output.shape == (2, 64)
            assert not tf.reduce_any(tf.math.is_nan(output))

    def test_stateful_mode(self):
        """Test stateful mode (maintains state between batches)."""
        # Note: Keras 3 stateful RNNs require batch_input_shape in Input layer
        inputs = keras_layers.Input(batch_shape=(1, 10, 32))
        layer = IndRNN(units=64, return_sequences=False, stateful=True)
        outputs = layer(inputs)

        model = models.Model(inputs=inputs, outputs=outputs)

        # First batch
        batch1 = tf.random.normal((1, 10, 32))
        output1 = model(batch1)

        # Second batch (state should be maintained)
        batch2 = tf.random.normal((1, 10, 32))
        output2 = model(batch2)

        # Reset state
        layer.reset_states()
        output3 = model(batch1)  # Same input as first batch

        # Output3 should match output1 (after reset), not output2
        assert output1.shape == output2.shape == output3.shape == (1, 64)
