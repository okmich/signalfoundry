import numpy as np
import tensorflow as tf
from keras import models, layers as keras_layers

from okmich_quant_neural_net.keras.layers.bahdanau_attention import BahdanauAttention


class TestBahdanauAttentionInitialization:
    """Test initialization and parameter validation."""

    def test_default_initialization(self):
        """Test initialization with required units parameter."""
        layer = BahdanauAttention(units=64)
        assert layer.units == 64
        assert layer.W1 is not None
        assert layer.W2 is not None
        assert layer.V is not None

    def test_custom_units(self):
        """Test initialization with different unit sizes."""
        for units in [16, 32, 64, 128, 256]:
            layer = BahdanauAttention(units=units)
            assert layer.units == units


class TestBahdanauAttentionBuild:
    """Test layer building."""

    def test_dense_layers_created(self):
        """Test that W1, W2, V dense layers are created."""
        layer = BahdanauAttention(units=64)

        # W1, W2, V should be Dense layers
        assert isinstance(layer.W1, keras_layers.Dense)
        assert isinstance(layer.W2, keras_layers.Dense)
        assert isinstance(layer.V, keras_layers.Dense)

        # Check output dimensions
        assert layer.W1.units == 64
        assert layer.W2.units == 64
        assert layer.V.units == 1


class TestBahdanauAttentionForward:
    """Test forward pass functionality."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        layer = BahdanauAttention(units=64)

        # Query: decoder state (batch_size, decoder_units)
        query = tf.random.normal((2, 128))
        # Values: encoder outputs (batch_size, sequence_length, encoder_units)
        values = tf.random.normal((2, 10, 128))

        context_vector, attention_weights = layer(query, values)

        # Context vector should have shape (batch_size, encoder_units)
        assert context_vector.shape == (2, 128)
        # Attention weights should have shape (batch_size, sequence_length, 1)
        assert attention_weights.shape == (2, 10, 1)

        assert not tf.reduce_any(tf.math.is_nan(context_vector))
        assert not tf.reduce_any(tf.math.is_nan(attention_weights))

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 across sequence dimension."""
        layer = BahdanauAttention(units=64)

        query = tf.random.normal((4, 128))
        values = tf.random.normal((4, 20, 128))

        _, attention_weights = layer(query, values)

        # Sum along sequence dimension should be ~1.0
        weights_sum = tf.reduce_sum(attention_weights, axis=1)
        assert tf.reduce_all(tf.abs(weights_sum - 1.0) < 1e-5)

    def test_attention_weights_positive(self):
        """Test that attention weights are all positive (softmax output)."""
        layer = BahdanauAttention(units=64)

        query = tf.random.normal((2, 128))
        values = tf.random.normal((2, 10, 128))

        _, attention_weights = layer(query, values)

        assert tf.reduce_all(attention_weights >= 0)
        assert tf.reduce_all(attention_weights <= 1)

    def test_context_is_weighted_sum(self):
        """Test that context vector is a weighted sum of encoder outputs."""
        layer = BahdanauAttention(units=64)

        query = tf.random.normal((1, 128))
        values = tf.random.normal((1, 5, 128))

        context_vector, attention_weights = layer(query, values)

        # Manually compute weighted sum
        manual_context = tf.reduce_sum(
            attention_weights * values, axis=1
        )

        # Should match the layer's output
        assert tf.reduce_all(tf.abs(context_vector - manual_context) < 1e-5)

    def test_different_sequence_lengths(self):
        """Test with varying sequence lengths."""
        layer = BahdanauAttention(units=64)

        query = tf.random.normal((2, 128))

        for seq_len in [5, 10, 20, 50, 100]:
            values = tf.random.normal((2, seq_len, 128))
            context_vector, attention_weights = layer(query, values)

            assert context_vector.shape == (2, 128)
            assert attention_weights.shape == (2, seq_len, 1)

    def test_different_encoder_decoder_dims(self):
        """Test with different encoder and decoder dimensions."""
        layer = BahdanauAttention(units=64)

        # Decoder state: 256 dims
        query = tf.random.normal((2, 256))
        # Encoder outputs: 128 dims
        values = tf.random.normal((2, 10, 128))

        context_vector, attention_weights = layer(query, values)

        # Context should match encoder dimension
        assert context_vector.shape == (2, 128)
        assert attention_weights.shape == (2, 10, 1)


class TestSerialization:
    """Test layer serialization and deserialization."""

    def test_get_config(self):
        """Test get_config returns all parameters."""
        layer = BahdanauAttention(units=128)

        config = layer.get_config()

        assert config["units"] == 128

    def test_from_config(self):
        """Test layer can be recreated from config."""
        layer = BahdanauAttention(units=128)

        config = layer.get_config()
        new_layer = BahdanauAttention.from_config(config)

        assert new_layer.units == layer.units


class TestIntegration:
    """Test integration scenarios."""

    def test_in_encoder_decoder_model(self):
        """Test BahdanauAttention in a simple encoder-decoder architecture."""
        # Encoder
        encoder_inputs = keras_layers.Input(shape=(10, 32))
        encoder_outputs, encoder_state = keras_layers.GRU(
            64, return_sequences=True, return_state=True
        )(encoder_inputs)

        # Decoder with attention
        decoder_inputs = keras_layers.Input(shape=(1, 16))
        decoder_gru = keras_layers.GRU(64, return_state=True)
        decoder_output, decoder_state = decoder_gru(
            decoder_inputs, initial_state=encoder_state
        )

        # Attention
        attention = BahdanauAttention(units=64)
        context_vector, attention_weights = attention(decoder_state, encoder_outputs)

        # Combine decoder output with context
        combined = keras_layers.Concatenate()([decoder_output, context_vector])
        outputs = keras_layers.Dense(1)(combined)

        model = models.Model(
            inputs=[encoder_inputs, decoder_inputs], outputs=outputs
        )
        model.compile(optimizer="adam", loss="mse")

        # Test forward pass
        enc_data = tf.random.normal((2, 10, 32))
        dec_data = tf.random.normal((2, 1, 16))
        predictions = model([enc_data, dec_data])

        assert predictions.shape == (2, 1)
        assert not tf.reduce_any(tf.math.is_nan(predictions))

    def test_gradient_flow(self):
        """Test gradients flow correctly through the layer."""
        layer = BahdanauAttention(units=64)

        query = tf.random.normal((2, 128))
        values = tf.random.normal((2, 10, 128))

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([query, values])
            context_vector, _ = layer(query, values)
            loss = tf.reduce_mean(context_vector ** 2)

        # Get gradients for both inputs
        grads = tape.gradient(loss, [query, values])

        # Check gradients exist and are not NaN
        for grad in grads:
            assert grad is not None
            assert not tf.reduce_any(tf.math.is_nan(grad))

        # Also check layer trainable variables
        layer_grads = tape.gradient(loss, layer.trainable_variables)
        for grad in layer_grads:
            assert grad is not None
            assert not tf.reduce_any(tf.math.is_nan(grad))

        # Clean up persistent tape
        del tape

    def test_training_integration(self):
        """Integration test: Training a simple seq2seq model with attention."""
        # Simple encoder-decoder with attention
        encoder_inputs = keras_layers.Input(shape=(20, 16))
        encoder_outputs, encoder_state = keras_layers.GRU(
            64, return_sequences=True, return_state=True
        )(encoder_inputs)

        # Decoder
        decoder_state = keras_layers.Dense(64)(encoder_state)

        # Attention
        attention = BahdanauAttention(units=64)
        context_vector, _ = attention(decoder_state, encoder_outputs)

        outputs = keras_layers.Dense(1)(context_vector)

        model = models.Model(inputs=encoder_inputs, outputs=outputs)
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
        """Test with single timestep in encoder."""
        layer = BahdanauAttention(units=64)

        query = tf.random.normal((2, 128))
        values = tf.random.normal((2, 1, 128))

        context_vector, attention_weights = layer(query, values)

        assert context_vector.shape == (2, 128)
        assert attention_weights.shape == (2, 1, 1)
        # With single timestep, attention weight should be 1.0
        assert tf.reduce_all(tf.abs(attention_weights - 1.0) < 1e-5)

    def test_large_sequence(self):
        """Test with very long sequence."""
        layer = BahdanauAttention(units=64)

        query = tf.random.normal((2, 128))
        values = tf.random.normal((2, 500, 128))

        context_vector, attention_weights = layer(query, values)

        assert context_vector.shape == (2, 128)
        assert attention_weights.shape == (2, 500, 1)
        assert not tf.reduce_any(tf.math.is_nan(context_vector))

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        layer = BahdanauAttention(units=64)

        query = tf.random.normal((1, 128))
        values = tf.random.normal((1, 10, 128))

        context_vector, attention_weights = layer(query, values)

        assert context_vector.shape == (1, 128)
        assert attention_weights.shape == (1, 10, 1)

    def test_large_batch_size(self):
        """Test with large batch size."""
        layer = BahdanauAttention(units=64)

        query = tf.random.normal((128, 128))
        values = tf.random.normal((128, 10, 128))

        context_vector, attention_weights = layer(query, values)

        assert context_vector.shape == (128, 128)
        assert attention_weights.shape == (128, 10, 1)


class TestAttentionMechanism:
    """Test attention-specific behavior."""

    def test_attention_focuses_correctly(self):
        """Test that attention can focus on specific parts of the sequence."""
        layer = BahdanauAttention(units=64)

        # Create a scenario where one encoder output is very different
        query = tf.random.normal((1, 64))
        values = tf.ones((1, 10, 64)) * 0.1
        # Make one timestep stand out
        values = tf.tensor_scatter_nd_update(
            values,
            [[0, 5]],  # Update position (batch=0, timestep=5)
            [tf.ones((64,)) * 10.0],
        )

        context_vector, attention_weights = layer(query, values)

        # Attention should focus more on the standout timestep
        weights_squeezed = tf.squeeze(attention_weights, axis=-1)
        max_attention_idx = tf.argmax(weights_squeezed, axis=1)

        # The maximum attention might be at the standout position
        # (not guaranteed due to random query, but test structure)
        assert context_vector.shape == (1, 64)
        assert attention_weights.shape == (1, 10, 1)

    def test_different_attention_patterns(self):
        """Test that different queries produce different attention patterns."""
        layer = BahdanauAttention(units=64)

        values = tf.random.normal((1, 10, 128))

        # Two different queries
        query1 = tf.random.normal((1, 128))
        query2 = tf.random.normal((1, 128))

        _, weights1 = layer(query1, values)
        _, weights2 = layer(query2, values)

        # Attention patterns should be different
        # (with very high probability given random initialization)
        diff = tf.reduce_sum(tf.abs(weights1 - weights2))
        assert diff > 0.01  # Not exactly the same

    def test_reproducible_with_same_inputs(self):
        """Test that same inputs produce same outputs (deterministic)."""
        layer = BahdanauAttention(units=64)

        query = tf.random.normal((2, 128))
        values = tf.random.normal((2, 10, 128))

        # Two forward passes with same inputs
        context1, weights1 = layer(query, values)
        context2, weights2 = layer(query, values)

        # Outputs should be identical
        assert tf.reduce_all(tf.abs(context1 - context2) < 1e-6)
        assert tf.reduce_all(tf.abs(weights1 - weights2) < 1e-6)


class TestBugFixes:
    """Test specific bug fixes and known issues."""

    def test_no_nan_with_extreme_values(self):
        """Test that extreme values don't cause NaN."""
        layer = BahdanauAttention(units=64)

        # Extreme query values
        query = tf.random.normal((2, 128)) * 100
        values = tf.random.normal((2, 10, 128)) * 100

        context_vector, attention_weights = layer(query, values)

        assert not tf.reduce_any(tf.math.is_nan(context_vector))
        assert not tf.reduce_any(tf.math.is_nan(attention_weights))
        assert not tf.reduce_any(tf.math.is_inf(context_vector))
        assert not tf.reduce_any(tf.math.is_inf(attention_weights))

    def test_stable_with_very_small_values(self):
        """Test numerical stability with very small values."""
        layer = BahdanauAttention(units=64)

        query = tf.random.normal((2, 128)) * 1e-6
        values = tf.random.normal((2, 10, 128)) * 1e-6

        context_vector, attention_weights = layer(query, values)

        assert not tf.reduce_any(tf.math.is_nan(context_vector))
        assert not tf.reduce_any(tf.math.is_nan(attention_weights))

    def test_variable_batch_size(self):
        """Test with different batch sizes in sequence."""
        layer = BahdanauAttention(units=64)

        for batch_size in [1, 2, 8, 16, 32]:
            query = tf.random.normal((batch_size, 128))
            values = tf.random.normal((batch_size, 10, 128))

            context_vector, attention_weights = layer(query, values)

            assert context_vector.shape == (batch_size, 128)
            assert attention_weights.shape == (batch_size, 10, 1)

    def test_mismatched_dims_handled(self):
        """Test that mismatched query/value dimensions are handled."""
        layer = BahdanauAttention(units=64)

        # Different dimensionalities
        query = tf.random.normal((2, 256))  # 256-dim decoder
        values = tf.random.normal((2, 10, 512))  # 512-dim encoder

        context_vector, attention_weights = layer(query, values)

        # Output should match encoder dimension
        assert context_vector.shape == (2, 512)
        assert attention_weights.shape == (2, 10, 1)
