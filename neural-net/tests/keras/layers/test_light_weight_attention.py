import numpy as np
import pytest
import tensorflow as tf
from keras import models, layers as keras_layers

from okmich_quant_neural_net.keras.layers import LightweightAttention


class TestLightweightAttentionInitialization:
    """Test initialization and parameter validation."""

    def test_default_initialization(self):
        """Test default parameters."""
        layer = LightweightAttention()
        assert layer.attn_type == "bahdanau"
        assert layer.heads == 4
        assert layer.return_attention_scores == True
        assert layer.supports_masking == True

    def test_custom_initialization(self):
        """Test custom parameters."""
        layer = LightweightAttention(
            attn_type="dot", heads=8, return_attention_scores=False
        )
        assert layer.attn_type == "dot"
        assert layer.heads == 8
        assert layer.return_attention_scores == False


class TestLightweightAttentionBuild:
    """Test layer building and weight initialization."""

    def test_build_bahdanau(self):
        """Test Bahdanau attention builds correctly."""
        layer = LightweightAttention(attn_type="bahdanau")
        layer.build((None, 10, 64))

        assert layer.units == 64
        assert layer.W is not None
        assert layer.U is not None
        assert layer.V is not None

    def test_build_dot_product(self):
        """Test dot-product attention builds correctly."""
        layer = LightweightAttention(attn_type="dot", heads=4)
        layer.build((None, 10, 64))

        assert layer.units == 64
        assert layer.head_dim == 16
        assert layer.Q_proj is not None
        assert layer.K_proj is not None
        assert layer.V_proj is not None

    def test_heads_not_divisible(self):
        """Test error when units not divisible by heads."""
        layer = LightweightAttention(attn_type="dot", heads=5)
        with pytest.raises(ValueError, match="must be divisible by heads"):
            layer.build((None, 10, 64))


class TestLightweightAttentionForward:
    """Test forward pass functionality."""

    def test_bahdanau_forward_basic(self):
        """Test basic Bahdanau forward pass."""
        layer = LightweightAttention(attn_type="bahdanau", return_attention_scores=True)
        inputs = tf.random.normal((2, 10, 64))

        context, scores = layer(inputs)

        assert context.shape == (2, 64)
        assert scores.shape == (2, 10)

    def test_dot_product_forward_basic(self):
        """Test basic dot-product forward pass."""
        layer = LightweightAttention(
            attn_type="dot", heads=4, return_attention_scores=True
        )
        inputs = tf.random.normal((2, 10, 64))

        context, scores = layer(inputs)

        assert context.shape == (2, 64)
        assert scores.shape == (2, 4, 10, 10)

    def test_forward_without_scores(self):
        """Test forward pass without returning attention scores."""
        layer = LightweightAttention(return_attention_scores=False)
        inputs = tf.random.normal((2, 10, 64))

        context = layer(inputs)

        assert context.shape == (2, 64)
        assert isinstance(context, tf.Tensor)

    def test_forward_with_mask(self):
        """Test forward pass with padding mask."""
        layer = LightweightAttention(attn_type="dot", heads=4)
        inputs = tf.random.normal((2, 10, 64))
        mask = tf.constant([[True] * 7 + [False] * 3, [True] * 5 + [False] * 5])

        context, _ = layer(inputs, mask=mask)

        assert context.shape == (2, 64)
        assert not tf.reduce_any(tf.math.is_nan(context))


class TestSerialization:
    """Test layer serialization and deserialization."""

    def test_get_config(self):
        """Test get_config returns all parameters."""
        layer = LightweightAttention(
            attn_type="dot", heads=8, return_attention_scores=False
        )

        config = layer.get_config()

        assert config["attn_type"] == "dot"
        assert config["heads"] == 8
        assert config["return_attention_scores"] == False

    def test_from_config(self):
        """Test layer can be recreated from config."""
        layer = LightweightAttention(attn_type="dot", heads=8)

        config = layer.get_config()
        new_layer = LightweightAttention.from_config(config)

        assert new_layer.attn_type == layer.attn_type
        assert new_layer.heads == layer.heads

    def test_compute_output_shape(self):
        """Test compute_output_shape returns correct shapes."""
        # Bahdanau with scores
        layer = LightweightAttention(attn_type="bahdanau", return_attention_scores=True)
        layer.build((None, 10, 64))
        out_shape = layer.compute_output_shape((2, 10, 64))
        assert out_shape[0] == (2, 64)
        assert out_shape[1] == (2, 10)

        # Dot with scores
        layer = LightweightAttention(
            attn_type="dot", heads=4, return_attention_scores=True
        )
        layer.build((None, 10, 64))
        out_shape = layer.compute_output_shape((2, 10, 64))
        assert out_shape[0] == (2, 64)
        assert out_shape[1] == (2, 4, 10, 10)

        # Without scores
        layer = LightweightAttention(return_attention_scores=False)
        layer.build((None, 10, 64))
        out_shape = layer.compute_output_shape((2, 10, 64))
        assert out_shape == (2, 64)


class TestIntegration:
    """Test integration scenarios."""

    def test_in_keras_model(self):
        """Test layer works in a Keras model."""
        inputs = keras_layers.Input(shape=(10, 64))
        attention = LightweightAttention(
            attn_type="dot", heads=4, return_attention_scores=False
        )
        context = attention(inputs)
        outputs = keras_layers.Dense(1)(context)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")

        # Test forward pass
        x = tf.random.normal((2, 10, 64))
        y = model(x)

        assert y.shape == (2, 1)

    def test_gradient_flow(self):
        """Test gradients flow correctly through the layer."""
        layer = LightweightAttention(
            attn_type="dot", heads=4, return_attention_scores=False
        )
        inputs = tf.random.normal((2, 10, 64))

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            context = layer(inputs)
            loss = tf.reduce_mean(context ** 2)

        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check all gradients are not None and not NaN
        for grad in gradients:
            assert grad is not None
            assert not tf.reduce_any(tf.math.is_nan(grad))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_timestep(self):
        """Test with single timestep sequence."""
        layer = LightweightAttention(return_attention_scores=False)
        inputs = tf.random.normal((2, 1, 64))

        context = layer(inputs)

        assert context.shape == (2, 64)

    def test_large_batch_size(self):
        """Test with large batch size."""
        layer = LightweightAttention(return_attention_scores=False)
        inputs = tf.random.normal((128, 10, 64))

        context = layer(inputs)

        assert context.shape == (128, 64)


class TestBugFixes:
    """Test specific bug fixes."""

    def test_mask_propagation(self):
        """
        BUG FIX #1: Mask propagation (supports_masking = True).

        Verify that masks propagate correctly through the layer in a Keras model.
        """
        layer = LightweightAttention(attn_type="dot", heads=4)

        # Check that supports_masking is True
        assert layer.supports_masking == True

        # Create a model with masking layer
        inputs = keras_layers.Input(shape=(None, 64))
        masked = keras_layers.Masking(mask_value=0.0)(inputs)
        attention = LightweightAttention(attn_type="dot", return_attention_scores=False)
        output = attention(masked)

        model = models.Model(inputs=inputs, outputs=output)

        # Create data with padding
        batch_data = np.random.randn(2, 10, 64).astype(np.float32)
        batch_data[0, 7:, :] = 0
        batch_data[1, 5:, :] = 0

        result = model(batch_data)
        assert result.shape == (2, 64)

    def test_output_shape_plain_tuples(self):
        """
        BUG FIX #2: Output shape returns plain tuples, not tf.TensorShape.

        Keras 3 expects plain Python tuples from compute_output_shape.
        """
        layer = LightweightAttention(
            attn_type="dot", heads=4, return_attention_scores=True
        )
        layer.build((None, 10, 64))

        output_shape = layer.compute_output_shape((32, 10, 64))

        # Should be tuple of tuples
        assert isinstance(output_shape, tuple)
        assert isinstance(output_shape[0], tuple)
        assert isinstance(output_shape[1], tuple)
        assert output_shape[0] == (32, 64)
        assert output_shape[1] == (32, 4, 10, 10)

        # Test without scores
        layer2 = LightweightAttention(return_attention_scores=False)
        layer2.build((None, 10, 64))
        output_shape2 = layer2.compute_output_shape((32, 10, 64))

        assert isinstance(output_shape2, tuple)
        assert output_shape2 == (32, 64)

    def test_bahdanau_with_variable_length(self):
        """Test Bahdanau works with different sequence lengths."""
        layer = LightweightAttention(
            attn_type="bahdanau", return_attention_scores=False
        )
        layer.build((None, 100, 64))

        # Test different lengths
        for seq_len in [20, 50, 100, 150]:
            inputs = tf.random.normal((2, seq_len, 64))
            output = layer(inputs)

            assert output.shape == (2, 64)
            assert not tf.reduce_any(tf.math.is_nan(output))

    def test_dot_with_variable_length(self):
        """Test dot-product works with different sequence lengths."""
        layer = LightweightAttention(
            attn_type="dot", heads=4, return_attention_scores=True
        )
        layer.build((None, 100, 64))

        # Test different lengths
        for seq_len in [10, 50, 100, 150]:
            inputs = tf.random.normal((2, seq_len, 64))
            context, attention_weights = layer(inputs)

            assert context.shape == (2, 64)
            assert attention_weights.shape == (2, 4, seq_len, seq_len)
            assert not tf.reduce_any(tf.math.is_nan(context))

    def test_training_integration(self):
        """Integration test: Training a simple model."""
        inputs = keras_layers.Input(shape=(None, 32))
        masked = keras_layers.Masking(mask_value=0.0)(inputs)
        x = keras_layers.Dense(64, activation="relu")(masked)

        attention = LightweightAttention(
            attn_type="dot", heads=4, return_attention_scores=False
        )
        context = attention(x)
        outputs = keras_layers.Dense(1)(context)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")

        # Test with variable lengths
        for seq_len in [20, 50]:
            batch_data = np.random.randn(8, seq_len, 32).astype(np.float32)

            # Add some padding
            for i in range(8):
                pad_len = np.random.randint(0, seq_len // 2)
                if pad_len > 0:
                    batch_data[i, -pad_len:, :] = 0.0

            y = np.random.randn(8, 1).astype(np.float32)

            loss = model.train_on_batch(batch_data, y)
            assert not np.isnan(loss)

            predictions = model.predict(batch_data, verbose=0)
            assert predictions.shape == (8, 1)
            assert not np.any(np.isnan(predictions))
