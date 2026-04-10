"""
Encoder-Decoder GRU with Attention (Seq2Seq Variant)
=====================================================

Architecture with encoder-decoder structure and Bahdanau attention mechanism.
Encoder compresses sequence context, attention weights relevant timesteps,
decoder focuses on classification from compressed representation.

Suitable for: liquidity forecasting, order book imbalance prediction, market microstructure


References:
"Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction" - Qin et al. (2017) https://arxiv.org/abs/1704.02971
Trading application: "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"** - Applied to stock markets - Liu et al. (2019)
https://dl.acm.org/doi/10.1145/3366424.3383297
"""

import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers, models

# Import Bahdanau attention layer from centralized location
from okmich_quant_neural_net.keras.layers.bahdanau_attention import BahdanauAttention
# Import task type and common utilities
from okmich_quant_neural_net.keras.paper2keras.common import (
    TaskType,
    create_output_layer_and_loss,
    get_optimizer,
    get_model_name,
)


def create_encoder_decoder_gru_attention(input_shape, num_classes, task_type=TaskType.CLASSIFICATION,
                                         encoder_units_1=128, encoder_units_2=64, encoder_dropout=0.3,
                                         attention_units=64, decoder_units=64, decoder_dropout=0.2, dense_units_1=64,
                                         dense_units_2=32, dense_dropout=0.2, learning_rate=0.001, l2_reg=0.0001):
    """
    Create an Encoder-Decoder GRU model with Bahdanau Attention.

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (sequence_length, n_features)
    num_classes : int
        Number of output classes for classification
    encoder_units_1 : int, default=128
        Number of units in first encoder GRU layer
    encoder_units_2 : int, default=64
        Number of units in second encoder GRU layer
    encoder_dropout : float, default=0.3
        Dropout rate after encoder layers
    attention_units : int, default=64
        Number of units in attention mechanism
    decoder_units : int, default=64
        Number of units in decoder GRU layer
    decoder_dropout : float, default=0.2
        Dropout rate after decoder layer
    dense_units_1 : int, default=64
        Number of units in first dense layer
    dense_units_2 : int, default=32
        Number of units in second dense layer
    dense_dropout : float, default=0.2
        Dropout rate after dense layers
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer
    l2_reg : float, default=0.0001
        L2 regularization factor

    Returns
    -------
    keras.Model
        Compiled Keras model
    """

    # Input layer
    inputs = layers.Input(shape=input_shape, name="input")

    # ============================================================
    # ENCODER
    # ============================================================

    # First encoder GRU layer (returns sequences for attention)
    encoder_gru1 = layers.GRU(
        encoder_units_1,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="encoder_gru_1",
    )(inputs)
    encoder_gru1 = layers.Dropout(encoder_dropout, name="encoder_dropout_1")(
        encoder_gru1
    )

    # Second encoder GRU layer (returns sequences and final state)
    encoder_gru2 = layers.GRU(
        encoder_units_2,
        return_sequences=True,
        return_state=True,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="encoder_gru_2",
    )
    encoder_outputs, encoder_state = encoder_gru2(encoder_gru1)
    encoder_outputs = layers.Dropout(encoder_dropout, name="encoder_dropout_2")(
        encoder_outputs
    )

    # ============================================================
    # ATTENTION MECHANISM
    # ============================================================

    # Bahdanau attention computes context vector from encoder outputs
    attention_layer = BahdanauAttention(attention_units, name="bahdanau_attention")
    context_vector, attention_weights = attention_layer(encoder_state, encoder_outputs)

    # ============================================================
    # DECODER
    # ============================================================

    # Expand context vector for decoder GRU input
    # Shape: (batch_size, 1, attention_context_dim)
    decoder_input = layers.RepeatVector(1, name="repeat_context")(context_vector)

    # Decoder GRU layer
    decoder_gru = layers.GRU(
        decoder_units,
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="decoder_gru",
    )(decoder_input, initial_state=encoder_state)
    decoder_gru = layers.Dropout(decoder_dropout, name="decoder_dropout")(decoder_gru)

    # Combine decoder output with context vector
    combined = layers.Concatenate(name="concat_decoder_context")(
        [decoder_gru, context_vector]
    )

    # ============================================================
    # CLASSIFICATION HEAD
    # ============================================================

    # First dense layer
    dense1 = layers.Dense(
        dense_units_1,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="dense_1",
    )(combined)
    dense1 = layers.Dropout(dense_dropout, name="dense_dropout_1")(dense1)

    # Second dense layer
    dense2 = layers.Dense(
        dense_units_2,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="dense_2",
    )(dense1)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        dense2, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("encoder_decoder_gru_attention", task_type)
    model = models.Model(
        inputs=inputs, outputs=outputs, name=model_name
    )

    # Compile with task-specific loss/metrics
    opt = get_optimizer("adam", learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


def create_attention_visualization_model(model):
    """
    Create a model that outputs both predictions and attention weights.
    Useful for interpreting which timesteps the model focuses on.

    Parameters
    ----------
    model : keras.Model
        Trained encoder-decoder model with attention

    Returns
    -------
    keras.Model
        Model that outputs (predictions, attention_weights)
    """
    # Get attention layer
    attention_layer = model.get_layer("bahdanau_attention")

    # Find the layer that feeds into attention (encoder outputs after dropout)
    encoder_outputs_layer = model.get_layer("encoder_dropout_2").output
    encoder_state_layer = model.get_layer("encoder_gru_2").output[1]

    # Create model that outputs attention weights
    # Note: This requires rebuilding the attention computation
    # For simplicity, we'll create a wrapper that captures attention during forward pass

    class AttentionModel(keras.Model):
        def __init__(self, base_model, **kwargs):
            super(AttentionModel, self).__init__(**kwargs)
            self.base_model = base_model
            self.attention_layer = base_model.get_layer("bahdanau_attention")

        def call(self, inputs):
            # Get encoder outputs
            encoder_gru1 = self.base_model.get_layer("encoder_gru_1")(inputs)
            encoder_dropout1 = self.base_model.get_layer("encoder_dropout_1")(
                encoder_gru1
            )

            encoder_gru2_layer = self.base_model.get_layer("encoder_gru_2")
            encoder_outputs, encoder_state = encoder_gru2_layer(encoder_dropout1)
            encoder_outputs = self.base_model.get_layer("encoder_dropout_2")(
                encoder_outputs
            )

            # Get attention weights
            context_vector, attention_weights = self.attention_layer(
                encoder_state, encoder_outputs
            )

            # Get final predictions
            predictions = self.base_model(inputs)

            return predictions, attention_weights

    return AttentionModel(model)


def create_tunable_encoder_decoder_gru_attention(input_shape, num_classes, task_type=TaskType.CLASSIFICATION):
    """
    Create a tunable version of the Encoder-Decoder GRU with Attention for hyperparameter optimization.

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (sequence_length, n_features)
    num_classes : int
        Number of output classes

    Returns
    -------
    function
        Model builder function for keras_tuner

    Example
    -------
    >>> tuner = kt.BayesianOptimization(
    ...     create_tunable_encoder_decoder_gru_attention(input_shape=(64, 10), num_classes=3),
    ...     objective='val_loss',
    ...     max_trials=20,
    ...     directory='tuner_results',
    ...     project_name='encoder_decoder_gru_attention'
    ... )
    >>> tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
    """

    def build_model(hp):
        # Hyperparameter search space
        encoder_units_1 = hp.Choice("encoder_units_1", values=[64, 128, 256])
        encoder_units_2 = hp.Choice("encoder_units_2", values=[32, 64, 128])
        encoder_dropout = hp.Float(
            "encoder_dropout", min_value=0.2, max_value=0.5, step=0.1
        )

        attention_units = hp.Choice("attention_units", values=[32, 64, 128])

        decoder_units = hp.Choice("decoder_units", values=[32, 64, 128])
        decoder_dropout = hp.Float(
            "decoder_dropout", min_value=0.1, max_value=0.4, step=0.1
        )

        dense_units_1 = hp.Int("dense_units_1", min_value=32, max_value=128, step=32)
        dense_units_2 = hp.Int("dense_units_2", min_value=16, max_value=64, step=16)
        dense_dropout = hp.Float(
            "dense_dropout", min_value=0.1, max_value=0.4, step=0.1
        )

        learning_rate = hp.Choice("learning_rate", values=[1e-4, 5e-4, 1e-3, 5e-3])
        l2_reg = hp.Choice("l2_reg", values=[1e-5, 1e-4, 1e-3])

        return create_encoder_decoder_gru_attention(
            input_shape=input_shape,
            num_classes=num_classes,
            task_type=task_type,
            encoder_units_1=encoder_units_1,
            encoder_units_2=encoder_units_2,
            encoder_dropout=encoder_dropout,
            attention_units=attention_units,
            decoder_units=decoder_units,
            decoder_dropout=decoder_dropout,
            dense_units_1=dense_units_1,
            dense_units_2=dense_units_2,
            dense_dropout=dense_dropout,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
        )

    return build_model


"""
HINTS & NOTES:
==============================================

1. ENCODER-DECODER ARCHITECTURE BENEFITS:
   - ENCODER: Compresses variable-length sequences into fixed representation
   - ATTENTION: Allows decoder to focus on relevant historical timesteps
   - DECODER: Transforms attended context into classification decision
   - Unlike vanilla seq2seq: This is adapted for classification, not generation

2. ATTENTION MECHANISM (BAHDANAU):
   - Computes alignment scores between decoder state and all encoder outputs
   - Produces attention weights via softmax (sums to 1.0)
   - Context vector = weighted sum of encoder outputs
   - Interpretable: Shows which timesteps model considers important

3. INPUT DATA PREPARATION:
   - Normalize/standardize features before training
   - Shape: (n_samples, sequence_length, n_features)
   - For order book data: Include bid/ask levels, volumes, imbalances
   - For liquidity: Include spreads, depth, trade intensity

4. SEQUENCE LENGTH CONSIDERATIONS:
   - Minimum: 32 timesteps (too short limits attention benefits)
   - Recommended: 50-100 timesteps
   - Maximum: 200+ (slower training, more memory)
   - Attention helps handle long sequences better than vanilla RNNs

5. ATTENTION UNITS:
   - Start with attention_units = encoder_units_2 (e.g., 64)
   - Increase if model underfits (128)
   - Decrease if overfitting or memory limited (32)
   - More units = more expressive attention, but more parameters

6. ENCODER-DECODER UNIT RATIOS:
   - encoder_units_1 > encoder_units_2 (e.g., 128 → 64)
   - decoder_units ≈ encoder_units_2 (e.g., 64)
   - This creates information bottleneck, forcing compression

7. CLASS IMBALANCE HANDLING:
   - Use class_weight in model.fit() for imbalanced classes
   - Example: class_weight={0: 1.0, 1: 2.5, 2: 1.5}
   - Consider focal loss for extreme imbalance
   - Monitor macro F1, not just accuracy

8. TRAINING TIPS:
   - Batch size: 32-64 recommended
   - Learning rate: 0.001 (reduce to 0.0005 if loss oscillates)
   - Gradient clipping: clipnorm=1.0 (important for seq2seq models!)
   - Use early stopping with patience=10-15
   - Attention models may need more epochs to converge (30-50)

9. HYPERPARAMETER TUNING PRIORITIES:
   High impact:
     - encoder_units_1 (64-256)
     - encoder_units_2 (32-128)
     - attention_units (32-128)
     - learning_rate (1e-4 to 5e-3)

   Medium impact:
     - decoder_units (32-128)
     - encoder_dropout (0.2-0.5)
     - dense_units_1

   Low impact:
     - decoder_dropout
     - dense_units_2

10. ATTENTION VISUALIZATION & INTERPRETABILITY:
    - Use create_attention_visualization_model() to extract attention weights
    - Visualize attention heatmap over time series
    - High attention = model focuses on that timestep for prediction
    - Useful for debugging and building trader confidence
    - Example: "Model predicts buy signal, focusing on t-15 (large volume spike)"

11. ARCHITECTURE VARIANTS:
    - Add Bidirectional GRU in encoder:
        encoder = layers.Bidirectional(GRU(...))

    - Use multiple attention heads (multi-head attention):
        Compute attention with different projection matrices, concatenate

    - Add self-attention in encoder before decoder attention:
        Layer that attends to its own outputs (like Transformer)

    - Use Luong attention instead of Bahdanau:
        Different scoring function: dot product, general, concat

    - Add skip connections from encoder to decoder:
        Concatenate encoder state with decoder output

12. LIQUIDITY FORECASTING USE CASE:
    - Input features: Bid/ask spreads, order book depth, trade intensity
    - Sequence: Last 50-100 market snapshots (1-5 second intervals)
    - Classes: High/Medium/Low liquidity regimes
    - Attention reveals: Which historical liquidity events matter most
    - Example: Sudden liquidity drop 30 seconds ago gets high attention

13. ORDER BOOK IMBALANCE PREDICTION:
    - Input features: Bid/ask volumes at each level, mid-price changes
    - Sequence: Last 60 order book states (100ms intervals)
    - Classes: Buy pressure/Neutral/Sell pressure
    - Attention reveals: Critical order flow changes that signal imbalance
    - Example: Large order at t-20 gets high attention for buy pressure prediction

14. COMPARISON WITH OTHER ARCHITECTURES:
    - vs GRU-only: Attention provides interpretability, handles long sequences better
    - vs LSTM: GRU has fewer parameters, trains faster, similar performance
    - vs Transformer: Seq2seq is more parameter-efficient for shorter sequences
    - vs CNN: Attention captures global dependencies better than local kernels

15. MEMORY OPTIMIZATION:
    - Use mixed precision training: 2x speedup, half memory
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)

    - Reduce encoder_units_1 from 128 to 64-96
    - Use smaller batch_size (16-32)
    - Gradient checkpointing for very long sequences

16. PRODUCTION DEPLOYMENT:
    - Model is moderately fast: ~20-100ms inference on CPU
    - Attention computation adds overhead vs vanilla GRU
    - Can save attention weights to database for post-hoc analysis
    - Deploy with TensorFlow Serving or ONNX Runtime

17. DEBUGGING TIPS:
    - If attention is uniform (all weights ~equal):
        * Model not learning useful attention
        * Try increasing attention_units
        * Check if features are normalized
        * Increase model capacity

    - If attention focuses on only 1-2 timesteps:
        * Model may be overfitting to spurious patterns
        * Increase encoder_dropout
        * Add more training data

    - If loss plateaus early:
        * Increase learning rate initially (0.005)
        * Use learning rate warmup
        * Check for data leakage

18. ADVANCED TRAINING TECHNIQUES:
    - Teacher forcing (not needed for classification variant)
    - Scheduled sampling (not applicable here)
    - Attention regularization: Add penalty for overly peaked attention
        L_attention = -entropy(attention_weights)
    - Curriculum learning: Train on shorter sequences first, then longer

19. COMPARISON: ENCODER-DECODER VS STANDARD GRU:
    Standard GRU:
    ✓ Simpler, faster training
    ✓ Fewer parameters
    ✗ No interpretability
    ✗ Worse on long sequences

    Encoder-Decoder with Attention:
    ✓ Interpretable attention weights
    ✓ Better long-range dependencies
    ✓ More expressive
    ✗ More parameters, slower training
    ✗ Harder to tune

20. WHEN TO USE THIS ARCHITECTURE:
    ✓ Need interpretability (which timesteps matter?)
    ✓ Long sequences (50-200 timesteps)
    ✓ Variable-importance patterns (some timesteps more critical)
    ✓ Liquidity, market microstructure, order flow prediction

    ✗ Very short sequences (<20 timesteps) - use simpler models
    ✗ Need real-time inference (<10ms) - use CNN or simple GRU
    ✗ Limited training data (<1000 samples) - too complex, will overfit
    """
