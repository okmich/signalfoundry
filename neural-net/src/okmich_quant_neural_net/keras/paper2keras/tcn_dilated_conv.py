"""
Temporal Convolutional Network (TCN) with Dilated Convolutions - Model Factory
===============================================================================

Architecture:
-------------
Input (32-100 timesteps, n_features)
    ↓
TCN Block (dilations=[1, 2, 4, 8, 16, 32])
    - Dilated convolutions with residual connections
    - Skip connections across blocks
    - Causal padding (no look-ahead bias)
    ↓
GlobalAveragePooling1D or last timestep
    ↓
Dense (64, ReLU) → Dropout (0.2)
    ↓
Dense (32, ReLU) → Dropout (0.2)
    ↓
Output (softmax for multi-class)

Key Features:
-------------
- Parallel processing (much faster than RNN/LSTM)
- Dilated convolutions capture multi-scale temporal patterns
- Large receptive field with fewer parameters
- Causal padding ensures no look-ahead bias
- Suitable for: price pattern recognition, support/resistance detection, regime classification

Receptive Field:
----------------
RF = 1 + 2 * (kernel_size - 1) * nb_stacks * sum(dilations)

For dilations=[1, 2, 4, 8, 16, 32], kernel_size=3, nb_stacks=1:
    RF = 127 timesteps (10+ hours of 5-min bars)

Usage:
------
1. Simple version (fixed hyperparameters):
   model = build_tcn(
       sequence_length=48,
       num_features=20,
       num_classes=3
   )

2. Tunable version (with KerasTuner):
   def model_builder(hp):
       return build_tcn_tunable(
           hp=hp,
           num_features=20,
           num_classes=3
       )

   tuner = keras_tuner.BayesianOptimization(
       model_builder,
       objective='val_loss',
       max_trials=20
   )


Reference:
- "Temporal Convolutional Networks for the Advance Prediction of ENSO" - Ham et al. (2019) - Applied to financial markets
https://www.nature.com/articles/s41586-019-1559-7
- Financial application: "Deep Learning for Financial Time Series Forecasting" - Borovykh et al. (2017)
https://arxiv.org/abs/1701.01887
"""

from keras import layers, models, optimizers, losses, metrics

from okmich_quant_neural_net.keras.metrics import (
    CausalRegimeAccuracy, RegimeTransitionRecall, RegimeTransitionPrecision,
)
from okmich_quant_neural_net.keras.paper2keras.common import (
    TaskType,
    create_output_layer_and_loss,
    get_optimizer,
    get_model_name,
)
# Import the existing TCN layer from the main codebase
from ..layers.tcn import TCN


# ============================================================================
# SIMPLE VERSION (Fixed Hyperparameters)
# ============================================================================


def build_tcn(sequence_length=48, num_features=20, num_classes=3, task_type=TaskType.CLASSIFICATION,
              nb_filters=64, kernel_size=3, nb_stacks=1, dilations=(1, 2, 4, 8, 16, 32), dropout_rate=0.2,
              use_skip_connections=True, use_batch_norm=False, use_layer_norm=False,
              dense1_units=64, dense2_units=32, dense_dropout=0.2,
              learning_rate=0.001, optimizer_name="adam", pooling="global"):
    """
    Build Temporal Convolutional Network (TCN) model (fixed hyperparameters).

    Args:
        sequence_length: Number of timesteps in input sequences
        num_features: Number of features per timestep
        num_classes: Number of output classes
        nb_filters: Number of convolutional filters in TCN (default: 64)
        kernel_size: Size of convolutional kernel (default: 3)
        nb_stacks: Number of stacks of residual blocks (default: 1)
        dilations: Tuple of dilation rates (default: (1, 2, 4, 8, 16, 32))
        dropout_rate: Dropout rate in TCN layers (default: 0.2)
        use_skip_connections: Use skip connections in TCN (default: True)
        use_batch_norm: Use batch normalization (default: False)
        use_layer_norm: Use layer normalization (default: False)
        dense1_units: Units in first dense layer (default: 64)
        dense2_units: Units in second dense layer (default: 32)
        dense_dropout: Dropout rate after dense layers (default: 0.2)
        learning_rate: Learning rate for optimizer (default: 0.001)
        optimizer_name: Optimizer to use ('adam', 'adamw', 'rmsprop') (default: 'adam')
        pooling: Temporal pooling method ('global' or 'last') (default: 'global')

    Returns:
        Compiled Keras model

    Example:
        >>> model = build_tcn(
        ...     sequence_length=48,
        ...     num_features=20,
        ...     num_classes=3,
        ...     dilations=(1, 2, 4, 8)
        ... )
        >>> model.summary()
    """

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # TCN layer
    x = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,
        padding="causal",  # No look-ahead bias
        use_skip_connections=use_skip_connections,
        dropout_rate=dropout_rate,
        return_sequences=(
                pooling == "global"
        ),  # True for global pooling, False for last timestep
        activation="relu",
        kernel_initializer="he_normal",
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm,
        name="tcn",
    )(inputs)

    # Pooling layer
    if pooling == "global":
        # Global average pooling across time dimension
        x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    # else: TCN already returned last timestep (return_sequences=False)

    # Dense layers
    x = layers.Dense(dense1_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dense_dropout, name="dropout_dense1")(x)

    x = layers.Dense(dense2_units, activation="relu", name="dense2")(x)
    x = layers.Dropout(dense_dropout, name="dropout_dense2")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("TCN_Classifier", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile model
    opt = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    # Print receptive field info
    receptive_field = 1 + 2 * (kernel_size - 1) * nb_stacks * sum(dilations)
    print(f"\nTCN Receptive Field: {receptive_field} timesteps")
    if sequence_length < receptive_field:
        print(
            f"⚠️  Warning: Sequence length ({sequence_length}) < Receptive field ({receptive_field})"
        )
        print(f"   Consider increasing sequence_length or reducing dilations")

    return model


# ============================================================================
# TUNABLE VERSION (With KerasTuner)
# ============================================================================


def build_tcn_tunable(hp, num_features, num_classes, task_type=TaskType.CLASSIFICATION,
                      sequence_length=None, max_sequence_length=100):
    """
    Build TCN model with hyperparameter tuning.

    This version allows KerasTuner to optimize:
    - Sequence length (if not provided as fixed parameter)
    - Number of filters
    - Kernel size
    - Dilation rates
    - Number of stacks
    - Dropout rates
    - Dense layer units
    - Pooling method
    - Learning rate
    - Optimizer choice

    Args:
        hp: KerasTuner HyperParameters object
        num_features: Number of features per timestep (fixed)
        num_classes: Number of output classes (fixed)
        sequence_length: If None, tunes sequence_length (32 to max_sequence_length).
                        If provided, uses this fixed value (for pre-created sequences).
        max_sequence_length: Maximum sequence length (used when sequence_length=None)

    Returns:
        Compiled Keras model with tunable hyperparameters

    Example:
        >>> import keras_tuner
        >>> # Example 1: Tune sequence_length
        >>> def model_builder(hp):
        ...     return build_tcn_tunable(
        ...         hp=hp,
        ...         num_features=20,
        ...         num_classes=3
        ...     )
        >>> # Example 2: Fixed sequence_length (for pre-created sequences)
        >>> def model_builder(hp):
        ...     return build_tcn_tunable(
        ...         hp=hp,
        ...         num_features=20,
        ...         num_classes=3,
        ...         sequence_length=48
        ...     )
        >>> tuner = keras_tuner.BayesianOptimization(
        ...     model_builder,
        ...     objective='val_accuracy',
        ...     max_trials=20,
        ...     directory='tuning_results',
        ...     project_name='tcn_classifier'
        ... )
    """

    # Tunable hyperparameters
    if sequence_length is None:
        # Tune sequence_length if not provided
        sequence_length = hp.Int(
            "sequence_length", min_value=32, max_value=max_sequence_length, step=16
        )
    # else: use the provided fixed sequence_length
    nb_filters = hp.Choice("nb_filters", values=[32, 64, 128])
    kernel_size = hp.Choice("kernel_size", values=[2, 3, 5])
    nb_stacks = hp.Int("nb_stacks", min_value=1, max_value=2, step=1)

    # Dilation strategy
    dilation_strategy = hp.Choice(
        "dilation_strategy", values=["small", "medium", "large"]
    )
    if dilation_strategy == "small":
        dilations = (1, 2, 4, 8)
    elif dilation_strategy == "medium":
        dilations = (1, 2, 4, 8, 16)
    else:  # large
        dilations = (1, 2, 4, 8, 16, 32)

    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.4, step=0.1)
    use_skip_connections = hp.Boolean("use_skip_connections", default=True)

    # Normalization
    normalization = hp.Choice("normalization", values=["none", "batch", "layer"])
    use_batch_norm = normalization == "batch"
    use_layer_norm = normalization == "layer"

    dense1_units = hp.Choice("dense1_units", values=[32, 64, 128])
    dense2_units = hp.Choice("dense2_units", values=[16, 32, 64])
    dense_dropout = hp.Float("dense_dropout", min_value=0.1, max_value=0.3, step=0.1)

    pooling = hp.Choice("pooling", values=["global", "last"])

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # TCN layer
    x = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,
        padding="causal",
        use_skip_connections=use_skip_connections,
        dropout_rate=dropout_rate,
        return_sequences=(pooling == "global"),
        activation="relu",
        kernel_initializer="he_normal",
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_layer_norm,
        name="tcn",
    )(inputs)

    # Pooling layer
    if pooling == "global":
        x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # Dense layers
    x = layers.Dense(dense1_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dense_dropout, name="dropout_dense1")(x)

    x = layers.Dense(dense2_units, activation="relu", name="dense2")(x)
    x = layers.Dropout(dense_dropout, name="dropout_dense2")(x)

    # Output layer
    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
        loss = losses.BinaryCrossentropy()
        output_metrics = [
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc"),
        ]
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
        loss = losses.SparseCategoricalCrossentropy()
        output_metrics = [
            CausalRegimeAccuracy(window_size=5, name="causal_regime_accuracy"),
            RegimeTransitionRecall(lag_tolerance=2, name="regime_transition_recall"),
            RegimeTransitionPrecision(lag_tolerance=2, name="regime_transition_precision"),
        ]

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name="TCN_Tunable")

    # Tunable optimizer
    optimizer_name = hp.Choice("optimizer", values=["adam", "adamw", "rmsprop"])
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
    )

    if optimizer_name == "adam":
        opt = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    elif optimizer_name == "adamw":
        opt = optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0)
    else:
        opt = optimizers.RMSprop(learning_rate=learning_rate, clipnorm=1.0)

    # Compile model
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


# ============================================================================
# HINTS AND BEST PRACTICES
# ============================================================================

"""
HINTS FOR USING TCN IN TRADING:
================================

1. RECEPTIVE FIELD CONSIDERATIONS:
   - TCN receptive field: RF = 1 + 2*(kernel_size-1)*nb_stacks*sum(dilations)
   - For dilations=(1,2,4,8,16,32), kernel_size=3, nb_stacks=1: RF=127
   - Ensure sequence_length >= receptive_field for full context
   - For 5-min bars:
     * dilations=(1,2,4,8) → RF=31 (~2.5 hours)
     * dilations=(1,2,4,8,16) → RF=63 (~5 hours)
     * dilations=(1,2,4,8,16,32) → RF=127 (~10 hours)

2. DILATIONS SELECTION:
   - Small patterns: (1, 2, 4, 8) - intraday trends
   - Medium patterns: (1, 2, 4, 8, 16) - daily patterns
   - Large patterns: (1, 2, 4, 8, 16, 32) - multi-day patterns
   - Must be powers of 2 for optimal performance

3. CAUSAL PADDING:
   - Always use 'causal' padding for trading (no look-ahead bias)
   - Model only sees past and present, never future
   - This is critical for realistic backtesting

4. POOLING STRATEGIES:
   - Global pooling: Averages all timesteps (more stable, less sensitive to latest data)
   - Last timestep: Uses only most recent info (more responsive, can be noisy)
   - For regime classification: Global pooling often works better
   - For next-bar prediction: Last timestep may be more appropriate

5. FEATURE ENGINEERING:
   - Normalize/standardize features BEFORE creating sequences
   - TCN works well with raw price data + indicators
   - No need to create sequences manually - just pass (batch, timesteps, features)
   - Example:
     ```python
     def feature_engineering_fn(train_data, test_data, train_labels, test_labels):
         scaler = StandardScaler()
         train_scaled = scaler.fit_transform(train_data)
         test_scaled = scaler.transform(test_data)

         # Convert to 3D if needed (already done in your pipeline)
         # Return as-is for TCN
         return train_scaled, test_scaled, train_labels, test_labels
     ```

6. TRAINING TIPS:
   - TCN trains faster than LSTM (parallel processing)
   - Use gradient clipping (clipnorm=1.0) to prevent exploding gradients
   - Batch size: 32-128 works well
   - Learning rate: Start with 0.001
   - Early stopping with patience=10-15 epochs

7. SKIP CONNECTIONS:
   - use_skip_connections=True (default): Better gradient flow, more stable
   - use_skip_connections=False: Simpler model, faster training
   - Recommended: Keep True unless you have memory constraints

8. NORMALIZATION:
   - Batch norm: Good for larger batches (>64), faster convergence
   - Layer norm: Better for smaller batches, more stable
   - None: Simplest, works well with proper data preprocessing
   - Start with none, add if needed

9. ADVANTAGES OVER LSTM:
   ✓ Parallel training (much faster)
   ✓ Larger receptive field with fewer parameters
   ✓ No vanishing gradients (residual connections)
   ✓ Naturally causal (no look-ahead bias risk)
   ✓ Better for pattern recognition in price data

10. WHEN TO USE TCN VS BILSTM:
    - Use TCN for:
      * Price pattern recognition
      * Support/resistance detection
      * Faster training needed
      * Very long sequences (>100 timesteps)

    - Use BiLSTM for:
      * When bidirectional context needed (careful with look-ahead!)
      * Smaller receptive fields sufficient
      * Order dependencies more important than patterns

11. COMMON PITFALLS:
    - Receptive field < sequence length: Model can't see full context
    - Too many dilations: Overfitting, slow training
    - Wrong padding: 'same' can cause look-ahead bias
    - Not using gradient clipping: Training instability

12. HYPERPARAMETER TUNING PRIORITIES:
    1. Dilations (most impact on receptive field)
    2. Number of filters (model capacity)
    3. Dropout rate (regularization)
    4. Learning rate (convergence speed)
    5. Dense layer units (final classification capacity)

13. ENSEMBLE WITH OTHER MODELS:
    - TCN captures patterns
    - LSTM captures sequential dependencies
    - Combine both for robust predictions
    - Use your ModelWalkForwardAnalysisOptimizer ensemble features
"""
