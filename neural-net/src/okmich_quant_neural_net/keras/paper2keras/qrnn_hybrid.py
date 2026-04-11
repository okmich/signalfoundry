"""
Quasi-Recurrent Neural Network (QRNN) - Model Factory
======================================================

Architecture:
-------------
Input (32-100 timesteps, n_features)
    ↓
QRNN Layer (128 units, window=2) → Dropout(0.3)
    ↓
QRNN Layer (128 units, window=2) → Dropout(0.3)
    ↓
QRNN Layer (64 units, window=2) → Dropout(0.2)
    ↓
GlobalAveragePooling1D
    ↓
Dense (64, ReLU) → Dropout(0.2)
    ↓
Dense (32, ReLU)
    ↓
Output (softmax for multi-class)

Key Features:
-------------
- Hybrid: Convolutions for feature extraction + recurrence for temporal modeling
- 2-20x faster than LSTM during training and inference
- Better parallelization than LSTM (convolutions are parallel)
- Simpler gating mechanism than LSTM (fewer parameters)
- Suitable for: real-time tick aggregation, intraday pattern recognition, streaming data

QRNN Advantages:
----------------
✓ Parallel convolutions (fast feature extraction)
✓ Sequential pooling (maintains temporal dependencies)
✓ Fewer parameters than LSTM
✓ Faster training and inference
✓ Good for long sequences
✓ Less prone to vanishing gradients

Pooling Strategies:
-------------------
- 'f' pooling (forget pooling): Like GRU, simpler
- 'fo' pooling (forget + output): Like LSTM, more expressive

Usage:
------
1. Simple version (fixed hyperparameters):
   model = build_qrnn(
       sequence_length=48,
       num_features=20,
       num_classes=3
   )

2. Tunable version (with KerasTuner):
   def model_builder(hp):
       return build_qrnn_tunable(
           hp=hp,
           num_features=20,
           num_classes=3
       )

   tuner = keras_tuner.BayesianOptimization(
       model_builder,
       objective='val_loss',
       max_trials=20
   )


References:
"Quasi-Recurrent Neural Networks" - Bradbury et al. (2016) https://arxiv.org/abs/1611.01576
Financial application: "Financial Time Series Forecasting with Deep Learning: A Systematic Literature Review: 2005-2019"** - Sezer et al. (2020) - Section on QRNN applications
https://www.sciencedirect.com/science/article/pii/S1568494620300910
"""

from keras import layers, models, optimizers, losses, metrics

# Import the QRNN layer
from ..layers import QRNN
from ..metrics import CausalRegimeAccuracy, RegimeTransitionRecall, RegimeTransitionPrecision
from .common import TaskType, create_output_layer_and_loss, get_optimizer, get_model_name


# ============================================================================
# SIMPLE VERSION (Fixed Hyperparameters)
# ============================================================================


def build_qrnn(sequence_length=48, num_features=20, num_classes=3, task_type=TaskType.CLASSIFICATION,
               qrnn1_units=128, qrnn2_units=128, qrnn3_units=64, window_size=2, pooling="fo",
               dropout_qrnn1=0.3, dropout_qrnn2=0.3, dropout_qrnn3=0.2,
               dense1_units=64, dense1_dropout=0.2, dense2_units=32, learning_rate=0.001, optimizer_name="adam"):
    """
    Build Quasi-Recurrent Neural Network (QRNN) model (fixed hyperparameters).

    Args:
        sequence_length: Number of timesteps in input sequences
        num_features: Number of features per timestep
        num_classes: Number of output classes
        qrnn1_units: Units in first QRNN layer (default: 128)
        qrnn2_units: Units in second QRNN layer (default: 128)
        qrnn3_units: Units in third QRNN layer (default: 64)
        window_size: Convolution window size (default: 2)
        pooling: Pooling type ('f' or 'fo') (default: 'fo')
        dropout_qrnn1: Dropout after first QRNN (default: 0.3)
        dropout_qrnn2: Dropout after second QRNN (default: 0.3)
        dropout_qrnn3: Dropout after third QRNN (default: 0.2)
        dense1_units: Units in first dense layer (default: 64)
        dense1_dropout: Dropout after first dense layer (default: 0.2)
        dense2_units: Units in second dense layer (default: 32)
        learning_rate: Learning rate for optimizer (default: 0.001)
        optimizer_name: Optimizer to use ('adam', 'adamw', 'rmsprop') (default: 'adam')

    Returns:
        Compiled Keras model

    Example:
        >>> model = build_qrnn(
        ...     sequence_length=48,
        ...     num_features=20,
        ...     num_classes=3
        ... )
        >>> model.summary()
    """

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # First QRNN layer
    x = QRNN(
        units=qrnn1_units,
        window_size=window_size,
        pooling=pooling,
        return_sequences=True,
        name="qrnn1",
    )(inputs)
    x = layers.Dropout(dropout_qrnn1, name="dropout_qrnn1")(x)

    # Second QRNN layer
    x = QRNN(
        units=qrnn2_units,
        window_size=window_size,
        pooling=pooling,
        return_sequences=True,
        name="qrnn2",
    )(x)
    x = layers.Dropout(dropout_qrnn2, name="dropout_qrnn2")(x)

    # Third QRNN layer (return_sequences=False to get final hidden state)
    x = QRNN(
        units=qrnn3_units,
        window_size=window_size,
        pooling=pooling,
        return_sequences=False,
        name="qrnn3",
    )(x)
    x = layers.Dropout(dropout_qrnn3, name="dropout_qrnn3")(x)

    # Dense layers
    x = layers.Dense(dense1_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dense1_dropout, name="dropout_dense1")(x)

    x = layers.Dense(dense2_units, activation="relu", name="dense2")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("QRNN_Hybrid", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile model
    opt = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    print(f"\nQRNN Model Configuration:")
    print(f"  Pooling type: {pooling}")
    print(f"  Window size: {window_size}")
    print(f"  QRNN units: [{qrnn1_units}, {qrnn2_units}, {qrnn3_units}]")

    return model


# ============================================================================
# TUNABLE VERSION (With KerasTuner)
# ============================================================================


def build_qrnn_tunable(hp, num_features, num_classes, task_type=TaskType.CLASSIFICATION,
                       sequence_length=None, max_sequence_length=100):
    """
    Build QRNN model with hyperparameter tuning.

    This version allows KerasTuner to optimize:
    - Sequence length (if not provided as fixed parameter)
    - QRNN units in each layer
    - Window size
    - Pooling strategy
    - Dropout rates
    - Dense layer units
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
        ...     return build_qrnn_tunable(
        ...         hp=hp,
        ...         num_features=20,
        ...         num_classes=3
        ...     )
        >>> # Example 2: Fixed sequence_length (for pre-created sequences)
        >>> def model_builder(hp):
        ...     return build_qrnn_tunable(
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
        ...     project_name='qrnn_classifier'
        ... )
    """

    # Tunable hyperparameters
    if sequence_length is None:
        # Tune sequence_length if not provided
        sequence_length = hp.Int(
            "sequence_length", min_value=32, max_value=max_sequence_length, step=16
        )
    # else: use the provided fixed sequence_length

    qrnn1_units = hp.Choice("qrnn1_units", values=[64, 128, 256])
    qrnn2_units = hp.Choice("qrnn2_units", values=[64, 128, 256])
    qrnn3_units = hp.Choice("qrnn3_units", values=[32, 64, 128])

    window_size = hp.Choice("window_size", values=[2, 3])
    pooling = hp.Choice("pooling", values=["f", "fo"])

    dropout_qrnn1 = hp.Float("dropout_qrnn1", min_value=0.2, max_value=0.5, step=0.1)
    dropout_qrnn2 = hp.Float("dropout_qrnn2", min_value=0.2, max_value=0.5, step=0.1)
    dropout_qrnn3 = hp.Float("dropout_qrnn3", min_value=0.1, max_value=0.4, step=0.1)

    dense1_units = hp.Choice("dense1_units", values=[32, 64, 128])
    dense1_dropout = hp.Float("dense1_dropout", min_value=0.1, max_value=0.3, step=0.1)
    dense2_units = hp.Choice("dense2_units", values=[16, 32, 64])

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # QRNN layers
    x = QRNN(
        units=qrnn1_units,
        window_size=window_size,
        pooling=pooling,
        return_sequences=True,
        name="qrnn1",
    )(inputs)
    x = layers.Dropout(dropout_qrnn1, name="dropout_qrnn1")(x)

    x = QRNN(
        units=qrnn2_units,
        window_size=window_size,
        pooling=pooling,
        return_sequences=True,
        name="qrnn2",
    )(x)
    x = layers.Dropout(dropout_qrnn2, name="dropout_qrnn2")(x)

    x = QRNN(
        units=qrnn3_units,
        window_size=window_size,
        pooling=pooling,
        return_sequences=False,
        name="qrnn3",
    )(x)
    x = layers.Dropout(dropout_qrnn3, name="dropout_qrnn3")(x)

    # Dense layers
    x = layers.Dense(dense1_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dense1_dropout, name="dropout_dense1")(x)

    x = layers.Dense(dense2_units, activation="relu", name="dense2")(x)

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
    model = models.Model(inputs=inputs, outputs=outputs, name="QRNN_Tunable")

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
HINTS FOR USING QRNN IN TRADING:
=================================

1. QRNN vs LSTM COMPARISON:
   QRNN Advantages:
   ✓ 2-20x faster training (convolutions are parallel)
   ✓ 2-10x faster inference
   ✓ Fewer parameters (~30% less)
   ✓ Better for long sequences (100+ timesteps)
   ✓ Good GPU utilization

   LSTM Advantages:
   ✓ More expressive gating
   ✓ Better for very complex temporal patterns
   ✓ More established/tested

2. POOLING STRATEGIES:
   - 'f' pooling (forget pooling):
     * Like GRU: h_t = f_t * h_{t-1} + (1-f_t) * z_t
     * Simpler, fewer parameters
     * Faster training
     * Good for most tasks

   - 'fo' pooling (forget + output):
     * Like LSTM: adds output gate
     * More expressive
     * Better for complex patterns
     * Recommended for trading

3. WINDOW SIZE SELECTION:
   - window_size=2: Standard QRNN, minimal parameters
   - window_size=3: Slightly more context, still fast
   - Recommendation: Start with 2, increase to 3 if needed

4. LAYER CONFIGURATION:
   - Typical: [128, 128, 64] - decreasing units
   - Deep: [256, 256, 128, 64] - more capacity
   - Wide: [256, 256, 128] - maintain capacity
   - Start with [128, 128, 64] and adjust

5. WHEN TO USE QRNN:
   ✓ Real-time tick aggregation (speed matters)
   ✓ Intraday pattern recognition
   ✓ Streaming data applications
   ✓ Long sequences (>100 timesteps)
   ✓ When LSTM is too slow
   ✓ GPU-accelerated environments

6. WHEN NOT TO USE QRNN:
   ✗ Very short sequences (<20 timesteps)
   ✗ When maximum accuracy is more important than speed
   ✗ When you need bidirectional processing
   ✗ CPU-only environments (LSTM might be better)

7. FEATURE ENGINEERING:
   - Standardize features before feeding to QRNN
   - QRNN works well with:
     * Price changes / returns
     * Volume ratios
     * Technical indicators
     * Order flow features
   - Example:
     ```python
     scaler = StandardScaler()
     X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features))
     X_train_scaled = X_train_scaled.reshape(-1, sequence_length, n_features)
     ```

8. TRAINING TIPS:
   - Batch size: 32-128 (larger is better for speed)
   - Learning rate: 0.0005 - 0.002
   - Use gradient clipping (clipnorm=1.0)
   - Early stopping with patience=10-15
   - QRNN trains faster than LSTM, so use more epochs

9. INFERENCE SPEED:
   - QRNN is 2-10x faster than LSTM at inference
   - Critical for real-time trading systems
   - Test inference latency:
     ```python
     import time
     start = time.time()
     pred = model.predict(X_test, batch_size=1)
     latency = (time.time() - start) * 1000  # ms
     print(f"Inference latency: {latency:.2f}ms")
     ```

10. SEQUENCE LENGTH CONSIDERATIONS:
    - QRNN scales better to long sequences than LSTM
    - For 5-min bars:
      * 48 timesteps = 4 hours
      * 96 timesteps = 8 hours
      * 144 timesteps = 12 hours
    - QRNN maintains speed even at 200+ timesteps

11. DROPOUT STRATEGIES:
    - Use higher dropout (0.3-0.5) on early QRNN layers
    - Lower dropout (0.2-0.3) on later layers
    - Prevents overfitting on noisy market data
    - QRNN is less prone to overfitting than LSTM

12. REAL-TIME TRADING APPLICATIONS:
    Example: Tick aggregation and prediction
    ```python
    # Build fast QRNN model
    model = build_qrnn(
        sequence_length=100,
        num_features=15,
        num_classes=3,
        window_size=2,
        pooling='f'  # Faster than 'fo'
    )

    # Real-time prediction loop
    while market_open:
        recent_ticks = get_last_n_ticks(100)
        features = extract_features(recent_ticks)
        prediction = model.predict(features, verbose=0)
        trade_signal = np.argmax(prediction)
        execute_trade(trade_signal)
    ```

13. MEMORY EFFICIENCY:
    - QRNN uses less memory than LSTM
    - Can batch larger sequences
    - Good for memory-constrained environments

14. ENSEMBLE STRATEGIES:
    Combine QRNN with:
    - TCN: QRNN for temporal, TCN for patterns
    - BiLSTM: QRNN for speed, BiLSTM for accuracy
    - Transformer: QRNN for baseline, Transformer for attention

15. DEBUGGING TIPS:
    - If training is unstable:
      * Reduce learning rate
      * Increase gradient clipping
      * Add more dropout
    - If model is slow:
      * Use 'f' pooling instead of 'fo'
      * Reduce window_size to 2
      * Reduce number of units
    - If underfitting:
      * Increase model capacity
      * Use 'fo' pooling
      * Add more QRNN layers
"""
