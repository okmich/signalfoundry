"""
Depthwise Separable Conv + GRU Hybrid (MobileNet-Inspired) - Model Factory
===========================================================================

Architecture:
-------------
Input (32-100 timesteps, n_features)
    ↓
DepthwiseSeparableConv1D (kernel=3, 64 filters)
    → BatchNorm → ReLU → Dropout(0.25)
    ↓
DepthwiseSeparableConv1D (kernel=5, 128 filters)
    → BatchNorm → ReLU → Dropout(0.25)
    ↓
GRU (64 units, return_sequences=True) → Dropout(0.3)
    ↓
GRU (32 units) → Dropout(0.2)
    ↓
Dense (64, ReLU) → Dropout(0.2)
    ↓
Output (softmax for multi-class)

Key Features:
-------------
- Depthwise separable convolutions = 8-10x fewer parameters than standard CNN
- SeparableConv1D = DepthwiseConv + PointwiseConv in one operation
- Extremely lightweight for edge deployment (mobile, embedded, low-power devices)
- Fast inference with reduced computational cost
- Suitable for: low-latency execution, mobile/embedded trading systems, real-time tick data
- Handles class imbalance with class weights and robust metrics

Technical Details:
------------------
- Depthwise convolution: Applies single filter per input channel
- Pointwise convolution: 1x1 convolution to combine channels
- Parameter reduction: ~8-10x compared to standard Conv1D
- Computation reduction: ~8-9x fewer FLOPs
- Ideal for resource-constrained environments

Usage:
------
1. Simple version (fixed hyperparameters):
   model = build_depthwise_separable_gru(
       sequence_length=48,
       num_features=20,
       num_classes=3
   )

2. Tunable version (with KerasTuner):
   def model_builder(hp):
       return build_depthwise_separable_gru_tunable(
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
"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" - Howard et al. (2017) https://arxiv.org/abs/1704.04861
Time series adaptation: "Temporal Convolutional Networks with Depthwise Separable Convolutions for Time Series Forecasting"** - Hewage et al. (2020) https://ieeexplore.ieee.org/document/9378374
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


# ============================================================================
# SIMPLE VERSION (Fixed Hyperparameters)
# ============================================================================


def build_depthwise_separable_gru(
        sequence_length=48,
        num_features=20,
        num_classes=3,
        task_type=TaskType.CLASSIFICATION,
        # Depthwise Separable Conv Block 1
        conv1_filters=64,
        conv1_kernel_size=3,
        conv1_dropout=0.25,
        # Depthwise Separable Conv Block 2
        conv2_filters=128,
        conv2_kernel_size=5,
        conv2_dropout=0.25,
        # GRU Layers
        gru1_units=64,
        gru2_units=32,
        dropout_gru1=0.3,
        dropout_gru2=0.2,
        # Dense Layer
        dense_units=64,
        dropout_dense=0.2,
        # Training
        learning_rate=0.001,
        optimizer_name="adam",
        class_weights=None,
):
    """
    Build Depthwise Separable Conv + GRU Hybrid model (MobileNet-inspired).

    This model combines the parameter efficiency of depthwise separable
    convolutions with the sequential modeling power of GRU. It's designed
    for edge deployment scenarios where model size and inference speed
    are critical constraints.

    Depthwise Separable Convolution:
    - Standard Conv: filters × kernel_size × input_channels operations
    - Depthwise Separable: kernel_size × input_channels + input_channels × filters
    - Reduction: ~8-10x fewer parameters, ~8-9x fewer FLOPs

    Args:
        sequence_length: Number of timesteps in input sequences
        num_features: Number of features per timestep
        num_classes: Number of output classes
        conv1_filters: Filters in first separable conv layer (default: 64)
        conv1_kernel_size: Kernel size for first separable conv (default: 3)
        conv1_dropout: Dropout after first conv block (default: 0.25)
        conv2_filters: Filters in second separable conv layer (default: 128)
        conv2_kernel_size: Kernel size for second separable conv (default: 5)
        conv2_dropout: Dropout after second conv block (default: 0.25)
        gru1_units: Units in first GRU layer (default: 64)
        gru2_units: Units in second GRU layer (default: 32)
        dropout_gru1: Dropout after first GRU (default: 0.3)
        dropout_gru2: Dropout after second GRU (default: 0.2)
        dense_units: Units in dense layer (default: 64)
        dropout_dense: Dropout after dense layer (default: 0.2)
        learning_rate: Learning rate for optimizer (default: 0.001)
        optimizer_name: Optimizer to use ('adam', 'adamw', 'rmsprop') (default: 'adam')
        class_weights: Dictionary of class weights for imbalanced data (optional)

    Returns:
        Compiled Keras model

    Example:
        >>> model = build_depthwise_separable_gru(
        ...     sequence_length=48,
        ...     num_features=20,
        ...     num_classes=3
        ... )
        >>> model.summary()
        >>> print(f"Total params: {model.count_params():,}")
    """

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # ========================================================================
    # Depthwise Separable Convolutional Blocks
    # ========================================================================
    # SeparableConv1D = DepthwiseConv1D + PointwiseConv1D (1x1 Conv)
    # This provides the same functionality with significantly fewer parameters

    # Block 1: Depthwise Separable Conv (kernel=3, 64 filters)
    x = layers.SeparableConv1D(
        filters=conv1_filters,
        kernel_size=conv1_kernel_size,
        padding="causal",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
        name="separable_conv1",
    )(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.Dropout(conv1_dropout, name="dropout_conv1")(x)

    # Block 2: Depthwise Separable Conv (kernel=5, 128 filters)
    x = layers.SeparableConv1D(
        filters=conv2_filters,
        kernel_size=conv2_kernel_size,
        padding="causal",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
        name="separable_conv2",
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.Dropout(conv2_dropout, name="dropout_conv2")(x)

    # ========================================================================
    # GRU Recurrent Layers
    # ========================================================================

    # First GRU layer (must return sequences for second GRU)
    x = layers.GRU(gru1_units, return_sequences=True, name="gru1")(x)
    x = layers.Dropout(dropout_gru1, name="dropout_gru1")(x)

    # Second GRU layer (returns only last output)
    x = layers.GRU(gru2_units, return_sequences=False, name="gru2")(x)
    x = layers.Dropout(dropout_gru2, name="dropout_gru2")(x)

    # ========================================================================
    # Dense Classification Head
    # ========================================================================

    # Dense layer
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_dense, name="dropout_dense")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("DepthwiseSeparable_GRU", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile model
    opt = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


# ============================================================================
# TUNABLE VERSION (With KerasTuner)
# ============================================================================


def build_depthwise_separable_gru_tunable(
        hp, num_features, num_classes, task_type=TaskType.CLASSIFICATION,
        sequence_length=None, max_sequence_length=100
):
    """
    Build Depthwise Separable Conv + GRU model with hyperparameter tuning.

    This version allows KerasTuner to optimize:
    - Sequence length (if not provided as fixed parameter)
    - Convolutional filters and kernel sizes
    - Dropout rates for conv and GRU layers
    - GRU units in each layer
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
        ...     return build_depthwise_separable_gru_tunable(
        ...         hp=hp,
        ...         num_features=20,
        ...         num_classes=3
        ...     )
        >>> # Example 2: Fixed sequence_length (for pre-created sequences)
        >>> def model_builder(hp):
        ...     return build_depthwise_separable_gru_tunable(
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
        ...     project_name='depthwise_separable_gru'
        ... )
    """

    # Tunable hyperparameters
    if sequence_length is None:
        # Tune sequence_length if not provided
        sequence_length = hp.Int(
            "sequence_length", min_value=32, max_value=max_sequence_length, step=16
        )
    # else: use the provided fixed sequence_length

    # Conv blocks
    conv1_filters = hp.Choice("conv1_filters", values=[32, 64, 128])
    conv1_kernel_size = hp.Choice("conv1_kernel_size", values=[3, 5, 7])
    conv1_dropout = hp.Float("conv1_dropout", min_value=0.1, max_value=0.4, step=0.05)

    conv2_filters = hp.Choice("conv2_filters", values=[64, 128, 256])
    conv2_kernel_size = hp.Choice("conv2_kernel_size", values=[3, 5, 7])
    conv2_dropout = hp.Float("conv2_dropout", min_value=0.1, max_value=0.4, step=0.05)

    # GRU layers
    gru1_units = hp.Choice("gru1_units", values=[32, 64, 128])
    gru2_units = hp.Choice("gru2_units", values=[16, 32, 64])
    dropout_gru1 = hp.Float("dropout_gru1", min_value=0.2, max_value=0.5, step=0.1)
    dropout_gru2 = hp.Float("dropout_gru2", min_value=0.1, max_value=0.4, step=0.1)

    # Dense layer
    dense_units = hp.Choice("dense_units", values=[32, 64, 128])
    dropout_dense = hp.Float("dropout_dense", min_value=0.1, max_value=0.4, step=0.1)

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # Depthwise Separable Conv Block 1
    x = layers.SeparableConv1D(
        filters=conv1_filters,
        kernel_size=conv1_kernel_size,
        padding="causal",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
        name="separable_conv1",
    )(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.Dropout(conv1_dropout, name="dropout_conv1")(x)

    # Depthwise Separable Conv Block 2
    x = layers.SeparableConv1D(
        filters=conv2_filters,
        kernel_size=conv2_kernel_size,
        padding="causal",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
        name="separable_conv2",
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.Dropout(conv2_dropout, name="dropout_conv2")(x)

    # GRU layers
    x = layers.GRU(gru1_units, return_sequences=True, name="gru1")(x)
    x = layers.Dropout(dropout_gru1, name="dropout_gru1")(x)

    x = layers.GRU(gru2_units, return_sequences=False, name="gru2")(x)
    x = layers.Dropout(dropout_gru2, name="dropout_gru2")(x)

    # Dense layer
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_dense, name="dropout_dense")(x)

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
    model = models.Model(
        inputs=inputs, outputs=outputs, name="DepthwiseSeparable_GRU_Tunable"
    )

    # Tunable optimizer
    optimizer_name = hp.Choice("optimizer", values=["adam", "adamw", "rmsprop"])
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
    )

    if optimizer_name == "adam":
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "adamw":
        opt = optimizers.AdamW(learning_rate=learning_rate)
    else:
        opt = optimizers.RMSprop(learning_rate=learning_rate)

    # Compile model
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


# ============================================================================
# HINTS AND BEST PRACTICES
# ============================================================================

"""
HINTS FOR USING THIS MODEL IN TRADING:
======================================

1. ARCHITECTURE BENEFITS - WHY DEPTHWISE SEPARABLE CONVOLUTIONS?

   Standard Conv1D:
   - Parameters: kernel_size × input_channels × output_filters
   - Example: 5 × 20 × 128 = 12,800 parameters

   Depthwise Separable Conv1D:
   - Depthwise: kernel_size × input_channels = 5 × 20 = 100
   - Pointwise: input_channels × output_filters = 20 × 128 = 2,560
   - Total: 100 + 2,560 = 2,660 parameters
   - Reduction: 12,800 / 2,660 = 4.8x fewer parameters!

   Benefits:
   - 8-10x parameter reduction (typically)
   - 8-9x computation reduction (FLOPs)
   - Faster inference (critical for real-time trading)
   - Lower memory footprint (fits on edge devices)
   - Easier to train (fewer parameters = less overfitting risk)

2. WHEN TO USE THIS MODEL:

   Ideal scenarios:
   - Low-latency execution requirements (< 10ms inference)
   - Mobile/embedded trading systems (phones, Raspberry Pi, FPGAs)
   - Real-time tick data processing (high-frequency updates)
   - Edge deployment (no cloud access, local inference)
   - Resource-constrained environments (limited RAM, CPU, battery)
   - High-throughput scenarios (processing many symbols simultaneously)

   Not ideal for:
   - Maximum accuracy at any cost (standard Conv/Transformer may be better)
   - Offline batch processing where latency doesn't matter
   - Scenarios with unlimited computational resources

3. DEPLOYMENT TARGETS:

   Mobile devices:
   - iOS: CoreML conversion, on-device inference
   - Android: TensorFlow Lite, on-device inference
   - Model size: ~1-5 MB (quantized INT8)
   - Inference: ~5-20ms on modern phones

   Embedded systems:
   - Raspberry Pi 4/5: TensorFlow Lite, ~10-50ms
   - NVIDIA Jetson Nano: TensorRT, ~2-10ms
   - ESP32/Arduino: TensorFlow Lite Micro (very constrained)

   Edge servers:
   - AWS Lambda, Google Cloud Functions
   - Docker containers with minimal resources
   - K8s pods with CPU-only nodes

4. MODEL OPTIMIZATION FOR EDGE DEPLOYMENT:

   Quantization (reduce precision):
   ```python
   import tensorflow as tf

   # Post-training quantization (easiest)
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   tflite_model = converter.convert()

   # INT8 quantization (most aggressive, 4x smaller)
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.target_spec.supported_types = [tf.int8]
   tflite_quant_model = converter.convert()

   # Save
   with open('model_quantized.tflite', 'wb') as f:
       f.write(tflite_quant_model)
   ```

   Pruning (remove unnecessary weights):
   ```python
   import tensorflow_model_optimization as tfmot

   # Define pruning schedule
   pruning_params = {
       'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
           initial_sparsity=0.0,
           final_sparsity=0.5,  # Remove 50% of weights
           begin_step=0,
           end_step=1000
       )
   }

   # Apply pruning
   model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
       model, **pruning_params
   )
   model_for_pruning.compile(...)

   # Train with pruning
   callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
   model_for_pruning.fit(..., callbacks=callbacks)

   # Strip pruning wrappers
   final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
   ```

5. SEQUENCE LENGTH SELECTION FOR LOW LATENCY:

   Shorter sequences = faster inference:
   - 32 timesteps: ~2-5ms inference (very fast)
   - 48 timesteps: ~3-8ms inference (fast)
   - 64 timesteps: ~5-12ms inference (moderate)
   - 96 timesteps: ~8-20ms inference (slower)

   Trade-off:
   - Shorter = faster but less context
   - Longer = more context but slower

   For 5-min bars:
   - 32 timesteps = 2.67 hours (good for intraday)
   - 48 timesteps = 4 hours (balanced)
   - 64 timesteps = 5.33 hours (more context)

6. FEATURE ENGINEERING FOR EDGE DEPLOYMENT:

   Keep it simple (reduce computation):
   - Avoid complex indicators (heavy computation)
   - Use fast indicators: SMA, EMA, RSI, price ratios
   - Pre-compute features server-side if possible
   - Use fixed-length windows (no dynamic sizing)

   Essential features for tick data:
   ```python
   def create_lightweight_features(df):
       # Price features (fast)
       df['returns'] = df['close'].pct_change()
       df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
       df['hl_ratio'] = df['high'] / df['low']
       df['co_ratio'] = df['close'] / df['open']

       # Volume features (fast)
       df['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()
       df['relative_volume'] = df['volume'] / df['volume_ma']

       # Momentum (fast)
       df['roc'] = df['close'].pct_change(periods=10)

       # Avoid: complex indicators like MACD, Bollinger Bands (if latency critical)

       return df
   ```

7. INFERENCE OPTIMIZATION:

   Batch predictions (when possible):
   ```python
   # Instead of predicting one-by-one
   # BAD: for x in X_samples: model.predict(x)

   # GOOD: batch predict
   predictions = model.predict(X_samples_batch)
   ```

   Model warm-up (avoid cold start):
   ```python
   # Warm up model at startup
   dummy_input = np.zeros((1, sequence_length, num_features), dtype=np.float32)
   _ = model.predict(dummy_input, verbose=0)
   ```

   Use TensorFlow Lite for production:
   ```python
   # Load TFLite model
   interpreter = tf.lite.Interpreter(model_path='model.tflite')
   interpreter.allocate_tensors()

   input_details = interpreter.get_input_details()
   output_details = interpreter.get_output_details()

   # Run inference
   interpreter.set_tensor(input_details[0]['index'], input_data)
   interpreter.invoke()
   output = interpreter.get_tensor(output_details[0]['index'])
   ```

8. TRAINING TIPS FOR EDGE MODELS:

   - Use smaller batch sizes (32-64) to simulate edge constraints
   - Train with mixed precision to match deployment
   - Use aggressive regularization (model must generalize well)
   - Early stopping (avoid overfitting, critical for small models)
   - Knowledge distillation (train large model, distill to small)

   Example knowledge distillation:
   ```python
   # Train large "teacher" model first
   teacher_model = build_large_model(...)
   teacher_model.fit(X_train, y_train, ...)

   # Create soft targets
   soft_targets = teacher_model.predict(X_train)

   # Train small "student" model (this depthwise model)
   student_model = build_depthwise_separable_gru(...)

   # Custom loss: combine hard targets + soft targets
   def distillation_loss(y_true, y_pred, soft_targets, temperature=3, alpha=0.5):
       hard_loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
       soft_loss = keras.losses.kl_divergence(
           tf.nn.softmax(soft_targets / temperature),
           tf.nn.softmax(y_pred / temperature)
       )
       return alpha * hard_loss + (1 - alpha) * soft_loss

   # Train student
   student_model.fit(X_train, y_train, ...)
   ```

9. HANDLING CLASS IMBALANCE (EDGE-SPECIFIC):

   On-device class weight adjustment:
   - Compute class weights on server
   - Pass weights to edge device as config
   - Apply during inference (adjust probability thresholds)

   Example:
   ```python
   # Server-side: compute class weights
   class_weights = compute_class_weight('balanced', ...)

   # Edge device: adjust predictions
   predictions = model.predict(X_test)
   adjusted_predictions = predictions * class_weights
   final_predictions = np.argmax(adjusted_predictions, axis=1)
   ```

10. MONITORING EDGE DEPLOYMENTS:

    Key metrics to track:
    - Inference latency (p50, p95, p99)
    - Memory usage (RAM, GPU memory)
    - CPU/GPU utilization
    - Battery drain (mobile devices)
    - Prediction accuracy (online learning feedback)
    - Model drift (compare to server baseline)

    Lightweight logging:
    ```python
    import time
    import psutil

    def monitor_inference(model, X):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        prediction = model.predict(X, verbose=0)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        latency_ms = (end_time - start_time) * 1000
        memory_delta_mb = end_memory - start_memory

        return prediction, {
            'latency_ms': latency_ms,
            'memory_delta_mb': memory_delta_mb
        }
    ```

11. WALK-FORWARD ANALYSIS INTEGRATION:

    ```python
    def feature_engineering_fn(train_data, test_data, train_labels, test_labels):
        from sklearn.preprocessing import StandardScaler

        # Scale (use StandardScaler for edge deployment simplicity)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Save scaler params for edge deployment
        scaler_params = {
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist()
        }
        # Save to JSON for edge device
        import json
        with open('scaler_params.json', 'w') as f:
            json.dump(scaler_params, f)

        # Create sequences
        def create_sequences(data, labels, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:i+seq_len])
                y.append(labels[i+seq_len])
            return np.array(X, dtype=np.float32), np.array(y)  # Use float32

        X_train, y_train = create_sequences(train_scaled, train_labels, sequence_length)
        X_test, y_test = create_sequences(test_scaled, test_labels, sequence_length)

        return X_train, X_test, y_train, y_test
    ```

12. COMMON PITFALLS:

    - Not testing on actual target hardware (always benchmark on device)
    - Over-optimizing for accuracy (latency/size matter more on edge)
    - Ignoring quantization accuracy loss (test quantized model performance)
    - Not considering battery drain (mobile deployment)
    - Forgetting model warm-up (first inference is slower)
    - Using float64 (use float32 or lower for edge)
    - Not handling variable sequence lengths (dynamic shapes are slow)
    - Ignoring network latency (if model needs server updates)

13. REAL-WORLD EDGE DEPLOYMENT EXAMPLE:

    ```python
    # Full pipeline for Raspberry Pi deployment

    # 1. Train and optimize model
    model = build_depthwise_separable_gru(...)
    model.fit(X_train, y_train, ...)

    # 2. Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # 3. Save
    with open('trading_model.tflite', 'wb') as f:
        f.write(tflite_model)

    # 4. Deploy to Raspberry Pi
    # On Pi:
    import tensorflow as tf
    import numpy as np

    # Load model
    interpreter = tf.lite.Interpreter(model_path='trading_model.tflite')
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Real-time inference loop
    while True:
        # Get latest market data
        market_data = fetch_latest_data()  # Your data source

        # Preprocess
        features = extract_features(market_data)
        input_tensor = create_sequence(features, seq_len=48)
        input_tensor = input_tensor.astype(np.float32)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Execute trade based on prediction
        execute_trade(prediction)

        # Sleep until next bar
        time.sleep(300)  # 5 minutes
    ```

14. PERFORMANCE BENCHMARKS (APPROXIMATE):

    Model size (this architecture):
    - FP32: ~500KB - 2MB
    - FP16: ~250KB - 1MB
    - INT8: ~125KB - 500KB

    Inference time (48 timesteps, 20 features):
    - High-end phone (A15, Snapdragon 8 Gen 2): 2-5ms
    - Mid-range phone (A13, Snapdragon 7): 5-15ms
    - Raspberry Pi 4: 10-30ms
    - Raspberry Pi 3: 30-100ms
    - ESP32: Not recommended (too slow)

    Compare to standard Conv1D:
    - 2-3x faster inference
    - 8-10x fewer parameters
    - Same or slightly lower accuracy

15. ALTERNATIVE OPTIMIZATIONS:

    If still too slow, consider:
    - Reduce num_features (feature selection)
    - Reduce sequence_length (shorter lookback)
    - Use 1D convolutions with larger strides (downsample)
    - Remove second GRU layer (single GRU)
    - Use quantization-aware training (better INT8 accuracy)
    - Use pruning + quantization combined
    - Consider simpler models (shallow MLP, decision trees)
"""
