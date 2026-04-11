"""
IndRNN (Independently Recurrent Neural Network) Stack - Model Factory
======================================================================

Architecture:
-------------
Input (32-100 timesteps, n_features)
    ↓
IndRNN (128 units, return_sequences=True) → BatchNorm → Dropout(0.3)
    ↓
IndRNN (128 units, return_sequences=True) → BatchNorm → Dropout(0.3)
    ↓
IndRNN (64 units, return_sequences=True) → BatchNorm → Dropout(0.2)
    ↓
IndRNN (64 units) → BatchNorm
    ↓
Dense (32, ReLU) → Dropout(0.2)
    ↓
Output (softmax for multi-class)

Key Features:
-------------
- Neurons process information independently (each neuron has its own recurrent weight)
- Can stack VERY deep (4+ layers) without vanishing gradients
- Longer memory than LSTM/GRU without complexity overhead
- Better gradient flow due to independent recurrence
- BatchNorm after each layer stabilizes training
- Suitable for: long-term trend following, regime persistence detection, multi-day patterns

Technical Details:
------------------
IndRNN formula: h_t = σ(W * x_t + u ⊙ h_{t-1} + b)
- W: input-to-hidden weights (matrix) - standard
- u: recurrent weights (VECTOR, not matrix!) - key difference from standard RNN
- ⊙: element-wise multiplication (Hadamard product)
- σ: activation function (ReLU works well, unlike vanilla RNN)

Benefits over LSTM/GRU:
- Fewer parameters (no gates, vector recurrent weights)
- Faster training and inference
- Can capture longer dependencies
- Better gradient flow in deep stacks
- No gradient vanishing even with 20+ layers

Usage:
------
1. Simple version (fixed hyperparameters):
   model = build_indrnn_stack(
       sequence_length=96,
       num_features=20,
       num_classes=3
   )

2. Tunable version (with KerasTuner):
   def model_builder(hp):
       return build_indrnn_stack_tunable(
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
- "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN" - Li et al. (2018) https://arxiv.org/abs/1803.04831
- Financial time series application: "Deep Recurrent Neural Networks for Financial Time Series"** - Chong et al. (2017) https://ieeexplore.ieee.org/document/8260883
"""

from keras import layers, models, optimizers, losses, metrics

# Import IndRNN layer from centralized location
from ..layers.indrnn import IndRNN
from ..metrics import CausalRegimeAccuracy, RegimeTransitionRecall, RegimeTransitionPrecision
from .common import TaskType, create_output_layer_and_loss, get_optimizer, get_model_name


# ============================================================================
# SIMPLE VERSION (Fixed Hyperparameters)
# ============================================================================


def build_indrnn_stack(sequence_length=96, num_features=20, num_classes=3, task_type=TaskType.CLASSIFICATION,
                       # IndRNN Layers
                       indrnn1_units=128, indrnn2_units=128, indrnn3_units=64, indrnn4_units=64, dropout_indrnn1=0.3,
                       dropout_indrnn2=0.3, dropout_indrnn3=0.2,
                       # Dense Layer
                       dense_units=32, dropout_dense=0.2,
                       # Recurrent weight clipping (for very long sequences)
                       recurrent_clip_min=None, recurrent_clip_max=None,
                       # Training
                       learning_rate=0.001, optimizer_name="adam", class_weights=None):
    """
    Build stacked IndRNN model (4 layers deep).

    IndRNN can be stacked very deep (4+ layers) without vanishing gradients
    due to independent recurrent connections. BatchNorm after each layer is
    crucial for training stability.

    This architecture is particularly effective for:
    - Long-term trend following (multi-day patterns)
    - Regime persistence detection (market states that persist)
    - Capturing dependencies beyond 100+ timesteps
    - Deep feature hierarchies in time series

    Args:
        sequence_length: Number of timesteps in input sequences (default: 96)
        num_features: Number of features per timestep
        num_classes: Number of output classes
        indrnn1_units: Units in first IndRNN layer (default: 128)
        indrnn2_units: Units in second IndRNN layer (default: 128)
        indrnn3_units: Units in third IndRNN layer (default: 64)
        indrnn4_units: Units in fourth IndRNN layer (default: 64)
        dropout_indrnn1: Dropout after layers 1-2 (default: 0.3)
        dropout_indrnn2: Dropout after layers 1-2 (default: 0.3)
        dropout_indrnn3: Dropout after layer 3 (default: 0.2)
        dense_units: Units in dense layer (default: 32)
        dropout_dense: Dropout after dense layer (default: 0.2)
        recurrent_clip_min: Min clip value for recurrent weights (optional)
        recurrent_clip_max: Max clip value for recurrent weights (optional)
        learning_rate: Learning rate for optimizer (default: 0.001)
        optimizer_name: Optimizer to use ('adam', 'adamw', 'rmsprop') (default: 'adam')
        class_weights: Dictionary of class weights for imbalanced data (optional)

    Returns:
        Compiled Keras model

    Example:
        >>> model = build_indrnn_stack(
        ...     sequence_length=96,
        ...     num_features=20,
        ...     num_classes=3
        ... )
        >>> model.summary()
    """

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # ========================================================================
    # Stacked IndRNN Layers (4 layers deep)
    # ========================================================================
    # IndRNN can be stacked much deeper than LSTM/GRU without gradient issues
    # BatchNorm is crucial after each layer for training stability

    # IndRNN Layer 1
    x = IndRNN(
        units=indrnn1_units,
        activation="relu",
        return_sequences=True,
        recurrent_clip_min=recurrent_clip_min,
        recurrent_clip_max=recurrent_clip_max,
        name="indrnn1",
    )(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(dropout_indrnn1, name="dropout1")(x)

    # IndRNN Layer 2
    x = IndRNN(
        units=indrnn2_units,
        activation="relu",
        return_sequences=True,
        recurrent_clip_min=recurrent_clip_min,
        recurrent_clip_max=recurrent_clip_max,
        name="indrnn2",
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(dropout_indrnn2, name="dropout2")(x)

    # IndRNN Layer 3
    x = IndRNN(
        units=indrnn3_units,
        activation="relu",
        return_sequences=True,
        recurrent_clip_min=recurrent_clip_min,
        recurrent_clip_max=recurrent_clip_max,
        name="indrnn3",
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Dropout(dropout_indrnn3, name="dropout3")(x)

    # IndRNN Layer 4 (final, no return_sequences)
    x = IndRNN(
        units=indrnn4_units,
        activation="relu",
        return_sequences=False,
        recurrent_clip_min=recurrent_clip_min,
        recurrent_clip_max=recurrent_clip_max,
        name="indrnn4",
    )(x)
    x = layers.BatchNormalization(name="bn4")(x)

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
    model_name = get_model_name("IndRNN_Stack", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile model
    opt = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


# ============================================================================
# TUNABLE VERSION (With KerasTuner)
# ============================================================================


def build_indrnn_stack_tunable(hp, num_features, num_classes, task_type=TaskType.CLASSIFICATION,
                               sequence_length=None, max_sequence_length=150):
    """
    Build stacked IndRNN model with hyperparameter tuning.

    This version allows KerasTuner to optimize:
    - Sequence length (if not provided as fixed parameter)
    - Number of IndRNN layers (2-6 layers)
    - Units in each layer
    - Dropout rates
    - Dense layer units
    - Learning rate
    - Optimizer choice

    Args:
        hp: KerasTuner HyperParameters object
        num_features: Number of features per timestep (fixed)
        num_classes: Number of output classes (fixed)
        sequence_length: If None, tunes sequence_length (48 to max_sequence_length).
                        If provided, uses this fixed value (for pre-created sequences).
        max_sequence_length: Maximum sequence length (used when sequence_length=None)

    Returns:
        Compiled Keras model with tunable hyperparameters

    Example:
        >>> import keras_tuner
        >>> # Example 1: Tune sequence_length
        >>> def model_builder(hp):
        ...     return build_indrnn_stack_tunable(
        ...         hp=hp,
        ...         num_features=20,
        ...         num_classes=3
        ...     )
        >>> # Example 2: Fixed sequence_length (for pre-created sequences)
        >>> def model_builder(hp):
        ...     return build_indrnn_stack_tunable(
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
        ...     project_name='indrnn_stack'
        ... )
    """

    # Tunable hyperparameters
    if sequence_length is None:
        # Tune sequence_length if not provided
        sequence_length = hp.Int(
            "sequence_length", min_value=48, max_value=max_sequence_length, step=16
        )
    # else: use the provided fixed sequence_length
    num_layers = hp.Int("num_layers", min_value=2, max_value=6)

    # Layer units (can vary per layer)
    units_choices = [64, 128, 256]
    dropout_rate = hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)

    dense_units = hp.Choice("dense_units", values=[16, 32, 64])
    dropout_dense = hp.Float("dropout_dense", min_value=0.1, max_value=0.4, step=0.1)

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # Dynamically stack IndRNN layers
    x = inputs
    for i in range(num_layers):
        layer_units = hp.Choice(f"layer_{i}_units", values=units_choices)
        return_sequences = i < num_layers - 1  # Last layer doesn't return sequences

        x = IndRNN(
            units=layer_units,
            activation="relu",
            return_sequences=return_sequences,
            name=f"indrnn_{i + 1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i + 1}")(x)

        if return_sequences:  # No dropout after last layer
            x = layers.Dropout(dropout_rate, name=f"dropout_{i + 1}")(x)

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
    model = models.Model(inputs=inputs, outputs=outputs, name="IndRNN_Stack_Tunable")

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
HINTS FOR USING IndRNN IN TRADING:
===================================

1. WHAT IS IndRNN AND WHY DOES IT MATTER?

   Traditional RNN/LSTM/GRU:
   - Recurrent connections form a matrix: h_t = σ(W_input * x_t + W_recurrent * h_{t-1} + b)
   - W_recurrent is a matrix (units × units)
   - Gradients flow through matrix multiplications
   - Vanishing/exploding gradients limit depth and sequence length

   IndRNN Innovation:
   - Recurrent connections are INDEPENDENT: h_t = σ(W * x_t + u ⊙ h_{t-1} + b)
   - u is a VECTOR (units,) not a matrix
   - Each neuron has its own recurrent weight
   - ⊙ is element-wise multiplication (Hadamard product)
   - Gradients flow independently per neuron
   - Can stack 20+ layers without vanishing gradients
   - Can process 1000+ timesteps effectively

   Key Benefits:
   - Longer memory: Captures dependencies beyond 100+ timesteps
   - Deeper stacking: 4-8 layers work well (vs 2-3 for LSTM)
   - Better gradients: Independent recurrence = better gradient flow
   - Fewer parameters: Vector vs matrix for recurrence
   - Faster training: Simpler operations than LSTM gates

2. WHEN TO USE IndRNN FOR TRADING:

   Ideal scenarios:
   - Long-term trend following (multi-day, multi-week patterns)
   - Regime persistence detection (market states that last)
   - Seasonal patterns (weekly, monthly cycles)
   - Long memory requirements (100+ timesteps)
   - Deep feature hierarchies (4+ layers needed)

   Examples:
   - Detecting trend reversals that develop over days
   - Identifying market regimes that persist (bull/bear markets)
   - Capturing weekly patterns in forex (Monday effect, Friday close)
   - Long-term momentum strategies (position trading)
   - Macro trend analysis (economic cycles)

   Not ideal for:
   - Very short sequences (< 32 timesteps) - overkill
   - When you need attention mechanisms - use Transformer
   - When bidirectional context is critical - use BiLSTM

3. SEQUENCE LENGTH SELECTION:

   IndRNN excels at LONG sequences:
   - 96 timesteps (8 hours of 5-min bars): Good starting point
   - 144 timesteps (12 hours): Captures full trading session
   - 288 timesteps (24 hours): Full day patterns
   - 500+ timesteps: Multi-day patterns (IndRNN can handle this!)

   For daily bars:
   - 20 timesteps = 1 month
   - 60 timesteps = 3 months (quarter)
   - 120 timesteps = 6 months
   - 252 timesteps = 1 year

   Tip: Start with 96-144 timesteps, then increase if needed.
   IndRNN won't suffer from vanishing gradients like LSTM would.

4. STACKING DEPTH:

   IndRNN can be stacked VERY deep:
   - 2 layers: Basic (like LSTM)
   - 4 layers: Recommended (default in this implementation)
   - 6 layers: Good for complex patterns
   - 8+ layers: Possible but diminishing returns

   Each layer learns different temporal abstractions:
   - Layer 1: Low-level patterns (local trends, reversals)
   - Layer 2: Medium-term patterns (intraday cycles)
   - Layer 3: Higher-level patterns (daily trends)
   - Layer 4: Abstract patterns (regime states)

5. BATCHNORM IS CRITICAL:

   ALWAYS use BatchNorm after each IndRNN layer:
   - Stabilizes training in deep stacks
   - Prevents internal covariate shift
   - Enables higher learning rates
   - Required for 4+ layer stacks

   Without BatchNorm:
   - Training becomes unstable
   - Gradients explode/vanish
   - Model fails to converge

6. RECURRENT WEIGHT INITIALIZATION:

   Critical for long sequences:

   For sequence_length T, recurrent weights should satisfy:
   - |u_i| < 2^(1/T) for stability

   Examples:
   - T=100: |u_i| < 1.007
   - T=500: |u_i| < 1.0014

   Implementation:
   ```python
   # For very long sequences, clip recurrent weights
   model = build_indrnn_stack(
       sequence_length=500,
       recurrent_clip_min=0,
       recurrent_clip_max=2**(1/500),  # ~1.0014
       ...
   )
   ```

   Default initialization (uniform [0, 1]) works for T < 200.

7. FEATURE ENGINEERING FOR LONG-TERM PATTERNS:

   Focus on features that capture persistence:
   ```python
   def create_persistence_features(df):
       # Trend strength (long-term)
       df['trend_20'] = df['close'].rolling(20).mean()
       df['trend_60'] = df['close'].rolling(60).mean()
       df['trend_strength'] = (df['close'] - df['trend_60']) / df['trend_60']

       # Regime indicators
       df['volatility_regime'] = df['returns'].rolling(60).std() / df['returns'].rolling(252).std()
       df['volume_regime'] = df['volume'].rolling(20).mean() / df['volume'].rolling(60).mean()

       # Momentum (multi-timeframe)
       df['momentum_5'] = df['close'].pct_change(5)
       df['momentum_20'] = df['close'].pct_change(20)
       df['momentum_60'] = df['close'].pct_change(60)

       # Trend persistence
       df['up_days'] = (df['close'] > df['close'].shift(1)).rolling(20).sum()
       df['trend_persistence'] = df['up_days'] / 20

       return df
   ```

8. TRAINING TIPS:

   Gradient clipping is essential:
   - Always use clipnorm=1.0 (default in this implementation)
   - Prevents gradient explosion in deep stacks

   Learning rate:
   - Start with 0.001 (default)
   - Use ReduceLROnPlateau if training stalls
   - Lower to 0.0001 for very deep stacks (6+ layers)

   Batch size:
   - 32-64: Standard, good for most cases
   - 16: For very long sequences (500+) to fit in memory
   - 128: For shorter sequences with lots of data

   Early stopping:
   - Patience=15-20 (longer than LSTM due to deeper stacks)
   - Monitor val_loss

   Example:
   ```python
   from keras.callbacks import EarlyStopping, ReduceLROnPlateau

   callbacks = [
       EarlyStopping(patience=20, restore_best_weights=True),
       ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6)
   ]

   history = model.fit(
       X_train, y_train,
       validation_data=(X_val, y_val),
       epochs=150,  # More epochs for deep stacks
       batch_size=32,
       callbacks=callbacks,
       verbose=1
   )
   ```

9. HANDLING CLASS IMBALANCE:

   Same strategies as other models, but consider:
   - Focal loss for severe imbalance (custom implementation)
   - Class weights (provided)
   - SMOTE oversampling (for minority classes)

   Long sequences + imbalance:
   ```python
   from imblearn.over_sampling import SMOTE

   # Reshape for SMOTE
   n_samples, seq_len, n_features = X_train.shape
   X_train_2d = X_train.reshape(n_samples, seq_len * n_features)

   # Apply SMOTE
   smote = SMOTE(random_state=42)
   X_train_balanced, y_train_balanced = smote.fit_resample(X_train_2d, y_train)

   # Reshape back
   X_train_balanced = X_train_balanced.reshape(-1, seq_len, n_features)
   ```

10. WALK-FORWARD ANALYSIS INTEGRATION:

    ```python
    def feature_engineering_fn(train_data, test_data, train_labels, test_labels):
        from sklearn.preprocessing import RobustScaler

        # Scale
        scaler = RobustScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Create sequences (longer for IndRNN)
        sequence_length = 96  # Or 144, 288 for longer patterns

        def create_sequences(data, labels, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:i+seq_len])
                y.append(labels[i+seq_len])
            return np.array(X, dtype=np.float32), np.array(y)

        X_train, y_train = create_sequences(train_scaled, train_labels, sequence_length)
        X_test, y_test = create_sequences(test_scaled, test_labels, sequence_length)

        return X_train, X_test, y_train, y_test
    ```

11. COMPARING IndRNN TO LSTM/GRU:

    Parameter comparison (for 128 units, 20 input features):
    - LSTM: ~66K parameters per layer (4 gates × weights)
    - GRU: ~50K parameters per layer (3 gates × weights)
    - IndRNN: ~3K parameters per layer (no gates, vector recurrence)

    Speed comparison:
    - IndRNN: ~1.5-2x faster than LSTM
    - IndRNN: ~1.2-1.5x faster than GRU
    - Speedup increases with sequence length

    Memory comparison:
    - IndRNN: Similar to GRU
    - LSTM: Highest memory (cell state + hidden state)

    Depth comparison:
    - IndRNN: 4-8 layers work well
    - LSTM/GRU: 2-3 layers typically max

    Sequence length:
    - IndRNN: 100-1000+ timesteps
    - LSTM: 50-200 timesteps (gradients degrade)
    - GRU: 50-300 timesteps

12. DEBUGGING TIPS:

    If loss doesn't decrease:
    - Check recurrent weight initialization (especially for T > 200)
    - Ensure BatchNorm is after EVERY layer
    - Reduce learning rate
    - Add gradient clipping (clipnorm=1.0)

    If loss explodes:
    - Add/reduce recurrent_clip_max
    - Lower learning rate
    - Check for NaN in input data
    - Increase BatchNorm momentum

    If overfitting:
    - Increase dropout (0.3-0.5)
    - Reduce number of layers
    - Add L2 regularization to Dense layers
    - Use more training data

    If underfitting:
    - Increase model depth (4→6 layers)
    - Increase sequence_length
    - Reduce dropout
    - Train for more epochs

13. ADVANCED TECHNIQUES:

    Bidirectional IndRNN (for non-causal tasks):
    ```python
    from keras.layers import Bidirectional

    # Note: Uses twice the parameters
    x = Bidirectional(IndRNN(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    ```

    Residual connections (for VERY deep stacks):
    ```python
    # Layer 1
    x1 = IndRNN(128, return_sequences=True)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)

    # Layer 2 with residual
    x2 = IndRNN(128, return_sequences=True)(x1)
    x2 = BatchNormalization()(x2)
    x2 = Add()([x1, x2])  # Residual connection
    x2 = Dropout(0.3)(x2)
    ```

    Hierarchical IndRNN (multi-scale):
    ```python
    # Process at different temporal resolutions
    # Branch 1: High-res (all timesteps)
    x1 = IndRNN(64, return_sequences=True)(inputs)
    x1 = GlobalAveragePooling1D()(x1)

    # Branch 2: Low-res (downsample)
    x2 = inputs[:, ::5, :]  # Every 5th timestep
    x2 = IndRNN(64, return_sequences=True)(x2)
    x2 = GlobalAveragePooling1D()(x2)

    # Concatenate
    x = Concatenate()([x1, x2])
    ```

14. MODEL INTERPRETATION:

    Analyzing recurrent weights:
    ```python
    # Get recurrent weights from each layer
    for layer in model.layers:
        if isinstance(layer, IndRNN):
            recurrent_weights = layer.cell.recurrent_kernel.numpy()
            print(f"{layer.name} recurrent weights:")
            print(f"  Mean: {recurrent_weights.mean():.4f}")
            print(f"  Std: {recurrent_weights.std():.4f}")
            print(f"  Min: {recurrent_weights.min():.4f}")
            print(f"  Max: {recurrent_weights.max():.4f}")

            # Visualize distribution
            import matplotlib.pyplot as plt
            plt.hist(recurrent_weights, bins=50)
            plt.title(f"{layer.name} Recurrent Weight Distribution")
            plt.show()
    ```

    Weights close to 1 indicate long memory neurons.
    Weights close to 0 indicate short memory neurons.

15. PRODUCTION DEPLOYMENT:

    Model saving/loading:
    ```python
    # Save
    model.save('indrnn_model.keras')

    # Load (custom objects needed)
    from keras.models import load_model
    model = load_model(
        'indrnn_model.keras',
        custom_objects={'IndRNN': IndRNN, 'IndRNNCell': IndRNNCell}
    )
    ```

    TensorFlow Lite conversion (for edge):
    ```python
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open('indrnn_model.tflite', 'wb') as f:
        f.write(tflite_model)
    ```

    Note: Custom layers may not be fully supported in TFLite.
    Test thoroughly before production deployment.
"""
