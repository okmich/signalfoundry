"""
Bidirectional LSTM with Attention - Model Factory
==================================================

Architecture:
-------------
Input (32-100 timesteps, n_features)
    ↓
Bidirectional LSTM (128 units) → Dropout (0.3)
    ↓
Bidirectional LSTM (64 units) → Dropout (0.3)
    ↓
Attention Layer (temporal weights)
    ↓
Dense (32, ReLU) → Dropout (0.2)
    ↓
Output (softmax for multi-class)

Key Features:
-------------
- Captures both forward and backward temporal dependencies
- Attention mechanism focuses on critical timeframes
- Suitable for: trend classification, regime detection
- Handles class imbalance with class weights and robust metrics

Usage:
------
1. Simple version (fixed hyperparameters):
   model = build_bilstm_attention(
       sequence_length=48,
       num_features=20,
       num_classes=3
   )

2. Tunable version (with KerasTuner):
   def model_builder(hp):
       return build_bilstm_attention_tunable(
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
Tsantekidis et al. (2017) "Algorithmic Trading Using Deep Neural Networks on High Frequency Data" - (https://arxiv.org/abs/1710.03870)
"""

from keras import layers, models

# Import the existing LightweightAttention layer from the main codebase
from ..layers import LightweightAttention

# Import task type and common utilities
from .common import TaskType, create_output_layer_and_loss, get_optimizer, get_model_name


# ============================================================================
# SIMPLE VERSION (Fixed Hyperparameters)
# ============================================================================


def build_bilstm_attention(sequence_length=48, num_features=20, num_classes=3, task_type=TaskType.CLASSIFICATION,
                           lstm1_units=128, lstm2_units=64, dropout_lstm=0.3,
                           attention_type="bahdanau", attention_heads=4, dense_units=32, dropout_dense=0.2,
                           learning_rate=0.001, optimizer_name="adam", class_weights=None):
    """
    Build Bidirectional LSTM with Attention model (fixed hyperparameters).

    Args:
        sequence_length: Number of timesteps in input sequences
        num_features: Number of features per timestep
        num_classes: Number of output classes
        lstm1_units: Units in first BiLSTM layer (default: 128)
        lstm2_units: Units in second BiLSTM layer (default: 64)
        dropout_lstm: Dropout rate after LSTM layers (default: 0.3)
        attention_type: Type of attention ('bahdanau' or 'dot') (default: 'bahdanau')
        attention_heads: Number of attention heads for dot-product attention (default: 4)
        dense_units: Units in dense layer (default: 32)
        dropout_dense: Dropout rate after dense layer (default: 0.2)
        learning_rate: Learning rate for optimizer (default: 0.001)
        optimizer_name: Optimizer to use ('adam', 'adamw', 'rmsprop') (default: 'adam')
        class_weights: Dictionary of class weights for imbalanced data (optional)

    Returns:
        Compiled Keras model

    Example:
        >>> model = build_bilstm_attention(
        ...     sequence_length=48,
        ...     num_features=20,
        ...     num_classes=3,
        ...     attention_type='bahdanau'
        ... )
        >>> model.summary()
    """

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # First LSTM layer
    x = layers.LSTM(lstm1_units, return_sequences=True, name="lstm1")(inputs)
    x = layers.Dropout(dropout_lstm, name="dropout1")(x)

    # Second LSTM layer
    x = layers.LSTM(lstm2_units, return_sequences=True, name="lstm2")(x)
    x = layers.Dropout(dropout_lstm, name="dropout2")(x)

    # Attention layer (using LightweightAttention from main codebase)
    x = LightweightAttention(attn_type=attention_type, heads=attention_heads, return_attention_scores=False,
                             name="lightweight_attention")(x)

    # Dense layer
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_dense, name="dropout3")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("LSTM_Attention", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile model
    opt = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


# ============================================================================
# TUNABLE VERSION (With KerasTuner)
# ============================================================================


def build_bilstm_attention_tunable(hp, num_features, num_classes, task_type=TaskType.CLASSIFICATION,
                                   sequence_length=None, max_sequence_length=100):
    """
    Build Bidirectional LSTM with Attention model with hyperparameter tuning.

    This version allows KerasTuner to optimize:
    - Sequence length (if not provided as fixed parameter)
    - LSTM units in each layer
    - Dropout rates
    - Attention mechanism type (bahdanau or dot-product)
    - Attention heads (for dot-product attention)
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
        ...     return build_bilstm_attention_tunable(
        ...         hp=hp,
        ...         num_features=20,
        ...         num_classes=3
        ...     )
        >>> # Example 2: Fixed sequence_length (for pre-created sequences)
        >>> def model_builder(hp):
        ...     return build_bilstm_attention_tunable(
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
        ...     project_name='bilstm_attention'
        ... )
    """

    # Tunable hyperparameters
    if sequence_length is None:
        # Tune sequence_length if not provided
        sequence_length = hp.Int(
            "sequence_length", min_value=32, max_value=max_sequence_length, step=16
        )
    # else: use the provided fixed sequence_length
    lstm1_units = hp.Choice("lstm1_units", values=[64, 128, 256])
    lstm2_units = hp.Choice("lstm2_units", values=[32, 64, 128])
    dropout_lstm = hp.Float("dropout_lstm", min_value=0.2, max_value=0.5, step=0.1)
    attention_type = hp.Choice("attention_type", values=["bahdanau", "dot"])
    attention_heads = hp.Choice("attention_heads", values=[2, 4, 8])
    dense_units = hp.Choice("dense_units", values=[16, 32, 64])
    dropout_dense = hp.Float("dropout_dense", min_value=0.1, max_value=0.4, step=0.1)

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # First LSTM layer
    x = layers.LSTM(lstm1_units, return_sequences=True, name="lstm1")(inputs)
    x = layers.Dropout(dropout_lstm, name="dropout1")(x)

    # Second LSTM layer
    x = layers.LSTM(lstm2_units, return_sequences=True, name="lstm2")(x)
    x = layers.Dropout(dropout_lstm, name="dropout2")(x)

    # Attention layer (using LightweightAttention from main codebase)
    x = LightweightAttention(attn_type=attention_type, heads=attention_heads, return_attention_scores=False,
                             name="lightweight_attention")(x)

    # Dense layer
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_dense, name="dropout3")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("LSTM_Attention_Tunable", task_type)
    model = models.Model(
        inputs=inputs, outputs=outputs, name=model_name
    )

    # Tunable optimizer
    optimizer_name = hp.Choice("optimizer", values=["adam", "adamw", "rmsprop"])
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
    )

    opt = get_optimizer(optimizer_name, learning_rate)

    # Compile model
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


# ============================================================================
# HINTS AND BEST PRACTICES
# ============================================================================

"""
HINTS FOR USING THIS MODEL IN TRADING:
======================================

1. SEQUENCE LENGTH SELECTION:
   - For 5-min bars:
     * 48 timesteps = 4 hours
     * 96 timesteps = 8 hours
     * 288 timesteps = 24 hours (1 day)
   - Tune based on your trading timeframe and market characteristics

2. HANDLING CLASS IMBALANCE:
   - Use class_weight parameter in model.fit()
   - Compute with: sklearn.utils.class_weight.compute_class_weight
   - Consider oversampling/undersampling before feeding to model
   - Focus on F1-score, balanced accuracy, or Cohen's Kappa

3. FEATURE ENGINEERING:
   - Normalize/standardize features BEFORE creating sequences
   - Use StandardScaler or MinMaxScaler fitted on training data only
   - Include both price-based and volume-based features
   - Consider adding lagged features

4. WALK-FORWARD ANALYSIS INTEGRATION:
   - The model expects 3D input: (batch_size, sequence_length, num_features)
   - Your feature engineering function should create sequences
   - Example:
     ```python
     def feature_engineering_fn(train_data, test_data, train_labels, test_labels):
         # Scale features
         scaler = StandardScaler()
         train_scaled = scaler.fit_transform(train_data)
         test_scaled = scaler.transform(test_data)

         # Create sequences
         def create_sequences(data, labels, seq_len):
             X, y = [], []
             for i in range(len(data) - seq_len):
                 X.append(data[i:i+seq_len])
                 y.append(labels[i+seq_len])
             return np.array(X), np.array(y)

         X_train, y_train = create_sequences(train_scaled, train_labels, sequence_length)
         X_test, y_test = create_sequences(test_scaled, test_labels, sequence_length)

         return X_train, X_test, y_train, y_test
     ```

5. ATTENTION WEIGHTS VISUALIZATION:
   - Set return_attention_scores=True in LightweightAttention
   - For Bahdanau: returns (context, attention_weights) where weights is (batch, timesteps)
   - For dot-product: returns (context, attention_weights) where weights is (batch, heads, timesteps, timesteps)
   - Visualize which timesteps the model focuses on
   - Useful for model interpretability and debugging
   - Attention types: 'bahdanau' (additive) or 'dot' (multi-head scaled dot-product)

6. REGULARIZATION:
   - Adjust dropout rates if overfitting occurs
   - Consider L2 regularization on Dense layers
   - Use early stopping with patience=5-10

7. TRAINING TIPS:
   - Start with learning_rate=0.001
   - Use ReduceLROnPlateau callback if loss plateaus
   - Monitor validation metrics, not just training metrics
   - Save best model using ModelCheckpoint callback

8. COMMON PITFALLS:
   - Data leakage: Never fit scaler on test data
   - Look-ahead bias: Ensure labels align with prediction time
   - Sequence creation: Account for sequence_length offset in labels
   - Memory: Large sequence_length × num_features can be memory-intensive

9. PERFORMANCE OPTIMIZATION:
   - Use mixed precision training for faster computation
   - Consider using CuDNN LSTM for GPU acceleration
   - Batch size: 32-128 typically works well
   - Gradient clipping if training is unstable

10. ENSEMBLE STRATEGIES:
    - Train multiple models with different random seeds
    - Combine with other architectures (TCN, Transformer)
    - Use ensemble voting/averaging for final predictions
"""
