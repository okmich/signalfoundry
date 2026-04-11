"""
GRU with Multi-Head Self-Attention (Transformer-Lite) - Model Factory
======================================================================

Architecture:
-------------
Input (32-100 timesteps, n_features)
    ↓
GRU (128 units, return_sequences=True) → Dropout (0.3)
    ↓
Multi-Head Attention (4 heads, key_dim=32)
    ↓
GRU (64 units) → Dropout (0.3)
    ↓
Dense (32, ReLU) → Dropout (0.2)
    ↓
Output (softmax for multi-class)

Key Features:
-------------
- Lighter than full Transformer (no positional encoding complexity)
- Multi-head attention highlights important timesteps across the sequence
- GRU provides efficient sequential processing with fewer parameters than LSTM
- Suitable for: money flow analysis, volume-price divergence detection
- Handles class imbalance with class weights and robust metrics

Usage:
------
1. Simple version (fixed hyperparameters):
   model = build_gru_multihead_attention(
       sequence_length=48,
       num_features=20,
       num_classes=3
   )

2. Tunable version (with KerasTuner):
   def model_builder(hp):
       return build_gru_multihead_attention_tunable(
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
"Enhancing Time Series Momentum Strategies Using Deep Neural Networks" - Fischer & Krauss (2018)
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3084950
"""

from keras import layers, models

# Import task type and common utilities
from .common import TaskType, create_output_layer_and_loss, get_optimizer, get_model_name


# ============================================================================
# SIMPLE VERSION (Fixed Hyperparameters)
# ============================================================================


def build_gru_multihead_attention(sequence_length=48, num_features=20, num_classes=3, task_type=TaskType.CLASSIFICATION,
                                  gru1_units=128, gru2_units=64, dropout_gru=0.3, num_heads=4, key_dim=32,
                                  attention_dropout=0.1, dense_units=32, dropout_dense=0.2, learning_rate=0.001,
                                  optimizer_name="adam", class_weights=None):
    """
    Build GRU with Multi-Head Self-Attention model (fixed hyperparameters).

    This model combines the efficiency of GRU with the attention mechanism
    from Transformers, creating a "Transformer-Lite" architecture. It's
    particularly effective for capturing both sequential dependencies and
    important temporal patterns in financial time series.

    Args:
        sequence_length: Number of timesteps in input sequences
        num_features: Number of features per timestep
        num_classes: Number of output classes
        gru1_units: Units in first GRU layer (default: 128)
        gru2_units: Units in second GRU layer (default: 64)
        dropout_gru: Dropout rate after GRU layers (default: 0.3)
        num_heads: Number of attention heads (default: 4)
        key_dim: Dimensionality of linearity for keys and queries (default: 32)
        attention_dropout: Dropout rate in attention mechanism (default: 0.1)
        dense_units: Units in dense layer (default: 32)
        dropout_dense: Dropout rate after dense layer (default: 0.2)
        learning_rate: Learning rate for optimizer (default: 0.001)
        optimizer_name: Optimizer to use ('adam', 'adamw', 'rmsprop') (default: 'adam')
        class_weights: Dictionary of class weights for imbalanced data (optional)

    Returns:
        Compiled Keras model

    Example:
        >>> model = build_gru_multihead_attention(
        ...     sequence_length=48,
        ...     num_features=20,
        ...     num_classes=3,
        ...     num_heads=4,
        ...     key_dim=32
        ... )
        >>> model.summary()
    """

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # First GRU layer (must return sequences for attention)
    x = layers.GRU(gru1_units, return_sequences=True, name="gru1")(inputs)
    x = layers.Dropout(dropout_gru, name="dropout1")(x)

    # Multi-Head Self-Attention
    # Query, Key, and Value are all the same (self-attention)
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=attention_dropout,
        name="multihead_attention",
    )(
        x, x
    )  # Self-attention: query=x, key=x, value=x

    # Add & Norm (residual connection)
    x = layers.Add(name="add_residual")([x, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6, name="layer_norm")(x)

    # Second GRU layer (returns only last output)
    x = layers.GRU(gru2_units, return_sequences=False, name="gru2")(x)
    x = layers.Dropout(dropout_gru, name="dropout2")(x)

    # Dense layer
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_dense, name="dropout3")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("GRU_MultiHeadAttention", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile model
    opt = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


# ============================================================================
# TUNABLE VERSION (With KerasTuner)
# ============================================================================

def build_gru_multihead_attention_tunable(hp, num_features, num_classes, task_type=TaskType.CLASSIFICATION,
                                          sequence_length=None, max_sequence_length=100):
    """
    Build GRU with Multi-Head Self-Attention model with hyperparameter tuning.

    This version allows KerasTuner to optimize:
    - Sequence length (if not provided as fixed parameter)
    - GRU units in each layer
    - Dropout rates
    - Number of attention heads
    - Key dimensionality for attention
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
        ...     return build_gru_multihead_attention_tunable(
        ...         hp=hp,
        ...         num_features=20,
        ...         num_classes=3
        ...     )
        >>> # Example 2: Fixed sequence_length (for pre-created sequences)
        >>> def model_builder(hp):
        ...     return build_gru_multihead_attention_tunable(
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
        ...     project_name='gru_multihead_attention'
        ... )
    """

    # Tunable hyperparameters
    if sequence_length is None:
        # Tune sequence_length if not provided
        sequence_length = hp.Int(
            "sequence_length", min_value=32, max_value=max_sequence_length, step=16
        )
    # else: use the provided fixed sequence_length
    gru1_units = hp.Choice("gru1_units", values=[64, 128, 256])
    gru2_units = hp.Choice("gru2_units", values=[32, 64, 128])
    dropout_gru = hp.Float("dropout_gru", min_value=0.2, max_value=0.5, step=0.1)
    num_heads = hp.Choice("num_heads", values=[2, 4, 8])
    key_dim = hp.Choice("key_dim", values=[16, 32, 64])
    attention_dropout = hp.Float(
        "attention_dropout", min_value=0.0, max_value=0.3, step=0.1
    )
    dense_units = hp.Choice("dense_units", values=[16, 32, 64])
    dropout_dense = hp.Float("dropout_dense", min_value=0.1, max_value=0.4, step=0.1)

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # First GRU layer (must return sequences for attention)
    x = layers.GRU(gru1_units, return_sequences=True, name="gru1")(inputs)
    x = layers.Dropout(dropout_gru, name="dropout1")(x)

    # Multi-Head Self-Attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=attention_dropout,
        name="multihead_attention",
    )(
        x, x
    )  # Self-attention: query=x, key=x, value=x

    # Add & Norm (residual connection)
    x = layers.Add(name="add_residual")([x, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6, name="layer_norm")(x)

    # Second GRU layer (returns only last output)
    x = layers.GRU(gru2_units, return_sequences=False, name="gru2")(x)
    x = layers.Dropout(dropout_gru, name="dropout2")(x)

    # Dense layer
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_dense, name="dropout3")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("GRU_MultiHeadAttention_Tunable", task_type)
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

1. ARCHITECTURE BENEFITS:
   - GRU is computationally lighter than LSTM (fewer parameters, faster training)
   - Multi-head attention captures different aspects of temporal patterns
   - No positional encoding needed (GRU already captures position)
   - Residual connection helps gradient flow and model stability

2. WHEN TO USE THIS MODEL:
   - Money flow analysis: detecting accumulation/distribution patterns
   - Volume-price divergence: attention can highlight volume anomalies
   - Multi-timeframe patterns: attention heads can focus on different scales
   - Order flow imbalance detection
   - Liquidity regime changes

3. SEQUENCE LENGTH SELECTION:
   - For 5-min bars:
     * 48 timesteps = 4 hours (intraday patterns)
     * 96 timesteps = 8 hours (full trading session)
     * 144 timesteps = 12 hours (extended patterns)
   - For volume/money flow: longer sequences capture accumulation phases
   - Use tunable version to optimize sequence length for your data

4. ATTENTION MECHANISM INSIGHTS:
   - num_heads=4: Good starting point for most tasks
   - num_heads=8: Better for complex multi-scale patterns
   - key_dim: Controls capacity of attention (32 is balanced, 64 for complex)
   - attention_dropout: Prevents overfitting in attention weights

5. FEATURE ENGINEERING FOR VOLUME/MONEY FLOW:
   Essential features to include:
   - Volume-related: volume, volume_ma, volume_std, relative_volume
   - Money flow: mfi (Money Flow Index), obv (On-Balance Volume)
   - Price-volume: vwap, price_volume_trend
   - Divergence indicators: rsi_divergence, macd_histogram
   - Order flow: bid_ask_spread, market_depth (if available)

   Example feature engineering:
   ```python
   def create_volume_features(df):
       df['volume_ma'] = df['volume'].rolling(20).mean()
       df['relative_volume'] = df['volume'] / df['volume_ma']
       df['price_volume_trend'] = (df['close'].pct_change() * df['volume']).cumsum()
       df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
       return df
   ```

6. HANDLING CLASS IMBALANCE:
   - Use class_weight parameter in model.fit()
   - Compute with: sklearn.utils.class_weight.compute_class_weight
   - For severe imbalance (>10:1), consider SMOTE or undersampling
   - Monitor precision/recall per class, not just accuracy
   - Use stratified splitting to maintain class distribution

7. TRAINING TIPS:
   - Start with learning_rate=0.001
   - Use ReduceLROnPlateau callback (patience=5, factor=0.5)
   - Early stopping with patience=10 (monitor val_loss)
   - Gradient clipping (clipnorm=1.0) prevents exploding gradients
   - Batch size: 32-64 for most cases, 128 for large datasets

   Example training setup:
   ```python
   from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

   callbacks = [
       EarlyStopping(patience=10, restore_best_weights=True),
       ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6),
       ModelCheckpoint('best_model.keras', save_best_only=True)
   ]

   history = model.fit(
       X_train, y_train,
       validation_data=(X_val, y_val),
       epochs=100,
       batch_size=32,
       class_weight=class_weights,
       callbacks=callbacks
   )
   ```

8. INTERPRETING ATTENTION WEIGHTS:
   To visualize what the model focuses on:
   ```python
   # Create a model that outputs attention weights
   from keras import Model

   attention_layer = model.get_layer('multihead_attention')
   attention_model = Model(
       inputs=model.input,
       outputs=[model.output, attention_layer.output]
   )

   # Get predictions and attention
   predictions, attention_output = attention_model.predict(X_test)

   # Attention weights show which timesteps are important
   # Useful for: debugging, feature importance, model interpretability
   ```

9. WALK-FORWARD ANALYSIS INTEGRATION:
   ```python
   def feature_engineering_fn(train_data, test_data, train_labels, test_labels):
       from sklearn.preprocessing import RobustScaler  # Better for financial data

       # Use RobustScaler (robust to outliers)
       scaler = RobustScaler()
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

10. COMMON PITFALLS:
    - Attention requires return_sequences=True in first GRU
    - num_heads must divide key_dim evenly (e.g., key_dim=32, num_heads=4)
    - Don't use too many heads (>8) without sufficient data
    - LayerNormalization is crucial for training stability
    - Residual connection prevents attention from dominating
    - GRU is sensitive to feature scaling - always normalize

11. PERFORMANCE OPTIMIZATION:
    - Mixed precision training: significant speedup on modern GPUs
    - CuDNN GRU: automatic on GPU with proper conditions
    - Reduce sequence_length if training is slow
    - Use smaller key_dim (16) for faster attention computation
    - Profile with TensorBoard to identify bottlenecks

    Example mixed precision:
    ```python
    from keras import mixed_precision

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # Build model (will use mixed precision automatically)
    model = build_gru_multihead_attention(...)
    ```

12. ENSEMBLE STRATEGIES:
    - Combine with TCN (for different temporal modeling)
    - Combine with CNN (for local pattern detection)
    - Train multiple models with different attention configurations
    - Use voting or averaging for final predictions
    - Stack predictions as features for meta-learner

13. DEBUGGING TIPS:
    - If loss doesn't decrease: check learning rate, normalization
    - If overfitting: increase dropout, add L2 regularization
    - If unstable training: reduce learning rate, use gradient clipping
    - If attention weights are uniform: increase key_dim or num_heads
    - Check attention output shape: should be (batch, seq_len, gru1_units*2)

14. MODEL VARIANTS TO CONSIDER:
    - Add more GRU layers (3-4 stacks) for deeper models
    - Use Bidirectional GRU for non-causal tasks
    - Add multiple attention layers (Transformer-style)
    - Combine with CNN layers before GRU (extract local features)
    - Use cross-attention with external features (e.g., market regime)

15. EVALUATION METRICS FOR IMBALANCED CLASSES:
    - Per-class precision, recall, F1-score
    - Macro-averaged F1 (treats all classes equally)
    - Cohen's Kappa (accounts for chance agreement)
    - Matthews Correlation Coefficient (robust to imbalance)
    - Confusion matrix analysis

    Example custom metric:
    ```python
    from sklearn.metrics import classification_report, cohen_kappa_score

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(classification_report(y_test, y_pred_classes))
    print(f"Cohen's Kappa: {cohen_kappa_score(y_test, y_pred_classes):.4f}")
    ```
"""
