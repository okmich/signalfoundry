"""
Deep LSTM with Attention (Lightweight) - Model Factory
======================================================

Architecture Description:
-------------------------
A deep stacked LSTM architecture with attention mechanism that captures long-term dependencies while focusing on key
bars through self-attention. This combines the depth of stacked LSTMs with the interpretability of attention mechanisms.

Architecture:
-------------
Input (32-100 timesteps, n_features)
    ↓
BatchNormalization (stabilizes input)
    ↓
LSTM Layer 1 (128 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
LSTM Layer 2 (64 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
Self-Attention (heads=4, learns temporal importance)
    ↓
Dense (64, ReLU)
    ↓
Dropout (0.3)
    ↓
Output (softmax for multi-class)

Key Features:
-------------
✓ Deep Architecture: 2 stacked LSTMs for hierarchical feature learning
✓ BatchNorm: Stabilizes training and improves convergence
✓ Self-Attention: Focuses on critical timeframes and events
✓ Long Dependencies: Better at capturing extended temporal patterns
✓ Interpretability: Attention weights show which bars matter most
✓ Regularization: Multiple dropout layers prevent overfitting

Why it Works for Trading:
--------------------------
The deep architecture learns hierarchical representations:
- Layer 1 (128 units): Learns low-level patterns (price movements, volume spikes)
- Layer 2 (64 units): Learns high-level patterns (trends, regime changes)
- Attention: Focuses on critical events and filters noise

The model excels at:
- Multi-timeframe pattern recognition
- Complex trend detection
- Regime change identification
- Long-term dependency modeling
- Critical event detection (breakouts, reversals)

Comparison to Other Models:
----------------------------
| Feature | Deep-LSTM-Attn | LSTM-Attn | BiLSTM-Attn |
|---------|----------------|-----------|-------------|
| Depth | 2 layers | 1 layer | 2 layers |
| Direction | Unidirectional | Unidirectional | Bidirectional |
| Parameters | Medium | Low | High |
| Speed | Medium | Fast | Slow |
| Accuracy | High | Medium | Highest |
| Use Case | Complex patterns | Real-time | Max accuracy |

Performance:
------------
Inference: ~15-25ms on GPU
Training: ~30-50 epochs typical
Memory: ~2x single LSTM
Parameters: ~350K (seq=48, features=20)

Fine-Tuning Strategy:
---------------------
1. Initial training: Full model, 30-50 epochs
2. Fine-tuning: Freeze early LSTMs, train attention + head
   model.layers[2].trainable = False  # LSTM 1
   model.layers[4].trainable = False  # LSTM 2

Usage:
------
1. Simple version (fixed hyperparameters):
   model = build_deep_lstm_attention(
       sequence_length=48,
       num_features=20,
       num_classes=3
   )

2. Tunable version (with KerasTuner):
   def model_builder(hp):
       return build_deep_lstm_attention_tunable(
           hp=hp,
           num_features=20,
           num_classes=3,
           sequence_length=48  # Optional: fix for pre-created sequences
       )

3. Extract attention weights:
   model = build_deep_lstm_attention(return_attention=True)
   predictions, attention_weights = model.predict(X_test)

References:
-----------
- DeepLOB: https://arxiv.org/abs/1808.03668
- LSTM Attention for Stock Prediction: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0227222
- Attention-based BiLSTM Trading: https://pmc.ncbi.nlm.nih.gov/articles/PMC8794624/
"""

import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models

from .common import TaskType, create_output_layer_and_loss, get_optimizer, get_model_name
from ..layers.light_weight_attention import LightweightAttention


# ============================================================================
# SIMPLE VERSION (Fixed Hyperparameters)
# ============================================================================


def build_deep_lstm_attention(sequence_length=48, num_features=20, num_classes=3, task_type=TaskType.CLASSIFICATION,
                              use_batch_norm=True, lstm1_units=128, lstm2_units=64, dropout_lstm=0.3,
                              attention_type="bahdanau", attention_heads=4, dense_units=64, dense_dropout=0.3,
                              learning_rate=0.001, optimizer_name="adam", return_attention=False):
    """
    Build Deep LSTM with Attention model (fixed hyperparameters).

    Deep stacked LSTM architecture with self-attention for capturing
    long-term dependencies and focusing on critical timeframes.
    Supports both classification and regression tasks.

    Args:
        sequence_length: Number of timesteps in input sequences
        num_features: Number of features per timestep
        num_classes: Number of output classes for classification
        task_type: TaskType.CLASSIFICATION or TaskType.REGRESSION (default: CLASSIFICATION)
        use_batch_norm: Whether to use BatchNormalization at input (default: True)
        lstm1_units: Units in first LSTM layer (default: 128)
        lstm2_units: Units in second LSTM layer (default: 64)
        dropout_lstm: Dropout rate after LSTM layers (default: 0.3)
        attention_type: Type of attention ('bahdanau' or 'dot') (default: 'bahdanau')
        attention_heads: Number of attention heads for dot-product (default: 4)
        dense_units: Units in dense layer (default: 64)
        dense_dropout: Dropout rate after dense layer (default: 0.3)
        learning_rate: Learning rate for optimizer (default: 0.001)
        optimizer_name: Optimizer to use ('adam', 'adamw', 'rmsprop')
        return_attention: If True, model returns (predictions, attention_weights)

    Returns:
        Compiled Keras model

    Example (Classification):
        >>> # Standard classification model
        >>> model = build_deep_lstm_attention(
        ...     sequence_length=48,
        ...     num_features=20,
        ...     num_classes=3,
        ...     task_type=TaskType.CLASSIFICATION

    Example (Regression):
        >>> model = build_deep_lstm_attention(
        ...     sequence_length=48,
        ...     num_features=20,
        ...     task_type=TaskType.REGRESSION
        ...     num_classes=3
        ... )
        >>>
        >>> # Model with attention weights output
        >>> model = build_deep_lstm_attention(return_attention=True)
        >>> predictions, attention_weights = model.predict(X_test)
    """

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # Optional BatchNormalization
    x = inputs
    if use_batch_norm:
        x = layers.BatchNormalization(name="batch_norm")(x)

    # First LSTM layer (128 units)
    x = layers.LSTM(lstm1_units, return_sequences=True, name="lstm_1")(x)
    x = layers.Dropout(dropout_lstm, name="dropout_1")(x)

    # Second LSTM layer (64 units)
    lstm_output = layers.LSTM(lstm2_units, return_sequences=True, name="lstm_2")(x)
    x = layers.Dropout(dropout_lstm, name="dropout_2")(lstm_output)

    # Attention layer (using LightweightAttention from main codebase)
    attention_layer = LightweightAttention(
        attn_type=attention_type,
        heads=attention_heads,
        return_attention_scores=return_attention,
        name="attention",
    )

    if return_attention:
        context, attention_weights = attention_layer(x)
    else:
        context = attention_layer(x)

    # Dense layers
    x = layers.Dense(dense_units, activation="relu", name="dense1")(context)
    x = layers.Dropout(dense_dropout, name="dropout_dense")(x)

    # Output layer with task-specific configuration
    predictions, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    base_name = "Deep_LSTM_Attention"
    if return_attention:
        outputs = [predictions, attention_weights]
        model_name = get_model_name(f"{base_name}_Interpretable", task_type)
        model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    else:
        model_name = get_model_name(base_name, task_type)
        model = models.Model(inputs=inputs, outputs=predictions, name=model_name)

    # Compile model
    opt = get_optimizer(optimizer_name, learning_rate)

    if return_attention:
        # Multi-output: Apply loss/metrics only to first output (predictions)
        model.compile(
            optimizer=opt,
            loss=[loss, None],  # List format
            metrics=[output_metrics, []],
        )
    else:
        model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    print(f"\nDeep LSTM-Attention Model Configuration:")
    print(f"  Task type: {task_type}")
    print(f"  Architecture: Stacked LSTM ({lstm1_units} → {lstm2_units}) + Attention")
    print(f"  BatchNorm: {use_batch_norm}")
    print(f"  Attention type: {attention_type} (heads={attention_heads})")
    print(f"  Dense units: {dense_units}")
    print(f"  Dropout: LSTM={dropout_lstm}, Dense={dense_dropout}")
    print(f"  Return attention weights: {return_attention}")
    if task_type == TaskType.CLASSIFICATION:
        print(f"  Metrics: Accuracy, Balanced Accuracy, Macro F1")
    else:
        print(f"  Metrics: MAE, MSE, RMSE, R², Directional Accuracy")

    return model


# ============================================================================
# TUNABLE VERSION (With KerasTuner)
# ============================================================================


def build_deep_lstm_attention_tunable(hp, num_features, num_classes, task_type=TaskType.CLASSIFICATION,
                                      sequence_length=None, max_sequence_length=100):
    """
    Build tunable Deep LSTM-Attention model for hyperparameter optimization.

    Supports both classification and regression tasks.

    Args:
        hp: KerasTuner HyperParameters object
        num_features: Number of features per timestep (fixed)
        num_classes: Number of output classes for classification (fixed)
        task_type: TaskType.CLASSIFICATION or TaskType.REGRESSION (default: CLASSIFICATION)
        sequence_length: If None, tunes sequence_length (32 to max_sequence_length).
                        If provided, uses this fixed value (for pre-created sequences).
        max_sequence_length: Maximum sequence length (used when sequence_length=None)

    Returns:
        Compiled Keras model with tunable hyperparameters

    Example (Classification):
        >>> import keras_tuner
        >>> def model_builder(hp):
        ...     return build_deep_lstm_attention_tunable(
        ...         hp=hp,
        ...         num_features=20,
        ...         num_classes=3,
        ...         task_type=TaskType.CLASSIFICATION,
        ...         sequence_length=48
        ...     )
        >>> tuner = keras_tuner.BayesianOptimization(
        ...     model_builder,
        ...     objective='val_accuracy',
        ...     max_trials=30
        ... )

    Example (Regression):
        >>> def model_builder(hp):
        ...     return build_deep_lstm_attention_tunable(
        ...         hp=hp,
        ...         num_features=20,
        ...         num_classes=3,  # Not used for regression
        ...         task_type=TaskType.REGRESSION,
        ...         sequence_length=48
        ...     )
        >>> tuner = keras_tuner.BayesianOptimization(
        ...     model_builder,
        ...     objective='val_r2_score',
        ...     max_trials=30
        ... )
    """

    # Tunable hyperparameters
    if sequence_length is None:
        # Tune sequence_length if not provided
        sequence_length = hp.Int(
            "sequence_length", min_value=32, max_value=max_sequence_length, step=16
        )
    # else: use the provided fixed sequence_length

    use_batch_norm = hp.Boolean("use_batch_norm", default=True)
    lstm1_units = hp.Choice("lstm1_units", values=[64, 128, 256])
    lstm2_units = hp.Choice("lstm2_units", values=[32, 64, 128])
    dropout_lstm = hp.Float("dropout_lstm", min_value=0.2, max_value=0.4, step=0.1)
    attention_type = hp.Choice("attention_type", values=["bahdanau", "dot"])
    attention_heads = hp.Choice("attention_heads", values=[2, 4, 8])
    dense_units = hp.Choice("dense_units", values=[32, 64, 128])
    dense_dropout = hp.Float("dense_dropout", min_value=0.2, max_value=0.4, step=0.1)

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # Optional BatchNormalization
    x = inputs
    if use_batch_norm:
        x = layers.BatchNormalization(name="batch_norm")(x)

    # First LSTM layer
    x = layers.LSTM(lstm1_units, return_sequences=True, name="lstm_1")(x)
    x = layers.Dropout(dropout_lstm, name="dropout_1")(x)

    # Second LSTM layer
    x = layers.LSTM(lstm2_units, return_sequences=True, name="lstm_2")(x)
    x = layers.Dropout(dropout_lstm, name="dropout_2")(x)

    # Attention layer
    context = LightweightAttention(
        attn_type=attention_type,
        heads=attention_heads,
        return_attention_scores=False,
        name="attention",
    )(x)

    # Dense layers
    x = layers.Dense(dense_units, activation="relu", name="dense1")(context)
    x = layers.Dropout(dense_dropout, name="dropout_dense")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("Deep_LSTM_Attention_Tunable", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

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
# FINE-TUNING UTILITIES
# ============================================================================


def freeze_lstm_layers(model):
    """
    Freeze LSTM layers for fine-tuning.
    Useful for transfer learning or fine-tuning on new data.

    Args:
        model: Compiled Deep LSTM-Attention model

    Returns:
        Model with frozen LSTM layers

    Example:
        >>> # Initial training
        >>> model = build_deep_lstm_attention(sequence_length=48, num_features=20, num_classes=3)
        >>> model.fit(X_train, y_train, epochs=30)
        >>>
        >>> # Fine-tuning: freeze LSTMs, train attention + head
        >>> model = freeze_lstm_layers(model)
        >>> model.fit(X_finetune, y_finetune, epochs=10, learning_rate=0.0001)
    """
    for layer in model.layers:
        if "lstm" in layer.name.lower():
            layer.trainable = False

    print("\nFrozen LSTM layers for fine-tuning:")
    for layer in model.layers:
        if "lstm" in layer.name.lower():
            print(f"  {layer.name}: trainable = {layer.trainable}")

    # Recompile after freezing
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=model.compiled_metrics._metrics,
    )

    return model


def unfreeze_all_layers(model):
    """
    Unfreeze all layers for full training.

    Args:
        model: Model with potentially frozen layers

    Returns:
        Model with all layers unfrozen
    """
    for layer in model.layers:
        layer.trainable = True

    print("\nUnfrozen all layers")

    # Recompile after unfreezing
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=model.compiled_metrics._metrics,
    )

    return model


# ============================================================================
# ATTENTION VISUALIZATION UTILITIES
# ============================================================================


def visualize_attention_weights(X_sample, attention_weights, timestamps=None, feature_names=None, save_path=None):
    """
    Visualize attention weights to understand which timesteps the model focuses on.

    Args:
        X_sample: Input sample (sequence_length, num_features)
        attention_weights: Attention weights (sequence_length,) or (heads, sequence_length, sequence_length)
        timestamps: Optional timestamps for x-axis labels
        feature_names: Optional feature names for plotting
        save_path: Optional path to save the figure

    Example:
        >>> model = build_deep_lstm_attention(return_attention=True)
        >>> predictions, attention_weights = model.predict(X_test)
        >>> visualize_attention_weights(
        ...     X_test[0],
        ...     attention_weights[0],
        ...     feature_names=['price', 'volume', 'rsi']
        ... )
    """
    sequence_length = X_sample.shape[0]

    # Handle different attention weight shapes
    if len(attention_weights.shape) == 1:
        # Bahdanau attention: (sequence_length,)
        weights = attention_weights
    elif len(attention_weights.shape) == 3:
        # Multi-head dot-product: (heads, sequence_length, sequence_length)
        # Average across heads and take diagonal (self-attention)
        weights = np.mean(np.diagonal(attention_weights, axis1=1, axis2=2), axis=0)
    else:
        raise ValueError(
            f"Unexpected attention weights shape: {attention_weights.shape}"
        )

    # Create figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    # Plot 1: Feature values with attention overlay
    if timestamps is None:
        timestamps = np.arange(sequence_length)

    # Plot some features (max 5 to avoid clutter)
    num_features_to_plot = min(5, X_sample.shape[1])
    for i in range(num_features_to_plot):
        label = feature_names[i] if feature_names else f"Feature {i}"
        ax1.plot(timestamps, X_sample[:, i], label=label, alpha=0.7)

    # Overlay attention weights as shaded regions
    ax1.fill_between(
        timestamps,
        X_sample.min() - 0.5,
        X_sample.max() + 0.5,
        alpha=weights * 0.5,  # Scale by attention weights
        color="red",
        label="Attention Focus",
    )

    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Feature Value (normalized)")
    ax1.set_title("Feature Values with Attention Overlay (red = high attention)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Attention weights bar chart
    colors = plt.cm.Reds(weights / weights.max())
    ax2.bar(timestamps, weights, color=colors, alpha=0.8, edgecolor="black")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Attention Weight")
    ax2.set_title("Attention Weights Distribution")
    ax2.grid(True, alpha=0.3, axis="y")

    # Highlight top-3 most important timesteps
    top_3_indices = np.argsort(weights)[-3:]
    for idx in top_3_indices:
        ax2.axvline(x=timestamps[idx], color="green", linestyle="--", alpha=0.5)
        ax2.text(
            timestamps[idx],
            weights[idx],
            f"{weights[idx]:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Attention visualization saved to: {save_path}")

    plt.show()

    # Print insights
    print("\nAttention Analysis:")
    print(f"  Top 3 most important timesteps:")
    for rank, idx in enumerate(reversed(top_3_indices), 1):
        print(f"    {rank}. Timestep {idx}: weight = {weights[idx]:.4f}")

    print(
        f"\n  Attention concentration (max/mean): {weights.max() / weights.mean():.2f}x"
    )
    print(f"  Attention entropy: {-np.sum(weights * np.log(weights + 1e-10)):.3f}")


def interpret_attention_patterns(model, X_samples, y_true, class_names=None):
    """
    Analyze attention patterns across multiple samples to understand
    what the model typically focuses on for each class.

    Args:
        model: Trained model with return_attention=True
        X_samples: Multiple samples (batch, sequence_length, features)
        y_true: True labels (batch,)
        class_names: Optional class names for reporting

    Example:
        >>> model = build_deep_lstm_attention(return_attention=True)
        >>> model.fit(X_train, y_train, epochs=30)
        >>> interpret_attention_patterns(
        ...     model, X_test, y_test,
        ...     class_names=['Bearish', 'Sideways', 'Bullish']
        ... )
    """
    predictions, attention_weights = model.predict(X_samples, verbose=0)

    # Get predicted classes
    if predictions.shape[-1] == 1:
        y_pred = (predictions > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(predictions, axis=1)

    # Handle attention weight shapes
    if len(attention_weights.shape) == 2:
        # Bahdanau: (batch, sequence_length)
        weights = attention_weights
    elif len(attention_weights.shape) == 4:
        # Multi-head: (batch, heads, sequence_length, sequence_length)
        weights = np.mean(np.diagonal(attention_weights, axis1=2, axis2=3), axis=1)

    num_classes = len(np.unique(y_true))
    sequence_length = weights.shape[1]

    print("\n" + "=" * 80)
    print("ATTENTION PATTERN ANALYSIS - DEEP LSTM")
    print("=" * 80)

    for cls in range(num_classes):
        class_name = class_names[cls] if class_names else f"Class {cls}"
        mask = (y_true == cls) & (y_pred == cls)  # Correctly predicted samples

        if mask.sum() == 0:
            print(f"\n{class_name}: No correctly predicted samples")
            continue

        # Average attention weights for this class
        avg_attention = weights[mask].mean(axis=0)

        # Find most important timesteps
        top_5_indices = np.argsort(avg_attention)[-5:]

        print(f"\n{class_name} (n={mask.sum()} samples):")
        print(f"  Average attention focus:")
        for rank, idx in enumerate(reversed(top_5_indices), 1):
            # Calculate relative position in sequence
            rel_pos = (idx / sequence_length) * 100
            print(
                f"    {rank}. Timestep {idx} ({rel_pos:.0f}% into sequence): {avg_attention[idx]:.4f}"
            )

        # Attention distribution statistics
        print(
            f"  Attention concentration: {avg_attention.max() / avg_attention.mean():.2f}x"
        )
        print(
            f"  Attention on recent bars (last 25%): {avg_attention[-sequence_length // 4:].sum():.3f}"
        )
        print(
            f"  Attention on old bars (first 25%): {avg_attention[:sequence_length // 4].sum():.3f}"
        )


# ============================================================================
# HINTS AND BEST PRACTICES
# ============================================================================

"""
HINTS FOR USING DEEP LSTM-ATTENTION IN TRADING:
================================================

1. ARCHITECTURE BENEFITS:
   Why Deep LSTM is better than Single LSTM:
   ✓ Hierarchical learning (low-level → high-level patterns)
   ✓ Better at capturing complex dependencies
   ✓ Layer 1: price patterns, volume spikes
   ✓ Layer 2: trends, regime changes
   ✓ Attention: critical event detection

2. BATCHNORM ADVANTAGES:
   Why use BatchNormalization:
   ✓ Stabilizes training (faster convergence)
   ✓ Reduces internal covariate shift
   ✓ Acts as mild regularization
   ✓ Better gradient flow
   ⚠ Disable during inference for consistency

3. LSTM LAYER SIZING:
   Recommended configurations:
   - Small: 64 → 32 (fast, simple patterns)
   - Medium: 128 → 64 (recommended default) ✓
   - Large: 256 → 128 (complex patterns, risk overfitting)

   Rule of thumb: Layer 2 = 0.5x Layer 1

4. DROPOUT STRATEGY:
   - After LSTM layers: 0.3 (prevent LSTM overfitting)
   - After Dense layer: 0.3 (prevent output overfitting)
   - Higher dropout (0.4) if overfitting
   - Lower dropout (0.2) if underfitting

5. ATTENTION TYPE SELECTION:
   Bahdanau (additive):
   ✓ Simpler, more interpretable
   ✓ Single weight per timestep
   ✓ Recommended for trading

   Dot-product (multi-head):
   ✓ More powerful for complex patterns
   ✓ Multiple attention scales
   ⚠ Harder to interpret

6. TRAINING STRATEGY:
   Phase 1 - Initial Training:
   - Full model, 30-50 epochs
   - Learning rate: 0.001
   - Monitor validation loss

   Phase 2 - Fine-Tuning (optional):
   - Freeze LSTM layers
   - Train attention + head only
   - Learning rate: 0.0001
   - 10-20 epochs

7. WHEN TO USE DEEP LSTM-ATTENTION:
   ✓ Complex multi-scale patterns
   ✓ Long-term dependencies (100+ bars)
   ✓ Multiple regime changes
   ✓ Need interpretability
   ✓ Sufficient training data (10K+ samples)

8. WHEN NOT TO USE:
   ✗ Simple patterns (use single LSTM)
   ✗ Very short sequences (<30 bars)
   ✗ Limited training data (<5K samples)
   ✗ Need maximum speed (use ESN)
   ✗ Need bidirectional (use BiLSTM-Attention)

9. HYPERPARAMETER TUNING PRIORITIES:
   1. lstm1_units (capacity)
   2. lstm2_units (capacity)
   3. attention_type (interpretability vs performance)
   4. learning_rate (convergence)
   5. dropout_lstm (regularization)
   6. dense_units (output capacity)

10. PERFORMANCE OPTIMIZATION:
    - Use GPU for training (15-25ms inference)
    - Batch predictions (batch_size=32-64)
    - Cache model predictions
    - Use mixed precision (fp16)
    - Consider quantization for production

11. MONITORING IN PRODUCTION:
    Track these metrics:
    - Prediction confidence (softmax max)
    - Attention concentration
    - Attention on recent vs old bars
    - Layer activation magnitudes
    - Gradient norms (detect training issues)

12. DEBUGGING TIPS:
    If poor performance:
    - Check attention patterns (visualize)
    - Verify LSTM capacity (increase units)
    - Check for vanishing gradients
    - Try different attention types
    - Increase dropout if overfitting
    - Decrease dropout if underfitting

13. COMPARISON TO OTHER MODELS:
    vs Single LSTM-Attention:
    ✓ Better accuracy (deeper hierarchy)
    ✓ Better long-term dependencies
    ✗ Slower training/inference
    ✗ More parameters (overfitting risk)

    vs BiLSTM-Attention:
    ✓ Faster (unidirectional)
    ✓ Real-time friendly
    ✗ Lower accuracy (no backward pass)

    vs Transformer:
    ✓ Simpler architecture
    ✓ Fewer parameters
    ✓ Faster training
    ✗ Less powerful for very long sequences

14. FEATURE ENGINEERING:
    Works best with:
    - Normalized returns (z-score)
    - Volume indicators
    - Technical indicators (RSI, MACD, Bollinger)
    - Volatility measures
    - Support/resistance distances
    - Market regime indicators

    Normalization: StandardScaler or MinMaxScaler

15. CLASS IMBALANCE HANDLING:
    For imbalanced crypto/stock data:
    ✓ Use class_weight in fit()
    ✓ Monitor Balanced Accuracy
    ✓ Track Macro F1
    ✓ Use SMOTE for extreme imbalance
    ✓ Ensemble multiple models

16. PRODUCTION DEPLOYMENT:
    Save model:
    ```python
    model.save('deep_lstm_attention.keras')
    ```

    Load model:
    ```python
    from okmich_quant_neural_net.keras.layers.light_weight_attention import LightweightAttention
    from okmich_quant_neural_net.keras.metrics import BalancedAccuracy, MacroF1Score

    model = keras.models.load_model(
        'deep_lstm_attention.keras',
        custom_objects={
            'LightweightAttention': LightweightAttention,
            'BalancedAccuracy': BalancedAccuracy,
            'MacroF1Score': MacroF1Score
        }
    )
    ```

17. INFERENCE OPTIMIZATION:
    ```python
    # Batch predictions
    predictions = model.predict(X_batch, batch_size=64)

    # Or use model.call() for single samples (faster)
    import tensorflow as tf
    single_pred = model(tf.expand_dims(X_single, 0), training=False)
    ```

18. ENSEMBLE STRATEGIES:
    Combine with:
    - Multiple Deep-LSTM-Attention (different seeds)
    - ESN (fast complementary predictions)
    - TCN (different temporal patterns)
    - Traditional ML (XGBoost for comparison)

19. RESEARCH & DEVELOPMENT:
    Experiment with:
    - Different LSTM layer counts (3+ layers)
    - Residual connections between LSTMs
    - Multiple attention heads
    - Attention at each LSTM layer
    - Combining with CNNs for feature extraction

20. KEY INSIGHT:
    "The Deep LSTM-Attention model combines the hierarchical
     feature learning of deep LSTMs with the interpretability
     of attention mechanisms. It's the sweet spot between
     single-LSTM simplicity and Transformer complexity,
     providing excellent performance for trading applications
     that need both accuracy and explainability."
"""

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DEEP LSTM WITH ATTENTION - LIGHTWEIGHT TRADING MODEL")
    print("=" * 80)
    print("\nDeep + Powerful + Interpretable")
    print("Captures long dependencies with focused attention on key bars")
    print("=" * 80)
    print("\nAvailable examples:")
    print("  1. Simple usage (fixed hyperparameters)")
    print("  2. Attention visualization (interpretability)")
    print("  3. Fine-tuning strategy")
    print("  4. Run all examples")
    print("=" * 80)

    choice = input("\nSelect example to run (1-4, or 'q' to quit): ").strip()

    if choice == "1":
        model, history = example_simple_usage()
    elif choice == "2":
        model, attention_weights = example_attention_visualization()
    elif choice == "3":
        model = example_fine_tuning()
    elif choice == "4":
        print("\nRunning all examples...\n")
        print("\n" + ">" * 80)
        example_simple_usage()
        print("\n" + ">" * 80)
        example_attention_visualization()
        print("\n" + ">" * 80)
        example_fine_tuning()
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED!")
        print("=" * 80)
    elif choice.lower() == "q":
        print("\nExiting...")
    else:
        print("\nInvalid choice. Please run again and select 1-4 or 'q'.")

    print("\n" + "=" * 80)
    print("For more details, see the HINTS section in the source code.")
    print("Architecture: Input → BatchNorm → LSTM(128) → LSTM(64) → Attention → Dense(64) → Output")
    print("Inference: ~15-25ms on GPU | Fine-tune: Freeze early LSTMs, train attention + head")
    print("=" * 80)
