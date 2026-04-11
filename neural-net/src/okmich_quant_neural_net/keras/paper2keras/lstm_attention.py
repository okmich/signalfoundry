"""
Attention-Augmented LSTM - Model Factory
=========================================

Architecture Description:
-------------------------
A simple yet powerful architecture that adds an Attention Mechanism on top of an LSTM. The attention layer learns which
specific timesteps in the lookback window are most important for the final prediction.

This is a middle ground between standard LSTM and full Transformer models, providing significant performance boost with
added interpretability.

Architecture:
-------------
Input (32-100 timesteps, n_features)
    ↓
LSTM (64 units, return_sequences=True)
    ↓
Attention Layer (learns temporal importance)
    ↓
Context Vector (weighted combination of LSTM outputs)
    ↓
Dense (32, ReLU) → Dropout (0.2)
    ↓
Output (softmax for multi-class)

Key Features:
-------------
✓ Interpretability: Attention weights show which bars the model focused on
✓ Focus: Model concentrates on critical events and ignores noise
✓ Performance: Often significantly improves standard LSTM performance
✓ Simplicity: Simpler than full Transformer, easier to train
✓ Visualization: Can plot attention weights to understand decisions

Why it Works for Trading:
--------------------------
Not all past price action is equally important. The model can learn to
"pay attention" to:
- High-volume breakout bars
- V-shaped reversal points
- Start of low-volatility periods
- Major support/resistance tests
- Regime change points

The attention weights provide interpretability - you can see which historical
bars the model considered most important for its prediction.

Comparison to BiLSTM with Attention:
------------------------------------
| Feature | Attention-LSTM | BiLSTM-Attention |
|---------|----------------|------------------|
| Direction | Unidirectional | Bidirectional |
| Complexity | Simple | More complex |
| Speed | Faster | Slower |
| Interpretability | High ✓ | Medium |
| Accuracy | Good | Better |
| Use Case | Real-time, interpretability | Maximum accuracy |

Usage:
------
1. Simple version (fixed hyperparameters):
   model = build_lstm_attention(
       sequence_length=48,
       num_features=20,
       num_classes=3
   )

2. Tunable version (with KerasTuner):
   def model_builder(hp):
       return build_lstm_attention_tunable(
           hp=hp,
           num_features=20,
           num_classes=3
       )

3. Extract attention weights:
   model = build_lstm_attention(return_attention=True)
   predictions, attention_weights = model.predict(X_test)
"""

import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models

from ..layers import LightweightAttention
from .common import TaskType, create_output_layer_and_loss, get_optimizer, get_model_name


# ============================================================================
# SIMPLE VERSION (Fixed Hyperparameters)
# ============================================================================
def build_lstm_attention(sequence_length=48, num_features=20, num_classes=3, task_type=TaskType.CLASSIFICATION,
                         lstm_units=64, attention_type="bahdanau", attention_heads=4, dense_units=32, dense_dropout=0.2,
                         learning_rate=0.001, optimizer_name="adam", return_attention=False):
    """
    Build Attention-Augmented LSTM model for classification or regression.

    Simple, interpretable architecture with attention mechanism.
    Focus on understanding which timesteps matter most for predictions.

    Args:
        sequence_length: Number of timesteps in input sequences
        num_features: Number of features per timestep
        num_classes: Number of output classes (for classification only)
        task_type: TaskType.CLASSIFICATION or TaskType.REGRESSION (default: CLASSIFICATION)
        lstm_units: Units in LSTM layer (default: 64)
        attention_type: Type of attention ('bahdanau' or 'dot') (default: 'bahdanau')
        attention_heads: Number of attention heads for dot-product (default: 4)
        dense_units: Units in dense layer (default: 32)
        dense_dropout: Dropout rate after dense layer (default: 0.2)
        learning_rate: Learning rate for optimizer (default: 0.001)
        optimizer_name: Optimizer to use ('adam', 'adamw', 'rmsprop')
        return_attention: If True, model returns (predictions, attention_weights)

    Returns:
        Compiled Keras model

    Example (Classification):
        >>> from okmich_quant_neural_net.keras.paper2keras.common import TaskType
        >>> model = build_lstm_attention(
        ...     sequence_length=48,
        ...     num_features=20,
        ...     num_classes=3,
        ...     task_type=TaskType.CLASSIFICATION
        ... )

    Example (Regression):
        >>> from okmich_quant_neural_net.keras.paper2keras.common import TaskType
        >>> model = build_lstm_attention(
        ...     sequence_length=48,
        ...     num_features=20,
        ...     task_type=TaskType.REGRESSION
        ... )

    Example (With attention weights):
        >>> model = build_lstm_attention(return_attention=True)
        >>> predictions, attention_weights = model.predict(X_test)
    """

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # LSTM layer (return sequences for attention)
    lstm_output = layers.LSTM(lstm_units, return_sequences=True, name="lstm")(inputs)

    # Attention layer (using LightweightAttention from main codebase)
    attention_layer = LightweightAttention(
        attn_type=attention_type,
        heads=attention_heads,
        return_attention_scores=return_attention,
        name="attention")

    if return_attention:
        context, attention_weights = attention_layer(lstm_output)
    else:
        context = attention_layer(lstm_output)

    # Dense layers
    x = layers.Dense(dense_units, activation="relu", name="dense1")(context)
    x = layers.Dropout(dense_dropout, name="dropout")(x)

    # Output layer (using common.py - handles both classification and regression!)
    predictions, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model with appropriate name
    base_model_name = "LSTM_Attention_Interpretable" if return_attention else "LSTM_Attention"
    model_name = get_model_name(base_model_name, task_type)

    if return_attention:
        outputs = [predictions, attention_weights]
        model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    else:
        model = models.Model(inputs=inputs, outputs=predictions, name=model_name)

    # Get optimizer (using common.py)
    opt = get_optimizer(optimizer_name, learning_rate)

    # Compile model
    if return_attention:
        # When returning attention, only use first output for loss/metrics
        # Use list format: [loss_for_predictions, loss_for_attention]
        model.compile(
            optimizer=opt,
            loss=[loss, None],  # Apply loss only to predictions, not attention
            metrics=[output_metrics, []],  # Metrics only for predictions
        )
    else:
        model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    # Print configuration
    print(f"\nLSTM-Attention Model Configuration:")
    print(f"  Task type: {task_type.value}")
    print(f"  LSTM units: {lstm_units}")
    print(f"  Attention type: {attention_type}")
    print(f"  Attention heads: {attention_heads}")
    print(f"  Return attention weights: {return_attention}")

    if task_type == TaskType.CLASSIFICATION:
        print(f"  Metrics: Accuracy, Balanced Accuracy, Macro F1 (class-imbalance robust)")
    else:  # Regression
        print(f"  Metrics: MAE, MSE, RMSE, R² Score, Directional Accuracy")

    return model


# ============================================================================
# TUNABLE VERSION (With KerasTuner)
# ============================================================================
def build_lstm_attention_tunable(hp, num_features, num_classes=3, task_type=TaskType.CLASSIFICATION,
                                 sequence_length=None, max_sequence_length=100):
    """
    Build tunable LSTM-Attention model for hyperparameter optimization.

    Args:
        hp: KerasTuner HyperParameters object
        num_features: Number of features per timestep (fixed)
        num_classes: Number of output classes (for classification only)
        task_type: TaskType.CLASSIFICATION or TaskType.REGRESSION (default: CLASSIFICATION)
        sequence_length: If None, tunes sequence_length (32 to max_sequence_length).
                        If provided, uses this fixed value (for pre-created sequences).
        max_sequence_length: Maximum sequence length (used when sequence_length=None)

    Returns:
        Compiled Keras model with tunable hyperparameters

    Example (Classification):
        >>> import keras_tuner
        >>> from okmich_quant_neural_net.keras.paper2keras.common import TaskType
        >>> def model_builder(hp):
        ...     return build_lstm_attention_tunable(
        ...         hp=hp,
        ...         num_features=20,
        ...         num_classes=3,
        ...         task_type=TaskType.CLASSIFICATION,
        ...         sequence_length=48  # Fixed for pre-created sequences
        ...     )
        >>> tuner = keras_tuner.BayesianOptimization(
        ...     model_builder,
        ...     objective='val_accuracy',
        ...     max_trials=20
        ... )

    Example (Regression):
        >>> def model_builder(hp):
        ...     return build_lstm_attention_tunable(
        ...         hp=hp,
        ...         num_features=20,
        ...         task_type=TaskType.REGRESSION,
        ...         sequence_length=48
        ...     )
        >>> tuner = keras_tuner.BayesianOptimization(
        ...     model_builder,
        ...     objective='val_r2_score',
        ...     direction='max',
        ...     max_trials=20
        ... )
    """

    # Tunable hyperparameters
    if sequence_length is None:
        # Tune sequence_length if not provided
        sequence_length = hp.Int(
            "sequence_length", min_value=32, max_value=max_sequence_length, step=16
        )
    # else: use the provided fixed sequence_length
    lstm_units = hp.Choice("lstm_units", values=[32, 64, 128, 256])
    attention_type = hp.Choice("attention_type", values=["bahdanau", "dot"])
    attention_heads = hp.Choice("attention_heads", values=[2, 4, 8])
    dense_units = hp.Choice("dense_units", values=[16, 32, 64, 128])
    dense_dropout = hp.Float("dense_dropout", min_value=0.1, max_value=0.4, step=0.1)

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # LSTM layer
    lstm_output = layers.LSTM(lstm_units, return_sequences=True, name="lstm")(inputs)

    # Attention layer
    context = LightweightAttention(
        attn_type=attention_type,
        heads=attention_heads,
        return_attention_scores=False,
        name="attention",
    )(lstm_output)

    # Dense layers
    x = layers.Dense(dense_units, activation="relu", name="dense1")(context)
    x = layers.Dropout(dense_dropout, name="dropout")(x)

    # Output layer (using common.py - handles both classification and regression!)
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model with appropriate name
    model_name = get_model_name("LSTM_Attention_Tunable", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Tunable optimizer
    optimizer_name = hp.Choice("optimizer", values=["adam", "adamw", "rmsprop"])

    # Task-specific learning rate ranges
    if task_type == TaskType.CLASSIFICATION:
        learning_rate = hp.Float(
            "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
        )
    else:  # Regression often needs lower learning rates
        learning_rate = hp.Float(
            "learning_rate", min_value=1e-5, max_value=1e-3, sampling="log"
        )

    # Get optimizer (using common.py)
    opt = get_optimizer(optimizer_name, learning_rate)

    # Compile model
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

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
        >>> model = build_lstm_attention(return_attention=True)
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
        >>> model = build_lstm_attention(return_attention=True)
        >>> model.fit(X_train, y_train, epochs=20)
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
    print("ATTENTION PATTERN ANALYSIS")
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
HINTS FOR USING LSTM-ATTENTION IN TRADING:
==========================================

1. INTERPRETABILITY BENEFITS:
   Key advantage over standard LSTM:
   - See which historical bars matter most
   - Understand model decision-making
   - Identify important events (breakouts, reversals)
   - Build trust in model predictions
   - Detect when model focuses on wrong features

2. ATTENTION PATTERNS TO LOOK FOR:
   Healthy patterns:
   ✓ Focus on recent bars (last 10-20% of sequence)
   ✓ Spikes at volume breakouts
   ✓ Attention at trend reversal points
   ✓ Focus on regime change moments

   Warning signs:
   ✗ Uniform attention (model not learning)
   ✗ Focus only on first/last bar (overly simple)
   ✗ Chaotic, random attention patterns
   ✗ No correlation with known events

3. BAHDANAU VS DOT-PRODUCT ATTENTION:
   Bahdanau (additive):
   - Simpler, easier to interpret
   - Single attention weight per timestep
   - Recommended for most trading tasks
   - Faster training

   Dot-product (multi-head):
   - More complex, multi-scale attention
   - Better for complex patterns
   - Harder to interpret (multiple heads)
   - Slightly slower

4. LSTM UNITS SELECTION:
   - Small (32-64): Fast, simple patterns
   - Medium (64-128): Recommended default
   - Large (128-256): Complex patterns, risk overfitting

   Start with 64 and increase if underfitting.

5. WHEN TO USE LSTM-ATTENTION:
   ✓ Need interpretability (understand decisions)
   ✓ Real-time trading (unidirectional is fine)
   ✓ Want faster training than BiLSTM
   ✓ Regulatory requirements (explainability)
   ✓ Building trust in model
   ✓ Debugging model behavior

6. WHEN NOT TO USE:
   ✗ Maximum accuracy needed (use BiLSTM-Attention)
   ✗ Don't care about interpretability
   ✗ Very short sequences (<20 timesteps)
   ✗ Need bidirectional information

7. TRAINING TIPS:
   - Batch size: 32-64
   - Epochs: 20-50
   - Learning rate: 0.0005-0.002
   - Use gradient clipping (clipnorm=1.0)
   - Monitor validation metrics
   - Early stopping patience: 10-15

8. FEATURE ENGINEERING:
   Attention works best with:
   - Price changes / returns
   - Volume indicators
   - Technical indicators (RSI, MACD)
   - Volatility measures
   - Support/resistance distances
   - Standardize all features!

9. VISUALIZATION BEST PRACTICES:
   - Plot attention for both correct and incorrect predictions
   - Average attention across samples per class
   - Look for consistent patterns
   - Compare attention to known market events
   - Save attention weights for post-analysis

10. CLASS IMBALANCE HANDLING:
    For imbalanced crypto/stock data:
    ✓ Use class_weight in fit()
    ✓ Monitor Balanced Accuracy (not just accuracy)
    ✓ Track Macro F1 (equal weight to all classes)
    ✓ Use confusion matrix analysis

11. PRODUCTION DEPLOYMENT:
    - Save model: model.save('lstm_attention.keras')
    - Save with custom layers:
      ```python
      from okmich_quant_ml.keras.layers import LightweightAttention
      model = keras.models.load_model(
          'model.keras',
          custom_objects={
              'LightweightAttention': LightweightAttention,
              'BalancedAccuracy': BalancedAccuracy,
              'MacroF1Score': MacroF1Score
          }
      )
      ```

12. ATTENTION WEIGHT ANALYSIS IN PRODUCTION:
    Log and analyze attention in real-time:
    ```python
    # During trading
    predictions, attention = model.predict(recent_bars)

    # Check if attention is reasonable
    if attention.max() / attention.mean() < 2.0:
        # Attention too diffuse - model uncertain
        reduce_position_size()

    # Check if focusing on recent bars
    recent_focus = attention[-10:].sum() / attention.sum()
    if recent_focus < 0.3:
        # Model ignoring recent data - warning!
        log_warning("Model not focusing on recent bars")
    ```

13. DEBUGGING WITH ATTENTION:
    If poor performance:
    - Visualize attention patterns
    - Check if attention focuses on noise
    - Verify attention aligns with intuition
    - Try different attention types
    - Increase LSTM capacity

14. ENSEMBLE STRATEGIES:
    Combine LSTM-Attention with:
    - ESN: LSTM-Attention for accuracy, ESN for speed
    - TCN: Different temporal patterns
    - Multiple LSTM-Attention with different seeds
    - MDN: Combine attention with uncertainty

15. COMPARISON TO OTHER MODELS:
    vs Standard LSTM:
    ✓ Better accuracy (attention helps focus)
    ✓ Interpretability (see what matters)
    ✗ Slightly slower

    vs BiLSTM-Attention:
    ✓ Simpler, faster
    ✓ Better interpretability
    ✗ Lower accuracy (no backward pass)

    vs Transformer:
    ✓ Much simpler
    ✓ Faster training
    ✓ Easier to interpret
    ✗ Less powerful for very long sequences

16. RESEARCH & DEVELOPMENT:
    Use attention visualization to:
    - Discover important market patterns
    - Validate feature engineering
    - Understand regime changes
    - Guide feature selection
    - Build domain knowledge

17. REGULATORY COMPLIANCE:
    Attention provides explainability:
    - Show which bars influenced decision
    - Demonstrate rational behavior
    - Audit model decisions
    - Meet explainability requirements

18. MONITORING IN PRODUCTION:
    Track these metrics:
    - Average attention concentration
    - Attention on recent vs old bars
    - Consistency of attention patterns
    - Correlation with volatility
    - Attention entropy (diversity)

19. HYPERPARAMETER TUNING PRIORITIES:
    1. lstm_units (capacity)
    2. attention_type (interpretability vs performance)
    3. dense_units (final classification capacity)
    4. learning_rate (convergence)
    5. attention_heads (for dot-product)

20. KEY INSIGHT:
    "The attention mechanism doesn't just improve accuracy -
     it transforms a black box LSTM into an interpretable
     decision-maker. You can see exactly which historical
     moments the model considers important, making it
     invaluable for understanding and trusting your trading
     algorithm."
"""
