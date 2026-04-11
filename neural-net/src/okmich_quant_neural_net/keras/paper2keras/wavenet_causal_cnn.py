"""
WaveNet-Inspired Causal CNN - Model Factory
============================================

Architecture:
-------------
Input (32-100 timesteps, n_features)
    ↓
Causal Conv1D Stack with Exponential Dilations
    - Conv1D (32 filters, kernel=2, dilation=1) → ReLU
    - Conv1D (32 filters, kernel=2, dilation=2) → ReLU
    - Conv1D (64 filters, kernel=2, dilation=4) → ReLU
    - Conv1D (64 filters, kernel=2, dilation=8) → ReLU
    - Residual connections + Skip connections
    ↓
Dual Pooling Strategy:
    - GlobalMaxPooling1D → Captures peaks/extremes
    - GlobalAvgPooling1D → Captures overall trends
    - Concatenate both
    ↓
Dense (128, ReLU) → Dropout (0.3)
    ↓
Dense (64, ReLU) → Dropout (0.2)
    ↓
Output (softmax for multi-class)

Key Features:
-------------
- Causal convolutions prevent lookahead bias (critical for trading)
- Exponentially large receptive fields with minimal parameters
- Dual pooling: Max captures extremes, Avg captures trends
- WaveNet-style architecture with residual connections
- Suitable for: high-frequency tick data, order flow imbalance, microstructure analysis

Receptive Field:
----------------
RF = 1 + 2 * (kernel_size - 1) * sum(dilations)

For kernel_size=2, dilations=[1, 2, 4, 8]:
    RF = 1 + 2 * (2-1) * (1+2+4+8) = 31 timesteps

Usage:
------
1. Simple version (fixed hyperparameters):
   model = build_wavenet_cnn(
       sequence_length=48,
       num_features=20,
       num_classes=3
   )

2. Tunable version (with KerasTuner):
   def model_builder(hp):
       return build_wavenet_cnn_tunable(
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
- "WaveNet: A Generative Model for Raw Audio" - van den Oord et al. (2016) - Original architecture
https://arxiv.org/abs/1609.03499
- "DeepVol: Volatility Forecasting from High-Frequency Data with Dilated Causal Convolutions" - Fernando Moreno-Pino, et al. (2024)
https://arxiv.org/abs/2210.04797
- Financial application: "Temporal Pattern Attention for Multivariate Time Series Forecasting" - Shih et al. (2019)
https://arxiv.org/abs/1809.04206
"""

import numpy as np
from keras import layers, models

# Import the existing TCN layer (WaveNet-based implementation)
from okmich_quant_neural_net.keras.layers import TCN

# Import task type and common utilities
from .common import TaskType, create_output_layer_and_loss, get_optimizer, get_model_name


# ============================================================================
# Internal helpers
# ============================================================================


def _resolve_tcn_filters(filters_list, dilations):
    """Normalize filters configuration and validate TCN constraints."""
    if isinstance(filters_list, tuple):
        filters_list = list(filters_list)

    if isinstance(filters_list, list):
        if len(filters_list) != len(dilations):
            raise ValueError(
                "filters_list must match dilations length when a list/tuple is provided. "
                f"Got len(filters_list)={len(filters_list)} and len(dilations)={len(dilations)}."
            )
        return filters_list

    if isinstance(filters_list, int):
        return filters_list

    raise TypeError(f"filters_list must be int, list, or tuple, got: {type(filters_list)}")


def _filters_from_strategy(filter_strategy, num_dilations):
    """Build a filters list that always matches the dilation schedule length."""
    if filter_strategy == "small":
        base_filters = [32, 32, 32, 32]
    elif filter_strategy == "medium":
        base_filters = [32, 32, 64, 64]
    elif filter_strategy == "large":
        base_filters = [32, 64, 128, 128]
    else:
        raise ValueError(f"Unknown filter_strategy: {filter_strategy}")

    if num_dilations <= len(base_filters):
        return base_filters[:num_dilations]

    return base_filters + [base_filters[-1]] * (num_dilations - len(base_filters))


# ============================================================================
# SIMPLE VERSION (Fixed Hyperparameters)
# ============================================================================


def build_wavenet_cnn(
        sequence_length=48,
        num_features=20,
        num_classes=3,
        task_type=TaskType.CLASSIFICATION,
        filters_list=(32, 32, 64, 64),
        kernel_size=2,
        dilations=(1, 2, 4, 8),
        dropout_rate=0.2,
        use_skip_connections=False,
        use_batch_norm=False,
        dense1_units=128,
        dense1_dropout=0.3,
        dense2_units=64,
        dense2_dropout=0.2,
        learning_rate=0.001,
        optimizer_name="adam",
):
    """
    Build WaveNet-inspired Causal CNN model (fixed hyperparameters).

    This model uses:
    - Causal convolutions (no look-ahead bias)
    - Exponential dilation rates for large receptive field
    - Dual pooling strategy (Max + Average)
    - Residual connections for stable training

    Args:
        sequence_length: Number of timesteps in input sequences
        num_features: Number of features per timestep
        num_classes: Number of output classes
        filters_list: List/tuple of filter sizes for each layer, or int for fixed filters
        kernel_size: Convolution kernel size (WaveNet uses 2) (default: 2)
        dilations: Tuple of dilation rates (default: (1, 2, 4, 8))
        dropout_rate: Dropout rate in conv layers (default: 0.2)
        use_skip_connections: Use skip connections (default: False)
        use_batch_norm: Use batch normalization (default: False)
        dense1_units: Units in first dense layer (default: 128)
        dense1_dropout: Dropout after first dense layer (default: 0.3)
        dense2_units: Units in second dense layer (default: 64)
        dense2_dropout: Dropout after second dense layer (default: 0.2)
        learning_rate: Learning rate for optimizer (default: 0.001)
        optimizer_name: Optimizer to use ('adam', 'adamw', 'rmsprop') (default: 'adam')

    Returns:
        Compiled Keras model

    Example:
        >>> model = build_wavenet_cnn(
        ...     sequence_length=48,
        ...     num_features=20,
        ...     num_classes=3
        ... )
        >>> model.summary()
    """

    tcn_filters = _resolve_tcn_filters(filters_list, dilations)
    if use_skip_connections and isinstance(tcn_filters, list) and len(set(tcn_filters)) > 1:
        raise ValueError(
            "use_skip_connections=True is only supported when all filters are identical. "
            "Set use_skip_connections=False or provide uniform filters_list values."
        )

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # TCN/WaveNet layer with specified filters per layer
    x = TCN(
        nb_filters=tcn_filters,
        kernel_size=kernel_size,
        nb_stacks=1,
        dilations=dilations,
        padding="causal",  # Critical: No look-ahead bias
        use_skip_connections=use_skip_connections,
        dropout_rate=dropout_rate,
        return_sequences=True,  # Need sequences for pooling
        activation="relu",
        kernel_initializer="he_normal",
        use_batch_norm=use_batch_norm,
        use_layer_norm=False,
        name="wavenet_tcn",
    )(inputs)

    # Dual pooling strategy: Max + Average
    # Max pooling captures peaks/extremes (important for price spikes, volume surges)
    max_pool = layers.GlobalMaxPooling1D(name="global_max_pool")(x)

    # Average pooling captures overall trends
    avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # Concatenate both pooling outputs
    x = layers.Concatenate(name="concat_pooling")([max_pool, avg_pool])

    # Dense layers
    x = layers.Dense(dense1_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dense1_dropout, name="dropout1")(x)

    x = layers.Dense(dense2_units, activation="relu", name="dense2")(x)
    x = layers.Dropout(dense2_dropout, name="dropout2")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("WaveNet_Causal_CNN", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile model with gradient clipping (important for stability)
    opt = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    # Calculate and display receptive field
    receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)
    print(f"\nWaveNet Receptive Field: {receptive_field} timesteps")
    if sequence_length < receptive_field:
        print(
            f"⚠️  Warning: Sequence length ({sequence_length}) < Receptive field ({receptive_field})"
        )
        print(f"   Consider increasing sequence_length or reducing dilations")

    return model


# ============================================================================
# TUNABLE VERSION (With KerasTuner)
# ============================================================================


def build_wavenet_cnn_tunable(hp, num_features, num_classes, task_type=TaskType.CLASSIFICATION, sequence_length=None,
                              max_sequence_length=100):
    """
    Build WaveNet-inspired Causal CNN with hyperparameter tuning.

    This version allows KerasTuner to optimize:
    - Sequence length (if not provided as fixed parameter)
    - Filter configuration strategy
    - Kernel size
    - Dilation strategy
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
        ...     return build_wavenet_cnn_tunable(
        ...         hp=hp,
        ...         num_features=20,
        ...         num_classes=3
        ...     )
        >>> # Example 2: Fixed sequence_length (for pre-created sequences)
        >>> def model_builder(hp):
        ...     return build_wavenet_cnn_tunable(
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
        ...     project_name='wavenet_cnn'
        ... )
    """

    # Tunable hyperparameters
    if sequence_length is None:
        # Tune sequence_length if not provided
        sequence_length = hp.Int("sequence_length", min_value=32, max_value=max_sequence_length, step=16)
    # else: use the provided fixed sequence_length
    kernel_size = hp.Choice("kernel_size", values=[2, 3])  # WaveNet uses 2

    # Filter configuration strategy
    filter_strategy = hp.Choice("filter_strategy", values=["small", "medium", "large"])

    # Dilation strategy
    dilation_strategy = hp.Choice("dilation_strategy", values=["shallow", "medium", "deep"])
    if dilation_strategy == "shallow":
        dilations = (1, 2, 4, 8)
    elif dilation_strategy == "medium":
        dilations = (1, 2, 4, 8, 16)
    else:  # deep
        dilations = (1, 2, 4, 8, 16, 32)
    filters_list = _filters_from_strategy(filter_strategy, len(dilations))

    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.4, step=0.1)
    use_skip_connections = hp.Boolean("use_skip_connections", default=False)
    if use_skip_connections and len(set(filters_list)) > 1:
        use_skip_connections = False
    use_batch_norm = hp.Boolean("use_batch_norm", default=False)

    dense1_units = hp.Choice("dense1_units", values=[64, 128, 256])
    dense1_dropout = hp.Float("dense1_dropout", min_value=0.2, max_value=0.5, step=0.1)
    dense2_units = hp.Choice("dense2_units", values=[32, 64, 128])
    dense2_dropout = hp.Float("dense2_dropout", min_value=0.1, max_value=0.4, step=0.1)

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # TCN/WaveNet layer
    x = TCN(
        nb_filters=filters_list,
        kernel_size=kernel_size,
        nb_stacks=1,
        dilations=dilations,
        padding="causal",
        use_skip_connections=use_skip_connections,
        dropout_rate=dropout_rate,
        return_sequences=True,
        activation="relu",
        kernel_initializer="he_normal",
        use_batch_norm=use_batch_norm,
        use_layer_norm=False,
        name="wavenet_tcn",
    )(inputs)

    # Dual pooling strategy
    max_pool = layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Concatenate(name="concat_pooling")([max_pool, avg_pool])

    # Dense layers
    x = layers.Dense(dense1_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dense1_dropout, name="dropout1")(x)

    x = layers.Dense(dense2_units, activation="relu", name="dense2")(x)
    x = layers.Dropout(dense2_dropout, name="dropout2")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(x, task_type, num_classes)

    # Create model
    model_name = get_model_name("WaveNet_Tunable", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Tunable optimizer
    optimizer_name = hp.Choice("optimizer", values=["adam", "adamw", "rmsprop"])
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    # Compile model with gradient clipping
    opt = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


# ============================================================================
# HINTS AND BEST PRACTICES
# ============================================================================

"""
HINTS FOR USING WAVENET CAUSAL CNN IN TRADING:
==============================================

1. WHEN TO USE WAVENET vs TCN:
   - WaveNet: More specific architecture, optimized for temporal patterns
   - TCN: More flexible, can tune more parameters
   - Use WaveNet for: Tick data, order flow, microstructure
   - Use TCN for: General regime classification, pattern recognition

2. DUAL POOLING STRATEGY:
   - Max Pooling: Captures extremes (price spikes, volume surges, flash crashes)
   - Avg Pooling: Captures overall trends (mean reversion levels)
   - Together: Comprehensive market state representation
   - Benefits: Robust to outliers, captures both extremes and trends

3. KERNEL SIZE SELECTION:
   - kernel_size=2: WaveNet original, minimal parameters, fast
   - kernel_size=3: Slightly larger receptive field, more pattern capacity
   - Recommendation: Start with 2, increase to 3 if underfitting

4. DILATION STRATEGIES:
   For high-frequency (ticks, seconds):
   - Shallow: (1, 2, 4, 8) → RF=31
   - Medium: (1, 2, 4, 8, 16) → RF=63

   For low-frequency (minutes, hours):
   - Medium: (1, 2, 4, 8, 16) → RF=63
   - Deep: (1, 2, 4, 8, 16, 32) → RF=127

5. FILTER PROGRESSION:
   - Small: (32, 32, 32, 32) - Simple patterns
   - Medium: (32, 32, 64, 64) - Balanced (recommended)
   - Large: (32, 64, 128, 128) - Complex patterns, risk of overfitting

6. CAUSAL CONVOLUTIONS:
   - CRITICAL: Always use padding='causal'
   - Ensures no look-ahead bias
   - Model only sees: t-n, ..., t-1, t (never t+1, t+2, ...)
   - Essential for realistic backtesting

7. HIGH-FREQUENCY TRADING USE CASES:
   ✓ Order flow imbalance detection
   ✓ Tick-by-tick price movement prediction
   ✓ Market microstructure analysis
   ✓ Quote dynamics modeling
   ✓ Spread prediction
   ✓ Liquidity regime classification

8. FEATURE ENGINEERING FOR TICK DATA:
   Essential features:
   - Bid-ask spread
   - Order book imbalance
   - Trade volume
   - Price changes
   - Quote updates frequency
   - Time between trades

   Example preprocessing:
   ```python
   def preprocess_tick_data(df):
       # Calculate spread
       df['spread'] = df['ask'] - df['bid']
       df['spread_pct'] = df['spread'] / df['mid_price']

       # Order imbalance
       df['imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])

       # Price changes
       df['price_change'] = df['mid_price'].diff()
       df['price_change_pct'] = df['mid_price'].pct_change()

       # Standardize
       scaler = StandardScaler()
       scaled = scaler.fit_transform(df[features])

       return scaled
   ```

9. TRAINING TIPS FOR TICK DATA:
   - Use smaller batch sizes (16-32) for tick data
   - Learning rate: 0.0001 - 0.001
   - Gradient clipping essential (clipnorm=1.0)
   - Monitor for overfitting (tick patterns can be noisy)
   - Use validation set from different time period

10. RECEPTIVE FIELD CONSIDERATIONS:
    For kernel_size=2:
    - dilations=(1,2,4,8): RF=31 ticks (~few seconds for high-freq)
    - dilations=(1,2,4,8,16): RF=63 ticks (~10-30 seconds)
    - dilations=(1,2,4,8,16,32): RF=127 ticks (~1-2 minutes)

    Ensure: sequence_length >= receptive_field

11. ADVANTAGES OVER RNN/LSTM:
    ✓ 5-10x faster training (parallel processing)
    ✓ No vanishing gradients
    ✓ Better at capturing local patterns
    ✓ More stable training
    ✓ Naturally causal (no bias risk)

12. DISADVANTAGES:
    ✗ Less flexible than attention mechanisms
    ✗ Fixed receptive field
    ✗ May miss very long-range dependencies
    ✗ More parameters than simple RNN

13. ENSEMBLE STRATEGIES:
    Combine WaveNet with:
    - LSTM: WaveNet for patterns, LSTM for long-term dependencies
    - Attention: WaveNet for local, Attention for global
    - TCN: Different dilation strategies

14. MONITORING & DIAGNOSTICS:
    - Track both max and avg pooling outputs separately
    - Visualize learned filters (what patterns are captured?)
    - Monitor receptive field vs actual pattern timescales
    - Check if model focuses on recent vs distant past

15. COMMON PITFALLS:
    ✗ Using padding='same' instead of 'causal' (CRITICAL!)
    ✗ Receptive field smaller than important pattern length
    ✗ Too many filters (overfitting on noise)
    ✗ Not using gradient clipping (training instability)
    ✗ Forgetting to standardize tick data features
"""
