"""
Residual LSTM Network (ResLSTM)
================================

Architecture with residual (skip) connections enabling very deep LSTM stacks.
Residual connections improve gradient flow and allow training deeper networks.

Suitable for: complex market microstructure, multi-factor models, deep feature learning


References:
-----------

Core Residual Learning Concepts:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. "Deep Residual Learning for Image Recognition" - He et al. (2015)
   https://arxiv.org/abs/1512.03385
   Foundational work introducing residual connections (skip connections) for training very deep networks.
   Won ILSVRC 2015. Inspired application of residual connections to LSTM architectures.

2. "Highway Networks" - Srivastava, Greff, Schmidhuber (2015)
   https://arxiv.org/abs/1505.00387
   Applied LSTM gating principles to feedforward networks, enabling hundreds of layers.
   Direct inspiration for Highway LSTM architectures.

Residual LSTM Architectures:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
3. "Residual LSTM: Design of a Deep Recurrent Architecture for Distant Speech Recognition"
   - Kim, El-Khamy, Lee (2017)
   https://arxiv.org/abs/1703.10135
   Introduces Residual LSTM with separated spatial and temporal shortcut paths to avoid
   gradient flow conflicts. Key architecture for deep recurrent networks.

4. "Recurrent Highway Networks" - Zilly et al. (2017)
   https://arxiv.org/abs/1607.03474
   Incorporates Highway layers inside recurrent transitions for superior depth scaling.
   Proceedings of ICML 2017.

5. "Highway-LSTM and Recurrent Highway Networks for Speech Recognition"
   - Pundak & Sainath (2016)
   https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45941.pdf
   Extended LSTM with highway networks for increased depth in time dimension.
   Google Research, INTERSPEECH 2016.

Financial Time Series Applications:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
6. "Stacked CNN-LSTM with Residual for Financial Time Series Prediction"
   - Fu, Wu, Wang (2023)
   https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4367410
   Stacks multiple CNN-LSTM blocks with residual connections for stock price prediction.
   Preserves shallow information while discovering complex non-linear features.

7. "Deep Learning for Stock Prediction Using Numerical and Textual Information"
   - Hu et al. (2018)
   https://ieeexplore.ieee.org/document/8456632
   Uses ResLSTM variant for financial market prediction combining multiple data sources.

8. "Deep learning for financial forecasting: A review of recent trends" (2025)
   - Comprehensive review of 187 studies (2020-2024)
   https://www.sciencedirect.com/science/article/pii/S1059056025008822
   Documents residual-based LSTM architectures (KM-CAE-DLSTM, NR-LSTM, SCLR) as
   significant advancement in financial forecasting, combining decomposition with deep learning.

9. "A deep learning framework for financial time series using stacked autoencoders
   and long-short term memory" - Bao et al. (2017)
   https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180944
   Early work on deep LSTM architectures for financial time series forecasting.
"""

from typing import Literal

import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers, models

from okmich_quant_neural_net.keras.paper2keras.common import (
    TaskType,
    create_output_layer_and_loss,
    get_optimizer,
    get_model_name,
)

ArchitectureType = Literal['simple', 'complex', 'deep']


# ============================================================================
# SIMPLE RESIDUAL LSTM (Matches User Specification)
# ============================================================================


def build_simple_residual_lstm(sequence_length=96, num_features=15, num_classes=3, task_type=TaskType.CLASSIFICATION,
                               num_lstm_layers=4, lstm_units=32, dropout_rate=0.15, dense_units=64, learning_rate=0.001,
                               optimizer_name="adam", use_layer_norm=True):
    """
    Build Simple Residual LSTM with direct residual connections and LayerNorm.

    Supports both classification and regression tasks with automatic configuration.

    Architecture:
        Input → LSTM(32) → Dropout → Residual Add → LayerNorm
             → LSTM(32) → Dropout → Residual Add → LayerNorm
             → LSTM(32) → Dropout → Residual Add → LayerNorm
             → LSTM(32) → Dropout → Residual Add → LayerNorm
             → Dense → Output

    This is a cleaner, simpler architecture compared to the complex version:
    - Direct residual connections (x + LSTM(x)) without intermediate Dense layers
    - LayerNorm after each residual block for stable training
    - Smaller LSTM units (32 vs 128) for efficiency
    - Lower dropout (0.1-0.2) to preserve gradient flow

    Args:
        sequence_length: Number of timesteps in input sequences (default: 96)
        num_features: Number of features per timestep (default: 15)
        num_classes: Number of output classes for classification (default: 3)
        task_type: TaskType.CLASSIFICATION or TaskType.REGRESSION (default: CLASSIFICATION)
        num_lstm_layers: Number of LSTM layers (3-5 recommended) (default: 4)
        lstm_units: Units in each LSTM layer (default: 32)
        dropout_rate: Dropout rate after each LSTM (0.1-0.2) (default: 0.15)
        dense_units: Units in final dense layer (default: 64)
        learning_rate: Learning rate (default: 0.001)
        optimizer_name: Optimizer to use ('adam', 'adamw', 'rmsprop')
        use_layer_norm: Use LayerNorm after residual connections (default: True)

    Returns:
        Compiled Keras model

    Example (Classification):
        >>> model = build_simple_residual_lstm(
        ...     sequence_length=96,
        ...     num_features=15,
        ...     num_classes=3,
        ...     task_type=TaskType.CLASSIFICATION,
        ...     num_lstm_layers=4,
        ...     lstm_units=32
        ... )

    Example (Regression):
        >>> model = build_simple_residual_lstm(
        ...     sequence_length=96,
        ...     num_features=15,
        ...     task_type=TaskType.REGRESSION,
        ...     num_lstm_layers=4,
        ...     lstm_units=32
        ... )
    """
    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    x = inputs

    # Stack LSTM layers with residual connections
    for i in range(num_lstm_layers):
        # LSTM layer
        lstm_out = layers.LSTM(
            lstm_units,
            return_sequences=True,
            name=f"lstm_{i + 1}",
        )(x)
        lstm_out = layers.Dropout(dropout_rate, name=f"dropout_{i + 1}")(lstm_out)

        # Residual connection
        if i == 0:
            # First layer: project input features to lstm_units
            x_projected = layers.TimeDistributed(
                layers.Dense(lstm_units), name="input_projection"
            )(x)
            x = layers.Add(name=f"residual_{i + 1}")([x_projected, lstm_out])
        else:
            # Direct addition (dimensions already match)
            x = layers.Add(name=f"residual_{i + 1}")([x, lstm_out])

        # LayerNormalization for stable training
        if use_layer_norm:
            x = layers.LayerNormalization(name=f"layer_norm_{i + 1}")(x)

    # Extract final hidden state (last timestep) — "where are we now at the end of this window"
    x = layers.Lambda(lambda t: t[:, -1, :], name="last_timestep")(x)

    # Final dense layer
    x = layers.Dense(dense_units, activation="relu", name="dense_final")(x)
    x = layers.Dropout(dropout_rate, name="dropout_final")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("Simple_ResLSTM", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile model
    opt = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    print(f"\nSimple Residual LSTM Configuration:")
    print(f"  Task type: {task_type}")
    print(f"  LSTM layers: {num_lstm_layers}")
    print(f"  LSTM units: {lstm_units}")
    print(f"  Dropout: {dropout_rate}")
    print(f"  LayerNorm: {use_layer_norm}")
    print(f"  Total parameters: {model.count_params():,}")
    if task_type == TaskType.CLASSIFICATION:
        print(f"  Metrics: Accuracy, Balanced Accuracy, Macro F1 (class-imbalance robust)")
    else:
        print(f"  Metrics: MAE, MSE, RMSE, R², Directional Accuracy")

    return model


def build_simple_residual_lstm_tunable(hp, num_features, num_classes, task_type=TaskType.CLASSIFICATION,
                                       sequence_length=None, max_sequence_length=128):
    """
    Build tunable Simple Residual LSTM for hyperparameter optimization.

    Supports both classification and regression tasks.

    Args:
        hp: KerasTuner HyperParameters object
        num_features: Number of features per timestep (fixed)
        num_classes: Number of output classes for classification (fixed)
        task_type: TaskType.CLASSIFICATION or TaskType.REGRESSION (default: CLASSIFICATION)
        sequence_length: If None, tunes sequence_length. If provided, uses fixed value.
        max_sequence_length: Maximum sequence length (used when sequence_length=None)

    Returns:
        Compiled Keras model with tunable hyperparameters

    Example (Classification):
        >>> import keras_tuner
        >>> def model_builder(hp):
        ...     return build_simple_residual_lstm_tunable(
        ...         hp=hp,
        ...         num_features=15,
        ...         num_classes=3,
        ...         task_type=TaskType.CLASSIFICATION,
        ...         sequence_length=96  # Fixed
        ...     )
        >>> tuner = keras_tuner.BayesianOptimization(
        ...     model_builder,
        ...     objective='val_accuracy',
        ...     max_trials=20
        ... )

    Example (Regression):
        >>> def model_builder(hp):
        ...     return build_simple_residual_lstm_tunable(
        ...         hp=hp,
        ...         num_features=15,
        ...         num_classes=3,  # Not used for regression
        ...         task_type=TaskType.REGRESSION,
        ...         sequence_length=96
        ...     )
        >>> tuner = keras_tuner.BayesianOptimization(
        ...     model_builder,
        ...     objective='val_r2_score',
        ...     max_trials=20
        ... )
    """
    # Tunable hyperparameters
    if sequence_length is None:
        sequence_length = hp.Int(
            "sequence_length", min_value=32, max_value=max_sequence_length, step=16
        )

    num_lstm_layers = hp.Int("num_lstm_layers", min_value=3, max_value=5, step=1)
    lstm_units = hp.Choice("lstm_units", values=[16, 32, 64])
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.25, step=0.05)
    dense_units = hp.Choice("dense_units", values=[32, 64, 128])
    use_layer_norm = hp.Boolean("use_layer_norm", default=True)

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    x = inputs

    # Stack LSTM layers with residual connections
    for i in range(num_lstm_layers):
        lstm_out = layers.LSTM(
            lstm_units,
            return_sequences=True,
            name=f"lstm_{i + 1}",
        )(x)
        lstm_out = layers.Dropout(dropout_rate, name=f"dropout_{i + 1}")(lstm_out)

        # Residual connection
        if i == 0:
            x_projected = layers.TimeDistributed(
                layers.Dense(lstm_units), name="input_projection"
            )(x)
            x = layers.Add(name=f"residual_{i + 1}")([x_projected, lstm_out])
        else:
            x = layers.Add(name=f"residual_{i + 1}")([x, lstm_out])

        # LayerNormalization
        if use_layer_norm:
            x = layers.LayerNormalization(name=f"layer_norm_{i + 1}")(x)

    # Extract final hidden state (last timestep) — "where are we now at the end of this window"
    x = layers.Lambda(lambda t: t[:, -1, :], name="last_timestep")(x)

    # Final dense layer
    x = layers.Dense(dense_units, activation="relu", name="dense_final")(x)
    x = layers.Dropout(dropout_rate, name="dropout_final")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("Simple_ResLSTM_Tunable", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Tunable optimizer
    optimizer_name = hp.Choice("optimizer", values=["adam", "adamw"])
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    opt = get_optimizer(optimizer_name, learning_rate)

    # Compile model
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


# ============================================================================
# FACTORY FUNCTION - Chooses Architecture
# ============================================================================


def build_residual_lstm_factory(architecture: ArchitectureType = 'simple', sequence_length=96, num_features=15,
                                num_classes=3, task_type=TaskType.CLASSIFICATION, **kwargs):
    """
    Factory function to build different ResLSTM architectures.

    Supports both classification and regression tasks.

    Architecture Types:
    -------------------
    'simple': Clean architecture with direct residual connections and LayerNorm
              - 3-5 LSTM layers, 32 units each
              - Direct x + LSTM(x) residual connections
              - LayerNorm after each block
              - Dropout 0.1-0.2
              - Best for: persistent regimes, medium-term dependencies (50-120 timesteps)
              - Faster training, fewer parameters (~50K)

    'complex': Original architecture with TimeDistributed Dense layers
               - 3 LSTM layers (128, 128, 64 units)
               - LSTM → Dense → LSTM → Residual with projection
               - More expressive but complex
               - Best for: complex patterns, high-dimensional features (20+)
               - More parameters (~300K)

    'deep': Very deep architecture with multiple residual blocks
            - 4+ blocks (8+ LSTM layers)
            - 128 units per layer
            - Best for: very complex patterns, large datasets (10K+ samples)
            - Most parameters (~500K+)

    Args:
        architecture: Type of architecture ('simple', 'complex', 'deep')
        sequence_length: Number of timesteps
        num_features: Number of features per timestep
        num_classes: Number of output classes for classification
        task_type: TaskType.CLASSIFICATION or TaskType.REGRESSION (default: CLASSIFICATION)
        **kwargs: Additional architecture-specific parameters

    Returns:
        Compiled Keras model

    Example (Classification):
        >>> # Simple architecture (recommended for most cases)
        >>> model = build_residual_lstm_factory(
        ...     architecture='simple',
        ...     sequence_length=96,
        ...     num_features=15,
        ...     num_classes=3,
        ...     task_type=TaskType.CLASSIFICATION
        ... )

    Example (Regression):
        >>> model = build_residual_lstm_factory(
        ...     architecture='simple',
        ...     sequence_length=96,
        ...     num_features=15,
        ...     task_type=TaskType.REGRESSION
        ... )
    """
    if architecture == 'simple':
        return build_simple_residual_lstm(
            sequence_length=sequence_length,
            num_features=num_features,
            num_classes=num_classes,
            task_type=task_type,
            **kwargs
        )

    elif architecture == 'complex':
        return create_residual_lstm(
            input_shape=(sequence_length, num_features),
            num_classes=num_classes,
            task_type=task_type,
            **kwargs
        )

    elif architecture == 'deep':
        return create_deep_residual_lstm(
            input_shape=(sequence_length, num_features),
            num_classes=num_classes,
            task_type=task_type,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: 'simple', 'complex', 'deep'"
        )


# ============================================================================
# ORIGINAL COMPLEX RESIDUAL LSTM (Retained for Compatibility)
# ============================================================================


def create_residual_lstm(input_shape, num_classes, task_type=TaskType.CLASSIFICATION,
                         lstm_units_1=128, lstm_units_2=128, lstm_units_3=64, dense_units_1=128,
                         dense_units_2=128, dense_units_3=64, dropout_lstm=0.3, dropout_final=0.2,
                         learning_rate=0.001, l2_reg=0.0001):
    """
    Create a Residual LSTM Network with skip connections.

    Supports both classification and regression tasks.

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (sequence_length, n_features)
    num_classes : int
        Number of output classes for classification
    task_type : TaskType
        TaskType.CLASSIFICATION or TaskType.REGRESSION (default: CLASSIFICATION)
    lstm_units_1 : int, default=128
        Number of units in first LSTM layer
    lstm_units_2 : int, default=128
        Number of units in second LSTM layer (should match lstm_units_1 for clean residual)
    lstm_units_3 : int, default=64
        Number of units in third LSTM layer
    dense_units_1 : int, default=128
        Number of units in first TimeDistributed Dense (should match lstm_units_1)
    dense_units_2 : int, default=128
        Number of units in second TimeDistributed Dense (should match lstm_units_2)
    dense_units_3 : int, default=64
        Number of units in final Dense layer
    dropout_lstm : float, default=0.3
        Dropout rate after LSTM layers
    dropout_final : float, default=0.2
        Dropout rate after final dense layer
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
    # BLOCK 1: First LSTM + Residual
    # ============================================================

    # First LSTM layer
    lstm1 = layers.LSTM(
        lstm_units_1,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="lstm_1",
    )(inputs)
    lstm1 = layers.Dropout(dropout_lstm, name="dropout_lstm_1")(lstm1)

    # TimeDistributed Dense (applied to each timestep)
    dense1 = layers.TimeDistributed(
        layers.Dense(dense_units_1, kernel_regularizer=keras.regularizers.l2(l2_reg)),
        name="time_dense_1",
    )(lstm1)

    # Second LSTM layer
    lstm2 = layers.LSTM(
        lstm_units_2,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="lstm_2",
    )(dense1)
    lstm2 = layers.Dropout(dropout_lstm, name="dropout_lstm_2")(lstm2)

    # Residual connection 1: Add lstm1 to lstm2
    if lstm_units_1 == lstm_units_2:
        residual1 = layers.Add(name="residual_1")([lstm1, lstm2])
    else:
        # Project lstm1 to match lstm2 dimensions
        lstm1_projected = layers.TimeDistributed(
            layers.Dense(lstm_units_2, name="residual_projection_1"),
            name="time_residual_projection_1",
        )(lstm1)
        residual1 = layers.Add(name="residual_1")([lstm1_projected, lstm2])

    # ============================================================
    # BLOCK 2: Third LSTM + Residual
    # ============================================================

    # TimeDistributed Dense
    dense2 = layers.TimeDistributed(
        layers.Dense(dense_units_2, kernel_regularizer=keras.regularizers.l2(l2_reg)),
        name="time_dense_2",
    )(residual1)

    # Third LSTM layer (final, no return_sequences)
    lstm3 = layers.LSTM(
        lstm_units_3,
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="lstm_3",
    )(dense2)
    lstm3 = layers.Dropout(dropout_lstm, name="dropout_lstm_3")(lstm3)

    # Residual connection 2: From residual1 (last timestep) to lstm3
    # Extract last timestep from residual1
    residual1_last = layers.Lambda(lambda x: x[:, -1, :], name="extract_last_timestep")(
        residual1
    )

    # Project to match lstm3 dimensions if needed
    if lstm_units_2 == lstm_units_3:
        residual2 = layers.Add(name="residual_2")([residual1_last, lstm3])
    else:
        residual1_projected = layers.Dense(lstm_units_3, name="residual_projection_2")(
            residual1_last
        )
        residual2 = layers.Add(name="residual_2")([residual1_projected, lstm3])

    # ============================================================
    # FINAL LAYERS
    # ============================================================

    # Final Dense layer
    x = layers.Dense(
        dense_units_3,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="dense_final",
    )(residual2)
    x = layers.Dropout(dropout_final, name="dropout_final")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("residual_lstm", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile model
    opt = get_optimizer("adam", learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


def create_deep_residual_lstm(input_shape, num_classes, task_type=TaskType.CLASSIFICATION, num_blocks=4,
                              lstm_units=128, dropout_lstm=0.3, learning_rate=0.001, l2_reg=0.0001):
    """
    Create a very deep Residual LSTM Network with multiple residual blocks.
    Each block consists of: LSTM → Dropout → Dense → LSTM → Dropout → Residual

    Supports both classification and regression tasks.

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (sequence_length, n_features)
    num_classes : int
        Number of output classes for classification
    task_type : TaskType
        TaskType.CLASSIFICATION or TaskType.REGRESSION (default: CLASSIFICATION)
    num_blocks : int, default=4
        Number of residual blocks (each block = 2 LSTM layers)
    lstm_units : int, default=128
        Number of units in each LSTM layer (same for all layers)
    dropout_lstm : float, default=0.3
        Dropout rate after LSTM layers
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer
    l2_reg : float, default=0.0001
        L2 regularization factor

    Returns
    -------
    keras.Model
        Compiled Keras model with multiple residual blocks
    """

    # Input layer
    inputs = layers.Input(shape=input_shape, name="input")

    x = inputs

    # Build multiple residual blocks
    for block_idx in range(num_blocks):
        # First LSTM in block
        lstm_a = layers.LSTM(
            lstm_units,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=f"block_{block_idx + 1}_lstm_a",
        )(x)
        lstm_a = layers.Dropout(dropout_lstm, name=f"block_{block_idx + 1}_dropout_a")(
            lstm_a
        )

        # TimeDistributed Dense
        dense = layers.TimeDistributed(
            layers.Dense(lstm_units, kernel_regularizer=keras.regularizers.l2(l2_reg)),
            name=f"block_{block_idx + 1}_dense",
        )(lstm_a)

        # Second LSTM in block
        lstm_b = layers.LSTM(
            lstm_units,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=f"block_{block_idx + 1}_lstm_b",
        )(dense)
        lstm_b = layers.Dropout(dropout_lstm, name=f"block_{block_idx + 1}_dropout_b")(
            lstm_b
        )

        # Residual connection: Add input of block to output
        if block_idx == 0:
            # First block: project input features to lstm_units
            x_projected = layers.TimeDistributed(
                layers.Dense(lstm_units), name="input_projection"
            )(x)
            x = layers.Add(name=f"block_{block_idx + 1}_residual")(
                [x_projected, lstm_b]
            )
        else:
            # Later blocks: direct addition (dimensions already match)
            x = layers.Add(name=f"block_{block_idx + 1}_residual")([x, lstm_b])

    # Final LSTM layer (no return_sequences)
    lstm_final = layers.LSTM(
        lstm_units // 2,
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="lstm_final",
    )(x)
    lstm_final = layers.Dropout(dropout_lstm, name="dropout_final")(lstm_final)

    # Dense layer
    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="dense_final",
    )(lstm_final)
    x = layers.Dropout(0.2, name="dropout_dense")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("deep_residual_lstm", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile
    opt = get_optimizer("adam", learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


def create_tunable_residual_lstm(input_shape, num_classes, task_type=TaskType.CLASSIFICATION):
    """
    Create a tunable version of the Residual LSTM for hyperparameter optimization.

    Supports both classification and regression tasks.

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (sequence_length, n_features)
    num_classes : int
        Number of output classes for classification
    task_type : TaskType
        TaskType.CLASSIFICATION or TaskType.REGRESSION (default: CLASSIFICATION)

    Returns
    -------
    function
        Model builder function for keras_tuner

    Example (Classification)
    -------
    >>> tuner = kt.BayesianOptimization(
    ...     create_tunable_residual_lstm(
    ...         input_shape=(64, 10),
    ...         num_classes=3,
    ...         task_type=TaskType.CLASSIFICATION
    ...     ),
    ...     objective='val_accuracy',
    ...     max_trials=20,
    ...     directory='tuner_results',
    ...     project_name='residual_lstm'
    ... )
    >>> tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50)

    Example (Regression)
    -------
    >>> tuner = kt.BayesianOptimization(
    ...     create_tunable_residual_lstm(
    ...         input_shape=(64, 10),
    ...         num_classes=3,  # Not used for regression
    ...         task_type=TaskType.REGRESSION
    ...     ),
    ...     objective='val_r2_score',
    ...     max_trials=20
    ... )
    """

    def build_model(hp):
        # Hyperparameter search space
        lstm_units_1 = hp.Choice("lstm_units_1", values=[64, 128, 256])
        lstm_units_2 = lstm_units_1  # Keep same for clean residual
        lstm_units_3 = hp.Choice("lstm_units_3", values=[32, 64, 128])

        dense_units_1 = lstm_units_1  # Match LSTM units
        dense_units_2 = lstm_units_2  # Match LSTM units
        dense_units_3 = hp.Int("dense_units_3", min_value=32, max_value=128, step=32)

        dropout_lstm = hp.Float("dropout_lstm", min_value=0.2, max_value=0.5, step=0.1)
        dropout_final = hp.Float(
            "dropout_final", min_value=0.1, max_value=0.4, step=0.1
        )

        learning_rate = hp.Choice("learning_rate", values=[1e-4, 5e-4, 1e-3, 5e-3])
        l2_reg = hp.Choice("l2_reg", values=[1e-5, 1e-4, 1e-3])

        return create_residual_lstm(
            input_shape=input_shape,
            num_classes=num_classes,
            task_type=task_type,
            lstm_units_1=lstm_units_1,
            lstm_units_2=lstm_units_2,
            lstm_units_3=lstm_units_3,
            dense_units_1=dense_units_1,
            dense_units_2=dense_units_2,
            dense_units_3=dense_units_3,
            dropout_lstm=dropout_lstm,
            dropout_final=dropout_final,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
        )

    return build_model


"""
1. RESIDUAL CONNECTIONS BENEFIT:
   - Enable training very deep LSTM networks (10+ layers)
   - Provide shortcut paths for gradients to flow backward
   - Reduce vanishing gradient problem
   - Allow model to learn identity mapping (if needed, residual = 0)
   - Inspired by ResNet architecture for CNNs

2. WHEN TO USE RESIDUAL LSTM:
   ✓ Complex patterns requiring deep networks
   ✓ Multi-factor models with many interacting features
   ✓ Market microstructure with intricate dependencies
   ✓ High-dimensional feature spaces (20+ features)
   ✓ When vanilla LSTM underfits

   ✗ Simple patterns (use vanilla LSTM or GRU)
   ✗ Small datasets (<5000 samples) - risk of overfitting
   ✗ Need fast inference - deeper networks are slower

3. ARCHITECTURE DESIGN PRINCIPLES:
   - Keep LSTM units constant within residual blocks (128, 128)
   - Use TimeDistributed Dense between LSTMs in same block
   - Apply projection layers when dimensions change (128 → 64)
   - Add residual connection every 1-2 LSTM layers
   - Use gradient clipping (clipnorm=1.0) - IMPORTANT!

4. INPUT DATA PREPARATION:
   - Normalize/standardize features before training
   - Shape: (n_samples, sequence_length, n_features)
   - For multi-factor models: Include technical, fundamental, sentiment features
   - For microstructure: Include order flow, bid-ask spreads, volumes

5. TIMEDBUTED DENSE LAYERS:
   - Applied independently to each timestep
   - Useful for feature transformation within sequences
   - Enables non-linear transformations between LSTMs
   - Helps match dimensions for residual connections

6. GRADIENT CLIPPING (CRITICAL!):
   - ResLSTM prone to gradient explosion due to residual paths
   - Always use clipnorm=1.0 or clipvalue=0.5
   - Monitor gradients during training
   - If loss becomes NaN, reduce learning rate or increase clipping

7. TRAINING TIPS:
   - Batch size: 32-64 recommended
   - Learning rate: 0.001 (reduce to 0.0005 for deep models)
   - Patience: 10-15 epochs (deep models need more time)
   - Use learning rate warmup for very deep models
   - Monitor both train and val loss closely

8. HYPERPARAMETER TUNING PRIORITIES:
   High impact:
     - lstm_units_1 (64-256)
     - lstm_units_3 (32-128)
     - dropout_lstm (0.2-0.5)
     - learning_rate (1e-4 to 5e-3)

   Medium impact:
     - dense_units_3 (32-128)
     - dropout_final (0.1-0.4)
     - l2_reg (1e-5 to 1e-3)

   Low impact:
     - dense_units_1/2 (should match lstm_units for clean residuals)

9. ARCHITECTURE VARIANTS:
   - Add BatchNormalization after each LSTM:
       lstm = layers.BatchNormalization()(lstm)
       Helps with gradient flow, faster convergence

   - Use Bidirectional LSTMs:
       lstm = layers.Bidirectional(LSTM(...))(x)
       Better context, but 2x parameters

   - Highway connections instead of additive residuals:
       gate = sigmoid(W_g * x + b_g)
       output = gate * LSTM(x) + (1 - gate) * x
       Learned gating of residual connections

   - Dense skip connections (connect every layer to every other):
       Like DenseNet architecture

10. DEEP RESIDUAL LSTM (10+ LAYERS):
    - Use create_deep_residual_lstm() function
    - Start with num_blocks=4 (8 LSTM layers)
    - Increase to num_blocks=6-8 for very complex patterns
    - Monitor overfitting carefully with deep models
    - Requires more training data (10k+ samples)

11. CLASS IMBALANCE HANDLING:
    - Use class_weight in model.fit()
    - Example: class_weight={0: 1.0, 1: 2.5, 2: 1.5}
    - Consider focal loss for extreme imbalance
    - Monitor macro F1, balanced accuracy, Cohen's kappa

12. COMPARISON: VANILLA LSTM VS RESLSTM:
    Vanilla LSTM (2-3 layers):
    ✓ Simpler, fewer parameters
    ✓ Faster training and inference
    ✓ Sufficient for many trading tasks
    ✗ Limited depth (vanishing gradients)
    ✗ May underfit complex patterns

    Residual LSTM (3-10+ layers):
    ✓ Can go much deeper (10+ layers)
    ✓ Better gradient flow
    ✓ More expressive, captures complex patterns
    ✓ Better for multi-factor models
    ✗ More parameters, slower training
    ✗ Requires more data to avoid overfitting
    ✗ Harder to tune

13. MEMORY OPTIMIZATION:
    - Use mixed precision training:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)

    - Reduce lstm_units from 128 to 64-96
    - Use smaller batch_size (16-32)
    - Gradient checkpointing for very deep models

14. MULTI-FACTOR MODELS:
    - Input features: Technical (RSI, MACD, Bollinger), Fundamental (P/E, EPS),
      Sentiment (news, social media), Alternative data (satellite, credit card)
    - ResLSTM can learn complex feature interactions across time
    - Each LSTM layer captures different abstraction levels
    - Early layers: Low-level patterns (individual indicators)
    - Deep layers: High-level patterns (macro regime shifts)

15. MARKET MICROSTRUCTURE:
    - Input features: Order flow, bid-ask spreads, depth, trade sizes,
      cancel ratios, order arrival rates
    - Sequence: Last 50-200 market snapshots (100ms-1s intervals)
    - Classes: Price direction, liquidity regimes, volatility states
    - ResLSTM captures intricate dependencies in order flow dynamics

16. DEBUGGING TIPS:
    - If loss becomes NaN:
        * Reduce learning rate (0.0005 or 0.0001)
        * Increase gradient clipping (clipnorm=0.5)
        * Check for extreme values in input data
        * Reduce model depth

    - If model underfits:
        * Increase depth (more blocks)
        * Increase lstm_units (128 → 256)
        * Reduce dropout (0.3 → 0.2)
        * Check feature engineering

    - If model overfits:
        * Increase dropout (0.3 → 0.4)
        * Increase l2_reg (1e-4 → 1e-3)
        * Add more training data
        * Reduce depth

17. PRODUCTION DEPLOYMENT:
    - Model is slower than CNN/GRU: ~50-200ms inference on CPU
    - Consider model quantization for faster inference
    - Use ONNX Runtime for optimized serving
    - Batch predictions for multiple instruments
    - Cache intermediate LSTM states if processing incremental updates

18. COMPARISON WITH OTHER ARCHITECTURES:
    - vs GRU: ResLSTM has more parameters, slightly more expressive
    - vs Transformer: ResLSTM better for shorter sequences (<100)
    - vs CNN-LSTM: ResLSTM better for pure temporal patterns
    - vs Attention: ResLSTM more parameter-efficient, less interpretable

19. ABLATION STUDY:
    - Train both vanilla LSTM and ResLSTM on same data
    - Compare:
        * Training stability (loss curves)
        * Final performance (accuracy, F1)
        * Gradient norms (should be more stable in ResLSTM)
        * Training time (ResLSTM slower)

20. ADVANCED TECHNIQUES:
    - Residual LSTM + Attention:
        Add attention mechanism after final LSTM
        Attention over all residual block outputs

    - Multi-scale ResLSTM:
        Parallel ResLSTM branches with different lstm_units
        Concatenate outputs before classification

    - ResLSTM Autoencoder:
        Encoder: ResLSTM downsampling
        Decoder: ResLSTM upsampling
        Useful for anomaly detection, feature learning

    - Stochastic Depth ResLSTM:
        Randomly drop residual blocks during training
        Reduces overfitting, improves generalization
"""
