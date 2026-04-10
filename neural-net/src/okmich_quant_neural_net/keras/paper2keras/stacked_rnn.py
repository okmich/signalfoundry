"""
Stacked RNN (GRU / LSTM / BiLSTM) - Model Factory
=================================================

Architecture:
-------------
Input (32-128 timesteps, n_features)
    |
RNN Layer 1 (128 units, return_sequences=True)
    |
Dropout (0.2)
    |
RNN Layer 2 (64 units, return_sequences=False)
    |
Dropout (0.2)
    |
Dense (32, ReLU)
    |
Output (softmax for multi-class / sigmoid for binary)

Supported RNN Types:
--------------------
- 'gru': Stacked GRU - Lightweight, fast, good baseline
- 'lstm': Stacked LSTM - Better long-term memory, more parameters
- 'bilstm': Stacked Bidirectional LSTM - Best pattern recognition, 2x parameters

Key Features:
-------------
- Captures temporal dependencies in sequential market data
- Configurable RNN type for different use cases
- Suitable for: trend classification, regime detection, momentum

Comparison:
-----------
| Type   | Params | Speed  | Memory   | Best For              |
|--------|--------|--------|----------|-----------------------|
| GRU    | 1x     | Fast   | Short    | Real-time, baseline   |
| LSTM   | 1.25x  | Medium | Long     | Regime detection      |
| BiLSTM | 2.5x   | Slow   | Full seq | Pattern recognition   |

Usage:
------
1. Simple version (fixed hyperparameters):
   model = build_stacked_rnn(
       sequence_length=96,
       num_features=15,
       num_classes=3,
       rnn_type='gru'
   )

2. Tunable version (with KerasTuner):
   def model_builder(hp):
       return build_stacked_rnn_tunable(
           hp=hp,
           num_features=15,
           num_classes=3,
           rnn_type='lstm'
       )

Reference:
----------
- Cho et al. (2014) "Learning Phrase Representations using RNN Encoder-Decoder"
  https://arxiv.org/abs/1406.1078
- Hochreiter & Schmidhuber (1997) "Long Short-Term Memory"
  https://www.bioinf.jku.at/publications/older/2604.pdf
- Schuster & Paliwal (1997) "Bidirectional Recurrent Neural Networks"
  https://ieeexplore.ieee.org/document/650093
- Fischer & Krauss (2018) "Deep learning with LSTM networks for financial market predictions"
  https://www.sciencedirect.com/science/article/abs/pii/S0377221717310652
"""

from typing import Literal

import numpy as np
from keras import layers, models

# Import from our common modules
from okmich_quant_neural_net.keras.paper2keras.common import (
    TaskType,
    create_output_layer_and_loss,
    get_optimizer,
    get_model_name,
)

RNNType = Literal['gru', 'lstm']


# ============================================================================
# SIMPLE VERSION (Fixed Hyperparameters)
# ============================================================================


def build_stacked_rnn(sequence_length: int = 96, num_features: int = 15, num_classes: int = 3,
                      task_type: TaskType = TaskType.CLASSIFICATION, rnn_type: RNNType = 'gru', rnn1_units: int = 128,
                      rnn2_units: int = 64, dropout_rate: float = 0.2, dense_units: int = 32,
                      learning_rate: float = 0.001, optimizer_name: str = "adam"):
    """
    Build a Stacked RNN model (GRU, LSTM, or BiLSTM) for classification or regression.

    Args:
        sequence_length: Number of timesteps in input sequences (default: 96).
        num_features: Number of features per timestep (default: 15).
        num_classes: Number of output classes (for classification only, default: 3).
        task_type: TaskType.CLASSIFICATION or TaskType.REGRESSION (default: CLASSIFICATION).
        rnn_type: Type of RNN - 'gru', 'lstm', or 'bilstm' (default: 'gru').
        rnn1_units: Units in first RNN layer (default: 128).
        rnn2_units: Units in second RNN layer (default: 64).
        dropout_rate: Dropout rate after RNN layers (default: 0.2).
        dense_units: Units in dense layer before output (default: 32).
        learning_rate: Learning rate for optimizer (default: 0.001).
        optimizer_name: Optimizer - 'adam', 'adamw', 'rmsprop', 'sgd' (default: 'adam').

    Returns:
        Compiled Keras model.

    Note:
        BiLSTM output dimensions are doubled (rnn1_units*2, rnn2_units*2).
        BiLSTM requires full sequence at inference (not for real-time streaming).

    Example (Classification):
        >>> from okmich_quant_neural_net.keras.paper2keras.common import TaskType
        >>> model = build_stacked_rnn(
        ...     sequence_length=96,
        ...     num_features=15,
        ...     num_classes=3,
        ...     task_type=TaskType.CLASSIFICATION,
        ...     rnn_type='gru'
        ... )

    Example (Regression):
        >>> model = build_stacked_rnn(
        ...     sequence_length=96,
        ...     num_features=15,
        ...     task_type=TaskType.REGRESSION,
        ...     rnn_type='lstm'
        ... )
    """
    rnn_type = rnn_type.lower()
    if rnn_type not in ('gru', 'lstm'):
        raise ValueError(f"rnn_type must be 'gru' or 'lstm', got '{rnn_type}'")

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # Select RNN layer type and model name
    if rnn_type == 'gru':
        RNNLayer = layers.GRU
        model_name = "Stacked_GRU"
    else:  # lstm
        RNNLayer = layers.LSTM
        model_name = "Stacked_LSTM"

    # First RNN layer - returns sequences
    x = RNNLayer(rnn1_units, return_sequences=True, name="rnn1")(inputs)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)

    # Second RNN layer - returns final state only
    x = RNNLayer(rnn2_units, return_sequences=False, name="rnn2")(x)
    x = layers.Dropout(dropout_rate, name="dropout2")(x)

    # Dense layer
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)

    # Output layer (using common.py - handles both classification and regression!)
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Update model name with task type suffix
    model_name = get_model_name(model_name, task_type)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Get optimizer (using common.py)
    opt = get_optimizer(optimizer_name, learning_rate)

    # Compile model
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


# ============================================================================
# TUNABLE VERSION (With KerasTuner)
# ============================================================================


def build_stacked_rnn_tunable(hp, num_features: int, num_classes: int = 3,
                              task_type: TaskType = TaskType.CLASSIFICATION,
                              rnn_type: RNNType = 'gru', sequence_length: int = None, max_sequence_length: int = 128):
    """
    Build a Stacked RNN model with hyperparameter tuning.

    This version allows KerasTuner to optimize:
    - Sequence length (lookback period) - optional, can be fixed
    - RNN units in each layer
    - Dropout rate
    - Dense layer units
    - Learning rate
    - Optimizer choice

    Args:
        hp: KerasTuner HyperParameters object.
        num_features: Number of features per timestep (fixed).
        num_classes: Number of output classes (fixed).
        rnn_type: Type of RNN - 'gru', 'lstm', or 'bilstm' (default: 'gru').
        sequence_length: If None, tunes sequence_length (32 to max_sequence_length).
                        If provided, uses this fixed value (for pre-created sequences).
                        (default: None - will tune it).
        max_sequence_length: Maximum sequence length to consider when tuning (default: 128).
                            Ignored if sequence_length is provided.
        metric: Metric - 'accuracy', 'precision', 'recall', 'auc' (default: 'accuracy').

    Returns:
        Compiled Keras model with tunable hyperparameters.

    Example 1 - Tune all parameters including sequence_length:
        >>> import keras_tuner
        >>> def model_builder(hp):
        ...     return build_stacked_rnn_tunable(
        ...         hp=hp,
        ...         num_features=15,
        ...         num_classes=3,
        ...         rnn_type='lstm',
        ...         max_sequence_length=96  # Will tune from 32 to 96
        ...     )
        >>> tuner = keras_tuner.BayesianOptimization(
        ...     model_builder,
        ...     objective='val_accuracy',
        ...     max_trials=20
        ... )

    Example 2 - Fixed sequence_length (for pre-created sequences):
        >>> def model_builder(hp):
        ...     return build_stacked_rnn_tunable(
        ...         hp=hp,
        ...         num_features=15,
        ...         num_classes=3,
        ...         rnn_type='lstm',
        ...         sequence_length=48  # Fixed! Won't tune this parameter
        ...     )
    """
    rnn_type = rnn_type.lower()
    if rnn_type not in ('gru', 'lstm'):
        raise ValueError(f"rnn_type must be 'gru' or 'lstm', got '{rnn_type}'")

    # Tunable hyperparameters
    if sequence_length is None:
        # Tune sequence_length if not provided
        sequence_length = hp.Int(
            "sequence_length", min_value=32, max_value=max_sequence_length, step=16
        )
    # else: use the provided fixed sequence_length
    rnn1_units = hp.Choice("rnn1_units", values=[64, 128, 256])
    rnn2_units = hp.Choice("rnn2_units", values=[32, 64, 128])
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.4, step=0.1)
    dense_units = hp.Choice("dense_units", values=[16, 32, 64])

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # Select RNN layer type and model name
    if rnn_type == 'gru':
        RNNLayer = layers.GRU
        model_name = "Stacked_GRU_Tunable"
    else:  # lstm
        RNNLayer = layers.LSTM
        model_name = "Stacked_LSTM_Tunable"

    # First RNN layer
    x = RNNLayer(rnn1_units, return_sequences=True, name="rnn1")(inputs)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)

    # Second RNN layer
    x = RNNLayer(rnn2_units, return_sequences=False, name="rnn2")(x)
    x = layers.Dropout(dropout_rate, name="dropout2")(x)

    # Dense layer
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)

    # Output layer (using common.py)
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Update model name with task type suffix
    model_name = get_model_name(model_name, task_type)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Tunable optimizer
    optimizer_name = hp.Choice("optimizer", values=["adam", "adamw", "rmsprop"])

    # Task-specific learning rate ranges
    if task_type == TaskType.CLASSIFICATION:
        learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    else:  # Regression often needs lower learning rates
        learning_rate = hp.Choice("learning_rate", values=[1e-3, 1e-4, 1e-5])

    # Get optimizer (using common.py)
    opt = get_optimizer(optimizer_name, learning_rate)

    # Compile model
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


# ============================================================================
# HINTS AND BEST PRACTICES
# ============================================================================

"""
HINTS FOR USING STACKED RNN IN TRADING:
=======================================

1. RNN TYPE SELECTION:

   GRU (Recommended Start):
   - Fastest training and inference
   - ~25% fewer parameters than LSTM
   - Good for short-term patterns
   - Best for: real-time signals, baseline models

   LSTM:
   - Separate cell state for long-term memory
   - Better for regime detection
   - More research literature available
   - Best for: multi-day patterns, regime classification

   BiLSTM:
   - Processes both directions
   - 2x parameters of LSTM
   - Requires full sequence (no streaming)
   - Best for: pattern recognition, historical labeling
   - WARNING: Has look-ahead in backward pass

2. SEQUENCE LENGTH SELECTION:
   For 5-minute bars:
   - 48 timesteps = 4 hours (intraday momentum)
   - 96 timesteps = 8 hours (full session) - RECOMMENDED
   - 144 timesteps = 12 hours (extended)
   - 288 timesteps = 24 hours (daily patterns)

3. LAYER SIZING:
   Decreasing pattern (128 -> 64) is recommended:
   - First layer: captures complex patterns
   - Second layer: compresses to representations

   Configurations:
   - Light: [64, 32] - fast, less capacity
   - Medium: [128, 64] - balanced (RECOMMENDED)
   - Heavy: [256, 128] - more capacity, overfitting risk

4. OPTIMIZER SELECTION:
   - Adam: Best general-purpose (RECOMMENDED)
   - Adagrad: Good for sparse features
   - RMSprop: Handles non-stationary objectives

   Learning rates: Start with 1e-3, tune from [1e-2, 1e-3, 1e-4]

5. DROPOUT STRATEGY:
   - 0.2: Light regularization (large datasets)
   - 0.3: Moderate (RECOMMENDED)
   - 0.4: Heavy (small datasets)

6. FEATURE ENGINEERING:
   Recommended features (15):
   - Price: returns, log_returns, high_low_range
   - Volume: volume_change, volume_ma_ratio
   - Technical: RSI, MACD, BB_position, ATR
   - Time: hour_of_day, day_of_week (encoded)

7. TRAINING TIPS:
   - Batch size: 32-64
   - Early stopping: patience=10-15
   - Learning rate schedule: ReduceLROnPlateau
   - Gradient clipping if unstable (clipnorm=1.0)

8. COMMON PITFALLS:
   - Not normalizing features (RNNs sensitive to scale)
   - Using raw prices instead of returns
   - Sequence too long causing vanishing gradients
   - Using BiLSTM for real-time (has look-ahead)
   - Forgetting look-ahead bias in labels

9. WALK-FORWARD VALIDATION:
   ```python
   def create_sequences(data, labels, seq_len):
       X, y = [], []
       for i in range(len(data) - seq_len):
           X.append(data[i:i+seq_len])
           y.append(labels[i+seq_len])
       return np.array(X), np.array(y)
   ```

10. ENSEMBLE STRATEGIES:
    - Train GRU, LSTM, BiLSTM and average predictions
    - Use different sequence lengths
    - Combine with CNN for feature extraction

11. PRODUCTION CONSIDERATIONS:
    | Type   | Size   | Inference | Memory  | Deploy         |
    |--------|--------|-----------|---------|----------------|
    | GRU    | ~200KB | <10ms     | Low     | Edge, realtime |
    | LSTM   | ~300KB | ~15ms     | Medium  | Cloud, batch   |
    | BiLSTM | ~600KB | ~30ms     | Higher  | Batch only     |

12. WHEN TO USE WHICH:
    - Quick prototype: GRU
    - Regime detection: LSTM
    - Pattern labeling: BiLSTM
    - Live trading: GRU or LSTM (not BiLSTM)
    - Backtesting: Any
"""

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STACKED RNN - MODEL FACTORY")
    print("Supports: GRU, LSTM, BiLSTM")
    print("=" * 80)
    print("\nAvailable examples:")
    print("  1. Simple usage (all three RNN types)")
    print("  2. Tunable usage (hyperparameter optimization)")
    print("  3. Training and evaluation")
    print("  4. Inference time comparison")
    print("  5. Run all examples")
    print("=" * 80)

    choice = input("\nSelect example to run (1-5, or 'q' to quit): ").strip()

    if choice == "1":
        model = example_simple_usage()
    elif choice == "2":
        tuner = example_tunable_usage()
    elif choice == "3":
        model, history = example_training()
    elif choice == "4":
        example_inference_comparison()
    elif choice == "5":
        print("\nRunning all examples...\n")
        print("\n" + ">" * 80)
        example_simple_usage()
        print("\n" + ">" * 80)
        example_tunable_usage()
        print("\n" + ">" * 80)
        example_training()
        print("\n" + ">" * 80)
        example_inference_comparison()
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED!")
        print("=" * 80)
    elif choice.lower() == "q":
        print("\nExiting...")
    else:
        print("\nInvalid choice. Please run again and select 1-5 or 'q'.")

    print("\n" + "=" * 80)
    print("For more details, see the HINTS section in the source code.")
    print("=" * 80)
