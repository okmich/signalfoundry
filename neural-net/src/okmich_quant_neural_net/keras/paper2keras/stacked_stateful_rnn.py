"""
Stateful RNN for Continuous Market State Tracking - Model Factory
=================================================================

Architecture:
-------------
Input (batch_size=1, timesteps=1, n_features) - One bar at a time
    |
Stateful RNN Layer 1 (128 units, stateful=True, return_sequences=True)
    |
Dropout (0.2)
    |
Stateful RNN Layer 2 (64 units, stateful=True, return_sequences=False)
    |
Dropout (0.2)
    |
Dense (32, ReLU)
    |
Output (softmax for multi-class / sigmoid for binary)

Supported RNN Types:
--------------------
- 'gru': Stacked Stateful GRU
- 'lstm': Stacked Stateful LSTM
- 'bilstm': Stacked Stateful Bidirectional LSTM

Key Features:
-------------
- Preserves hidden state between predictions (true online learning)
- Mimics how a trader maintains mental picture of evolving market
- Processes one bar at a time with continuous context
- Not limited to fixed lookback window
- Requires manual state reset on session boundaries

Stateful vs Stateless:
----------------------
Stateless (default):
  - Hidden state resets to zeros for each batch
  - Fixed lookback window (e.g., last 96 bars)
  - Faster training, simpler data preparation

Stateful:
  - Hidden state carries over between batches
  - Unlimited effective memory
  - Captures evolving regimes, momentum build-up
  - Requires batch_size=1 and manual state management

Usage:
------
1. Simple version (fixed hyperparameters):
   model = build_stateful_rnn(
       num_features=15,
       num_classes=3,
       rnn_type='gru'
   )

2. Tunable version (with KerasTuner):
   def model_builder(hp):
       return build_stateful_rnn_tunable(
           hp=hp,
           num_features=15,
           num_classes=3,
           rnn_type='gru'
       )

3. Live trading with ModelManager:
   from okmich_quant_neural_net.keras.stateful import ModelManager

   model = build_stateful_rnn(num_features=15, num_classes=3)
   manager = ModelManager(model, stability_period=5)

   # Session start
   manager.initialize_state(warmup_data)

   # Each bar
   prediction = manager.predict(current_bar)

   # Session end
   manager.reset_state()

Reference:
----------
- "The comparison stateless and stateful LSTM architectures for short-term stock price forecasting"
  https://www.growingscience.com/ijds/Vol8/ijdns_2024_9.pdf
- Gers et al. (2000) "Learning to Forget: Continual Prediction with LSTM"
  https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
"""

from typing import Literal

import numpy as np
from keras import layers, models

# Import task type and common utilities
from .common import TaskType, create_output_layer_and_loss, get_optimizer, get_model_name

RNNType = Literal['gru', 'lstm']


# ============================================================================
# SIMPLE VERSION (Fixed Hyperparameters)
# ============================================================================


def build_stateful_rnn(num_features: int = 15, num_classes: int = 3, task_type: TaskType = TaskType.CLASSIFICATION,
                       rnn_type: RNNType = 'gru', rnn1_units: int = 128, rnn2_units: int = 64,
                       dropout_rate: float = 0.2, dense_units: int = 32, learning_rate: float = 0.001,
                       optimizer_name: str = "adam", batch_size: int = 1, ):
    """
    Build a Stateful RNN model for continuous market state tracking.

    This model preserves hidden state between predictions, enabling true
    online learning where the model maintains context as new bars arrive.

    Args:
        num_features: Number of features per timestep.
        num_classes: Number of output classes.
        rnn_type: Type of RNN ('gru', 'lstm', 'bilstm').
        rnn1_units: Units in first RNN layer (default: 128).
        rnn2_units: Units in second RNN layer (default: 64).
        dropout_rate: Dropout rate after RNN layers (default: 0.2).
        dense_units: Units in dense layer before output (default: 32).
        learning_rate: Learning rate for optimizer (default: 0.001).
        optimizer_name: Optimizer ('adam', 'adagrad', 'rmsprop').
        batch_size: Batch size for stateful model (default: 1).

    Returns:
        Compiled stateful Keras model.

    Note:
        - Stateful models require fixed batch_size at build time
        - Use model.reset_states() on session boundaries
        - Feed one timestep at a time: shape (batch_size, 1, num_features)

    Example:
        >>> model = build_stateful_rnn(
        ...     num_features=15,
        ...     num_classes=3,
        ...     rnn_type='gru'
        ... )
        >>> # Feed one bar at a time
        >>> prediction = model.predict(bar_data.reshape(1, 1, -1))
        >>> # Reset on new session
        >>> model.reset_states()
    """
    rnn_type = rnn_type.lower()
    if rnn_type not in ('gru', 'lstm'):
        raise ValueError(f"rnn_type must be 'gru' or 'lstm', got '{rnn_type}'")

    # Input layer with fixed batch size for stateful model
    # Shape: (batch_size, None, features) - supports both training chunks and live inference
    # During training: feed (batch_size, chunk_size, features)
    # During live inference: feed (batch_size, 1, features)
    inputs = layers.Input(batch_shape=(batch_size, None, num_features), name="input_bar")

    # Select RNN layer type
    if rnn_type == 'gru':
        RNNLayer = layers.GRU
        model_name = "Stateful_GRU"
    else:  # lstm
        RNNLayer = layers.LSTM
        model_name = "Stateful_LSTM"

    # First RNN layer - stateful, returns sequences
    x = RNNLayer(rnn1_units, return_sequences=True, stateful=True, name="rnn1")(inputs)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)

    # Second RNN layer - stateful, returns final state only
    x = RNNLayer(rnn2_units, return_sequences=False, stateful=True, name="rnn2")(x)
    x = layers.Dropout(dropout_rate, name="dropout2")(x)

    # Dense layer
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Update model name based on task type
    base_model_name = model_name
    model_name = get_model_name(base_model_name, task_type)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile model
    opt = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


# ============================================================================
# INFERENCE MODEL — batch_size=1 copy for ModelManager / live trading
# ============================================================================


def build_inference_model(trained_model):
    """
    Build a ``batch_size=1`` inference copy of a trained stateful model and
    transfer its weights.

    Training uses ``batch_size=N`` (N parallel lanes) for throughput.
    Live inference and evaluation use ``batch_size=1`` (one bar at a time via
    ``ModelManager``).  The two models share the same architecture and weights
    — only the fixed batch dimension differs.

    Parameters
    ----------
    trained_model :
        A compiled stateful model returned by ``build_stateful_rnn`` with any
        ``batch_size >= 1``.

    Returns
    -------
    inference_model :
        An identical architecture compiled with ``batch_size=1``, with weights
        copied from ``trained_model``.  Ready to pass to ``ModelManager``.

    Example
    -------
    >>> train_model = build_stateful_rnn(num_features=26, num_classes=3,
    ...                                  rnn_type='gru', batch_size=4)
    >>> trainer = StatefulTrainer(train_model, ...)
    >>> trainer.fit(X_train, y_train, X_val, y_val)
    >>>
    >>> inference_model = build_inference_model(train_model)
    >>> manager = ModelManager(inference_model, stability_period=50)
    >>> manager.initialize_state(warmup_data)
    >>> pred = manager.predict(bar)
    """
    layer_names = {l.name for l in trained_model.layers}

    # --- num_features ---
    num_features = trained_model.input_shape[-1]

    # --- rnn_type and units ---
    rnn1_layer = trained_model.get_layer('rnn1')
    rnn_type = 'gru' if isinstance(rnn1_layer, layers.GRU) else 'lstm'
    rnn1_units = rnn1_layer.units
    rnn2_units = trained_model.get_layer('rnn2').units

    # --- dropout and dense ---
    dropout_rate = trained_model.get_layer('dropout1').rate
    dense_units = trained_model.get_layer('dense1').units

    # --- num_classes and task_type from output layer ---
    output_cfg = trained_model.get_layer('output').get_config()
    num_classes = output_cfg['units']
    activation = output_cfg.get('activation', 'softmax')
    task_type = TaskType.REGRESSION if activation == 'linear' else TaskType.CLASSIFICATION

    # --- build batch_size=1 model and transfer weights ---
    inference_model = build_stateful_rnn(
        num_features=num_features,
        num_classes=num_classes,
        task_type=task_type,
        rnn_type=rnn_type,
        rnn1_units=rnn1_units,
        rnn2_units=rnn2_units,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        batch_size=1,
    )
    inference_model.set_weights(trained_model.get_weights())
    return inference_model


# ============================================================================
# TUNABLE VERSION (With KerasTuner)
# ============================================================================


def build_stateful_rnn_tunable(hp, num_features: int, num_classes: int, task_type: TaskType = TaskType.CLASSIFICATION,
                               rnn_type: RNNType = 'gru', batch_size: int = 1):
    """
    Build a Stateful RNN model with hyperparameter tuning.

    This version allows KerasTuner to optimize:
    - RNN units in each layer
    - Dropout rate
    - Dense layer units
    - Learning rate
    - Optimizer choice

    Args:
        hp: KerasTuner HyperParameters object.
        num_features: Number of features per timestep (fixed).
        num_classes: Number of output classes (fixed).
        rnn_type: Type of RNN ('gru', 'lstm', 'bilstm').
        metric: Metric ('accuracy', 'precision', 'recall', 'auc').
        batch_size: Batch size for stateful model (default: 1).

    Returns:
        Compiled stateful Keras model with tunable hyperparameters.

    Example:
        >>> import keras_tuner
        >>> def model_builder(hp):
        ...     return build_stateful_rnn_tunable(
        ...         hp=hp,
        ...         num_features=15,
        ...         num_classes=3,
        ...         rnn_type='gru'
        ...     )
        >>> tuner = keras_tuner.BayesianOptimization(
        ...     model_builder,
        ...     objective='val_accuracy',
        ...     max_trials=20
        ... )
    """
    rnn_type = rnn_type.lower()
    if rnn_type not in ('gru', 'lstm'):
        raise ValueError(f"rnn_type must be 'gru' or 'lstm', got '{rnn_type}'")

    # Tunable hyperparameters
    rnn1_units = hp.Choice("rnn1_units", values=[64, 128, 256])
    rnn2_units = hp.Choice("rnn2_units", values=[32, 64, 128])
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.4, step=0.1)
    dense_units = hp.Choice("dense_units", values=[16, 32, 64])

    # Input layer with fixed batch size, variable timesteps
    inputs = layers.Input(batch_shape=(batch_size, None, num_features), name="input_bar")

    # Select RNN layer type
    if rnn_type == 'gru':
        RNNLayer = layers.GRU
        model_name = "Stateful_GRU_Tunable"
    else:  # lstm
        RNNLayer = layers.LSTM
        model_name = "Stateful_LSTM_Tunable"

    # First RNN layer
    x = RNNLayer(rnn1_units, return_sequences=True, stateful=True, name="rnn1")(inputs)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)

    # Second RNN layer
    x = RNNLayer(rnn2_units, return_sequences=False, stateful=True, name="rnn2")(x)
    x = layers.Dropout(dropout_rate, name="dropout2")(x)

    # Dense layer
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Update model name based on task type
    base_model_name = model_name
    model_name = get_model_name(base_model_name, task_type)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Tunable optimizer
    optimizer_name = hp.Choice("optimizer", values=["adam", "adagrad", "rmsprop"])
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    # Compile model
    opt = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def prepare_stateful_training_data(X: np.ndarray, y: np.ndarray):
    """
    Prepare data for stateful model training.

    Stateful models require data to be fed one timestep at a time with
    batch_size=1. This function reshapes sequence data appropriately.

    Args:
        X: Input data of shape (num_samples, sequence_length, num_features)
           or (num_samples, num_features) for single timestep.
        y: Labels of shape (num_samples,) or (num_samples, sequence_length).

    Returns:
        X_stateful: Reshaped to (num_samples * sequence_length, 1, num_features)
        y_stateful: Reshaped to match X_stateful if needed.

    Example:
        >>> X = np.random.randn(100, 48, 15)  # 100 samples, 48 timesteps, 15 features
        >>> y = np.random.randint(0, 3, 100)
        >>> X_train, y_train = prepare_stateful_training_data(X, y)
        >>> # X_train: (4800, 1, 15), y_train: (4800,)
    """
    if X.ndim == 2:
        # Already single timestep: (samples, features) -> (samples, 1, features)
        return X.reshape(-1, 1, X.shape[-1]), y

    # Sequence data: (samples, seq_len, features) -> (samples*seq_len, 1, features)
    num_samples, seq_len, num_features = X.shape
    X_stateful = X.reshape(-1, 1, num_features)

    # Repeat labels for each timestep (label applies to end of sequence)
    # For training, we typically only care about the final prediction
    if y.ndim == 1:
        y_stateful = np.repeat(y, seq_len)
    else:
        y_stateful = y.reshape(-1)

    return X_stateful, y_stateful


class StatefulTrainingCallback:
    """
    Callback helper for training stateful models.

    Resets model state at the start of each epoch and optionally
    between sequences within an epoch.

    Usage:
        >>> callback = StatefulTrainingCallback(model, reset_per_sequence=True, sequence_length=48)
        >>> for epoch in range(epochs):
        ...     callback.on_epoch_begin()
        ...     for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
        ...         callback.on_batch_begin(batch_idx)
        ...         model.train_on_batch(X_batch, y_batch)
    """

    def __init__(self, model, reset_per_sequence: bool = True, sequence_length: int = 1):
        self.model = model
        self.reset_per_sequence = reset_per_sequence
        self.sequence_length = sequence_length
        self.batch_counter = 0

    def on_epoch_begin(self):
        """Reset state at the start of each epoch."""
        self.model.reset_states()
        self.batch_counter = 0

    def on_batch_begin(self, batch_idx: int):
        """Reset state at sequence boundaries if configured."""
        if self.reset_per_sequence and batch_idx > 0:
            if batch_idx % self.sequence_length == 0:
                self.model.reset_states()
        self.batch_counter = batch_idx


# ============================================================================
# USAGE EXAMPLES
# ============================================================================


def example_simple_usage():
    """Example: Using the simple (fixed) version."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Stateful RNN (Fixed Hyperparameters)")
    print("=" * 80)

    num_features = 15
    num_classes = 3

    # Build all three RNN types
    for rnn_type in ['gru', 'lstm']:
        print(f"\n--- Building Stateful {rnn_type.upper()} ---")
        model = build_stateful_rnn(
            num_features=num_features,
            num_classes=num_classes,
            rnn_type=rnn_type,
            rnn1_units=128,
            rnn2_units=64,
            dropout_rate=0.2,
            dense_units=32,
        )
        model.summary()
        print(f"Total parameters: {model.count_params():,}")

    # Demonstrate stateful inference
    print("\n--- Demonstrating Stateful Inference ---")
    model = build_stateful_rnn(num_features=num_features, num_classes=num_classes, rnn_type='gru')

    # Simulate feeding bars one at a time
    print("\nSimulating live inference (5 bars):")
    for i in range(5):
        bar_data = np.random.randn(1, 1, num_features).astype(np.float32)
        prediction = model.predict(bar_data, verbose=0)
        print(f"  Bar {i + 1}: prediction = {prediction[0]}")

    # Reset state (new session)
    print("\nResetting state (new trading session)...")
    model.reset_states()

    # Continue inference
    print("Continuing inference after reset:")
    for i in range(3):
        bar_data = np.random.randn(1, 1, num_features).astype(np.float32)
        prediction = model.predict(bar_data, verbose=0)
        print(f"  Bar {i + 1}: prediction = {prediction[0]}")

    return model


def example_with_model_manager():
    """Example: Using with ModelManager for live trading."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Stateful RNN with ModelManager")
    print("=" * 80)

    from okmich_quant_neural_net.keras.stateful import ModelManager

    num_features = 15
    num_classes = 3

    # Build stateful model
    model = build_stateful_rnn(
        num_features=num_features,
        num_classes=num_classes,
        rnn_type='gru',
    )

    # Create manager with stability period
    manager = ModelManager(model, stability_period=5)

    # Simulate session start with warmup data
    print("\nSimulating trading session...")
    warmup_data = np.random.randn(20, num_features).astype(np.float32)

    print(f"Warming up with {len(warmup_data)} historical bars...")
    manager.initialize_state(warmup_data)

    # Simulate live bars
    print("\nProcessing live bars:")
    for i in range(10):
        bar_data = np.random.randn(num_features).astype(np.float32)
        prediction = manager.predict(bar_data)

        if prediction is None:
            print(f"  Bar {i + 1}: Stabilizing...")
        else:
            pred_class = np.argmax(prediction[0])
            print(f"  Bar {i + 1}: Class {pred_class} (probs: {prediction[0].round(3)})")

    # End of session
    print("\nEnd of trading session - resetting state...")
    manager.reset_state()
    print(f"Manager state: is_warmed_up={manager.is_warmed_up}, is_stable={manager.is_stable}")

    return model, manager


def example_training():
    """Example: Training a stateful model."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Training Stateful RNN")
    print("=" * 80)

    num_features = 15
    num_classes = 3
    sequence_length = 48
    num_sequences = 100

    # Generate synthetic training data
    print(f"\nGenerating {num_sequences} sequences of length {sequence_length}...")
    X_raw = np.random.randn(num_sequences, sequence_length, num_features).astype(np.float32)
    y_raw = np.random.randint(0, num_classes, num_sequences)

    # Prepare for stateful training
    X_train, y_train = prepare_stateful_training_data(X_raw, y_raw)
    print(f"Prepared data shape: X={X_train.shape}, y={y_train.shape}")

    # Build model
    model = build_stateful_rnn(
        num_features=num_features,
        num_classes=num_classes,
        rnn_type='gru',
    )

    # Training loop with manual state reset
    print("\nTraining with manual state management...")
    epochs = 3
    callback = StatefulTrainingCallback(model, reset_per_sequence=True, sequence_length=sequence_length)

    for epoch in range(epochs):
        callback.on_epoch_begin()
        epoch_loss = 0

        for i in range(len(X_train)):
            callback.on_batch_begin(i)
            loss = model.train_on_batch(X_train[i:i + 1], y_train[i:i + 1])
            epoch_loss += loss[0] if isinstance(loss, list) else loss

        avg_loss = epoch_loss / len(X_train)
        print(f"  Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.4f}")

    print("\nTraining complete!")
    return model


# ============================================================================
# HINTS AND BEST PRACTICES
# ============================================================================

"""
HINTS FOR USING STATEFUL RNN IN TRADING:
========================================

1. WHEN TO USE STATEFUL VS STATELESS:
   Stateful (this model):
   - Live inference: feed one bar at a time
   - Unlimited effective memory
   - Captures evolving regimes
   - More complex training/state management

   Stateless (standard models):
   - Batch training/inference
   - Fixed lookback window
   - Simpler to train and deploy
   - Better for backtesting

2. STATE MANAGEMENT RULES:
   - ALWAYS reset state at session boundaries (new trading day)
   - CONSIDER resetting on major regime changes
   - NEVER reset mid-sequence during training
   - Use ModelManager for production deployment

3. DATA PREPARATION:
   - Training: reshape (samples, seq_len, features) -> (samples*seq_len, 1, features)
   - Inference: feed (1, 1, features) one bar at a time
   - Warmup: feed historical data to initialize state

4. TRAINING CONSIDERATIONS:
   - Cannot shuffle data (sequence order matters)
   - Reset state at epoch boundaries
   - Reset state between independent sequences
   - Use smaller learning rates (state accumulates errors)

5. WARMUP STRATEGY:
   Recommended warmup length:
   - 20-50 bars for short-term patterns
   - 100-200 bars for regime detection
   - Previous session's data is ideal

6. RNN TYPE SELECTION:
   - GRU: Fastest, good for most cases (recommended start)
   - LSTM: Better long-term memory, more parameters
   - BiLSTM: Best pattern recognition, but has look-ahead in backward pass

   Note: BiLSTM stateful is unusual - the backward pass still processes
   forward in time for stateful models, limiting its bidirectional benefit.

7. STABILITY PERIOD:
   After reset, the model needs time to "warm up" its internal state.
   Typical stability periods:
   - 5-10 bars for short-term models
   - 20-30 bars for regime models
   - Match to your warmup strategy

8. BATCH SIZE:
   - Must be fixed at model build time
   - batch_size=1 for live inference
   - Cannot change without rebuilding model

9. MEMORY CONSIDERATIONS:
   - Stateful models hold state tensors in memory
   - State size = batch_size * units * (1 for GRU, 2 for LSTM)
   - BiLSTM doubles the state size

10. DEBUGGING TIPS:
    - Print state values to verify they're changing
    - Compare predictions before/after reset
    - Verify warmup is actually running (not skipped)

11. COMMON PITFALLS:
    - Forgetting to reset state between sessions
    - Shuffling training data
    - Using batch_size > 1 for live inference
    - Not warming up before first prediction
    - Expecting BiLSTM to work like stateless BiLSTM

12. PRODUCTION CHECKLIST:
    [ ] Model built with batch_size=1
    [ ] ModelManager configured with stability_period
    [ ] Warmup data pipeline ready
    [ ] Reset schedule defined (daily, weekly, etc.)
    [ ] Monitoring for state drift
    [ ] Fallback for model failures

13. COMPARISON TABLE:
    | Aspect          | Stateful        | Stateless       |
    |-----------------|-----------------|-----------------|
    | Memory          | Unlimited       | Fixed window    |
    | Training        | Complex         | Simple          |
    | Live inference  | Natural fit     | Requires window |
    | Batch size      | Fixed (usually 1)| Flexible       |
    | Data prep       | Sequential      | Can shuffle     |
    | State management| Manual          | Automatic       |
"""

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STATEFUL RNN - MODEL FACTORY")
    print("Supports: GRU, LSTM, BiLSTM")
    print("=" * 80)
    print("\nAvailable examples:")
    print("  1. Simple usage (all three RNN types)")
    print("  2. With ModelManager (live trading simulation)")
    print("  3. Training stateful model")
    print("  4. Run all examples")
    print("=" * 80)

    choice = input("\nSelect example to run (1-4, or 'q' to quit): ").strip()

    if choice == "1":
        model = example_simple_usage()
    elif choice == "2":
        model, manager = example_with_model_manager()
    elif choice == "3":
        model = example_training()
    elif choice == "4":
        print("\nRunning all examples...\n")
        print("\n" + ">" * 80)
        example_simple_usage()
        print("\n" + ">" * 80)
        example_with_model_manager()
        print("\n" + ">" * 80)
        example_training()
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED!")
        print("=" * 80)
    elif choice.lower() == "q":
        print("\nExiting...")
    else:
        print("\nInvalid choice. Please run again and select 1-4 or 'q'.")

    print("\n" + "=" * 80)
    print("For more details, see the HINTS section in the source code.")
    print("=" * 80)
