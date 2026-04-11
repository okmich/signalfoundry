"""
Hierarchical Multi-Scale CNN (Pyramid Architecture)
====================================================

Architecture with parallel branches capturing patterns at different timescales.
Multiple kernel sizes process micro and macro trends simultaneously.

Suitable for: multi-timeframe momentum, volatility regime changes, pattern recognition


References:
- "Multi-Scale Convolutional Neural Networks for Time Series Classification" - Cui et al. (2016)
https://arxiv.org/abs/1603.06995
- Financial application: "Deep Learning for Stock Market Prediction from Financial News Articles"** - Vargas et al. (2017)
https://ieeexplore.ieee.org/document/8259701
"""

import keras_tuner as kt
from keras import layers, models
from keras.regularizers import l2

from .common import TaskType, create_output_layer_and_loss, get_optimizer, get_model_name


def create_hierarchical_multiscale_cnn(input_shape, num_classes, task_type=TaskType.CLASSIFICATION,
                                       branch_filters=32, kernel_sizes=(2, 4, 8, 16), pool_size=2,
                                       conv_filters_1=128, conv_filters_2=64, conv_kernel_size=3, conv_dropout=0.3,
                                       dense_units_1=128, dense_units_2=64, dense_dropout_1=0.3, dense_dropout_2=0.2,
                                       learning_rate=0.001, l2_reg=0.0001, jit_compile=False):
    """
    Create a Hierarchical Multi-Scale CNN with pyramid architecture.

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (sequence_length, n_features)
    num_classes : int
        Number of output classes for classification
    branch_filters : int, default=32
        Number of filters in each parallel branch
    kernel_sizes : tuple, default=(2, 4, 8, 16)
        Kernel sizes for parallel branches (captures different timescales)
    pool_size : int, default=2
        Pool size for MaxPooling1D in branches
    conv_filters_1 : int, default=128
        Number of filters in first Conv1D after concatenation
    conv_filters_2 : int, default=64
        Number of filters in second Conv1D
    conv_kernel_size : int, default=3
        Kernel size for Conv1D layers after concatenation
    conv_dropout : float, default=0.3
        Dropout rate after Conv1D layers
    dense_units_1 : int, default=128
        Number of units in first dense layer
    dense_units_2 : int, default=64
        Number of units in second dense layer
    dense_dropout_1 : float, default=0.3
        Dropout rate after first dense layer
    dense_dropout_2 : float, default=0.2
        Dropout rate after second dense layer
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer
    l2_reg : float, default=0.0001
        L2 regularization factor
    jit_compile : bool, default=False
        Whether to enable XLA JIT compilation in model.compile

    Returns
    -------
    keras.Model
        Compiled Keras model
    """

    # Input layer
    inputs = layers.Input(shape=input_shape, name="input")

    # Parallel branches with different kernel sizes (multi-scale feature extraction)
    branches = []
    for i, kernel_size in enumerate(kernel_sizes):
        branch = layers.Conv1D(
            filters=branch_filters,
            kernel_size=kernel_size,
            padding="causal",
            activation="relu",
            kernel_regularizer=l2(l2_reg),
            name=f"branch_{i + 1}_conv_k{kernel_size}",
        )(inputs)
        branch = layers.MaxPooling1D(pool_size=pool_size, name=f"branch_{i + 1}_pool")(
            branch
        )
        branches.append(branch)

    # Concatenate all branches
    concatenated = layers.Concatenate(name="concat_branches")(branches)

    # First Conv1D layer after concatenation
    conv1 = layers.Conv1D(
        filters=conv_filters_1,
        kernel_size=conv_kernel_size,
        padding="causal",
        kernel_regularizer=l2(l2_reg),
        name="conv1d_1",
    )(concatenated)
    conv1 = layers.Activation("relu", name="relu_1")(conv1)
    conv1 = layers.Dropout(conv_dropout, name="conv_dropout_1")(conv1)

    # Second Conv1D layer
    conv2 = layers.Conv1D(
        filters=conv_filters_2,
        kernel_size=conv_kernel_size,
        padding="causal",
        kernel_regularizer=l2(l2_reg),
        name="conv1d_2",
    )(conv1)
    conv2 = layers.Activation("relu", name="relu_2")(conv2)
    conv2 = layers.Dropout(conv_dropout, name="conv_dropout_2")(conv2)

    # Global pooling (both max and average, then concatenate)
    global_max_pool = layers.GlobalMaxPooling1D(name="global_max_pool")(conv2)
    global_avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(conv2)
    pooled = layers.Concatenate(name="concat_global_pools")(
        [global_max_pool, global_avg_pool]
    )

    # First Dense layer
    dense1 = layers.Dense(
        dense_units_1,
        activation="relu",
        kernel_regularizer=l2(l2_reg),
        name="dense_1",
    )(pooled)
    dense1 = layers.Dropout(dense_dropout_1, name="dense_dropout_1")(dense1)

    # Second Dense layer
    x = layers.Dense(
        dense_units_2,
        activation="relu",
        kernel_regularizer=l2(l2_reg),
        name="dense_2",
    )(dense1)
    x = layers.Dropout(dense_dropout_2, name="dense_dropout_2")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("hierarchical_multiscale_cnn", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile
    opt = get_optimizer("adam", learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics, jit_compile=jit_compile)

    return model


def create_tunable_hierarchical_multiscale_cnn(
        input_shape, num_classes, task_type=TaskType.CLASSIFICATION, jit_compile=False
):
    """
    Create a tunable version of the Hierarchical Multi-Scale CNN for hyperparameter optimization.

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (sequence_length, n_features)
    num_classes : int
        Number of output classes

    Returns
    -------
    function
        Model builder function for keras_tuner

    Example
    -------
    >>> tuner = kt.BayesianOptimization(
    ...     create_tunable_hierarchical_multiscale_cnn(input_shape=(64, 10), num_classes=3),
    ...     objective='val_loss',
    ...     max_trials=20,
    ...     directory='tuner_results',
    ...     project_name='hierarchical_multiscale_cnn'
    ... )
    >>> tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
    """

    def build_model(hp):
        # Hyperparameter search space
        branch_filters = hp.Choice("branch_filters", values=[16, 32, 64])

        # Option to tune kernel sizes
        use_small_kernels = hp.Boolean("use_small_kernels")
        if use_small_kernels:
            kernel_sizes = (2, 4, 6, 8)
        else:
            kernel_sizes = (2, 4, 8, 16)

        pool_size = hp.Choice("pool_size", values=[2, 3])

        conv_filters_1 = hp.Choice("conv_filters_1", values=[64, 128, 256])
        conv_filters_2 = hp.Choice("conv_filters_2", values=[32, 64, 128])
        conv_kernel_size = hp.Choice("conv_kernel_size", values=[3, 5])
        conv_dropout = hp.Float("conv_dropout", min_value=0.2, max_value=0.5, step=0.1)

        dense_units_1 = hp.Int("dense_units_1", min_value=64, max_value=256, step=64)
        dense_units_2 = hp.Int("dense_units_2", min_value=32, max_value=128, step=32)
        dense_dropout_1 = hp.Float(
            "dense_dropout_1", min_value=0.2, max_value=0.5, step=0.1
        )
        dense_dropout_2 = hp.Float(
            "dense_dropout_2", min_value=0.1, max_value=0.4, step=0.1
        )

        learning_rate = hp.Choice("learning_rate", values=[1e-4, 5e-4, 1e-3, 5e-3])
        l2_reg = hp.Choice("l2_reg", values=[1e-5, 1e-4, 1e-3])

        return create_hierarchical_multiscale_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            task_type=task_type,
            branch_filters=branch_filters,
            kernel_sizes=kernel_sizes,
            pool_size=pool_size,
            conv_filters_1=conv_filters_1,
            conv_filters_2=conv_filters_2,
            conv_kernel_size=conv_kernel_size,
            conv_dropout=conv_dropout,
            dense_units_1=dense_units_1,
            dense_units_2=dense_units_2,
            dense_dropout_1=dense_dropout_1,
            dense_dropout_2=dense_dropout_2,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
            jit_compile=jit_compile,
        )

    return build_model


# ============================================================================
# HINTS AND BEST PRACTICES
# ============================================================================

"""
1. MULTI-SCALE ARCHITECTURE BENEFITS:
   - Parallel branches capture patterns at different timescales simultaneously
   - Small kernels (2-4): Micro trends, rapid reversals, scalping signals
   - Medium kernels (6-8): Swing trading patterns, intraday trends
   - Large kernels (12-16): Macro trends, regime changes, position trading
   - No need to manually engineer multi-timeframe features!

2. KERNEL SIZE SELECTION FOR TRADING:
   For 5-minute bars:
     - kernel=2 (10 min):  Very short-term momentum
     - kernel=4 (20 min):  Scalping patterns
     - kernel=6 (30 min):  Intraday support/resistance
     - kernel=12 (1 hour): Session trends
     - kernel=24 (2 hours): Daily regime shifts

   For 1-minute bars:
     - kernel=5 (5 min):   Ultra short-term
     - kernel=15 (15 min): Short-term patterns
     - kernel=30 (30 min): Medium-term patterns
     - kernel=60 (1 hour): Long-term patterns

3. INPUT DATA PREPARATION:
   - Normalize/standardize features before training
   - Ensure sequence_length is divisible by pool_size
   - Recommended lengths: 32, 64, 96, 128 (for pool_size=2)
   - Minimum length should be > largest kernel size

4. GLOBAL POOLING STRATEGY:
   - GlobalMaxPooling: Captures strongest activations (peaks, extremes)
   - GlobalAveragePooling: Captures overall trends (smoothed signals)
   - Concatenating both: Gets best of both worlds
   - This replaces flattening, reducing parameters significantly

5. BRANCH FILTERS:
   - Start with 32 filters per branch
   - Increase to 48-64 if model underfits
   - Decrease to 16 if overfitting or memory limited
   - All branches should have equal filters for balanced contribution

6. CLASS IMBALANCE HANDLING:
   - Use class_weight in model.fit() for imbalanced classes
   - Example: class_weight={0: 1.0, 1: 3.0, 2: 1.5}
   - Consider focal loss for extreme imbalance
   - Monitor macro F1 and per-class recall

7. TRAINING TIPS:
   - Batch size: 32-64 recommended
   - Learning rate: 0.001 (reduce to 0.0005 if loss oscillates)
   - Use early stopping with patience=10-15
   - ReduceLROnPlateau helps escape local minima
   - Gradient clipping: clipnorm=1.0 if gradients explode

8. HYPERPARAMETER TUNING PRIORITIES:
   High impact:
     - branch_filters (16-64)
     - conv_filters_1 (64-256)
     - dense_units_1 (64-256)
     - learning_rate (1e-4 to 5e-3)

   Medium impact:
     - kernel_sizes (experiment with different combinations)
     - conv_dropout (0.2-0.5)
     - dense_dropout_1 (0.2-0.5)

   Low impact:
     - pool_size (usually 2 is optimal)
     - conv_kernel_size (3 or 5)
     - dense_units_2

9. ARCHITECTURE VARIANTS:
   - Add BatchNormalization after Conv layers for faster convergence:
       conv = layers.BatchNormalization()(conv)

   - Add residual connections between Conv layers:
       conv2 = layers.Add()([conv1_projection, conv2])

   - Use SeparableConv1D for efficiency:
       layers.SeparableConv1D(filters, kernel_size, ...)

   - Add more branches for finer-grained scales:
       kernel_sizes = (2, 3, 4, 6, 8, 12, 16)

   - Replace MaxPool with strided convolutions:
       Conv1D(filters, kernel_size, strides=2)

10. MEMORY OPTIMIZATION:
    - Use mixed precision training for 2x speedup:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)

    - Reduce branch_filters from 32 to 16 if GPU memory limited
    - Use smaller batch_size (16-32)
    - Consider using functional model with shared Conv layers

11. INTERPRETABILITY:
    - Extract features from each branch to see which timescale is most predictive
    - Visualize branch activations to understand learned patterns
    - Use attention mechanisms to weight branches dynamically
    - Analyze which kernel sizes contribute most to correct predictions

12. COMPARISON WITH OTHER ARCHITECTURES:
    - vs Single-scale CNN: Better captures multi-timeframe information
    - vs LSTM: Faster training, more parallelizable, no vanishing gradients
    - vs Transformer: More parameter-efficient for shorter sequences
    - vs TCN: Similar multi-scale capability, but pyramid is more explicit

13. MULTI-TIMEFRAME MOMENTUM CLASSIFICATION:
    - Use this architecture for: Bullish/Neutral/Bearish classification
    - Small kernels detect: Quick reversals, breakout signals
    - Large kernels detect: Trend continuation, regime persistence
    - Model learns to combine signals across timeframes automatically

14. VOLATILITY REGIME CLASSIFICATION:
    - Classes: Low/Medium/High volatility regimes
    - Small kernels: Detect sudden volatility spikes
    - Large kernels: Identify sustained regime changes
    - Add volatility features (ATR, Bollinger width) to input

15. PRODUCTION DEPLOYMENT:
    - Model is fast: ~10-50ms inference on CPU
    - No recurrent layers: Fully parallelizable
    - Easy to quantize: Use TFLite for mobile/edge deployment
    - Batch predictions for multiple instruments simultaneously
    """

