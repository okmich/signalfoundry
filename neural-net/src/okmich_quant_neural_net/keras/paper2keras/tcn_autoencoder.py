"""
Temporal Convolutional Autoencoder (TCN-AE)
============================================

Architecture using TCN-based autoencoder to learn compressed, meaningful representations in an unsupervised/semi-supervised manner.
TCN's dilated causal convolutions capture long-range dependencies efficiently, making it ideal for denoising noisy financial data.

Two-phase approach:
1. Pre-training: Train autoencoder on reconstruction task (unsupervised)
2. Fine-tuning: Use encoder representations for classification (supervised)

Suitable for: noisy financial data, long sequences, parallel training, denoising
"""

import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers, models

# Import custom TCN layer
from okmich_quant_neural_net.keras.layers.tcn import TCN
# Import task type and common utilities
from .common import TaskType, create_output_layer_and_loss, get_optimizer, get_model_name


def create_tcn_autoencoder(input_shape, encoding_dim=32, nb_filters=64, kernel_size=3, dilations=(1, 2, 4, 8, 16, 32),
                           nb_stacks=1, dropout_rate=0.2, use_skip_connections=True, learning_rate=0.001, l2_reg=0.0001,
                           bottleneck_activation="relu"):
    """
    Create a TCN Autoencoder for unsupervised representation learning.

    This model uses dilated causal convolutions to learn a compressed representation,
    forcing it to capture the most salient temporal patterns while filtering noise.

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (sequence_length, n_features)
    encoding_dim : int, default=32
        Size of the compressed bottleneck representation
    nb_filters : int, default=64
        Number of filters in TCN convolutional layers
    kernel_size : int, default=3
        Size of the convolutional kernel
    dilations : tuple, default=(1, 2, 4, 8, 16, 32)
        Dilation rates for TCN layers (captures different timescales)
    nb_stacks : int, default=1
        Number of stacks of residual blocks
    dropout_rate : float, default=0.2
        Dropout rate in TCN layers
    use_skip_connections : bool, default=True
        Whether to use skip connections in TCN
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer
    l2_reg : float, default=0.0001
        L2 regularization factor
    bottleneck_activation : str, default="relu"
        Activation for the bottleneck Dense layer. Use "linear" for clustering
        tasks (gives continuous, signed latent space suitable for k-means/GMM).

    Returns
    -------
    keras.Model
        Compiled autoencoder model for reconstruction
    keras.Model
        Encoder model (for extracting representations)
    """

    sequence_length, n_features = input_shape

    # ============================================================
    # ENCODER
    # ============================================================

    inputs = layers.Input(shape=input_shape, name="input")

    # TCN Encoder (with return_sequences=True to get full sequence)
    encoded = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,
        padding="causal",
        use_skip_connections=use_skip_connections,
        dropout_rate=dropout_rate,
        return_sequences=True,
        activation="relu",
        name="encoder_tcn",
    )(inputs)

    # Global pooling to compress sequence dimension
    # Use both max and average pooling for richer representation
    global_max_pool = layers.GlobalMaxPooling1D(name="encoder_global_max_pool")(encoded)
    global_avg_pool = layers.GlobalAveragePooling1D(name="encoder_global_avg_pool")(
        encoded
    )
    pooled = layers.Concatenate(name="encoder_concat_pools")(
        [global_max_pool, global_avg_pool]
    )

    # Bottleneck: Compressed representation
    bottleneck = layers.Dense(
        encoding_dim,
        activation=bottleneck_activation,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="bottleneck",
    )(pooled)

    # ============================================================
    # DECODER
    # ============================================================

    # Expand bottleneck to match TCN input requirements
    decoded = layers.Dense(
        nb_filters,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="decoder_expand",
    )(bottleneck)

    # Repeat for each timestep
    decoded = layers.RepeatVector(sequence_length, name="repeat_encoding")(decoded)

    # TCN Decoder (reconstructs sequence)
    decoded = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,
        padding="causal",
        use_skip_connections=use_skip_connections,
        dropout_rate=dropout_rate,
        return_sequences=True,
        activation="relu",
        name="decoder_tcn",
    )(decoded)

    # Reconstruct original features at each timestep
    decoded = layers.TimeDistributed(layers.Dense(n_features), name="reconstruction")(
        decoded
    )

    # ============================================================
    # CREATE MODELS
    # ============================================================

    # Full autoencoder (for pre-training)
    autoencoder = models.Model(inputs=inputs, outputs=decoded, name="tcn_autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="mse",
        metrics=["mae"],
    )

    # Encoder only (for extracting representations)
    encoder = models.Model(inputs=inputs, outputs=bottleneck, name="encoder")

    return autoencoder, encoder


def create_classifier_from_encoder(encoder, num_classes, task_type=TaskType.CLASSIFICATION,
                                   dense_units_1=64, dense_units_2=32, dropout=0.2,
                                   learning_rate=0.001, l2_reg=0.0001, freeze_encoder=True):
    """
    Create a classifier using pre-trained encoder representations.

    Parameters
    ----------
    encoder : keras.Model
        Pre-trained encoder model
    num_classes : int
        Number of output classes for classification
    dense_units_1 : int, default=64
        Number of units in first dense layer
    dense_units_2 : int, default=32
        Number of units in second dense layer
    dropout : float, default=0.2
        Dropout rate
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer
    l2_reg : float, default=0.0001
        L2 regularization factor
    freeze_encoder : bool, default=True
        If True, freeze encoder weights during training

    Returns
    -------
    keras.Model
        Compiled classifier model
    """

    # Freeze encoder layers if requested
    if freeze_encoder:
        for layer in encoder.layers:
            layer.trainable = False

    # Get encoder input and output
    inputs = encoder.input
    encoded = encoder.output

    # Classification head
    x = layers.Dense(
        dense_units_1,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="classifier_dense_1",
    )(encoded)
    x = layers.Dropout(dropout, name="classifier_dropout_1")(x)

    x = layers.Dense(
        dense_units_2,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="classifier_dense_2",
    )(x)
    x = layers.Dropout(dropout, name="classifier_dropout_2")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create classifier model
    model_name = get_model_name("classifier", task_type)
    classifier = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile with optimizer and task-specific loss/metrics
    opt = get_optimizer("adam", learning_rate)
    classifier.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return classifier


def create_end_to_end_tcn_model(input_shape, num_classes, task_type=TaskType.CLASSIFICATION, encoding_dim=32,
                                nb_filters=64, kernel_size=3, dilations=(1, 2, 4, 8, 16, 32), nb_stacks=1,
                                dropout_rate=0.2, use_skip_connections=True, dense_units_1=64, dense_units_2=32,
                                learning_rate=0.001, l2_reg=0.0001, bottleneck_activation="relu"):
    """
    Create an end-to-end TCN Autoencoder Classifier (no pre-training).

    This variant trains encoder and classifier jointly from scratch.

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (sequence_length, n_features)
    num_classes : int
        Number of output classes
    encoding_dim : int, default=32
        Size of the compressed representation
    nb_filters : int, default=64
        Number of filters in TCN layers
    kernel_size : int, default=3
        Kernel size for TCN
    dilations : tuple, default=(1, 2, 4, 8, 16, 32)
        Dilation rates for TCN
    nb_stacks : int, default=1
        Number of stacks in TCN
    dropout_rate : float, default=0.2
        Dropout rate in TCN
    use_skip_connections : bool, default=True
        Use skip connections in TCN
    dense_units_1 : int, default=64
        First dense layer units
    dense_units_2 : int, default=32
        Second dense layer units
    learning_rate : float, default=0.001
        Learning rate
    l2_reg : float, default=0.0001
        L2 regularization

    Returns
    -------
    keras.Model
        Compiled end-to-end model
    """

    # Input
    inputs = layers.Input(shape=input_shape, name="input")

    # TCN Encoder
    encoded = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,
        padding="causal",
        use_skip_connections=use_skip_connections,
        dropout_rate=dropout_rate,
        return_sequences=True,
        activation="relu",
        name="encoder_tcn",
    )(inputs)

    # Global pooling
    global_max_pool = layers.GlobalMaxPooling1D(name="global_max_pool")(encoded)
    global_avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(encoded)
    pooled = layers.Concatenate(name="concat_pools")([global_max_pool, global_avg_pool])

    # Bottleneck
    bottleneck = layers.Dense(
        encoding_dim,
        activation=bottleneck_activation,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="bottleneck",
    )(pooled)

    # Classifier head
    x = layers.Dense(
        dense_units_1,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="classifier_dense_1",
    )(bottleneck)
    x = layers.Dropout(dropout_rate, name="classifier_dropout_1")(x)

    x = layers.Dense(
        dense_units_2,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="classifier_dense_2",
    )(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout_2")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("end_to_end_tcn_classifier", task_type)
    model = models.Model(
        inputs=inputs, outputs=outputs, name=model_name
    )

    # Compile with task-specific loss/metrics
    opt = get_optimizer("adam", learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


def create_tunable_tcn_autoencoder_classifier(input_shape, num_classes, task_type=TaskType.CLASSIFICATION,
                                              bottleneck_activation="relu"):
    """
    Create a tunable end-to-end TCN Autoencoder Classifier for hyperparameter optimization.

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
    ...     create_tunable_tcn_autoencoder_classifier(input_shape=(96, 20), num_classes=3),
    ...     objective='val_loss',
    ...     max_trials=20,
    ...     directory='tuner_results',
    ...     project_name='tcn_autoencoder_classifier'
    ... )
    >>> tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
    """

    def build_model(hp):
        # Hyperparameter search space
        encoding_dim = hp.Choice("encoding_dim", values=[16, 32, 64])
        nb_filters = hp.Choice("nb_filters", values=[32, 64, 128])
        kernel_size = hp.Choice("kernel_size", values=[2, 3, 4])

        # Dilations - choose between shallow and deep
        use_deep_dilations = hp.Boolean("use_deep_dilations")
        if use_deep_dilations:
            dilations = (1, 2, 4, 8, 16, 32)
        else:
            dilations = (1, 2, 4, 8)

        nb_stacks = hp.Choice("nb_stacks", values=[1, 2])
        dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.4, step=0.1)
        use_skip_connections = hp.Boolean("use_skip_connections")

        dense_units_1 = hp.Int("dense_units_1", min_value=32, max_value=128, step=32)
        dense_units_2 = hp.Int("dense_units_2", min_value=16, max_value=64, step=16)

        learning_rate = hp.Choice("learning_rate", values=[1e-4, 5e-4, 1e-3, 5e-3])
        l2_reg = hp.Choice("l2_reg", values=[1e-5, 1e-4, 1e-3])

        return create_end_to_end_tcn_model(
            input_shape=input_shape,
            num_classes=num_classes,
            task_type=task_type,
            encoding_dim=encoding_dim,
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            dilations=dilations,
            nb_stacks=nb_stacks,
            dropout_rate=dropout_rate,
            use_skip_connections=use_skip_connections,
            dense_units_1=dense_units_1,
            dense_units_2=dense_units_2,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
            bottleneck_activation=bottleneck_activation,
        )

    return build_model


# ============================================================================
# HINTS AND BEST PRACTICES
# ============================================================================

""""
1. TCN AUTOENCODER BENEFITS:
   ✓ Parallel training (10x faster than LSTM on GPUs)
   ✓ No vanishing gradients (residual connections)
   ✓ Flexible receptive field (controlled by dilations)
   ✓ Excellent for denoising noisy financial data
   ✓ Learns multi-scale patterns via dilated convolutions
   ✓ More memory efficient than LSTM

2. DILATED CONVOLUTIONS:
   - Dilation rate: How many timesteps to skip between kernel elements
   - Example: dilation=1 (normal conv), dilation=2 (skip 1), dilation=4 (skip 3)
   - Dilations (1, 2, 4, 8, 16, 32) capture patterns at 6 different timescales
   - Receptive field = 1 + 2 * (kernel_size - 1) * nb_stacks * sum(dilations)
   - For kernel_size=3, dilations=(1,2,4,8,16,32): receptive field = 127 timesteps

3. RECEPTIVE FIELD SELECTION:
   - Should cover relevant historical context
   - For 5-min bars: 127 timesteps = ~10.5 hours (good for intraday)
   - For 1-min bars: 127 timesteps = ~2 hours (good for short-term)
   - Adjust dilations to match your trading timeframe
   - Shallow: (1, 2, 4, 8) - receptive field = 31
   - Medium: (1, 2, 4, 8, 16) - receptive field = 63
   - Deep: (1, 2, 4, 8, 16, 32) - receptive field = 127

4. TWO-PHASE TRAINING STRATEGY:
   PHASE 1 - Pre-training (Unsupervised):
   ✓ Train on large unlabeled dataset (5k-50k samples)
   ✓ Goal: Learn denoised, multi-scale representations
   ✓ Reconstruction loss: MSE between input and output
   ✓ Epochs: 30-100
   ✓ Batch size: 64-128 (TCN benefits from larger batches)

   PHASE 2 - Fine-tuning (Supervised):
   ✓ Freeze encoder, train classifier (1k-5k labeled samples)
   ✓ Goal: Learn classification from learned representations
   ✓ Epochs: 20-50
   ✓ Batch size: 32-64

   OPTIONAL - Full Fine-tuning:
   ✓ Unfreeze encoder, end-to-end training
   ✓ Learning rate: 0.0001 (10x lower)
   ✓ Epochs: 5-10

5. WHY TCN FOR FINANCIAL DATA:
   ✓ Financial data is 90%+ noise
   ✓ TCN's dilated convolutions capture multi-scale patterns
   ✓ Causal padding ensures no look-ahead bias
   ✓ Residual connections prevent gradient vanishing
   ✓ Parallel computation = faster experimentation
   ✓ Better than LSTM for sequences > 50 timesteps

6. INPUT DATA PREPARATION:
   CRITICAL: Normalize data before training!
   - Use StandardScaler or MinMaxScaler per feature
   - TCN sensitive to scale differences (like all deep learning)
   - Example:
       from sklearn.preprocessing import StandardScaler
       scaler = StandardScaler()
       X_scaled = scaler.fit_transform(X.reshape(-1, n_features))
       X_scaled = X_scaled.reshape(-1, sequence_length, n_features)

7. ENCODING DIMENSION (BOTTLENECK SIZE):
   - Too small (8-16): Information loss, underfitting
   - Optimal (32-64): Good compression, filters noise
   - Too large (128+): Less compression, may not denoise
   - Rule: encoding_dim ≈ sqrt(sequence_length * n_features)
   - For (96, 20): sqrt(1920) ≈ 44, use 32-64

8. NB_FILTERS SELECTION:
   - Start with 64 filters
   - Increase to 128 if model underfits
   - Decrease to 32 if overfitting or memory limited
   - More filters = more capacity, but slower training

9. SKIP CONNECTIONS:
   - use_skip_connections=True (recommended)
   - Adds residual connections within TCN
   - Improves gradient flow
   - Better reconstruction quality
   - Minimal computational overhead

10. TRAINING TIPS:
    Pre-training:
    - Batch size: 64-128 (TCN benefits from parallelism)
    - Learning rate: 0.001
    - Gradient clipping: clipnorm=1.0 (always use!)
    - Monitor: val_loss (reconstruction MSE)
    - Good reconstruction: MSE < 0.1 (normalized data)

    Classification:
    - Batch size: 32-64
    - Learning rate: 0.001 (frozen), 0.0001 (unfrozen)
    - Patience: 7-10 epochs
    - Monitor: val_accuracy, val_auc

11. HYPERPARAMETER TUNING PRIORITIES:
    High impact:
      - encoding_dim (16-64)
      - nb_filters (32-128)
      - dilations (affects receptive field)
      - learning_rate (1e-4 to 5e-3)

    Medium impact:
      - kernel_size (2-4)
      - nb_stacks (1-2)
      - dropout_rate (0.1-0.4)
      - use_skip_connections (usually True)

    Low impact:
      - dense_units (classifier head)
      - l2_reg (usually 1e-4 works)

12. COMPARISON: TCN-AE VS LSTM-AE:
    TCN Autoencoder:
    ✓ 10x faster training (parallel)
    ✓ Better for long sequences (96+)
    ✓ Flexible receptive field
    ✓ No vanishing gradients
    ✓ More memory efficient
    ✗ Less interpretable than recurrent

    LSTM Autoencoder:
    ✓ True sequential processing
    ✓ Maintains hidden state
    ✓ Better for short sequences (<50)
    ✓ More interpretable dynamics
    ✗ Slower training (sequential)
    ✗ Vanishing gradient issues
    ✗ Higher memory usage

13. DENOISING CAPABILITY:
    - TCN autoencoder excels at denoising
    - Dilated convolutions capture true signals across scales
    - Bottleneck filters out high-frequency noise
    - Test denoising quality:
        * Add Gaussian noise to validation data
        * Measure reconstruction MSE
        * Should be < noisy MSE

14. USE CASES IN TRADING:
    a) Regime Detection:
       - Pre-train on years of unlabeled OHLCV data
       - Fine-tune on labeled regime changes
       - TCN captures multi-scale regime patterns

    b) Volatility Forecasting:
       - Pre-train on high-frequency tick data
       - Fine-tune on volatility labels
       - Denoises microstructure noise

    c) Order Book Imbalance:
       - Pre-train on order book snapshots
       - Fine-tune on imbalance direction
       - TCN captures order flow at multiple timescales

    d) Multi-Instrument Pattern Learning:
       - Pre-train on multiple correlated instruments
       - Transfer encoder to new instrument
       - Shared patterns across markets

15. TRANSFER LEARNING:
    - Pre-train on one asset class (stocks)
    - Transfer encoder to another (crypto, forex)
    - Fine-tune on target market
    - Useful when target has limited data

16. MONITORING OVERFITTING:
    - Track train vs val reconstruction loss
    - Large gap = overfitting:
        * Increase dropout (0.2 → 0.3)
        * Add more unlabeled data
        * Reduce nb_filters or encoding_dim
        * Increase l2_reg

17. DEBUGGING TIPS:
    - If reconstruction loss not decreasing:
        * Check data normalization
        * Increase nb_filters (32 → 64)
        * Reduce encoding_dim temporarily
        * Check for NaN in data

    - If classifier not learning:
        * Verify encoder was pre-trained
        * Try unfreezing encoder
        * Increase dense_units
        * Check class imbalance

    - If loss becomes NaN:
        * Reduce learning rate
        * Increase gradient clipping
        * Check input data scale

18. PRODUCTION DEPLOYMENT:
    - Fast inference: ~10-50ms on CPU
    - Parallelizable: Batch predictions
    - Save separately:
        encoder.save('tcn_encoder.h5')
        classifier.save('tcn_classifier.h5')
    - Inference pipeline:
        1. Normalize input
        2. Extract features: encoder.predict(X)
        3. Classify: classifier.predict(X)

19. ADVANCED TECHNIQUES:
    - Multi-task Learning:
        Joint loss = α * L_reconstruction + β * L_classification
        Train both tasks simultaneously

    - Variational TCN Autoencoder:
        Add sampling in bottleneck: z ~ N(μ, σ)
        Loss = reconstruction + KL_divergence
        More robust representations

    - Adversarial Training:
        Add discriminator on latent space
        Forces encoder to learn canonical distributions

    - Progressive Dilations:
        Start with shallow dilations (1,2,4,8)
        Progressively add deeper (16,32) during training
        Curriculum learning for temporal hierarchy

20. WHEN TO USE TCN-AE:
    ✓ Long sequences (96-200 timesteps)
    ✓ Need fast training (GPU available)
    ✓ Multi-scale patterns important
    ✓ Noisy financial data
    ✓ Limited labeled data, abundant unlabeled
    ✓ Need denoising capability

    ✗ Very short sequences (<30 timesteps) - use simpler models
    ✗ Need interpretable recurrent dynamics - use LSTM
    ✗ No GPU available - training slower without parallelism
    ✗ Large labeled dataset - use end-to-end supervised
"""
