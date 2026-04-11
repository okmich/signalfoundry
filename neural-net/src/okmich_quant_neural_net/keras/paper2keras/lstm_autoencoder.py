"""
LSTM Autoencoder for Representation Learning
=============================================

Architecture using LSTM-based autoencoder to learn compressed, meaningful representations in an unsupervised/semi-supervised manner.
The encoder learns to denoise input and capture salient features, which are then used for classification.

Two-phase approach:
1. Pre-training: Train autoencoder on reconstruction task (unsupervised)
2. Fine-tuning: Use encoder representations for classification (supervised)

Suitable for: noisy financial data, feature learning, denoising, transfer learning
"""

import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers, models

# Import task type and common utilities
from .common import TaskType, create_output_layer_and_loss, get_optimizer, get_model_name


def create_lstm_autoencoder(input_shape, encoding_dim=32, encoder_units_1=128, encoder_units_2=64, decoder_units_1=64,
                            decoder_units_2=128, dropout=0.2, learning_rate=0.001, l2_reg=0.0001,
                            bottleneck_activation="relu"):
    """
    Create an LSTM Autoencoder for unsupervised representation learning.

    This model learns to reconstruct the input sequence, forcing it to learn
    a compressed representation in the bottleneck layer.

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (sequence_length, n_features)
    encoding_dim : int, default=32
        Size of the compressed bottleneck representation
    encoder_units_1 : int, default=128
        Number of units in first encoder LSTM layer
    encoder_units_2 : int, default=64
        Number of units in second encoder LSTM layer
    decoder_units_1 : int, default=64
        Number of units in first decoder LSTM layer
    decoder_units_2 : int, default=128
        Number of units in second decoder LSTM layer
    dropout : float, default=0.2
        Dropout rate after LSTM layers
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

    # First encoder LSTM
    encoded = layers.LSTM(
        encoder_units_1,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="encoder_lstm_1",
    )(inputs)
    encoded = layers.Dropout(dropout, name="encoder_dropout_1")(encoded)

    # Second encoder LSTM
    encoded = layers.LSTM(
        encoder_units_2,
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="encoder_lstm_2",
    )(encoded)
    encoded = layers.Dropout(dropout, name="encoder_dropout_2")(encoded)

    # Bottleneck: Compressed representation
    encoded = layers.Dense(
        encoding_dim,
        activation=bottleneck_activation,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="bottleneck",
    )(encoded)

    # ============================================================
    # DECODER
    # ============================================================

    # Repeat the encoded representation for each timestep
    decoded = layers.RepeatVector(sequence_length, name="repeat_encoding")(encoded)

    # First decoder LSTM
    decoded = layers.LSTM(
        decoder_units_1,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="decoder_lstm_1",
    )(decoded)
    decoded = layers.Dropout(dropout, name="decoder_dropout_1")(decoded)

    # Second decoder LSTM
    decoded = layers.LSTM(
        decoder_units_2,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="decoder_lstm_2",
    )(decoded)
    decoded = layers.Dropout(dropout, name="decoder_dropout_2")(decoded)

    # Reconstruct original features at each timestep
    decoded = layers.TimeDistributed(layers.Dense(n_features), name="reconstruction")(
        decoded
    )

    # Full autoencoder (for pre-training)
    autoencoder = models.Model(inputs=inputs, outputs=decoded, name="lstm_autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    # Encoder only (for extracting representations)
    encoder = models.Model(inputs=inputs, outputs=encoded, name="encoder")

    return autoencoder, encoder


def create_classifier_from_encoder(encoder, num_classes, task_type=TaskType.CLASSIFICATION, dense_units=32,
                                   dropout=0.2, learning_rate=0.001, l2_reg=0.0001, freeze_encoder=True):
    """
    Create a classifier using pre-trained encoder representations.

    Parameters
    ----------
    encoder : keras.Model
        Pre-trained encoder model
    num_classes : int
        Number of output classes for classification
    dense_units : int, default=32
        Number of units in dense layer before output
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
        dense_units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="classifier_dense",
    )(encoded)
    x = layers.Dropout(dropout, name="classifier_dropout")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create classifier model
    model_name = get_model_name("classifier", task_type)
    classifier = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile with task-specific loss/metrics
    opt = get_optimizer("adam", learning_rate)
    classifier.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return classifier


def create_end_to_end_model(input_shape, num_classes, task_type=TaskType.CLASSIFICATION, encoding_dim=32,
                            encoder_units_1=128, encoder_units_2=64, dense_units=32, dropout=0.2, learning_rate=0.001,
                            l2_reg=0.0001, bottleneck_activation="relu"):
    """
    Create an end-to-end LSTM Autoencoder Classifier (no pre-training).

    This variant trains encoder and classifier jointly from scratch.

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (sequence_length, n_features)
    num_classes : int
        Number of output classes
    encoding_dim : int, default=32
        Size of the compressed representation
    encoder_units_1 : int, default=128
        Number of units in first encoder LSTM
    encoder_units_2 : int, default=64
        Number of units in second encoder LSTM
    dense_units : int, default=32
        Number of units in classifier dense layer
    dropout : float, default=0.2
        Dropout rate
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

    # Encoder
    encoded = layers.LSTM(
        encoder_units_1,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="encoder_lstm_1",
    )(inputs)
    encoded = layers.Dropout(dropout, name="encoder_dropout_1")(encoded)

    encoded = layers.LSTM(
        encoder_units_2,
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="encoder_lstm_2",
    )(encoded)
    encoded = layers.Dropout(dropout, name="encoder_dropout_2")(encoded)

    # Bottleneck
    encoded = layers.Dense(
        encoding_dim,
        activation=bottleneck_activation,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="bottleneck",
    )(encoded)

    # Classifier head
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="classifier_dense",
    )(encoded)
    x = layers.Dropout(dropout, name="classifier_dropout")(x)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("end_to_end_autoencoder_classifier", task_type)
    model = models.Model(
        inputs=inputs, outputs=outputs, name=model_name
    )

    # Compile with task-specific loss/metrics
    opt = get_optimizer("adam", learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


def create_tunable_lstm_autoencoder_classifier(input_shape, num_classes, task_type=TaskType.CLASSIFICATION,
                                               bottleneck_activation="relu"):
    """
    Create a tunable end-to-end LSTM Autoencoder Classifier for hyperparameter optimization.

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
    ...     create_tunable_lstm_autoencoder_classifier(input_shape=(96, 20), num_classes=3),
    ...     objective='val_loss',
    ...     max_trials=20,
    ...     directory='tuner_results',
    ...     project_name='lstm_autoencoder_classifier'
    ... )
    >>> tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
    """

    def build_model(hp):
        # Hyperparameter search space
        encoding_dim = hp.Choice("encoding_dim", values=[16, 32, 64])
        encoder_units_1 = hp.Choice("encoder_units_1", values=[64, 128, 256])
        encoder_units_2 = hp.Choice("encoder_units_2", values=[32, 64, 128])
        dense_units = hp.Int("dense_units", min_value=16, max_value=64, step=16)
        dropout = hp.Float("dropout", min_value=0.1, max_value=0.4, step=0.1)
        learning_rate = hp.Choice("learning_rate", values=[1e-4, 5e-4, 1e-3, 5e-3])
        l2_reg = hp.Choice("l2_reg", values=[1e-5, 1e-4, 1e-3])

        return create_end_to_end_model(
            input_shape=input_shape,
            num_classes=num_classes,
            task_type=task_type,
            encoding_dim=encoding_dim,
            encoder_units_1=encoder_units_1,
            encoder_units_2=encoder_units_2,
            dense_units=dense_units,
            dropout=dropout,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
            bottleneck_activation=bottleneck_activation,
        )

    return build_model


"""
1. TWO-PHASE TRAINING STRATEGY:
   PHASE 1 - Pre-training (Unsupervised):
   ✓ Train autoencoder on large unlabeled dataset (5k-50k samples)
   ✓ Goal: Learn robust, denoised representations
   ✓ Loss: MSE between input and reconstruction
   ✓ Epochs: 30-100 (until reconstruction stabilizes)

   PHASE 2 - Fine-tuning (Supervised):
   ✓ Freeze encoder, train classifier on labeled data (1k-5k samples)
   ✓ Goal: Learn classification from learned representations
   ✓ Epochs: 20-50

   OPTIONAL - Full Fine-tuning:
   ✓ Unfreeze encoder, fine-tune entire model with low LR
   ✓ Use learning_rate = 0.0001 (10x lower than initial)
   ✓ Epochs: 5-10

2. BENEFITS OF TWO-PHASE APPROACH:
   ✓ Works with small labeled datasets (1k samples vs 10k+ for end-to-end)
   ✓ Learns denoised representations (filters out market noise)
   ✓ More robust to overfitting
   ✓ Can leverage large unlabeled historical data
   ✓ Better generalization to unseen market conditions

3. WHEN TO USE AUTOENCODER PRE-TRAINING:
   ✓ Limited labeled data (<5k samples)
   ✓ Abundant unlabeled data (>10k samples)
   ✓ Very noisy financial data
   ✓ Need robust feature representations
   ✓ Transfer learning across different instruments/markets

   ✗ Large labeled dataset (>10k samples) - use end-to-end
   ✗ Simple patterns - use simpler models
   ✗ Need very fast training - pre-training adds time

4. ENCODING DIMENSION (BOTTLENECK SIZE):
   - Too small (8-16): May lose important information, underfitting
   - Optimal (32-64): Good compression, captures essential patterns
   - Too large (128+): Less compression, may not filter noise effectively
   - Rule of thumb: encoding_dim ≈ sqrt(sequence_length * n_features)
   - For (96, 20) input: sqrt(96*20) ≈ 44, use 32-64

5. RECONSTRUCTION LOSS MONITORING:
   - Good reconstruction: MSE < 0.1, MAE < 0.2 (for normalized data)
   - If reconstruction poor (MSE > 0.5):
       * Increase encoding_dim
       * Increase encoder/decoder units
       * Train longer
       * Check data normalization

6. INPUT DATA PREPARATION:
   CRITICAL: Normalize data before training autoencoder!
   - Use StandardScaler or MinMaxScaler per feature
   - Autoencoder very sensitive to scale differences
   - Example:
       from sklearn.preprocessing import StandardScaler
       scaler = StandardScaler()
       X_scaled = scaler.fit_transform(X.reshape(-1, n_features))
       X_scaled = X_scaled.reshape(-1, sequence_length, n_features)

7. DENOISING CAPABILITY:
   - Autoencoder learns to filter market noise automatically
   - Compression forces model to keep only salient features
   - Test denoising: Add Gaussian noise to validation data, measure reconstruction
   - Useful for: High-frequency data, order book snapshots, tick data

8. UNSUPERVISED PRE-TRAINING DATA:
   - Use ALL available historical data (labeled or not)
   - Multiple instruments: Concatenate data from correlated assets
   - Data augmentation: Add small noise, time shifts, scaling
   - No labels needed: Makes use of vast historical archives

9. TRAINING TIPS:
   Pre-training:
   - Batch size: 64-128 (larger for stability)
   - Learning rate: 0.001
   - Patience: 5-7 epochs
   - Monitor: val_loss (reconstruction MSE)

   Classification:
   - Batch size: 32-64
   - Learning rate: 0.001 (frozen), 0.0001 (unfrozen)
   - Patience: 7-10 epochs
   - Monitor: val_accuracy, val_auc

10. HYPERPARAMETER TUNING PRIORITIES:
    High impact:
      - encoding_dim (16-64)
      - encoder_units_1 (64-256)
      - encoder_units_2 (32-128)
      - learning_rate (1e-4 to 5e-3)

    Medium impact:
      - dropout (0.1-0.4)
      - dense_units (16-64)
      - decoder units (for reconstruction quality)

    Low impact:
      - l2_reg (usually 1e-4 works well)

11. ARCHITECTURE VARIANTS:
    - Variational Autoencoder (VAE):
        Add sampling layer in bottleneck: z ~ N(μ, σ)
        Loss = reconstruction_loss + KL_divergence
        Better for generation, more robust representations

    - Convolutional Autoencoder:
        Replace LSTM with Conv1D layers
        Faster, better for local patterns

    - Stacked Autoencoders:
        Train multiple autoencoders hierarchically
        Each learns different abstraction level

    - Adversarial Autoencoder:
        Add discriminator to match latent distribution
        More robust representations

12. TRANSFER LEARNING ACROSS MARKETS:
    - Pre-train on one asset class (e.g., stocks)
    - Transfer encoder to another (e.g., crypto, forex)
    - Fine-tune on target market data
    - Useful when target market has limited data

13. SEMI-SUPERVISED LEARNING:
    - Pre-train on mix of labeled + unlabeled data
    - During pre-training, use both reconstruction and classification loss
    - Joint loss: L = L_reconstruction + α * L_classification
    - α controls trade-off (start with 0.1-0.5)

14. REPRESENTATION ANALYSIS:
    - Visualize learned representations with t-SNE or UMAP
    - Check if classes cluster in encoding space
    - Good clustering = encoder learned discriminative features
    - Example:
        from sklearn.manifold import TSNE
        encoded = encoder.predict(X_val)
        tsne = TSNE(n_components=2).fit_transform(encoded)
        plt.scatter(tsne[:, 0], tsne[:, 1], c=y_val)

15. COMPARISON: PRE-TRAINING VS END-TO-END:
    Pre-training (Two-phase):
    ✓ Better with small labeled data
    ✓ More robust, less overfitting
    ✓ Denoising capability
    ✓ Transfer learning possible
    ✗ Slower training (two phases)
    ✗ More complex pipeline

    End-to-end:
    ✓ Simpler pipeline
    ✓ Faster training (single phase)
    ✓ Better with large labeled data
    ✗ May overfit with small data
    ✗ No denoising pre-training

16. USE CASES IN TRADING:
    a) Regime Detection:
       - Pre-train on years of unlabeled market data
       - Fine-tune on labeled regime transitions
       - Encoder learns market dynamics, classifier detects regimes

    b) Order Flow Classification:
       - Pre-train on order book snapshots (unlabeled)
       - Fine-tune on labeled buy/sell pressure
       - Denoises microstructure noise

    c) Multi-Asset Feature Learning:
       - Pre-train on multiple correlated assets
       - Shared encoder learns common patterns
       - Fine-tune per asset for specific predictions

17. MONITORING OVERFITTING:
    - Track train vs val reconstruction loss (pre-training)
    - Track train vs val accuracy (classification)
    - Large gap = overfitting:
        * Increase dropout (0.2 → 0.3)
        * Add more unlabeled data for pre-training
        * Reduce encoding_dim
        * Add L2 regularization

18. PRODUCTION DEPLOYMENT:
    - Save encoder separately: encoder.save('encoder.h5')
    - Save classifier: classifier.save('classifier.h5')
    - Inference pipeline:
        1. Load encoder + classifier
        2. Preprocess input (normalize with saved scaler)
        3. Extract features: features = encoder.predict(X)
        4. Classify: predictions = classifier.predict(X)
    - Moderate speed: ~30-100ms on CPU

19. DEBUGGING TIPS:
    - If reconstruction loss not decreasing:
        * Check data normalization
        * Increase model capacity (more units)
        * Reduce encoding_dim temporarily
        * Check for NaN in data

    - If classifier not learning:
        * Verify encoder was pre-trained
        * Try unfreezing encoder earlier
        * Increase dense_units
        * Check class imbalance

    - If encoder learns trivial identity:
        * Encoding_dim too large (no compression)
        * Reduce encoding_dim
        * Add noise to input during pre-training

20. ADVANCED TECHNIQUES:
    - Contractive Autoencoder:
        Add penalty on Jacobian of encoder
        Forces smooth latent space
        Loss += λ * ||∂h/∂x||²

    - Denoising Autoencoder:
        Add noise to input, train to reconstruct clean output
        Input: X + noise, Target: X
        More robust representations

    - Sequence-to-Sequence Autoencoder:
        Predict future sequence instead of reconstruct
        Input: X[0:T], Target: X[T:2T]
        Useful for forecasting tasks

    - Multi-task Learning:
        Train autoencoder + classifier jointly
        Loss = L_reconstruction + β * L_classification
        Balances reconstruction and discrimination
"""
