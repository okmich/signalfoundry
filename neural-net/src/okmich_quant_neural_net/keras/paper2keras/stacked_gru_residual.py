"""
Stacked GRU with Residual Connections
======================================

Architecture designed for time-series classification with faster training than LSTM and residual connections to prevent
vanishing gradients.

Suitable for: volatility prediction, momentum classification, regime detection

Reference:
Selvin et al. (2017) "Stock Price Prediction Using Deep Learning Algorithm and its Comparison" - (https://ieeexplore.ieee.org/document/8259364)
"""

import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers, models

from .common import TaskType, create_output_layer_and_loss, get_optimizer, get_model_name


def create_stacked_gru_residual(input_shape, num_classes, task_type=TaskType.CLASSIFICATION,
                                gru_units_1=128, gru_units_2=128, gru_units_3=64, dense_units=32,
                                dropout_1=0.3, dropout_2=0.3, dropout_3=0.2, learning_rate=0.001, l2_reg=0.0001):
    """
    Create a Stacked GRU model with residual connections.

    Supports both classification and regression tasks.

    Parameters
    ----------
    input_shape : tuple
        Shape of input data (sequence_length, n_features)
    num_classes : int
        Number of output classes for classification
    task_type : TaskType
        TaskType.CLASSIFICATION or TaskType.REGRESSION (default: CLASSIFICATION)
    gru_units_1 : int, default=128
        Number of units in first GRU layer
    gru_units_2 : int, default=128
        Number of units in second GRU layer (should match gru_units_1 for residual)
    gru_units_3 : int, default=64
        Number of units in third GRU layer
    dense_units : int, default=32
        Number of units in dense layer
    dropout_1 : float, default=0.3
        Dropout rate after first GRU
    dropout_2 : float, default=0.3
        Dropout rate after second GRU
    dropout_3 : float, default=0.2
        Dropout rate after third GRU
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

    # First GRU layer
    gru1 = layers.GRU(
        gru_units_1,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="gru_1",
    )(inputs)
    gru1 = layers.Dropout(dropout_1, name="dropout_1")(gru1)

    # Second GRU layer with residual connection
    gru2 = layers.GRU(
        gru_units_2,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="gru_2",
    )(gru1)
    gru2 = layers.Dropout(dropout_2, name="dropout_2")(gru2)

    # Add residual connection (skip connection from gru1 to gru2)
    # Only works if gru_units_1 == gru_units_2
    if gru_units_1 == gru_units_2:
        gru2_residual = layers.Add(name="residual_connection")([gru1, gru2])
    else:
        # If units don't match, use a projection layer
        gru1_projected = layers.Dense(gru_units_2, name="residual_projection")(gru1)
        gru2_residual = layers.Add(name="residual_connection")([gru1_projected, gru2])

    # Third GRU layer (no return_sequences, outputs last timestep)
    gru3 = layers.GRU(
        gru_units_3,
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="gru_3",
    )(gru2_residual)
    gru3 = layers.Dropout(dropout_3, name="dropout_3")(gru3)

    # Dense layer
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name="dense",
    )(gru3)

    # Output layer with task-specific configuration
    outputs, loss, output_metrics = create_output_layer_and_loss(
        x, task_type, num_classes
    )

    # Create model
    model_name = get_model_name("stacked_gru_residual", task_type)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    # Compile
    opt = get_optimizer("adam", learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=output_metrics)

    return model


def create_tunable_stacked_gru_residual(input_shape, num_classes, task_type=TaskType.CLASSIFICATION):
    """
    Create a tunable version of the Stacked GRU model for hyperparameter optimization.

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

    Example
    -------
    >>> tuner = kt.BayesianOptimization(
    ...     create_tunable_stacked_gru_residual(
    ...         input_shape=(50, 10),
    ...         num_classes=3,
    ...         task_type=TaskType.CLASSIFICATION
    ...     ),
    ...     objective='val_accuracy',
    ...     max_trials=20,
    ...     directory='tuner_results',
    ...     project_name='stacked_gru_residual'
    ... )
    >>> tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
    """

    def build_model(hp):
        # Hyperparameter search space
        gru_units_1 = hp.Int("gru_units_1", min_value=64, max_value=256, step=64)
        gru_units_2 = gru_units_1  # Keep same for residual connection
        gru_units_3 = hp.Int("gru_units_3", min_value=32, max_value=128, step=32)
        dense_units = hp.Int("dense_units", min_value=16, max_value=64, step=16)

        dropout_1 = hp.Float("dropout_1", min_value=0.2, max_value=0.5, step=0.1)
        dropout_2 = hp.Float("dropout_2", min_value=0.2, max_value=0.5, step=0.1)
        dropout_3 = hp.Float("dropout_3", min_value=0.1, max_value=0.4, step=0.1)

        learning_rate = hp.Choice("learning_rate", values=[1e-4, 5e-4, 1e-3, 5e-3])
        l2_reg = hp.Choice("l2_reg", values=[1e-5, 1e-4, 1e-3])

        return create_stacked_gru_residual(
            input_shape=input_shape,
            num_classes=num_classes,
            task_type=task_type,
            gru_units_1=gru_units_1,
            gru_units_2=gru_units_2,
            gru_units_3=gru_units_3,
            dense_units=dense_units,
            dropout_1=dropout_1,
            dropout_2=dropout_2,
            dropout_3=dropout_3,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
        )

    return build_model


# ============================================================================
# HINTS AND BEST PRACTICES
# ============================================================================
"""
1. RESIDUAL CONNECTIONS:
   - GRU layers 1 and 2 must have the same number of units for clean residuals
   - If different sizes needed, a projection layer is automatically added
   - Residuals help with gradient flow in deep networks

2. INPUT DATA PREPARATION:
   - Normalize/standardize features before training
   - Shape: (n_samples, sequence_length, n_features)
   - For 5-min trading data: sequence_length might be 32-100 bars

3. CLASS IMBALANCE HANDLING:
   - Use class_weight parameter in model.fit() for imbalanced data
   - Consider SMOTE or other oversampling techniques
   - Monitor macro F1 score alongside accuracy

4. TRAINING TIPS:
   - Start with learning_rate=0.001, reduce if loss oscillates
   - Use batch_size=32 or 64 for stability
   - Apply gradient clipping if gradients explode: optimizer.clipnorm=1.0

5. HYPERPARAMETER TUNING:
   - Focus on: GRU units, dropout rates, learning rate
   - Use 20-50 trials for thorough search
   - Monitor both val_loss and val_auc

6. ARCHITECTURE VARIANTS:
   - Add BatchNormalization after GRU layers for faster convergence
   - Use bidirectional GRUs for better context: layers.Bidirectional(GRU(...))
   - Add attention mechanism after final GRU for interpretability

7. COMPARISON WITH LSTM:
   - GRU: ~25% fewer parameters, faster training
   - LSTM: Better for very long sequences (100+ timesteps)
   - For 5-min bars (32-100 steps), GRU is usually sufficient
    """
