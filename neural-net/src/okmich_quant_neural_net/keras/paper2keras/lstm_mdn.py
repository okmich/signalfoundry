"""
LSTM-Mixture Density Network (LSTM-MDN) - Model Factory
========================================================

Architecture Description:
-------------------------
This architecture combines an LSTM with a Mixture Density Network (MDN) layer.
Instead of predicting a single class or value, the MDN predicts the parameters of a probability distribution (typically a mixture of Gaussians).

This provides:
✓ Probabilistic forecasts with uncertainty quantification
✓ Multi-modal predictions (multiple possible outcomes)
✓ Risk-aware position sizing based on confidence
✓ Better tail risk modeling

Architecture:
-------------
Input (32-100 timesteps, n_features)
    ↓
LSTM (128 units, return_sequences=True) → Dropout (0.2)
    ↓
LSTM (64 units, return_sequences=False) → Dropout (0.2)
    ↓
MDN Output Layer
    ↓
Output: [π1, π2, π3, μ1, μ2, μ3, σ1, σ2, σ3]
    - π (pi): Mixing coefficients (weights of each Gaussian)
    - μ (mu): Means of each Gaussian component
    - σ (sigma): Standard deviations of each Gaussian component

Trading Application:
--------------------
Instead of: "Price will go up"

MDN predicts: "60% chance of small upward move (mean=+0.1%, std=0.2%),
               30% chance of large downward move (mean=-0.8%, std=0.4%),
               10% chance of sideways move (mean=0%, std=0.1%)"

This allows:
✓ Position sizing based on confidence and risk
✓ Stop-loss placement based on tail probabilities
✓ Risk/reward assessment from predicted distribution
✓ Regime detection from mixture components
✓ Uncertainty quantification (know when model is uncertain)

Key Advantages:
---------------
✓ Probabilistic predictions (not just point estimates)
✓ Uncertainty quantification (critical for risk management)
✓ Multi-modal distributions (capture complex patterns)
✓ Risk-aware trading (size positions by confidence)
✓ Better tail risk modeling (extreme moves)
✓ Regime-aware (different Gaussians = different regimes)

Metrics for Class Imbalance:
-----------------------------
For classification tasks, we use robust metrics:
✓ Balanced Accuracy (equal weight to all classes)
✓ Macro F1-score (average F1 across classes)
✓ Cohen's Kappa (accounts for chance agreement)
✓ Per-class Precision/Recall
✓ Matthews Correlation Coefficient (MCC)

Usage:
------
1. Regression (predict return distributions):
   model = build_lstm_mdn_regression(
       sequence_length=48,
       num_features=20,
       num_mixtures=3
   )

2. Classification (predict directional probabilities):
   model = build_lstm_mdn_classification(
       sequence_length=48,
       num_features=20,
       num_classes=3,
       num_mixtures=3
   )

3. Tunable version (with KerasTuner):
   def model_builder(hp):
       return build_lstm_mdn_tunable(
           hp=hp,
           num_features=20,
           task='regression'
       )
"""

import numpy as np
from keras import layers, models, optimizers, losses, metrics

# Import MDN components
from ..layers.mdn import MixtureDensityLayer, mdn_loss, split_mdn_params
from ..metrics import BalancedAccuracy, MacroF1Score


# ============================================================================
# REGRESSION VERSION (Predict Return Distributions)
# ============================================================================


def build_lstm_mdn_regression(sequence_length=48, num_features=20, num_mixtures=3, output_dim=1,
                              lstm1_units=128, lstm2_units=64, dropout1=0.2, dropout2=0.2, learning_rate=0.001,
                              optimizer_name="adam"):
    """
    Build LSTM-MDN for regression (predict return distributions).

    This version predicts probability distributions over continuous values
    (e.g., returns, prices) with uncertainty quantification.

    Args:
        sequence_length: Number of timesteps
        num_features: Number of features per timestep
        num_mixtures: Number of Gaussian components (default: 3)
            - More mixtures = more complex distributions
            - 3-5 typically sufficient for returns
        output_dim: Dimensionality of output (default: 1 for returns)
        lstm1_units: Units in first LSTM (default: 128)
        lstm2_units: Units in second LSTM (default: 64)
        dropout1: Dropout after first LSTM (default: 0.2)
        dropout2: Dropout after second LSTM (default: 0.2)
        learning_rate: Learning rate (default: 0.001)
        optimizer_name: Optimizer ('adam', 'adamw', 'rmsprop')

    Returns:
        Compiled Keras model

    Example:
        >>> model = build_lstm_mdn_regression(
        ...     sequence_length=48,
        ...     num_features=20,
        ...     num_mixtures=3
        ... )
        >>> # Train
        >>> model.fit(X_train, y_train_returns, epochs=50)
        >>> # Predict distributions
        >>> params = model.predict(X_test)
        >>> pi, mu, sigma = split_mdn_params(params, 1, 3)
    """

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # LSTM layers
    x = layers.LSTM(lstm1_units, return_sequences=True, name="lstm1")(inputs)
    x = layers.Dropout(dropout1, name="dropout1")(x)

    x = layers.LSTM(lstm2_units, return_sequences=False, name="lstm2")(x)
    x = layers.Dropout(dropout2, name="dropout2")(x)

    # MDN output layer
    outputs = MixtureDensityLayer(
        output_dim=output_dim, num_mixtures=num_mixtures, name="mdn_output"
    )(x)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name="LSTM_MDN_Regression")

    # Select optimizer
    if optimizer_name.lower() == "adam":
        opt = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    elif optimizer_name.lower() == "adamw":
        opt = optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0)
    else:
        opt = optimizers.RMSprop(learning_rate=learning_rate, clipnorm=1.0)

    # Compile with MDN loss
    model.compile(
        optimizer=opt, loss=mdn_loss(output_dim=output_dim, num_mixtures=num_mixtures)
    )

    print(f"\nLSTM-MDN Regression Model Configuration:")
    print(f"  Number of mixtures: {num_mixtures}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Output parameters: {num_mixtures * (1 + 2 * output_dim)}")
    print(f"  Loss: Negative Log-Likelihood")

    return model


# ============================================================================
# CLASSIFICATION VERSION (Predict Directional Probabilities)
# ============================================================================


def build_lstm_mdn_classification(sequence_length=48, num_features=20, num_classes=3, num_mixtures=3,
                                  lstm1_units=128, lstm2_units=64, dropout1=0.2, dropout2=0.2, dense_units=64,
                                  dense_dropout=0.2, learning_rate=0.001, optimizer_name="adam", class_weights=None):
    """
    Build LSTM-MDN for classification with uncertainty quantification.

    This version uses MDN to learn uncertainty-aware class probabilities.
    The model first predicts a continuous distribution, then maps it to
    class probabilities with confidence scores.

    Args:
        sequence_length: Number of timesteps
        num_features: Number of features per timestep
        num_classes: Number of output classes
        num_mixtures: Number of Gaussian components for uncertainty (default: 3)
        lstm1_units: Units in first LSTM (default: 128)
        lstm2_units: Units in second LSTM (default: 64)
        dropout1: Dropout after first LSTM (default: 0.2)
        dropout2: Dropout after second LSTM (default: 0.2)
        dense_units: Units in dense layer (default: 64)
        dense_dropout: Dropout after dense layer (default: 0.2)
        learning_rate: Learning rate (default: 0.001)
        optimizer_name: Optimizer to use
        class_weights: Dictionary of class weights for imbalance

    Returns:
        Compiled Keras model with class-imbalance robust metrics

    Example:
        >>> # Compute class weights for imbalanced data
        >>> from sklearn.utils.class_weight import compute_class_weight
        >>> class_weights = compute_class_weight(
        ...     'balanced',
        ...     classes=np.unique(y_train),
        ...     y=y_train
        ... )
        >>> class_weights = {i: w for i, w in enumerate(class_weights)}
        >>>
        >>> model = build_lstm_mdn_classification(
        ...     sequence_length=48,
        ...     num_features=20,
        ...     num_classes=3,
        ...     class_weights=class_weights
        ... )
    """

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # LSTM layers
    x = layers.LSTM(lstm1_units, return_sequences=True, name="lstm1")(inputs)
    x = layers.Dropout(dropout1, name="dropout1")(x)

    x = layers.LSTM(lstm2_units, return_sequences=False, name="lstm2")(x)
    x = layers.Dropout(dropout2, name="dropout2")(x)

    # MDN layer for uncertainty
    mdn_params = MixtureDensityLayer(
        output_dim=1, num_mixtures=num_mixtures, name="mdn_uncertainty"
    )(x)

    # Dense layer on top of MDN features
    x = layers.Concatenate(name="concat_features")([x, mdn_params])
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dense_dropout, name="dropout_dense")(x)

    # Classification output
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name="LSTM_MDN_Classification")

    # Select optimizer
    if optimizer_name.lower() == "adam":
        opt = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    elif optimizer_name.lower() == "adamw":
        opt = optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0)
    else:
        opt = optimizers.RMSprop(learning_rate=learning_rate, clipnorm=1.0)

    # Compile with class-imbalance robust metrics
    model.compile(
        optimizer=opt,
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[
            metrics.SparseCategoricalAccuracy(name="accuracy"),
            BalancedAccuracy(num_classes=num_classes, name="balanced_accuracy"),
            MacroF1Score(num_classes=num_classes, name="macro_f1"),
        ],
    )

    print(f"\nLSTM-MDN Classification Model Configuration:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Uncertainty mixtures: {num_mixtures}")
    print(f"  Metrics: Accuracy, Balanced Accuracy, Macro F1 (class-imbalance robust)")

    return model


# ============================================================================
# TUNABLE VERSION (With KerasTuner)
# ============================================================================


def build_lstm_mdn_tunable(hp, num_features, task="regression", num_classes=None, sequence_length=None,
                           max_sequence_length=100):
    """
    Build tunable LSTM-MDN model for hyperparameter optimization.

    Args:
        hp: KerasTuner HyperParameters object
        num_features: Number of features per timestep (fixed)
        task: 'regression' or 'classification'
        num_classes: Number of classes (for classification)
        sequence_length: If None, tunes sequence_length (32 to max_sequence_length).
                        If provided, uses this fixed value (for pre-created sequences).
        max_sequence_length: Maximum sequence length (used when sequence_length=None)

    Returns:
        Compiled Keras model

    Example:
        >>> import keras_tuner
        >>> # Example 1: Tune sequence_length
        >>> def model_builder(hp):
        ...     return build_lstm_mdn_tunable(
        ...         hp=hp,
        ...         num_features=20,
        ...         task='regression'
        ...     )
        >>> # Example 2: Fixed sequence_length (for pre-created sequences)
        >>> def model_builder(hp):
        ...     return build_lstm_mdn_tunable(
        ...         hp=hp,
        ...         num_features=20,
        ...         task='regression',
        ...         sequence_length=48
        ...     )
        >>> tuner = keras_tuner.BayesianOptimization(
        ...     model_builder,
        ...     objective='val_loss',
        ...     max_trials=20
        ... )
    """

    # Tunable hyperparameters
    if sequence_length is None:
        # Tune sequence_length if not provided
        sequence_length = hp.Int("sequence_length", 32, max_sequence_length, step=16)
    # else: use the provided fixed sequence_length
    num_mixtures = hp.Choice("num_mixtures", [2, 3, 4, 5])
    lstm1_units = hp.Choice("lstm1_units", [64, 128, 256])
    lstm2_units = hp.Choice("lstm2_units", [32, 64, 128])
    dropout1 = hp.Float("dropout1", 0.1, 0.4, step=0.1)
    dropout2 = hp.Float("dropout2", 0.1, 0.4, step=0.1)
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    optimizer_name = hp.Choice("optimizer", ["adam", "adamw", "rmsprop"])

    # Input layer
    inputs = layers.Input(shape=(sequence_length, num_features), name="input_sequences")

    # LSTM layers
    x = layers.LSTM(lstm1_units, return_sequences=True, name="lstm1")(inputs)
    x = layers.Dropout(dropout1, name="dropout1")(x)

    x = layers.LSTM(lstm2_units, return_sequences=False, name="lstm2")(x)
    x = layers.Dropout(dropout2, name="dropout2")(x)

    if task == "regression":
        # MDN output for regression
        outputs = MixtureDensityLayer(
            output_dim=1, num_mixtures=num_mixtures, name="mdn_output"
        )(x)

        # Select optimizer
        if optimizer_name == "adam":
            opt = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        elif optimizer_name == "adamw":
            opt = optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0)
        else:
            opt = optimizers.RMSprop(learning_rate=learning_rate, clipnorm=1.0)

        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name="LSTM_MDN_Tunable")

        # Compile with MDN loss
        model.compile(
            optimizer=opt, loss=mdn_loss(output_dim=1, num_mixtures=num_mixtures)
        )

    else:  # classification
        dense_units = hp.Choice("dense_units", [32, 64, 128])
        dense_dropout = hp.Float("dense_dropout", 0.1, 0.3, step=0.1)

        # MDN layer for uncertainty
        mdn_params = MixtureDensityLayer(
            output_dim=1, num_mixtures=num_mixtures, name="mdn_uncertainty"
        )(x)

        # Dense layer
        x = layers.Concatenate(name="concat_features")([x, mdn_params])
        x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
        x = layers.Dropout(dense_dropout, name="dropout_dense")(x)

        # Classification output
        outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name="LSTM_MDN_Tunable")

        # Select optimizer
        if optimizer_name == "adam":
            opt = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        elif optimizer_name == "adamw":
            opt = optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0)
        else:
            opt = optimizers.RMSprop(learning_rate=learning_rate, clipnorm=1.0)

        # Compile with class-imbalance metrics
        model.compile(
            optimizer=opt,
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=[
                metrics.SparseCategoricalAccuracy(name="accuracy"),
                BalancedAccuracy(num_classes=num_classes),
                MacroF1Score(num_classes=num_classes),
            ],
        )

    return model


# ============================================================================
# HINTS AND BEST PRACTICES
# ============================================================================

"""
HINTS FOR USING LSTM-MDN IN TRADING:
=====================================

1. MDN BASICS:
   - MDN predicts distribution parameters, not point estimates
   - Each mixture component = one possible market regime
   - Pi (π): Weight of each regime
   - Mu (μ): Expected outcome for each regime
   - Sigma (σ): Uncertainty/risk for each regime

2. NUMBER OF MIXTURES:
   - 2: Binary regimes (e.g., volatile vs calm)
   - 3: Recommended for most crypto/stock trading
   - 4-5: Complex multi-regime markets
   - Too many (>5): Risk of overfitting

   Each mixture should represent a meaningful regime:
   - Mixture 1: Small moves (sideways market)
   - Mixture 2: Upward trends
   - Mixture 3: Downward trends

3. REGRESSION VS CLASSIFICATION:
   Use Regression when:
   - Need exact return predictions
   - Want to model return distributions
   - Risk/reward ratios are important
   - Position sizing based on expected value

   Use Classification when:
   - Only need direction (up/down/sideways)
   - Working with discrete labels
   - Simplicity preferred over precision

4. CLASS IMBALANCE HANDLING:
   For imbalanced classification:
   ✓ Use class_weight parameter in fit()
   ✓ Monitor Balanced Accuracy (not just accuracy)
   ✓ Track Macro F1 (equal weight to all classes)
   ✓ Use per-class precision/recall
   ✓ Cohen's Kappa for overall agreement

5. POSITION SIZING WITH MDN:
   Use uncertainty for risk management:

   ```python
   # Get predictions
   params = model.predict(X)
   pi, mu, sigma = split_mdn_params(params, 1, num_mixtures)

   # Expected return (weighted mean)
   expected_return = np.sum(pi * mu, axis=1)

   # Risk (total variance)
   uncertainty = get_mdn_uncertainty(params, 1, num_mixtures)
   risk = np.sqrt(uncertainty['total_variance'])

   # Position size = f(expected_return, risk, confidence)
   position_size = (expected_return / risk) * confidence
   ```

6. UNCERTAINTY INTERPRETATION:
   Total Variance (aleatoric + epistemic):
   - Low (<0.0001): Model is confident
   - Medium (0.0001-0.001): Normal uncertainty
   - High (>0.001): Model is uncertain - reduce position!

   Entropy (regime uncertainty):
   - Low (<0.5): Clear regime (one dominant mixture)
   - Medium (0.5-1.0): Mixed regime
   - High (>1.0): Unclear regime - stay out!

   Dominant Component:
   - High (>0.7): Strong conviction
   - Medium (0.4-0.7): Moderate conviction
   - Low (<0.4): Weak conviction - small position

7. RISK MANAGEMENT:
   Stop-loss placement:
   - Use sigma (σ) from predicted distribution
   - Stop-loss = entry ± (2-3) * sigma
   - Higher sigma = wider stops

   Take-profit:
   - Target = entry + expected_return
   - Adjust by confidence level

   Position sizing:
   - Kelly criterion with MDN variance
   - Scale down in high uncertainty regimes

8. TRAINING TIPS:
   - Batch size: 32-64
   - Epochs: 50-100 (MDN needs more training)
   - Learning rate: 0.0005-0.002
   - Use gradient clipping (clipnorm=1.0)
   - Monitor validation loss carefully
   - Early stopping patience: 15-20

9. FEATURE ENGINEERING:
   MDN works well with:
   - Returns (not raw prices)
   - Volatility measures
   - Volume indicators
   - Regime indicators
   - Standardize features!

10. COMMON PITFALLS:
    ✗ Using raw prices (use returns!)
    ✗ Too many mixtures (overfitting)
    ✗ Forgetting to extract parameters with split_mdn_params()
    ✗ Not using uncertainty for risk management
    ✗ Ignoring entropy (regime uncertainty)

11. DEBUGGING MDN:
    If poor performance:
    - Reduce num_mixtures (try 2-3)
    - Increase training epochs
    - Check feature scaling
    - Visualize predicted distributions
    - Monitor component weights (pi)

    If unstable training:
    - Lower learning rate
    - Increase gradient clipping
    - Simplify LSTM architecture
    - Check for outliers in targets

12. WALK-FORWARD INTEGRATION:
    ```python
    def model_builder_fn(hp):
        return build_lstm_mdn_tunable(
            hp=hp,
            num_features=20,
            task='regression'
        )

    optimizer = ModelWalkForwardAnalysisOptimizer(
        model_builder_fn=model_builder_fn,
        tuning_epochs=50,  # MDN needs more epochs
        # ... other params
    )
    ```

13. PRODUCTION DEPLOYMENT:
    - Save model with: model.save()
    - Load with: keras.models.load_model()
    - Remember to include custom objects:
      ```python
      from okmich_quant_ml.keras.layers.mdn import MixtureDensityLayer
      model = keras.models.load_model(
          'model.keras',
          custom_objects={'MixtureDensityLayer': MixtureDensityLayer}
      )
      ```

14. ADVANCED TECHNIQUES:
    - Temperature scaling for calibration
    - Ensemble multiple MDNs
    - Conditional MDN (different mixtures per regime)
    - Time-varying mixtures

15. COMPARISON TO OTHER MODELS:
    vs Standard LSTM:
    ✓ MDN provides uncertainty
    ✓ Better risk management
    ✓ Multi-modal predictions
    ✗ Slower to train
    ✗ More complex

    vs Ensemble:
    ✓ Single model (faster inference)
    ✓ Principled uncertainty
    ✗ More training required

16. MONITORING IN PRODUCTION:
    Track these metrics:
    - Average uncertainty over time
    - Dominant component distribution
    - Prediction vs reality (calibration)
    - Component usage (are all mixtures used?)

17. REGIME DETECTION:
    Use mixture weights to detect regimes:
    ```python
    # If component 0 dominant: sideways market
    # If component 1 dominant: uptrend
    # If component 2 dominant: downtrend
    regime = np.argmax(pi, axis=1)
    ```

18. ENSEMBLE WITH OTHER MODELS:
    Combine MDN with:
    - ESN: MDN for uncertainty, ESN for speed
    - TCN: MDN for distribution, TCN for patterns
    - Multiple MDNs with different num_mixtures

19. HYPERPARAMETER TUNING PRIORITIES:
    1. num_mixtures (most important)
    2. LSTM units (capacity)
    3. Learning rate (convergence)
    4. Dropout (regularization)
    5. Number of epochs (MDN needs more)

20. KEY INSIGHT:
    "MDN transforms predictions from 'price will be X' to
     'there's a 60% chance price will be around X ± Y, and
     a 40% chance it will be around Z ± W'. This probabilistic
     view is exactly what traders need for position sizing and
     risk management."
"""
