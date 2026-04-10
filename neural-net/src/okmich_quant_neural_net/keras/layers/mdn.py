"""
Mixture Density Network (MDN) Layer
====================================

MDN outputs parameters of a mixture of probability distributions instead of point predictions. This provides
probabilistic forecasts with uncertainty quantification - critical for risk-aware trading.

Key Advantages for Trading:
----------------------------
✓ Uncertainty quantification (know when model is confident vs uncertain)
✓ Multi-modal predictions (capture multiple possible outcomes)
✓ Risk-aware position sizing (size based on confidence and distribution)
✓ Better tail risk modeling (capture extreme moves)
✓ Regime-aware predictions (different Gaussians for different regimes)

Architecture:
-------------
Traditional Network:
    Input → LSTM → Dense → Single Output (point estimate)
                             ⚠️ No uncertainty information

Mixture Density Network:
    Input → LSTM → MDN Layer → Distribution Parameters
                                ✓ Full probability distribution
                                ✓ Uncertainty quantification

MDN outputs for each mixture component:
- pi (π): Mixing coefficient (weight of this component)
- mu (μ): Mean of the Gaussian
- sigma (σ): Standard deviation of the Gaussian

For 3 mixtures predicting 1D output (e.g., next return):
    Output = [π1, π2, π3, μ1, μ2, μ3, σ1, σ2, σ3]
    Total: 9 values

Trading Example:
----------------
Instead of: "Return will be +0.5%"
MDN predicts: "60% chance of +0.1% (σ=0.2%), 30% chance of -0.8% (σ=0.4%),
               10% chance of +2% (σ=1.0%)"

This allows:
- Position sizing based on confidence
- Risk assessment from σ values
- Regime detection from mixture weights
- Stop-loss placement based on tail probabilities
"""

import numpy as np
import tensorflow as tf
from keras import layers
from keras.src.saving import register_keras_serializable


@register_keras_serializable()
class MixtureDensityLayer(layers.Layer):
    """
    Mixture Density Network output layer.

    Outputs parameters for a mixture of Gaussian distributions:
    - Mixing coefficients (π): Weights for each component (sum to 1)
    - Means (μ): Center of each Gaussian
    - Std deviations (σ): Spread of each Gaussian

    Args:
        output_dim: Dimensionality of output variable (default: 1 for returns)
        num_mixtures: Number of Gaussian components in mixture (default: 3)
        epsilon: Small constant for numerical stability (default: 1e-6)

    Input shape:
        (batch_size, features)

    Output shape:
        (batch_size, output_params) where:
        output_params = num_mixtures * (1 + 2*output_dim)
        For output_dim=1, num_mixtures=3: output_params = 9

    Example:
        >>> mdn = MixtureDensityLayer(output_dim=1, num_mixtures=3)
        >>> params = mdn(lstm_output)  # (batch, 9)
        >>> pi, mu, sigma = split_mdn_params(params, 1, 3)
    """

    def __init__(self, output_dim=1, num_mixtures=3, epsilon=1e-6, **kwargs):
        super(MixtureDensityLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_mixtures = num_mixtures
        self.epsilon = epsilon

        # Total number of output parameters
        # For each mixture: 1 pi + output_dim mu's + output_dim sigma's
        self.num_params = num_mixtures * (1 + 2 * output_dim)

        # Dense layer to output all parameters
        self.dense = None

    def build(self, input_shape):
        # Create dense layer
        self.dense = layers.Dense(
            self.num_params, activation="linear", name="mdn_params"
        )
        self.dense.build(input_shape)
        super(MixtureDensityLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Forward pass through MDN layer.

        Returns raw parameters that need to be split and transformed:
        - pi (mixing coefficients) → softmax
        - mu (means) → linear
        - sigma (std devs) → exp (ensure positive)
        """
        return self.dense(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_params)

    def get_config(self):
        config = super(MixtureDensityLayer, self).get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "num_mixtures": self.num_mixtures,
                "epsilon": self.epsilon,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def split_mdn_params(params, output_dim, num_mixtures, epsilon=1e-6):
    """
    Split MDN output into mixing coefficients, means, and std deviations.

    Args:
        params: Raw MDN output (batch, num_params)
        output_dim: Dimensionality of output variable
        num_mixtures: Number of mixture components
        epsilon: Small constant for numerical stability

    Returns:
        pi: Mixing coefficients (batch, num_mixtures)
        mu: Means (batch, num_mixtures, output_dim)
        sigma: Std deviations (batch, num_mixtures, output_dim)

    Example:
        >>> params = model.predict(X_test)  # (100, 9) for 3 mixtures, 1D output
        >>> pi, mu, sigma = split_mdn_params(params, 1, 3)
        >>> # pi: (100, 3), mu: (100, 3, 1), sigma: (100, 3, 1)
    """
    # Split parameters
    pi_params = params[:, :num_mixtures]
    mu_params = params[:, num_mixtures: num_mixtures * (1 + output_dim)]
    sigma_params = params[:, num_mixtures * (1 + output_dim):]

    # Transform parameters
    # Pi: softmax to ensure they sum to 1
    pi = tf.nn.softmax(pi_params, axis=-1)

    # Mu: linear (no transformation)
    mu = tf.reshape(mu_params, [-1, num_mixtures, output_dim])

    # Sigma: exp to ensure positive, add epsilon for stability
    sigma = tf.exp(sigma_params) + epsilon
    sigma = tf.reshape(sigma, [-1, num_mixtures, output_dim])

    return pi, mu, sigma


def mdn_loss(output_dim, num_mixtures, epsilon=1e-6):
    """
    Create MDN loss function (negative log-likelihood).

    The loss is the negative log-likelihood of the true value under
    the predicted mixture of Gaussians.

    Args:
        output_dim: Dimensionality of output variable
        num_mixtures: Number of mixture components
        epsilon: Small constant for numerical stability

    Returns:
        Loss function compatible with model.compile()

    Example:
        >>> loss_fn = mdn_loss(output_dim=1, num_mixtures=3)
        >>> model.compile(optimizer='adam', loss=loss_fn)
    """

    def loss(y_true, y_pred):
        """
        Compute negative log-likelihood.

        Args:
            y_true: True values (batch, output_dim)
            y_pred: MDN parameters (batch, num_params)

        Returns:
            Scalar loss value
        """
        # Split parameters
        pi, mu, sigma = split_mdn_params(y_pred, output_dim, num_mixtures, epsilon)

        # Reshape y_true to match mu shape
        y_true = tf.reshape(y_true, [-1, 1, output_dim])

        # Compute log-Gaussian probability for each component in log-space to
        # avoid underflow when output_dim > 1 (reduce_prod can zero-out easily).
        # log p(y|k) = sum_d [ -0.5*log(2π) - log(σ_kd) - 0.5*((y_d-μ_kd)/σ_kd)² ]
        log_gaussian = (
                -0.5 * tf.math.log(2.0 * np.pi)
                - tf.math.log(sigma)
                - 0.5 * tf.square((y_true - mu) / sigma)
        )
        log_gaussian = tf.reduce_sum(log_gaussian, axis=-1)  # (batch, num_mixtures)

        # log mixture probability: logsumexp(log(π) + log p(y|k))
        log_mixture = tf.reduce_logsumexp(
            tf.math.log(pi + epsilon) + log_gaussian, axis=-1
        )  # (batch,)

        return tf.reduce_mean(-log_mixture)

    return loss


def sample_from_mdn(pi, mu, sigma, num_samples=1):
    """
    Sample from the predicted mixture distribution.

    Args:
        pi: Mixing coefficients (batch, num_mixtures)
        mu: Means (batch, num_mixtures, output_dim)
        sigma: Std deviations (batch, num_mixtures, output_dim)
        num_samples: Number of samples to draw per batch item

    Returns:
        Samples from the mixture (batch, num_samples, output_dim)

    Example:
        >>> pi, mu, sigma = split_mdn_params(params, 1, 3)
        >>> samples = sample_from_mdn(pi, mu, sigma, num_samples=1000)
        >>> # Use samples for Monte Carlo simulation
    """
    batch_size = tf.shape(pi)[0]
    num_mixtures = tf.shape(pi)[1]
    output_dim = tf.shape(mu)[2]

    # Sample mixture components according to pi
    # For each sample, choose which Gaussian to sample from
    component_indices = tf.random.categorical(
        tf.math.log(pi + 1e-8), num_samples=num_samples
    )  # (batch, num_samples)

    # Gather selected mu and sigma
    batch_indices = tf.tile(
        tf.expand_dims(tf.range(batch_size), axis=1), [1, num_samples]
    )

    indices = tf.stack([batch_indices, component_indices], axis=-1)

    selected_mu = tf.gather_nd(mu, indices)  # (batch, num_samples, output_dim)
    selected_sigma = tf.gather_nd(sigma, indices)  # (batch, num_samples, output_dim)

    # Sample from selected Gaussians
    epsilon = tf.random.normal(tf.shape(selected_mu))
    samples = selected_mu + selected_sigma * epsilon

    return samples


def get_mdn_predictions(params, output_dim, num_mixtures, method="mean"):
    """
    Get point predictions from MDN parameters.

    Args:
        params: MDN output parameters
        output_dim: Dimensionality of output
        num_mixtures: Number of mixture components
        method: Prediction method
            - 'mean': Weighted mean of mixture
            - 'mode': Most probable component
            - 'sample': Random sample from distribution

    Returns:
        Predictions (batch, output_dim)

    Example:
        >>> params = model.predict(X_test)
        >>> predictions = get_mdn_predictions(params, 1, 3, method='mean')
    """
    pi, mu, sigma = split_mdn_params(params, output_dim, num_mixtures)

    if method == "mean":
        # Weighted mean of all components
        pi_expanded = tf.expand_dims(pi, axis=-1)  # (batch, num_mixtures, 1)
        weighted_mu = pi_expanded * mu  # (batch, num_mixtures, output_dim)
        prediction = tf.reduce_sum(weighted_mu, axis=1)  # (batch, output_dim)

    elif method == "mode":
        # Most probable component (highest pi)
        mode_indices = tf.argmax(pi, axis=1)  # (batch,)
        batch_indices = tf.range(tf.shape(pi)[0])
        indices = tf.stack([batch_indices, mode_indices], axis=1)
        prediction = tf.gather_nd(mu, indices)  # (batch, output_dim)

    elif method == "sample":
        # Random sample
        samples = sample_from_mdn(pi, mu, sigma, num_samples=1)
        prediction = samples[:, 0, :]  # (batch, output_dim)

    else:
        raise ValueError(f"Unknown method: {method}")

    return prediction


def get_mdn_uncertainty(params, output_dim, num_mixtures):
    """
    Extract uncertainty measures from MDN.

    Args:
        params: MDN output parameters
        output_dim: Dimensionality of output
        num_mixtures: Number of mixture components

    Returns:
        Dictionary with uncertainty measures:
        - 'total_variance': Total variance of mixture
        - 'entropy': Entropy of mixing coefficients (regime uncertainty)
        - 'dominant_component': Weight of most probable component

    Example:
        >>> params = model.predict(X_test)
        >>> uncertainty = get_mdn_uncertainty(params, 1, 3)
        >>> high_uncertainty = uncertainty['total_variance'] > threshold
    """
    pi, mu, sigma = split_mdn_params(params, output_dim, num_mixtures)

    # 1. Total variance (aleatoric + epistemic uncertainty)
    # Var[X] = E[Var[X|component]] + Var[E[X|component]]

    # Weighted mean
    pi_expanded = tf.expand_dims(pi, axis=-1)
    mean = tf.reduce_sum(pi_expanded * mu, axis=1)  # (batch, output_dim)

    # Within-component variance (aleatoric)
    within_var = tf.reduce_sum(pi_expanded * tf.square(sigma), axis=1)

    # Between-component variance (epistemic)
    mean_expanded = tf.expand_dims(mean, axis=1)  # (batch, 1, output_dim)
    between_var = tf.reduce_sum(pi_expanded * tf.square(mu - mean_expanded), axis=1)

    total_variance = within_var + between_var  # (batch, output_dim)
    total_variance = tf.reduce_mean(total_variance, axis=1)  # (batch,)

    # 2. Entropy of mixing coefficients (regime uncertainty)
    entropy = -tf.reduce_sum(pi * tf.math.log(pi + 1e-8), axis=1)  # (batch,)

    # 3. Dominant component weight
    dominant_component = tf.reduce_max(pi, axis=1)  # (batch,)

    return {
        "total_variance": total_variance,
        "entropy": entropy,
        "dominant_component": dominant_component,
    }
