"""
Custom Keras Metrics for Classification and Regression Tasks.

This module provides custom metrics that are useful for trading and financial
time series prediction tasks, handling both classification and regression scenarios.

Classification Metrics:
    - BalancedAccuracy: Robust to class imbalance
    - MacroF1Score: Robust to class imbalance

Regression Metrics:
    - R2Score: Coefficient of determination
    - DirectionalAccuracy: Sign prediction accuracy (critical for trading)
    - MAPE: Mean Absolute Percentage Error

Regime Metrics:
    - CausalRegimeAccuracy: Temporally-persistent accuracy via causal sliding window
    - RegimeTransitionRecall: Detection rate of true regime changes (causal lag tolerance)
    - RegimeTransitionPrecision: Precision of predicted regime transitions

Usage:
    >>> from okmich_quant_neural_net.keras.metrics import R2Score, DirectionalAccuracy
    >>> model.compile(
    ...     optimizer='adam',
    ...     loss='mse',
    ...     metrics=[R2Score(), DirectionalAccuracy()]
    ... )
"""

import tensorflow as tf
from keras import metrics


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

class BalancedAccuracy(metrics.Metric):
    """
    Balanced accuracy metric - robust to class imbalance.

    Balanced accuracy is the average of recall obtained on each class.
    Unlike standard accuracy, it gives equal weight to each class regardless
    of their frequency in the dataset.

    Formula:
        balanced_accuracy = mean([recall_class_0, recall_class_1, ...])

    This is particularly useful for trading classification where class imbalance
    is common (e.g., more neutral samples than strong trends).

    Args:
        num_classes: Number of classes in the classification task
        name: Metric name (default: 'balanced_accuracy')

    Example:
        >>> metric = BalancedAccuracy(num_classes=3)
        >>> model.compile(..., metrics=[metric])
    """

    def __init__(self, num_classes: int, name: str = "balanced_accuracy", **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(
            name="confusion_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
        cm = tf.math.confusion_matrix(
            y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32
        )
        self.confusion_matrix.assign_add(cm)

    def result(self):
        per_class_recall = tf.linalg.diag_part(self.confusion_matrix) / (
                tf.reduce_sum(self.confusion_matrix, axis=1) + 1e-7
        )
        return tf.reduce_mean(per_class_recall)

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))


class MacroF1Score(metrics.Metric):
    """
    Macro F1-score - robust to class imbalance.

    Computes F1-score for each class independently, then averages them.
    This gives equal weight to each class regardless of support.

    Formula:
        f1_class = 2 * (precision * recall) / (precision + recall)
        macro_f1 = mean([f1_class_0, f1_class_1, ...])

    Useful when you care equally about performance on all classes,
    even rare ones (e.g., strong bullish/bearish moves).

    Implementation uses confusion-matrix accumulation (vectorized, no Python
    loops over classes) — consistent with BalancedAccuracy.

    Args:
        num_classes: Number of classes in the classification task
        name: Metric name (default: 'macro_f1')

    Example:
        >>> metric = MacroF1Score(num_classes=3)
        >>> model.compile(..., metrics=[metric])
    """

    def __init__(self, num_classes: int, name: str = "macro_f1", **kwargs):
        super(MacroF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(
            name="confusion_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
        cm = tf.math.confusion_matrix(
            y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32
        )
        self.confusion_matrix.assign_add(cm)

    def result(self):
        # TP per class: diagonal
        tp = tf.linalg.diag_part(self.confusion_matrix)
        # FP per class: predicted as c but not truly c (column sum minus TP)
        fp = tf.reduce_sum(self.confusion_matrix, axis=0) - tp
        # FN per class: truly c but predicted as other (row sum minus TP)
        fn = tf.reduce_sum(self.confusion_matrix, axis=1) - tp
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1_per_class = 2 * precision * recall / (precision + recall + 1e-7)
        return tf.reduce_mean(f1_per_class)

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))


# ============================================================================
# REGRESSION METRICS
# ============================================================================

class R2Score(metrics.Metric):
    """
    Coefficient of determination (R² score) for regression.

    R² measures the proportion of variance in the dependent variable that is
    predictable from the independent variable(s).

    Formula:
        R² = 1 - (SS_res / SS_tot)
        where:
            SS_res = sum of squared residuals
            SS_tot = total sum of squares = sum(y_true^2) - sum(y_true)^2 / n

    Interpretation:
        R² = 1.0 → Perfect predictions
        R² = 0.0 → Model predicts as well as the mean
        R² < 0.0 → Model is worse than predicting the mean

    For trading models, R² > 0.1 is often considered good due to market noise.

    Implementation note: SS_tot is computed via sufficient statistics
    (sum_y, sum_y2, count) so that the result is batch-order invariant.
    Computing SS_tot as sum((y - running_mean)^2) per batch is incorrect
    because the running mean changes with each batch, making R² depend on
    the order batches are presented.

    Args:
        name: Metric name (default: 'r2_score')

    Example:
        >>> metric = R2Score()
        >>> model.compile(..., metrics=[metric])
    """

    def __init__(self, name: str = 'r2_score', **kwargs):
        super(R2Score, self).__init__(name=name, **kwargs)
        self.ss_res = self.add_weight(name='ss_res', initializer='zeros')
        self.sum_y = self.add_weight(name='sum_y', initializer='zeros')
        self.sum_y2 = self.add_weight(name='sum_y2', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        # Accumulate sufficient statistics
        residuals = y_true - y_pred
        self.ss_res.assign_add(tf.reduce_sum(residuals ** 2))
        self.sum_y.assign_add(tf.reduce_sum(y_true))
        self.sum_y2.assign_add(tf.reduce_sum(y_true ** 2))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        # SS_tot = sum(y^2) - sum(y)^2 / n  (computational formula, batch-order invariant)
        ss_tot = self.sum_y2 - (self.sum_y ** 2) / (self.count + 1e-7)
        return 1.0 - (self.ss_res / (ss_tot + 1e-7))

    def reset_state(self):
        self.ss_res.assign(0.0)
        self.sum_y.assign(0.0)
        self.sum_y2.assign(0.0)
        self.count.assign(0.0)


class DirectionalAccuracy(metrics.Metric):
    """
    Directional accuracy for regression predictions.

    Measures how often the sign of the prediction matches the sign of the
    true value. This is critical for trading regression models where getting
    the direction right matters as much as (or more than) the exact magnitude.

    Formula:
        directional_accuracy = (correct_sign_predictions) / (total_predictions)

    For trading:
        - Values > 0.5 indicate the model has predictive power
        - Values > 0.55 are considered good
        - Values > 0.6 are excellent for financial markets

    Args:
        name: Metric name (default: 'directional_accuracy')

    Example:
        >>> metric = DirectionalAccuracy()
        >>> model.compile(..., metrics=[metric])
    """

    def __init__(self, name: str = 'directional_accuracy', **kwargs):
        super(DirectionalAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        # Check if signs match
        same_sign = tf.equal(tf.sign(y_true), tf.sign(y_pred))
        self.correct.assign_add(tf.reduce_sum(tf.cast(same_sign, tf.float32)))
        self.total.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.correct / (self.total + 1e-7)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


class MAPE(metrics.Metric):
    """
    Mean Absolute Percentage Error.

    MAPE expresses error as a percentage of the true value, making it
    scale-independent and easier to interpret across different assets.

    Formula:
        MAPE = mean(|y_true - y_pred| / |y_true|) * 100

    Note: Skips samples where y_true is very close to zero to avoid division issues.

    Interpretation:
        MAPE < 10%: Excellent
        MAPE 10-20%: Good
        MAPE 20-50%: Acceptable
        MAPE > 50%: Poor

    Args:
        name: Metric name (default: 'mape')

    Example:
        >>> metric = MAPE()
        >>> model.compile(..., metrics=[metric])
    """

    def __init__(self, name: str = 'mape', **kwargs):
        super(MAPE, self).__init__(name=name, **kwargs)
        self.sum_ape = self.add_weight(name='sum_ape', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        # Avoid division by zero - skip samples where y_true is near zero
        mask = tf.abs(y_true) > 1e-7
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)

        # Calculate absolute percentage error
        ape = tf.abs((y_true_masked - y_pred_masked) / y_true_masked)
        self.sum_ape.assign_add(tf.reduce_sum(ape))
        self.count.assign_add(tf.cast(tf.size(y_true_masked), tf.float32))

    def result(self):
        return (self.sum_ape / (self.count + 1e-7)) * 100.0

    def reset_state(self):
        self.sum_ape.assign(0.0)
        self.count.assign(0.0)


# ============================================================================
# REGIME METRICS
# ============================================================================

class CausalRegimeAccuracy(metrics.Metric):
    """
    Causally-aware accuracy for regime classification over sequences.

    The score at time t is the average accuracy of the prediction window
    [t-window_size+1, ..., t]. This rewards models that stay correct for
    consecutive bars rather than flickering between classes.

    Early timesteps that have not yet accumulated a full window are normalised
    by the number of available bars (min(t+1, window_size)) so they are not
    unfairly penalised.

    Args:
        window_size: Number of consecutive bars in the causal window (default 5).
        name: Metric name (default 'causal_regime_accuracy').

    Example:
        >>> metric = CausalRegimeAccuracy(window_size=5)
        >>> model.compile(..., metrics=[metric])
    """

    def __init__(self, window_size=5, name='causal_regime_accuracy', **kwargs):
        super(CausalRegimeAccuracy, self).__init__(name=name, **kwargs)
        self.window_size = window_size
        self.total_windowed_accuracy = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred_classes = tf.argmax(y_pred, axis=-1)  # int64
        if len(y_true.shape) == 3:  # (batch, seq, classes) one-hot
            true_classes = tf.argmax(y_true, axis=-1)
        elif len(y_true.shape) == 2:  # (batch, seq) integer labels
            true_classes = tf.cast(y_true, tf.int64)
        else:  # (batch,) scalar — expand to (1, batch)
            true_classes = tf.cast(tf.expand_dims(y_true, 0), tf.int64)
            pred_classes = tf.expand_dims(pred_classes, 0)

        # Binary correctness map: [batch, seq]
        is_correct = tf.cast(tf.equal(pred_classes, true_classes), tf.float32)

        # Causal zero-padding at the start so conv1d sees only past + current
        paddings = [[0, 0], [self.window_size - 1, 0]]
        x_padded = tf.pad(is_correct, paddings, mode='constant', constant_values=0.0)
        x_padded = tf.expand_dims(x_padded, axis=-1)  # [batch, seq+pad, 1]

        filters = tf.ones((self.window_size, 1, 1), dtype=tf.float32)
        window_sums = tf.nn.conv1d(
            input=x_padded, filters=filters, stride=1, padding='VALID'
        )  # [batch, seq, 1]

        # Normalise early steps by their actual available history
        seq_len = tf.shape(is_correct)[1]
        positions = tf.cast(tf.range(1, seq_len + 1), tf.float32)
        effective_window = tf.minimum(positions, tf.cast(self.window_size, tf.float32))
        effective_window = tf.reshape(effective_window, [1, -1, 1])

        windowed_acc = tf.squeeze(window_sums / effective_window, axis=-1)

        self.total_windowed_accuracy.assign_add(tf.reduce_sum(windowed_acc))
        self.count.assign_add(tf.cast(tf.reduce_prod(tf.shape(windowed_acc)), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total_windowed_accuracy, self.count)

    def reset_state(self):
        self.total_windowed_accuracy.assign(0.0)
        self.count.assign(0.0)


class RegimeTransitionRecall(metrics.Metric):
    """
    Recall for true regime-change events along a sequence.

    A true transition at bar t is considered "caught" if the model predicts
    a transition anywhere in [t, ..., t+lag_tolerance] (causal only — no
    future leakage; early false calls before t are not credited).

    Boundary contract: index 0 is never counted as a transition event.
    The shift uses self-reference (true_shifted[0] = true_classes[0]),
    so the difference at t=0 is always zero. This is an explicit design
    choice: the first bar has no predecessor, so regime continuity from
    before the sequence cannot be assessed.

    Args:
        lag_tolerance: How many bars after a true transition the model may
            still receive credit (default 2).
        name: Metric name (default 'regime_transition_recall').

    Example:
        >>> metric = RegimeTransitionRecall(lag_tolerance=2)
        >>> model.compile(..., metrics=[metric])
    """

    def __init__(self, lag_tolerance=2, name='regime_transition_recall', **kwargs):
        super(RegimeTransitionRecall, self).__init__(name=name, **kwargs)
        self.lag_tolerance = lag_tolerance
        self.transitions_caught = self.add_weight(name='caught', initializer='zeros')
        self.total_true_transitions = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred_classes = tf.argmax(y_pred, axis=-1)  # int64
        if len(y_true.shape) == 3:  # (batch, seq, classes) one-hot
            true_classes = tf.argmax(y_true, axis=-1)
        elif len(y_true.shape) == 2:  # (batch, seq) integer labels
            true_classes = tf.cast(y_true, tf.int64)
        else:  # (batch,) scalar — expand to (1, batch)
            true_classes = tf.cast(tf.expand_dims(y_true, 0), tf.int64)
            pred_classes = tf.expand_dims(pred_classes, 0)

        # Detect transitions by comparing each bar with its predecessor
        true_shifted = tf.concat([true_classes[:, :1], true_classes], axis=1)[:, :-1]
        pred_shifted = tf.concat([pred_classes[:, :1], pred_classes], axis=1)[:, :-1]

        true_edges = tf.cast(tf.not_equal(true_classes, true_shifted), tf.float32)
        pred_edges = tf.cast(tf.not_equal(pred_classes, pred_shifted), tf.float32)

        # Causally dilate predicted edges: position t covers [t-lag, ..., t]
        pred_edges_exp = tf.expand_dims(pred_edges, axis=-1)
        padded = tf.pad(pred_edges_exp, [[0, 0], [self.lag_tolerance, 0], [0, 0]])
        dilated = tf.nn.max_pool1d(
            padded, ksize=self.lag_tolerance + 1, strides=1, padding='VALID'
        )
        dilated_pred_edges = tf.squeeze(dilated, axis=-1)

        self.transitions_caught.assign_add(
            tf.reduce_sum(true_edges * dilated_pred_edges)
        )
        self.total_true_transitions.assign_add(tf.reduce_sum(true_edges))

    def result(self):
        return tf.math.divide_no_nan(self.transitions_caught, self.total_true_transitions)

    def reset_state(self):
        self.transitions_caught.assign(0.0)
        self.total_true_transitions.assign(0.0)


class RegimeTransitionPrecision(metrics.Metric):
    """
    Precision for predicted regime-change events along a sequence.

    Complements RegimeTransitionRecall: asks "of all predicted transitions,
    how many had a real one nearby?" A true transition at t is considered
    nearby a prediction at p if p is in [t, ..., t+lag_tolerance] (causal).

    Boundary contract: index 0 is never counted as a transition event for
    either true or predicted sequences. The shift uses self-reference
    (classes_shifted[0] = classes[0]), so the difference at t=0 is always
    zero. This mirrors RegimeTransitionRecall's boundary policy.

    Args:
        lag_tolerance: Backward window around a true transition within which
            a predicted transition counts as correct (default 2).
        name: Metric name (default 'regime_transition_precision').

    Example:
        >>> metric = RegimeTransitionPrecision(lag_tolerance=2)
        >>> model.compile(..., metrics=[metric])
    """

    def __init__(self, lag_tolerance=2, name='regime_transition_precision', **kwargs):
        super(RegimeTransitionPrecision, self).__init__(name=name, **kwargs)
        self.lag_tolerance = lag_tolerance
        self.correct_predictions = self.add_weight(name='correct', initializer='zeros')
        self.total_predictions = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred_classes = tf.argmax(y_pred, axis=-1)  # int64
        if len(y_true.shape) == 3:  # (batch, seq, classes) one-hot
            true_classes = tf.argmax(y_true, axis=-1)
        elif len(y_true.shape) == 2:  # (batch, seq) integer labels
            true_classes = tf.cast(y_true, tf.int64)
        else:  # (batch,) scalar — expand to (1, batch)
            true_classes = tf.cast(tf.expand_dims(y_true, 0), tf.int64)
            pred_classes = tf.expand_dims(pred_classes, 0)

        true_shifted = tf.concat([true_classes[:, :1], true_classes], axis=1)[:, :-1]
        pred_shifted = tf.concat([pred_classes[:, :1], pred_classes], axis=1)[:, :-1]

        true_edges = tf.cast(tf.not_equal(true_classes, true_shifted), tf.float32)
        pred_edges = tf.cast(tf.not_equal(pred_classes, pred_shifted), tf.float32)

        # Causally dilate true edges so a predicted transition is "correct" if
        # a real one occurred within [t-lag, ..., t]
        true_edges_exp = tf.expand_dims(true_edges, axis=-1)
        padded = tf.pad(true_edges_exp, [[0, 0], [self.lag_tolerance, 0], [0, 0]])
        dilated = tf.nn.max_pool1d(
            padded, ksize=self.lag_tolerance + 1, strides=1, padding='VALID'
        )
        dilated_true_edges = tf.squeeze(dilated, axis=-1)

        self.correct_predictions.assign_add(
            tf.reduce_sum(pred_edges * dilated_true_edges)
        )
        self.total_predictions.assign_add(tf.reduce_sum(pred_edges))

    def result(self):
        return tf.math.divide_no_nan(self.correct_predictions, self.total_predictions)

    def reset_state(self):
        self.correct_predictions.assign(0.0)
        self.total_predictions.assign(0.0)
