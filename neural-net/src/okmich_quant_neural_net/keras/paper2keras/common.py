from enum import Enum
from typing import Tuple, List

import keras
from keras import layers, losses, metrics

from ..metrics import (
    R2Score, DirectionalAccuracy,  # noqa: F401 — re-exported for callers
    CausalRegimeAccuracy, RegimeTransitionRecall, RegimeTransitionPrecision,
)


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

    def __str__(self):
        return self.value


def create_output_layer_and_loss(x, task_type: TaskType, num_classes: int = None, output_name: str = "output") \
        -> Tuple[layers.Layer, losses.Loss, List[metrics.Metric]]:
    """
    Create the output layer, loss function, and metrics based on task type.

    This function encapsulates ALL the conditional logic for classification vs regression, making model definition files
    clean and consistent across all architectures. Instead of 30+ lines of if/else in each model file, you have ONE line.

    Args:
        x: The input tensor (output from the last hidden layer)
        task_type: TaskType.CLASSIFICATION or TaskType.REGRESSION
        num_classes: Number of classes (required for classification, ignored for regression)
        output_name: Name for the output layer (default: "output")

    Returns:
        Tuple of (output_layer, loss_function, metrics_list):
            - output_layer: Keras Dense layer with appropriate activation
            - loss_function: Keras loss function appropriate for the task
            - metrics_list: List of Keras metrics appropriate for the task

    Raises:
        TypeError: If task_type is not a TaskType enum
        ValueError: If num_classes is missing for classification task

    Example (Classification):
        >>> from okmich_quant_neural_net.keras.paper2keras.common import (
        ...     TaskType, create_output_layer_and_loss
        ... )
        >>> x = layers.Dense(64, activation='relu')(inputs)
        >>> output, loss, metrics = create_output_layer_and_loss(
        ...     x, TaskType.CLASSIFICATION, num_classes=3
        ... )
        >>> model = models.Model(inputs=inputs, outputs=output)
        >>> model.compile(optimizer='adam', loss=loss, metrics=metrics)

    Example (Regression):
        >>> x = layers.Dense(64, activation='relu')(inputs)
        >>> output, loss, metrics = create_output_layer_and_loss(
        ...     x, TaskType.REGRESSION
        ... )
        >>> model = models.Model(inputs=inputs, outputs=output)
        >>> model.compile(optimizer='adam', loss=loss, metrics=metrics)
    """
    # Validate inputs
    if not isinstance(task_type, TaskType):
        raise TypeError(
            f"task_type must be TaskType enum, got: {type(task_type)}. "
            f"Use TaskType.CLASSIFICATION or TaskType.REGRESSION"
        )

    if task_type == TaskType.CLASSIFICATION:
        if num_classes is None:
            raise ValueError("num_classes is required for classification tasks")

        if num_classes == 2:
            # Binary classification
            output_layer = layers.Dense(1, activation='sigmoid', name=output_name)(x)
            loss_fn = losses.BinaryCrossentropy()
            metrics_list = [
                metrics.BinaryAccuracy(name='accuracy'),
                metrics.Precision(name='precision'),
                metrics.Recall(name='recall'),
                metrics.AUC(name='auc'),
            ]
        else:
            # Multi-class classification
            output_layer = layers.Dense(num_classes, activation='softmax', name=output_name)(x)
            loss_fn = losses.SparseCategoricalCrossentropy()
            metrics_list = [
                CausalRegimeAccuracy(window_size=5, name='causal_regime_accuracy'),
                RegimeTransitionRecall(lag_tolerance=2, name='regime_transition_recall'),
                RegimeTransitionPrecision(lag_tolerance=2, name='regime_transition_precision'),
            ]
    elif task_type == TaskType.REGRESSION:
        # Single continuous output
        output_layer = layers.Dense(1, activation='linear', name=output_name)(x)
        # Huber loss (robust to outliers, recommended for trading)
        # delta=1.0 means errors < 1.0 use squared loss, errors > 1.0 use absolute loss
        loss_fn = losses.Huber(delta=1.0)

        # Regression metrics
        metrics_list = [
            metrics.MeanAbsoluteError(name='mae'),
            metrics.MeanSquaredError(name='mse'),
            metrics.RootMeanSquaredError(name='rmse'),
            R2Score(name='r2_score'),
            DirectionalAccuracy(name='directional_accuracy'),
        ]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    return output_layer, loss_fn, metrics_list


def get_optimizer(optimizer_name: str, learning_rate: float, clipnorm: float = 1.0) -> keras.optimizers.Optimizer:
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        return keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    elif optimizer_name == 'adamw':
        return keras.optimizers.AdamW(learning_rate=learning_rate, clipnorm=clipnorm)
    elif optimizer_name == 'rmsprop':
        return keras.optimizers.RMSprop(learning_rate=learning_rate, clipnorm=clipnorm)
    elif optimizer_name == 'sgd':
        return keras.optimizers.SGD(
            learning_rate=learning_rate,
            clipnorm=clipnorm,
            momentum=0.9,
            nesterov=True
        )
    else:
        return keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)


def get_model_name(base_name: str, task_type: TaskType) -> str:
    if task_type == TaskType.REGRESSION:
        return f"{base_name}_Regression"
    else:
        return base_name
