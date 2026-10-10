"""
Technical Analyst Attention Network (TAAN)
==========================================

Two-level structured attention over a set of indicator matrices. Each indicator (RSI, Stochastic, ZigZag, ...) is a
matrix of ``R`` lookback periods x ``F`` recent timesteps. Level 1 applies an independent HierarchicalCellAttention per
indicator, selecting the most relevant periods (rows). The selected rows are stacked into one ``(M, H1 * F)`` matrix and
level 2 applies a second HierarchicalCellAttention across the ``M`` indicators. An MLP head classifies.

With the paper's configuration (one 8-period ZigZag matrix, eight 14-period oscillator matrices, F = 10, 4 level-1
heads, 3 level-2 heads, dense 128 -> 64, 3 classes) the model has exactly 26,006 trainable parameters, matching the
count reported in Table I of the paper. ``build_attention_model`` returns a weight-sharing model whose outputs are the
attention weights of both levels, for interpretability analysis.

Suitable for: turning-point classification, meta-labelling of a base signal, regime classification over an indicator
panel.

Reference:
    Tavassolian, Yousefloei, Abdoos, Vahidi Asl (2026). "Technical Analyst Attention Network (TAAN): An Interpretable
    Deep Learning Model for Algorithmic Trading in Crypto and Forex Markets." IEEE IICAI.
    DOI 10.1109/IICAI70155.2026.11620664
"""

import re
from enum import StrEnum
from typing import Dict, Mapping, Optional, Tuple

import keras
from keras import layers, models
from keras.regularizers import l2

from ..layers.hierarchical_cell_attention import HierarchicalCellAttention
from ..metrics import BalancedAccuracy, MacroF1Score
from .common import TaskType, create_output_layer_and_loss, get_optimizer, get_model_name

SUPPORTED_TASK_TYPES = (TaskType.CLASSIFICATION, TaskType.REGRESSION)


class TaanLayer(StrEnum):
    """Names of the layers ``create_taan`` builds. Used for construction, validation and attention extraction."""
    LEVEL1_PREFIX = "l1_hca_"  # + indicator name
    LEVEL1_FLAT_PREFIX = "l1_flat_"  # + indicator name
    LEVEL2 = "l2_hca"
    LEVEL2_STACK = "l2_stack"
    LEVEL2_FLAT = "l2_flat"
    DENSE_1 = "dense_1"
    DENSE_2 = "dense_2"
    DROPOUT_1 = "dense_dropout_1"
    DROPOUT_2 = "dense_dropout_2"
    OUTPUT = "output"  # created by create_output_layer_and_loss


class AttentionOutput(StrEnum):
    """Keys of the dict returned by the model from ``build_attention_model``."""
    LEVEL1_PREFIX = "l1_attn_"  # + indicator name
    LEVEL2 = "l2_attn"


_PREFIXED_LAYERS = (TaanLayer.LEVEL1_PREFIX, TaanLayer.LEVEL1_FLAT_PREFIX)
_FIXED_LAYER_NAMES = frozenset(str(member) for member in TaanLayer if member not in _PREFIXED_LAYERS)
# Layer names become TensorFlow graph scope names during tracing, which only accept this character set.
_VALID_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _validate_matrix_shapes(matrix_shapes: Mapping[str, Tuple[int, int]]) -> int:
    """Validate names and shapes up front so errors name the offending indicator; return the shared column count."""
    if not matrix_shapes:
        raise ValueError("matrix_shapes must contain at least one indicator matrix")
    cols = set()
    for name, shape in matrix_shapes.items():
        if not isinstance(name, str) or not _VALID_NAME.match(name):
            raise ValueError(f"indicator name {name!r} must be a non-empty string of letters, digits, '_', '.', '-' "
                             f"(it becomes a Keras layer name)")
        if name in _FIXED_LAYER_NAMES or name.startswith(tuple(str(prefix) for prefix in _PREFIXED_LAYERS)):
            raise ValueError(f"indicator name {name!r} collides with an internal TAAN layer name (see TaanLayer)")
        if not (isinstance(shape, tuple) and len(shape) == 2 and all(isinstance(d, int) and d > 0 for d in shape)):
            raise ValueError(f"matrix_shapes[{name!r}] must be a (rows, cols) tuple of positive ints, got {shape!r}")
        cols.add(shape[1])
    if len(cols) != 1:
        raise ValueError(f"all indicator matrices must share the same number of columns, got {sorted(cols)}")
    return cols.pop()


def _validate_task(task_type, num_classes: Optional[int]) -> None:
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"TAAN supports {[str(t) for t in SUPPORTED_TASK_TYPES]}, got {task_type}")
    if task_type == TaskType.CLASSIFICATION and not (isinstance(num_classes, int) and num_classes >= 2):
        raise ValueError(f"num_classes must be an int >= 2 for classification, got {num_classes!r}")


def create_taan(matrix_shapes: Mapping[str, Tuple[int, int]], num_classes: Optional[int],
                task_type=TaskType.CLASSIFICATION, level1_heads=4, level2_heads=3, temperature=1.0, dense_units_1=128,
                dense_units_2=64, dense_dropout=0.0, learning_rate=0.001, l2_reg=0.0, jit_compile=False):
    """
    Create a compiled TAAN model.

    Parameters
    ----------
    matrix_shapes : Mapping[str, tuple[int, int]]
        Ordered mapping of indicator name -> (rows, cols). Each entry becomes one named model input of that shape, so
        the model is trained with ``model.fit({name: array, ...}, y)``. ALWAYS feed a dict. Keras binds a positional
        list by sorted input name, not by insertion order; it raises only when the two orders differ, and otherwise
        silently mis-binds matrices with equal row counts. All matrices must share the same ``cols``. Names are
        restricted to letters, digits, ``_``, ``.`` and ``-`` and must not collide with ``TaanLayer`` names.
    num_classes : int or None
        Number of output classes (2 -> sigmoid / binary cross-entropy, >= 3 -> softmax / sparse cross-entropy).
        Ignored (may be None) for TaskType.REGRESSION.
    task_type : TaskType
        TaskType.CLASSIFICATION (default) or TaskType.REGRESSION. Other task types raise ValueError.
    level1_heads : int, default=4
        Periods selected per indicator by the intra-indicator attention.
    level2_heads : int, default=3
        Indicators selected by the inter-indicator attention.
    temperature : float, default=1.0
        Softmax temperature shared by both attention levels.
    dense_units_1, dense_units_2 : int, default=128, 64
        Width of the two hidden layers of the classification head.
    dense_dropout : float, default=0.0
        Dropout after each hidden layer (the paper uses none).
    learning_rate : float, default=0.001
        Adam learning rate.
    l2_reg : float, default=0.0
        L2 regularisation on the dense layers (the paper uses none).
    jit_compile : bool, default=False
        Whether to enable XLA JIT compilation in model.compile.

    Returns
    -------
    keras.Model
        Compiled Keras model with one named input per indicator matrix. Logged metrics: multi-class -> ``accuracy``,
        ``macro_f1``, ``balanced_accuracy``; binary -> ``accuracy``, ``precision``, ``recall``, ``auc``; regression ->
        the shared regression set (``mae``, ``mse``, ``rmse``, ``r2_score``, ``directional_accuracy``).
    """
    _validate_task(task_type, num_classes)
    n_cols = _validate_matrix_shapes(matrix_shapes)

    inputs = {}
    selected = []
    for name, (rows, _) in matrix_shapes.items():
        x = layers.Input(shape=(rows, n_cols), name=name)
        inputs[name] = x
        out, _ = HierarchicalCellAttention(num_heads=level1_heads, temperature=temperature,
                                           name=f"{TaanLayer.LEVEL1_PREFIX}{name}")(x)
        # (B, H1, F) -> (B, 1, H1 * F): one row per indicator in the level-2 matrix
        selected.append(layers.Reshape((1, level1_heads * n_cols), name=f"{TaanLayer.LEVEL1_FLAT_PREFIX}{name}")(out))

    level2_input = layers.Concatenate(axis=1, name=TaanLayer.LEVEL2_STACK)(selected)  # (B, M, H1 * F); M = 1 is fine
    level2_out, _ = HierarchicalCellAttention(num_heads=level2_heads, temperature=temperature,
                                              name=TaanLayer.LEVEL2)(level2_input)

    x = layers.Flatten(name=TaanLayer.LEVEL2_FLAT)(level2_out)  # (B, H2 * H1 * F)
    regularizer = l2(l2_reg) if l2_reg else None
    x = layers.Dense(dense_units_1, activation="relu", kernel_regularizer=regularizer, name=TaanLayer.DENSE_1)(x)
    if dense_dropout:
        x = layers.Dropout(dense_dropout, name=TaanLayer.DROPOUT_1)(x)
    x = layers.Dense(dense_units_2, activation="relu", kernel_regularizer=regularizer, name=TaanLayer.DENSE_2)(x)
    if dense_dropout:
        x = layers.Dropout(dense_dropout, name=TaanLayer.DROPOUT_2)(x)

    outputs, loss, output_metrics = create_output_layer_and_loss(x, task_type, num_classes,
                                                                 output_name=TaanLayer.OUTPUT)
    if task_type == TaskType.CLASSIFICATION and num_classes > 2:
        # The shared helper attaches regime-transition metrics to multi-class heads; they are meaningless for sparse
        # per-bar event labels, so use the imbalance-aware set instead. The binary and regression sets are kept.
        output_metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                          MacroF1Score(num_classes=num_classes), BalancedAccuracy(num_classes=num_classes)]

    model = models.Model(inputs=inputs, outputs=outputs, name=get_model_name("taan", task_type))
    model.compile(optimizer=get_optimizer("adam", learning_rate), loss=loss, metrics=output_metrics,
                  jit_compile=jit_compile)
    return model


def default_tuner_objective(num_classes: Optional[int], task_type=TaskType.CLASSIFICATION) -> Tuple[str, str]:
    """Return ``(metric_name, direction)`` for keras_tuner matching the metrics ``create_taan`` logs for this task."""
    _validate_task(task_type, num_classes)
    if task_type == TaskType.REGRESSION:
        return "val_loss", "min"
    if num_classes == 2:
        return "val_auc", "max"
    return "val_macro_f1", "max"


def build_attention_model(model: keras.Model) -> keras.Model:
    """
    Return a model sharing ``model``'s weights whose outputs are the attention weights of both levels.

    Outputs are a dict: ``l1_attn_<name>`` -> (B, R_name, level1_heads) for every indicator, and ``l2_attn`` ->
    (B, M, level2_heads), where the M rows follow the insertion order of ``matrix_shapes`` given to ``create_taan``.
    Like the parent, the returned model takes a dict of arrays keyed by indicator name.
    """
    outputs: Dict[str, keras.KerasTensor] = {}
    level1_prefix = str(TaanLayer.LEVEL1_PREFIX)
    found = []
    for layer in model.layers:
        if not isinstance(layer, HierarchicalCellAttention):
            continue
        found.append(layer.name)
        _, attn = layer.output  # the layer returns (selected rows, attention weights)
        if layer.name == TaanLayer.LEVEL2:
            outputs[str(AttentionOutput.LEVEL2)] = attn
        elif layer.name.startswith(level1_prefix):
            outputs[f"{AttentionOutput.LEVEL1_PREFIX}{layer.name[len(level1_prefix):]}"] = attn
    if str(AttentionOutput.LEVEL2) not in outputs:
        raise ValueError(f"not a TAAN model: missing level-2 layer {str(TaanLayer.LEVEL2)!r}; "
                         f"HierarchicalCellAttention layers found: {found}")
    # Keep the parent's dict input structure (keyed by indicator name) so both models are fed the same way.
    inputs = {tensor.name: tensor for tensor in model.inputs}
    return models.Model(inputs=inputs, outputs=outputs, name=f"{model.name}_attention")


def create_tunable_taan(matrix_shapes: Mapping[str, Tuple[int, int]], num_classes: Optional[int],
                        task_type=TaskType.CLASSIFICATION, jit_compile=False):
    """
    Create a tunable version of TAAN for keras_tuner hyperparameter optimisation.

    Example
    -------
    >>> metric, direction = default_tuner_objective(num_classes=3)
    >>> tuner = kt.BayesianOptimization(
    ...     create_tunable_taan(matrix_shapes={"rsi": (14, 10), "zigzag": (8, 10)}, num_classes=3),
    ...     objective=kt.Objective(metric, direction=direction),
    ...     max_trials=20,
    ...     directory='tuner_results',
    ...     project_name='taan'
    ... )
    >>> tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
    """

    def build_model(hp):
        level1_heads = hp.Choice("level1_heads", values=[2, 3, 4, 6])
        level2_heads = hp.Choice("level2_heads", values=[2, 3, 4])
        temperature = hp.Choice("temperature", values=[0.5, 1.0, 2.0])
        dense_units_1 = hp.Int("dense_units_1", min_value=64, max_value=256, step=64)
        dense_units_2 = hp.Int("dense_units_2", min_value=32, max_value=128, step=32)
        dense_dropout = hp.Float("dense_dropout", min_value=0.0, max_value=0.4, step=0.1)
        learning_rate = hp.Choice("learning_rate", values=[1e-4, 5e-4, 1e-3, 5e-3])
        l2_reg = hp.Choice("l2_reg", values=[0.0, 1e-5, 1e-4, 1e-3])
        return create_taan(matrix_shapes=matrix_shapes, num_classes=num_classes, task_type=task_type,
                           level1_heads=level1_heads, level2_heads=level2_heads, temperature=temperature,
                           dense_units_1=dense_units_1, dense_units_2=dense_units_2, dense_dropout=dense_dropout,
                           learning_rate=learning_rate, l2_reg=l2_reg, jit_compile=jit_compile)

    return build_model


# ============================================================================
# HINTS AND BEST PRACTICES
# ============================================================================
"""
1. INPUT LAYOUT:
   - One matrix per indicator: rows = lookback periods (e.g. 5, 7, 10, ..., 90), cols = recent timesteps, with the
     LAST column being the current bar. Column F-1 at time t must only use data up to and including t.
   - All matrices share the column count; row counts may differ (the paper uses 8 for ZigZag and 14 for oscillators).
   - Bounded oscillators (RSI, Stochastic, ADX, DI) should be divided by their natural bound; anything unbounded
     (ZigZag distances) needs a CAUSAL normaliser (rolling or train-frozen min-max). Full-sample min-max leaks.
   - ALWAYS feed the model a dict keyed by indicator name. A positional list is bound by Keras in sorted-name order,
     which silently mis-binds matrices with equal row counts. Indicator names become layer names and must not collide
     with the TaanLayer enum members (l1_hca_*, l1_flat_*, l2_hca, l2_stack, l2_flat, dense_1, dense_2,
     dense_dropout_*, output).

2. INTERPRETABILITY:
   - build_attention_model(model).predict(inputs) returns the per-head row weights at both levels.
   - Level-2 weights averaged over heads give a per-sample "which indicator family am I listening to" vector; it is
     causal if the model was trained only on earlier data, so it can serve as a regime descriptor.
   - Attention is not importance: confirm any claim with an ablation (zero the top-attended matrix, measure the drop).

3. CLASS IMBALANCE:
   - Turning-point labels are ~85% Hold. Use class_weight in fit() and monitor macro_f1 / balanced_accuracy (multi-
     class) or auc (binary), never raw accuracy. Early-stop on the metric default_tuner_objective() names, with a
     CHRONOLOGICAL validation split.

4. TEMPERATURE:
   - tau < 1 makes each head close to a hard top-1 row selection (more interpretable, noisier gradients).
   - tau > 1 blends rows (smoother, less selective). The paper treats it as a fixed hyperparameter.

5. PARAMETER BUDGET:
   - Attention weights are tiny (R*F + R*H per matrix); the MLP head dominates. With F=10 and heads 4/3 the head
     sees H2*H1*F = 120 inputs. Keep dense_units small; the paper's 128/64 is already most of the 26k parameters.

6. DEVIATIONS FROM THE PAPER (deliberate, shared with the rest of paper2keras):
   - get_optimizer applies gradient clipping (clipnorm=1.0); the paper does not mention clipping.
   - Dropout and L2 default to 0 to match the paper, but are exposed for the tunable variant.
   - Only CLASSIFICATION and REGRESSION are supported; POSTERIOR_DISTILLATION is rejected because the imbalance-aware
     metrics expect integer labels.
"""
