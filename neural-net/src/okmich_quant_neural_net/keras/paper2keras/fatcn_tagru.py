"""
FATCN-TAGRU Ablation Suite — Model Factory
===========================================

Full ablation of the architecture from:
    Fu Y. & Xiao H. (2022). "Stock Price Prediction Model Based on Dual Attention
    and TCN." SSRN 4282842.

All variants are adapted for strict causal inference (no look-ahead bias).
See extras/FATCN_TAGRU_Spec.docx for the full design specification.

Architecture pipeline (full model):
------------------------------------
Input (T timesteps, 7 features)
    ↓
[FA]  FeatureAttention   — pointwise tanh+softmax over feature dim (causal by construction)
    ↓
[TCN] Causal Dilated TCN — dilations [1,2,4,8], causal padding, residual connections
    ↓
[GRU] Unidirectional GRU × 2 — return_sequences=True for the first, True for the second
    ↓
[TA]  LightweightAttention(causal=True) — pools GRU hidden states to a single vector
    ↓
Dense head → Output

Ablation variants (this file)
------------------------------
Variant  Name           FA   TCN  GRU  TA    Builder
  3      TCN-GRU        —    ✓    ✓    —     build_tcn_gru / build_tcn_gru_tunable
  4      TAGRU          —    —    ✓    ✓     build_tagru   / build_tagru_tunable
  5      FATCN          ✓    ✓    —    —     build_fatcn   / build_fatcn_tunable
  6      FATCN-GRU      ✓    ✓    ✓    —     build_fatcn_gru / build_fatcn_gru_tunable
  7      TCN-TAGRU      —    ✓    ✓    ✓     build_tcn_tagru / build_tcn_tagru_tunable
  8      FATCN-TAGRU    ✓    ✓    ✓    ✓     build_fatcn_tagru / build_fatcn_tagru_tunable

Variants 1 (GRU) and 2 (TCN) are in depthwise_separable_gru.py and tcn_dilated_conv.py.

Causality notes
---------------
- FeatureAttention: softmax over the FEATURE axis at each timestep — no cross-time
  dependency, inherently causal.
- TCN: causal padding (left-pad only) enforced via padding="causal".
- GRU: unidirectional only — bidirectional GRU would read future bars.
- LightweightAttention(causal=True): upper-triangular mask (dot) or lower-triangular
  masked per-row softmax (bahdanau); context = last timestep's output only.

Usage
-----
Simple (fixed hyperparameters):
    model = build_fatcn_tagru(sequence_length=60, num_features=7, num_classes=3)

Tunable (KerasTuner):
    def model_builder(hp):
        return build_fatcn_tagru_tunable(hp, num_features=7, num_classes=3,
                                         sequence_length=60)
    tuner = keras_tuner.BayesianOptimization(model_builder, objective="val_loss",
                                             max_trials=30)
"""

from keras import layers, models

from okmich_quant_neural_net.keras.layers.feature_attention import FeatureAttention
from okmich_quant_neural_net.keras.layers.light_weight_attention import LightweightAttention
from okmich_quant_neural_net.keras.layers.tcn import TCN
from okmich_quant_neural_net.keras.paper2keras.common import (
    TaskType,
    create_output_layer_and_loss,
    get_optimizer,
    get_model_name,
)


# ============================================================================
# Internal builder helpers
# ============================================================================

def _tcn_block(x, nb_filters, kernel_size, dilations, dropout_rate,
               nb_stacks=1, name="tcn"):
    """Apply a causal TCN and return sequences."""
    return TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,
        padding="causal",
        use_skip_connections=True,
        dropout_rate=dropout_rate,
        return_sequences=True,
        activation="relu",
        kernel_initializer="he_normal",
        name=name,
    )(x)


def _gru_block(x, units1, units2, dropout1, dropout2):
    """Two-layer unidirectional GRU; second layer returns sequences."""
    x = layers.GRU(units1, return_sequences=True, name="gru1")(x)
    x = layers.Dropout(dropout1, name="dropout_gru1")(x)
    x = layers.GRU(units2, return_sequences=True, name="gru2")(x)
    x = layers.Dropout(dropout2, name="dropout_gru2")(x)
    return x


def _temporal_attention(x, attn_units, heads, attn_type, dropout):
    """LightweightAttention pool + dropout → (B, attn_units)."""
    # Project GRU output to attn_units if sizes differ
    if x.shape[-1] != attn_units:
        x = layers.Dense(attn_units, use_bias=False, name="ta_proj")(x)
    attn = LightweightAttention(
        attn_type=attn_type, heads=heads, causal=True,
        return_attention_scores=False, name="temporal_attention"
    )
    context = attn(x)  # (B, attn_units)
    return layers.Dropout(dropout, name="dropout_ta")(context)


def _dense_head(x, dense_units, dense_dropout):
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    return layers.Dropout(dense_dropout, name="dropout_dense")(x)


def _compile(inputs, x, task_type, num_classes, model_name,
             optimizer_name, learning_rate):
    outputs, loss, output_metrics = create_output_layer_and_loss(x, task_type, num_classes)
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(optimizer=get_optimizer(optimizer_name, learning_rate),
                  loss=loss, metrics=output_metrics)
    return model


# ============================================================================
# Variant 3 — TCN-GRU
# ============================================================================

def build_tcn_gru(
        sequence_length=60,
        num_features=7,
        num_classes=3,
        task_type=TaskType.CLASSIFICATION,
        # TCN
        nb_filters=64,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        tcn_dropout=0.1,
        # GRU
        gru1_units=64,
        gru2_units=64,
        gru_dropout1=0.1,
        gru_dropout2=0.1,
        # Head
        dense_units=64,
        dense_dropout=0.2,
        # Training
        learning_rate=1e-3,
        optimizer_name="adamw",
):
    """TCN → GRU (no attention). Ablation variant 3."""
    inputs = layers.Input(shape=(sequence_length, num_features), name="input")
    x = _tcn_block(inputs, nb_filters, kernel_size, dilations, tcn_dropout)
    x = _gru_block(x, gru1_units, gru2_units, gru_dropout1, gru_dropout2)
    x = x[:, -1, :]  # last timestep
    x = _dense_head(x, dense_units, dense_dropout)
    return _compile(inputs, x, task_type, num_classes,
                    get_model_name("TCN_GRU", task_type), optimizer_name, learning_rate)


def build_tcn_gru_tunable(
        hp, num_features, num_classes,
        task_type=TaskType.CLASSIFICATION,
        sequence_length=None, max_sequence_length=100,
):
    """Tunable TCN-GRU (variant 3)."""
    if sequence_length is None:
        sequence_length = hp.Int("sequence_length", 32, max_sequence_length, step=16)

    nb_filters = hp.Choice("nb_filters", [32, 64, 128])
    kernel_size = hp.Choice("kernel_size", [3, 5])
    dilations = hp.Choice("dilations", ["1,2,4,8", "1,2,4,8,16"])
    dilations = tuple(int(d) for d in dilations.split(","))
    tcn_dropout = hp.Float("tcn_dropout", 0.0, 0.3, step=0.1)
    gru1_units = hp.Choice("gru1_units", [32, 64, 128])
    gru2_units = hp.Choice("gru2_units", [32, 64, 128])
    gru_dropout1 = hp.Float("gru_dropout1", 0.0, 0.3, step=0.1)
    gru_dropout2 = hp.Float("gru_dropout2", 0.0, 0.3, step=0.1)
    dense_units = hp.Choice("dense_units", [32, 64, 128])
    dense_dropout = hp.Float("dense_dropout", 0.1, 0.4, step=0.1)
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    return build_tcn_gru(
        sequence_length=sequence_length, num_features=num_features,
        num_classes=num_classes, task_type=task_type,
        nb_filters=nb_filters, kernel_size=kernel_size, dilations=dilations,
        tcn_dropout=tcn_dropout, gru1_units=gru1_units, gru2_units=gru2_units,
        gru_dropout1=gru_dropout1, gru_dropout2=gru_dropout2,
        dense_units=dense_units, dense_dropout=dense_dropout, learning_rate=lr,
    )


# ============================================================================
# Variant 4 — TAGRU  (Temporal Attention + GRU)
# ============================================================================

def build_tagru(
        sequence_length=60,
        num_features=7,
        num_classes=3,
        task_type=TaskType.CLASSIFICATION,
        # GRU
        gru1_units=64,
        gru2_units=64,
        gru_dropout1=0.1,
        gru_dropout2=0.1,
        # Temporal Attention
        attn_type="dot",
        attn_heads=4,
        attn_dropout=0.1,
        # Head
        dense_units=64,
        dense_dropout=0.2,
        # Training
        learning_rate=1e-3,
        optimizer_name="adamw",
):
    """GRU → Temporal Attention (causal). Ablation variant 4."""
    inputs = layers.Input(shape=(sequence_length, num_features), name="input")
    x = _gru_block(inputs, gru1_units, gru2_units, gru_dropout1, gru_dropout2)
    x = _temporal_attention(x, gru2_units, attn_heads, attn_type, attn_dropout)
    x = _dense_head(x, dense_units, dense_dropout)
    return _compile(inputs, x, task_type, num_classes,
                    get_model_name("TAGRU", task_type), optimizer_name, learning_rate)


def build_tagru_tunable(
        hp, num_features, num_classes,
        task_type=TaskType.CLASSIFICATION,
        sequence_length=None, max_sequence_length=100,
):
    """Tunable TAGRU (variant 4)."""
    if sequence_length is None:
        sequence_length = hp.Int("sequence_length", 32, max_sequence_length, step=16)

    gru1_units = hp.Choice("gru1_units", [32, 64, 128])
    gru2_units = hp.Choice("gru2_units", [32, 64, 128])
    gru_dropout1 = hp.Float("gru_dropout1", 0.0, 0.3, step=0.1)
    gru_dropout2 = hp.Float("gru_dropout2", 0.0, 0.3, step=0.1)
    attn_type = hp.Choice("attn_type", ["dot", "bahdanau", "decay"])
    attn_heads = hp.Choice("attn_heads", [2, 4, 8])
    attn_dropout = hp.Float("attn_dropout", 0.0, 0.3, step=0.1)
    dense_units = hp.Choice("dense_units", [32, 64, 128])
    dense_dropout = hp.Float("dense_dropout", 0.1, 0.4, step=0.1)
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    return build_tagru(
        sequence_length=sequence_length, num_features=num_features,
        num_classes=num_classes, task_type=task_type,
        gru1_units=gru1_units, gru2_units=gru2_units,
        gru_dropout1=gru_dropout1, gru_dropout2=gru_dropout2,
        attn_type=attn_type, attn_heads=attn_heads, attn_dropout=attn_dropout,
        dense_units=dense_units, dense_dropout=dense_dropout, learning_rate=lr,
    )


# ============================================================================
# Variant 5 — FATCN  (Feature Attention + TCN)
# ============================================================================

def build_fatcn(
        sequence_length=60,
        num_features=7,
        num_classes=3,
        task_type=TaskType.CLASSIFICATION,
        # TCN
        nb_filters=64,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        tcn_dropout=0.1,
        # Head
        dense_units=64,
        dense_dropout=0.2,
        # Training
        learning_rate=1e-3,
        optimizer_name="adamw",
):
    """Feature Attention → TCN (no GRU, no temporal attention). Ablation variant 5."""
    inputs = layers.Input(shape=(sequence_length, num_features), name="input")
    x = FeatureAttention(name="feature_attention")(inputs)
    x = _tcn_block(x, nb_filters, kernel_size, dilations, tcn_dropout)
    x = x[:, -1, :]  # last timestep (TCN is causal — this is the current bar)
    x = _dense_head(x, dense_units, dense_dropout)
    return _compile(inputs, x, task_type, num_classes,
                    get_model_name("FATCN", task_type), optimizer_name, learning_rate)


def build_fatcn_tunable(
        hp, num_features, num_classes,
        task_type=TaskType.CLASSIFICATION,
        sequence_length=None, max_sequence_length=100,
):
    """Tunable FATCN (variant 5)."""
    if sequence_length is None:
        sequence_length = hp.Int("sequence_length", 32, max_sequence_length, step=16)

    nb_filters = hp.Choice("nb_filters", [32, 64, 128])
    kernel_size = hp.Choice("kernel_size", [3, 5])
    dilations = hp.Choice("dilations", ["1,2,4,8", "1,2,4,8,16"])
    dilations = tuple(int(d) for d in dilations.split(","))
    tcn_dropout = hp.Float("tcn_dropout", 0.0, 0.3, step=0.1)
    dense_units = hp.Choice("dense_units", [32, 64, 128])
    dense_dropout = hp.Float("dense_dropout", 0.1, 0.4, step=0.1)
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    return build_fatcn(
        sequence_length=sequence_length, num_features=num_features,
        num_classes=num_classes, task_type=task_type,
        nb_filters=nb_filters, kernel_size=kernel_size, dilations=dilations,
        tcn_dropout=tcn_dropout, dense_units=dense_units,
        dense_dropout=dense_dropout, learning_rate=lr,
    )


# ============================================================================
# Variant 6 — FATCN-GRU  (Feature Attention + TCN + GRU)
# ============================================================================

def build_fatcn_gru(
        sequence_length=60,
        num_features=7,
        num_classes=3,
        task_type=TaskType.CLASSIFICATION,
        # TCN
        nb_filters=64,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        tcn_dropout=0.1,
        # GRU
        gru1_units=64,
        gru2_units=64,
        gru_dropout1=0.1,
        gru_dropout2=0.1,
        # Head
        dense_units=64,
        dense_dropout=0.2,
        # Training
        learning_rate=1e-3,
        optimizer_name="adamw",
):
    """Feature Attention → TCN → GRU (no temporal attention). Ablation variant 6."""
    inputs = layers.Input(shape=(sequence_length, num_features), name="input")
    x = FeatureAttention(name="feature_attention")(inputs)
    x = _tcn_block(x, nb_filters, kernel_size, dilations, tcn_dropout)
    x = _gru_block(x, gru1_units, gru2_units, gru_dropout1, gru_dropout2)
    x = x[:, -1, :]
    x = _dense_head(x, dense_units, dense_dropout)
    return _compile(inputs, x, task_type, num_classes,
                    get_model_name("FATCN_GRU", task_type), optimizer_name, learning_rate)


def build_fatcn_gru_tunable(
        hp, num_features, num_classes,
        task_type=TaskType.CLASSIFICATION,
        sequence_length=None, max_sequence_length=100,
):
    """Tunable FATCN-GRU (variant 6)."""
    if sequence_length is None:
        sequence_length = hp.Int("sequence_length", 32, max_sequence_length, step=16)

    nb_filters = hp.Choice("nb_filters", [32, 64, 128])
    kernel_size = hp.Choice("kernel_size", [3, 5])
    dilations = hp.Choice("dilations", ["1,2,4,8", "1,2,4,8,16"])
    dilations = tuple(int(d) for d in dilations.split(","))
    tcn_dropout = hp.Float("tcn_dropout", 0.0, 0.3, step=0.1)
    gru1_units = hp.Choice("gru1_units", [32, 64, 128])
    gru2_units = hp.Choice("gru2_units", [32, 64, 128])
    gru_dropout1 = hp.Float("gru_dropout1", 0.0, 0.3, step=0.1)
    gru_dropout2 = hp.Float("gru_dropout2", 0.0, 0.3, step=0.1)
    dense_units = hp.Choice("dense_units", [32, 64, 128])
    dense_dropout = hp.Float("dense_dropout", 0.1, 0.4, step=0.1)
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    return build_fatcn_gru(
        sequence_length=sequence_length, num_features=num_features,
        num_classes=num_classes, task_type=task_type,
        nb_filters=nb_filters, kernel_size=kernel_size, dilations=dilations,
        tcn_dropout=tcn_dropout, gru1_units=gru1_units, gru2_units=gru2_units,
        gru_dropout1=gru_dropout1, gru_dropout2=gru_dropout2,
        dense_units=dense_units, dense_dropout=dense_dropout, learning_rate=lr,
    )


# ============================================================================
# Variant 7 — TCN-TAGRU  (TCN + Temporal Attention + GRU)
# ============================================================================

def build_tcn_tagru(
        sequence_length=60,
        num_features=7,
        num_classes=3,
        task_type=TaskType.CLASSIFICATION,
        # TCN
        nb_filters=64,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        tcn_dropout=0.1,
        # GRU
        gru1_units=64,
        gru2_units=64,
        gru_dropout1=0.1,
        gru_dropout2=0.1,
        # Temporal Attention
        attn_type="dot",
        attn_heads=4,
        attn_dropout=0.1,
        # Head
        dense_units=64,
        dense_dropout=0.2,
        # Training
        learning_rate=1e-3,
        optimizer_name="adamw",
):
    """TCN → GRU → Temporal Attention (causal). Ablation variant 7."""
    inputs = layers.Input(shape=(sequence_length, num_features), name="input")
    x = _tcn_block(inputs, nb_filters, kernel_size, dilations, tcn_dropout)
    x = _gru_block(x, gru1_units, gru2_units, gru_dropout1, gru_dropout2)
    x = _temporal_attention(x, gru2_units, attn_heads, attn_type, attn_dropout)
    x = _dense_head(x, dense_units, dense_dropout)
    return _compile(inputs, x, task_type, num_classes,
                    get_model_name("TCN_TAGRU", task_type), optimizer_name, learning_rate)


def build_tcn_tagru_tunable(
        hp, num_features, num_classes,
        task_type=TaskType.CLASSIFICATION,
        sequence_length=None, max_sequence_length=100,
):
    """Tunable TCN-TAGRU (variant 7)."""
    if sequence_length is None:
        sequence_length = hp.Int("sequence_length", 32, max_sequence_length, step=16)

    nb_filters = hp.Choice("nb_filters", [32, 64, 128])
    kernel_size = hp.Choice("kernel_size", [3, 5])
    dilations = hp.Choice("dilations", ["1,2,4,8", "1,2,4,8,16"])
    dilations = tuple(int(d) for d in dilations.split(","))
    tcn_dropout = hp.Float("tcn_dropout", 0.0, 0.3, step=0.1)
    gru1_units = hp.Choice("gru1_units", [32, 64, 128])
    gru2_units = hp.Choice("gru2_units", [32, 64, 128])
    gru_dropout1 = hp.Float("gru_dropout1", 0.0, 0.3, step=0.1)
    gru_dropout2 = hp.Float("gru_dropout2", 0.0, 0.3, step=0.1)
    attn_type = hp.Choice("attn_type", ["dot", "bahdanau", "decay"])
    attn_heads = hp.Choice("attn_heads", [2, 4, 8])
    attn_dropout = hp.Float("attn_dropout", 0.0, 0.3, step=0.1)
    dense_units = hp.Choice("dense_units", [32, 64, 128])
    dense_dropout = hp.Float("dense_dropout", 0.1, 0.4, step=0.1)
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    return build_tcn_tagru(
        sequence_length=sequence_length, num_features=num_features,
        num_classes=num_classes, task_type=task_type,
        nb_filters=nb_filters, kernel_size=kernel_size, dilations=dilations,
        tcn_dropout=tcn_dropout, gru1_units=gru1_units, gru2_units=gru2_units,
        gru_dropout1=gru_dropout1, gru_dropout2=gru_dropout2,
        attn_type=attn_type, attn_heads=attn_heads, attn_dropout=attn_dropout,
        dense_units=dense_units, dense_dropout=dense_dropout, learning_rate=lr,
    )


# ============================================================================
# Variant 8 — FATCN-TAGRU  (full model)
# ============================================================================

def build_fatcn_tagru(
        sequence_length=60,
        num_features=7,
        num_classes=3,
        task_type=TaskType.CLASSIFICATION,
        # TCN
        nb_filters=64,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        tcn_dropout=0.1,
        # GRU
        gru1_units=64,
        gru2_units=64,
        gru_dropout1=0.1,
        gru_dropout2=0.1,
        # Temporal Attention
        attn_type="dot",
        attn_heads=4,
        attn_dropout=0.1,
        # Head
        dense_units=64,
        dense_dropout=0.2,
        # Training
        learning_rate=1e-3,
        optimizer_name="adamw",
):
    """Full FATCN-TAGRU model: FA → TCN → GRU → TA (causal). Ablation variant 8.

    Args:
        sequence_length: Input window length T (default 60 bars).
        num_features:    Features per bar F (paper uses 7).
        num_classes:     Output classes (3 for Long/Flat/Short).
        nb_filters:      TCN filter count (default 64).
        kernel_size:     TCN kernel size (default 3).
        dilations:       TCN dilation schedule (default (1,2,4,8)).
        tcn_dropout:     Dropout within TCN residual blocks (default 0.1).
        gru1_units:      First GRU hidden size (default 64).
        gru2_units:      Second GRU hidden size (default 64).
        gru_dropout1/2:  Dropout after each GRU layer (default 0.1).
        attn_type:       "dot", "bahdanau", or "decay" (default "dot").
        attn_heads:      Attention heads for dot type (default 4).
        attn_dropout:    Dropout after attention pooling (default 0.1).
        dense_units:     Classification head hidden size (default 64).
        dense_dropout:   Dropout in head (default 0.2).
        learning_rate:   Optimizer LR (default 1e-3).
        optimizer_name:  "adam", "adamw", or "rmsprop" (default "adamw").

    Returns:
        Compiled Keras model.
    """
    inputs = layers.Input(shape=(sequence_length, num_features), name="input")

    # Stage 1 — Feature Attention
    x = FeatureAttention(name="feature_attention")(inputs)

    # Stage 2 — Causal TCN
    x = _tcn_block(x, nb_filters, kernel_size, dilations, tcn_dropout)

    # Stage 3 — Unidirectional GRU
    x = _gru_block(x, gru1_units, gru2_units, gru_dropout1, gru_dropout2)

    # Stage 4 — Causal Temporal Attention
    x = _temporal_attention(x, gru2_units, attn_heads, attn_type, attn_dropout)

    # Stage 5 — Dense head
    x = _dense_head(x, dense_units, dense_dropout)

    return _compile(inputs, x, task_type, num_classes,
                    get_model_name("FATCN_TAGRU", task_type), optimizer_name, learning_rate)


def build_fatcn_tagru_tunable(hp, num_features, num_classes, task_type=TaskType.CLASSIFICATION, sequence_length=None,
                              max_sequence_length=100):
    """Tunable full FATCN-TAGRU (variant 8).

    Args:
        hp:                  KerasTuner HyperParameters object.
        num_features:        Fixed number of features per timestep.
        num_classes:         Fixed number of output classes.
        task_type:           TaskType enum (default CLASSIFICATION).
        sequence_length:     If None, tunes T; if set, fixes T (pre-created sequences).
        max_sequence_length: Upper bound when tuning T (default 100).

    Returns:
        Compiled Keras model with tunable hyperparameters.

    Example:
        >>> import keras_tuner
        >>> def model_builder(hp):
        ...     return build_fatcn_tagru_tunable(
        ...         hp, num_features=7, num_classes=3, sequence_length=60
        ...     )
        >>> tuner = keras_tuner.BayesianOptimization(
        ...     model_builder, objective="val_loss", max_trials=30
        ... )
    """
    if sequence_length is None:
        sequence_length = hp.Int("sequence_length", 32, max_sequence_length, step=16)

    nb_filters = hp.Choice("nb_filters", [32, 64, 128])
    kernel_size = hp.Choice("kernel_size", [3, 5])
    dilations = hp.Choice("dilations", ["1,2,4,8", "1,2,4,8,16"])
    dilations = tuple(int(d) for d in dilations.split(","))
    tcn_dropout = hp.Float("tcn_dropout", 0.0, 0.3, step=0.1)
    gru1_units = hp.Choice("gru1_units", [32, 64, 128])
    gru2_units = hp.Choice("gru2_units", [32, 64, 128])
    gru_dropout1 = hp.Float("gru_dropout1", 0.0, 0.3, step=0.1)
    gru_dropout2 = hp.Float("gru_dropout2", 0.0, 0.3, step=0.1)
    attn_type = hp.Choice("attn_type", ["dot", "bahdanau", "decay"])
    attn_heads = hp.Choice("attn_heads", [2, 4, 8])
    attn_dropout = hp.Float("attn_dropout", 0.0, 0.3, step=0.1)
    dense_units = hp.Choice("dense_units", [32, 64, 128])
    dense_dropout = hp.Float("dense_dropout", 0.1, 0.4, step=0.1)
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    return build_fatcn_tagru(
        sequence_length=sequence_length, num_features=num_features,
        num_classes=num_classes, task_type=task_type,
        nb_filters=nb_filters, kernel_size=kernel_size, dilations=dilations,
        tcn_dropout=tcn_dropout, gru1_units=gru1_units, gru2_units=gru2_units,
        gru_dropout1=gru_dropout1, gru_dropout2=gru_dropout2,
        attn_type=attn_type, attn_heads=attn_heads, attn_dropout=attn_dropout,
        dense_units=dense_units, dense_dropout=dense_dropout, learning_rate=lr,
    )
