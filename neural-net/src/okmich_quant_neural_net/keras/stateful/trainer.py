"""
StatefulTrainer — Correct training loop for stateful RNN models.

The core constraint of stateful training
-----------------------------------------
A stateful RNN carries its hidden state forward from one batch to the next.
This is only meaningful if sample *i* in batch *n+1* is the direct temporal
continuation of sample *i* in batch *n*.  Two consequences:

1. The time series must be split into **parallel lanes** before batching.
   Each lane is a contiguous, non-overlapping slice of the series.  All lanes
   are advanced together, one chunk at a time.

2. State is reset **between epochs**, never between chunks within an epoch.
   Resetting between chunks throws away exactly the context that stateful
   training is meant to preserve.

Lane layout (batch_size=3, chunk_size=4, T=24):

    Lane 0:  t[ 0.. 3]  t[ 4.. 7]  t[ 8..11]   ... (8 chunks)
    Lane 1:  t[ 8..11]  t[12..15]  t[16..19]
    Lane 2:  t[16..19]  t[20..23]  ...

Each training step feeds one (batch_size, chunk_size, features) block and
the model state at the end of that block seeds the next.

Validation
----------
Before the validation pass the training state is discarded (reset), the
val data is processed sequentially in the same lane structure, then state
is reset again before the next training epoch begins.  Because training
state is always reset at epoch start this is safe and correct.

Label modes
-----------
``y_mode='last'``  — label at the last timestep of each chunk.
    Use with models whose second RNN layer has ``return_sequences=False``.
    Output shape per chunk: (batch_size,) or (batch_size, num_classes).

``y_mode='all'``   — label at every timestep in the chunk.
    Use with seq2seq models (``return_sequences=True``, TimeDistributed head).
    Output shape per chunk: (batch_size, chunk_size) for integer labels.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reset_layer_state(layer) -> None:
    """
    Reset state on a single stateful layer with Keras-version compatibility.

    Keras 3 and TF 2.x expose ``reset_states()`` (plural) on recurrent layers.
    Some custom or third-party layers may only expose ``reset_state()``
    (singular).  This helper tries both names, warning once if neither exists.
    """
    if hasattr(layer, 'reset_states'):
        layer.reset_states()
    elif hasattr(layer, 'reset_state'):
        layer.reset_state()
    else:
        logger.warning(
            f"Layer '{layer.name}' has no reset_states() or reset_state() method. "
            "State was not reset — this may cause incorrect training behaviour."
        )


# ---------------------------------------------------------------------------
# Lane builder — standalone utility, usable outside StatefulTrainer
# ---------------------------------------------------------------------------

def build_stateful_lanes(X: np.ndarray, y: np.ndarray, batch_size: int, chunk_size: int,
                         y_mode: str = 'last') -> tuple[np.ndarray, np.ndarray]:
    """
    Slice a flat time series into ordered, non-overlapping chunks laid out across ``batch_size`` parallel lanes.

    Parameters
    ----------
    X : (T, num_features)  — flat feature array (NOT pre-sequenced)
    y : (T,) or (T, num_classes)  — per-timestep labels
    batch_size : int
        Number of parallel lanes.  Must match the ``batch_size`` the model was built with (encoded in ``batch_shape``).
    chunk_size : int
        Number of timesteps per training chunk (the "sequence length" for the training loop).  Smaller values give more
        gradient updates per epoch; larger values preserve longer context within a single forward pass.
    y_mode : {'last', 'all'}
        'last' — only the final timestep label of each chunk is returned. Use with ``return_sequences=False`` models.
        'all'  — all timestep labels are returned. Use with seq2seq (``return_sequences=True``) models.

    Returns
    -------
    X_chunks : (n_chunks, batch_size, chunk_size, num_features)
    y_chunks : 'last' → (n_chunks, batch_size)
               'all'  → (n_chunks, batch_size, chunk_size)

    Notes
    -----
    The leading ``T`` timesteps are truncated to the largest multiple of ``batch_size * chunk_size``.
    Any trailing timesteps are silently discarded.  Log a warning if the discarded fraction exceeds 5 %.
    """
    if X.ndim != 2:
        raise ValueError(
            f"X must be a flat time series of shape (T, num_features), got {X.shape}. "
            "Do NOT pre-sequence the data before passing to StatefulTrainer."
        )
    if y_mode not in ('last', 'all'):
        raise ValueError(f"y_mode must be 'last' or 'all', got '{y_mode}'")

    T = len(X)
    block = batch_size * chunk_size
    usable = (T // block) * block

    if usable == 0:
        raise ValueError(
            f"Time series too short: T={T} cannot fit even one block of "
            f"batch_size={batch_size} × chunk_size={chunk_size}={block}."
        )

    discarded = T - usable
    if discarded / T > 0.05:
        logger.warning(
            f"build_stateful_lanes: discarding {discarded}/{T} timesteps "
            f"({100 * discarded / T:.1f}%) to align to batch_size×chunk_size={block}. "
            "Consider adjusting chunk_size."
        )

    X = X[:usable]  # (usable, features)
    y = y[:usable]

    lane_len = usable // batch_size  # timesteps per lane
    n_chunks = lane_len // chunk_size

    # Reshape to (batch_size, n_chunks, chunk_size, features)
    nf = X.shape[1]
    X_lanes = X.reshape(batch_size, n_chunks, chunk_size, nf)
    # Transpose → (n_chunks, batch_size, chunk_size, features)
    X_chunks = X_lanes.transpose(1, 0, 2, 3)

    if y.ndim == 1:
        y_lanes = y.reshape(batch_size, n_chunks, chunk_size)
    else:
        y_lanes = y.reshape(batch_size, n_chunks, chunk_size, y.shape[1])

    if y_mode == 'last':
        # Take label at the last timestep of each chunk.
        # y.ndim==1: y_lanes (batch_size, n_chunks, chunk_size)         → y_last (batch_size, n_chunks)
        # y.ndim==2: y_lanes (batch_size, n_chunks, chunk_size, n_cls)  → y_last (batch_size, n_chunks, n_cls)
        y_last = y_lanes[:, :, -1]
        if y.ndim == 1:
            y_chunks = y_last.transpose(1, 0)  # (n_chunks, batch_size)
        else:
            y_chunks = y_last.transpose(1, 0, 2)  # (n_chunks, batch_size, n_cls)
    else:
        if y.ndim == 1:
            y_chunks = y_lanes.transpose(1, 0, 2)  # (n_chunks, batch_size, chunk_size)
        else:
            y_chunks = y_lanes.transpose(1, 0, 2, 3)  # (n_chunks, batch_size, chunk_size, n_cls)

    return X_chunks, y_chunks


# ---------------------------------------------------------------------------
# StatefulTrainer
# ---------------------------------------------------------------------------

class StatefulTrainer:
    """
    Correct training loop for stateful RNN models.

    Stateful models carry hidden state between batches, so data ordering and state reset timing are not handled by
    ``model.fit()``.  This class owns that responsibility:

    * Splits the flat time series into ordered parallel lanes.
    * Calls ``train_on_batch`` for each chunk in temporal order.
    * Resets state **between epochs** (not between chunks).
    * Runs a full sequential validation pass with its own state reset.
    * Implements patience-based early stopping with optional weight restore.

    Parameters
    ----------
    model :
        A compiled Keras model built with ``stateful=True`` and a fixed ``batch_shape`` matching ``batch_size``.
    chunk_size : int
        Timesteps per training chunk.  Equivalent to ``sequence_length`` in stateless training.  Typical range: 32–128.
    batch_size : int
        Number of parallel lanes.  Must equal the ``batch_size`` encoded in the model's ``batch_shape``.
        Default 1 (required for the existing ``build_stateful_rnn`` factory which always uses ``batch_size=1``).
    y_mode : {'last', 'all'}
        Label selection per chunk — see ``build_stateful_lanes`` docstring.
    epochs : int
        Maximum training epochs.
    patience : int
        Epochs without validation loss improvement before early stopping. Ignored when no validation data is provided.
    min_delta : float
        Minimum absolute improvement in validation loss to count as progress.
    restore_best_weights : bool
        If True, load weights from the epoch with lowest validation loss on stop (early or full).
    verbose : int
        0 = silent, 1 = one line per epoch.
    """

    def __init__(self, model, chunk_size: int, batch_size: int = 1, y_mode: str = 'last', epochs: int = 50,
                 patience: int = 10, min_delta: float = 1e-4, restore_best_weights: bool = True, verbose: int = 1):
        self.model = model
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.y_mode = y_mode
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        # Collect all stateful layers for state reset
        # In Keras 3, Functional models don't have reset_states() directly
        self._stateful_layers = [
            layer for layer in model.layers
            if hasattr(layer, 'stateful') and getattr(layer, 'stateful', False)
        ]
        if not self._stateful_layers:
            logger.warning("No stateful layers found in model. Ensure model was built with stateful=True.")

        # State set during fit()
        self.history: dict[str, list] = defaultdict(list)
        self.best_epoch: int = -1
        self._best_weights: Optional[list] = None
        self._best_val_loss: float = float('inf')
        self._stopped_early: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> dict:
        """
        Run the stateful training loop.

        Parameters
        ----------
        X_train : (T_train, num_features)
            Flat, chronologically ordered training features.
        y_train : (T_train,) or (T_train, num_classes)
            Per-timestep training labels.
        X_val : (T_val, num_features), optional
        y_val : (T_val,) or (T_val, num_classes), optional
            Validation data.  Early stopping requires validation data.

        Returns
        -------
        dict  — training history keyed by metric name, e.g.
            {'loss': [...], 'val_loss': [...], 'causal_regime_accuracy': [...]}
        """
        self.history = defaultdict(list)
        self._best_val_loss = float('inf')
        self._best_weights = None
        self._stopped_early = False
        self.best_epoch = -1
        _patience_counter = 0

        X_tr_chunks, y_tr_chunks = build_stateful_lanes(
            X_train, y_train, self.batch_size, self.chunk_size, self.y_mode
        )
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_va_chunks, y_va_chunks = build_stateful_lanes(
                X_val, y_val, self.batch_size, self.chunk_size, self.y_mode
            )

        n_train_chunks = len(X_tr_chunks)

        for epoch in range(self.epochs):
            # ---- Training pass ----
            self._reset_states()
            train_sums: dict[str, float] = {}

            for X_chunk, y_chunk in zip(X_tr_chunks, y_tr_chunks):
                result = self.model.train_on_batch(
                    X_chunk, y_chunk, return_dict=True
                )
                for k, v in result.items():
                    train_sums[k] = train_sums.get(k, 0.0) + float(v)

            train_metrics = {k: v / n_train_chunks for k, v in train_sums.items()}
            for k, v in train_metrics.items():
                self.history[k].append(v)

            # ---- Validation pass ----
            if has_val:
                val_metrics = self._eval_pass(X_va_chunks, y_va_chunks)
                for k, v in val_metrics.items():
                    self.history[f'val_{k}'].append(v)
                val_loss = val_metrics['loss']
            else:
                val_loss = None

            # ---- Logging ----
            if self.verbose >= 1:
                self._log_epoch(epoch, train_metrics, val_metrics if has_val else None)

            # ---- Early stopping ----
            if has_val:
                if val_loss < self._best_val_loss - self.min_delta:
                    self._best_val_loss = val_loss
                    self.best_epoch = epoch
                    _patience_counter = 0
                    if self.restore_best_weights:
                        self._best_weights = [
                            w.numpy().copy() for w in self.model.weights
                        ]
                else:
                    _patience_counter += 1

                if _patience_counter >= self.patience:
                    if self.verbose >= 1:
                        print(
                            f"\nEarly stopping at epoch {epoch}. "
                            f"Best epoch: {self.best_epoch} "
                            f"(val_loss={self._best_val_loss:.4f})"
                        )
                    self._stopped_early = True
                    break

        # ---- Restore best weights ----
        if self.restore_best_weights and self._best_weights is not None:
            for w, bw in zip(self.model.weights, self._best_weights):
                w.assign(bw)
            if self.verbose >= 1 and has_val:
                print(f"Restored best weights from epoch {self.best_epoch}.")

        # Reset state after training so the model is clean for inference
        self._reset_states()
        return dict(self.history)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Run a stateful evaluation pass and return mean metrics.

        Resets model state before and after the pass.

        Parameters
        ----------
        X : (T, num_features)
        y : (T,) or (T, num_classes)

        Returns
        -------
        dict  — mean metric values over all chunks, e.g.
            {'loss': 0.45, 'causal_regime_accuracy': 0.72, ...}
        """
        X_chunks, y_chunks = build_stateful_lanes(
            X, y, self.batch_size, self.chunk_size, self.y_mode
        )
        return self._eval_pass(X_chunks, y_chunks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_states(self) -> None:
        """Reset hidden states of all stateful layers (Keras-version compatible)."""
        for layer in self._stateful_layers:
            _reset_layer_state(layer)

    def _eval_pass(
            self,
            X_chunks: np.ndarray,
            y_chunks: np.ndarray,
    ) -> dict:
        """Sequential evaluation over pre-built chunks with state reset."""
        self._reset_states()
        sums: dict[str, float] = {}
        n = len(X_chunks)

        for X_chunk, y_chunk in zip(X_chunks, y_chunks):
            result = self.model.test_on_batch(
                X_chunk, y_chunk, return_dict=True
            )
            for k, v in result.items():
                sums[k] = sums.get(k, 0.0) + float(v)

        self._reset_states()
        return {k: v / n for k, v in sums.items()}

    @staticmethod
    def _log_epoch(epoch: int, train: dict, val: Optional[dict]) -> None:
        parts = [f"Epoch {epoch:4d}"]
        for k, v in train.items():
            parts.append(f"{k}={v:.4f}")
        if val:
            for k, v in val.items():
                parts.append(f"val_{k}={v:.4f}")
        print("  ".join(parts))
