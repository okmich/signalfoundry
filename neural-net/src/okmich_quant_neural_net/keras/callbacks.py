"""
Custom Keras Callbacks for Regime Classification Training.

This module provides training utilities designed around regime-aware metrics.
Use them together with the regime metrics in :mod:`okmich_quant_neural_net.keras.metrics`.

Diagnostics:
    - diagnose_model: Interpret a combination of metric values as a failure mode.
    - DiagnosticTracker: Accumulates per-epoch diagnoses and detects stalled modes.

Callbacks:
    - RegimeEarlyStopping: Early stopping with hard/soft failure gate logic,
      trajectory logging, and integrated DiagnosticTracker.

Typical usage::

    from okmich_quant_neural_net.keras.metrics import (
        CausalRegimeAccuracy,
        RegimeTransitionRecall,
        RegimeTransitionPrecision,
    )
    from okmich_quant_neural_net.keras.callbacks import RegimeEarlyStopping

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            CausalRegimeAccuracy(window_size=5),
            RegimeTransitionRecall(lag_tolerance=2),
            RegimeTransitionPrecision(lag_tolerance=2),
        ],
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        callbacks=[RegimeEarlyStopping(patience=5)],
    )
"""

from collections import deque

import tensorflow as tf


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diagnose_model(causal, trans, precision=None,
                   train_causal=None, train_trans=None):
    """
    Interpret the three regime metric values as named failure modes.

    Returns a list of all matching diagnoses (multi-label).  Optionally
    compares train vs validation metrics to detect overfitting.

    Args:
        causal: Validation causal regime accuracy.
        trans: Validation regime transition recall.
        precision: Validation regime transition precision (optional).
        train_causal: Training causal accuracy (optional, for overfit detection).
        train_trans: Training transition recall (optional).

    Returns:
        List[str]: One or more diagnosis strings.
    """
    findings = []

    # Core failure modes — derived from the 3 regime metrics only
    if causal < 0.5 and trans > 0.5:
        findings.append("FLICKERING: Model switches too often (1-bar regimes)")
    if causal > 0.75 and trans < 0.2:
        findings.append("STUCK: Model never transitions (majority class predictor)")
    if causal > 0.65 and 0.2 <= trans < 0.5:
        findings.append("LAGGY: Persistent but misses transitions (increase lag_tolerance?)")
    if causal < 0.45 and trans < 0.3:
        findings.append("RANDOM: Model not learning (check features/labels)")
    if causal > 0.7 and trans > 0.5:
        findings.append("HEALTHY: Good regime detection")

    # Precision-based diagnoses
    if precision is not None:
        if trans > 0.7 and precision < 0.3:
            findings.append(
                "HALLUCINATING: Predicts transitions everywhere "
                "(high recall, low precision)"
            )
        if precision > 0.8 and trans < 0.3:
            findings.append(
                "CONSERVATIVE: Few predictions but mostly correct "
                "(high precision, low recall)"
            )

    # Overfit detection
    if train_causal is not None:
        causal_gap = train_causal - causal
        if causal_gap > 0.15:
            findings.append(f"OVERFITTING: Train-val causal gap = {causal_gap:.2f}")
    if train_trans is not None:
        trans_gap = train_trans - trans
        if trans_gap > 0.2:
            findings.append(
                f"OVERFIT_TRANSITIONS: Train-val transition gap = {trans_gap:.2f}"
            )

    if not findings:
        findings.append(
            f"UNCLEAR (causal={causal:.2f}, trans={trans:.2f}): Review architecture"
        )

    return findings


class DiagnosticTracker:
    """
    Accumulates per-epoch regime diagnostics to detect persistent failure modes
    and training trajectory patterns.

    Args:
        max_history: Maximum number of epochs to retain (default 200).
    """

    def __init__(self, max_history=200):
        self.history = deque(maxlen=max_history)

    def log(self, epoch, causal, trans, precision=None,
            train_causal=None, train_trans=None):
        """Run diagnose_model for the current epoch and store the result."""
        findings = diagnose_model(
            causal, trans, precision,
            train_causal, train_trans,
        )
        self.history.append({
            'epoch': epoch,
            'findings': findings,
            'metrics': {
                'causal': causal, 'trans': trans, 'precision': precision,
            },
        })
        return findings

    def trend(self, last_n=5):
        """
        Report whether the model is stable, stalled, or still evolving.

        Returns:
            str: One of 'INSUFFICIENT_DATA', 'STABLE_HEALTHY for N epochs',
                 'STALLED in [...] for N epochs', or 'EVOLVING'.
        """
        if len(self.history) < last_n:
            return "INSUFFICIENT_DATA"

        recent = [set(h['findings']) for h in list(self.history)[-last_n:]]
        if all(r == recent[0] for r in recent):
            modes = list(recent[0])
            if any("HEALTHY" in m for m in modes):
                return f"STABLE_HEALTHY for {last_n} epochs"
            return f"STALLED in {modes} for {last_n} epochs"
        return "EVOLVING"

    def summary(self):
        """Print a compact summary of the training trajectory (last 10 epochs)."""
        if not self.history:
            print("No history recorded.")
            return

        print(f"\n{'=' * 60}")
        print("DIAGNOSTIC SUMMARY")
        print(f"{'=' * 60}")
        print(f"Epochs tracked: {len(self.history)}")
        print(f"Current trend: {self.trend()}")
        print(f"\nRecent history:")
        for entry in list(self.history)[-10:]:
            modes = ", ".join(entry['findings'])
            m = entry['metrics']
            prec_str = f" prec={m['precision']:.3f}" if m['precision'] is not None else ""
            print(
                f"  Epoch {entry['epoch']:3d}: "
                f"causal={m['causal']:.3f} trans={m['trans']:.3f}{prec_str} -> {modes}"
            )


# ---------------------------------------------------------------------------
# Early Stopping Callback
# ---------------------------------------------------------------------------

class RegimeEarlyStopping(tf.keras.callbacks.Callback):
    """
    Early stopping with explicit failure-mode detection for regime models.

    Gate logic
    ----------
    A model state at epoch t is considered *valid* only when ALL of the
    following hold simultaneously:

    * ``causal_regime_accuracy >= min_causal``
    * ``regime_transition_recall >= min_trans``
    * No *hard* failure mode is active

    Hard failure modes (invalidate the score)
    ------------------------------------------
    * FLICKERING  — low causal but high transition recall (flips every bar)
    * STUCK       — high causal but near-zero transition recall (majority predictor)
    * RANDOM      — both causal and transition recall are very low

    Soft warnings (logged but do not invalidate)
    ---------------------------------------------
    * LAGGY — model transitions but below the min_trans threshold

    Scoring (only when gates pass)
    ------------------------------
    ``score = 0.4 * causal + 0.6 * trans``

    Args:
        patience: Epochs without improvement before stopping (default 5).
        min_causal: Minimum causal regime accuracy gate (default 0.65).
        min_trans: Minimum transition recall gate (default 0.50).
        laggy_upper: Upper bound on transition recall to classify as LAGGY
            (default 0.75).
        warmup_epochs: Epochs to skip before applying early stopping (default 15).
        max_no_valid_epochs: Hard stop if no valid model found this many epochs
            after warmup (default 30).
        restore_best_weights: Restore weights from best valid epoch on stop
            (default True).
    """

    def __init__(
            self,
            patience=5,
            min_causal=0.65,
            min_trans=0.50,
            laggy_upper=0.75,
            warmup_epochs=15,
            max_no_valid_epochs=30,
            restore_best_weights=True,
    ):
        super().__init__()
        self.patience = patience
        self.min_causal = min_causal
        self.min_trans = min_trans
        self.laggy_upper = laggy_upper
        self.warmup_epochs = warmup_epochs
        self.max_no_valid_epochs = max_no_valid_epochs
        self.restore_best_weights = restore_best_weights

        # Runtime state (reset on_train_begin is handled by Keras)
        self.best_valid_score = -float('inf')
        self.best_weights = None
        self.best_epoch = -1
        self.wait = 0
        self._stopped_early = False

        self.failure_history = deque(maxlen=patience * 2)
        self.current_failure_modes = set()
        self.tracker = DiagnosticTracker()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        causal = logs.get('val_causal_regime_accuracy', 0.0)
        trans = logs.get('val_regime_transition_recall', 0.0)
        prec = logs.get('val_regime_transition_precision')

        # --- Failure mode detection ---
        hard_failures = set()
        warnings = set()

        if causal < 0.5 and trans > 0.5:
            hard_failures.add("FLICKERING")
        if causal > 0.75 and trans < self.min_trans:
            hard_failures.add("STUCK")
        if causal < 0.45 and trans < 0.3:
            hard_failures.add("RANDOM")
        if causal > 0.6 and self.min_trans <= trans < self.laggy_upper:
            warnings.add("LAGGY")

        all_status = hard_failures | warnings

        # --- Gate check and scoring ---
        gates_pass = (
                causal >= self.min_causal
                and trans >= self.min_trans
                and not hard_failures
        )
        score = (0.4 * causal + 0.6 * trans) if gates_pass else -float('inf')

        # Build status label — only show HEALTHY when gates actually pass
        if gates_pass:
            status_str = "HEALTHY"
        elif all_status:
            status_str = "|".join(sorted(all_status))
        else:
            gate_misses = []
            if causal < self.min_causal:
                gate_misses.append(f"causal<{self.min_causal}")
            if trans < self.min_trans:
                gate_misses.append(f"trans<{self.min_trans}")
            status_str = "BELOW_GATES:" + ",".join(gate_misses)

        # --- Integrated diagnostics ---
        self.tracker.log(epoch, causal, trans, prec)

        # --- Update best model ---
        if gates_pass and score > self.best_valid_score:
            self.best_valid_score = score
            self.best_epoch = epoch
            self.wait = 0
            self.current_failure_modes.clear()
            if self.restore_best_weights:
                self.best_weights = [w.numpy() for w in self.model.weights]
        else:
            self.wait += 1
            if all_status:
                self.current_failure_modes.update(all_status)
                self.failure_history.append((epoch, sorted(all_status)))

        # --- Per-epoch log line ---
        warn_tag = f" [WARN: {','.join(sorted(warnings))}]" if warnings else ""
        prec_str = f" Prec={prec:.3f}" if prec is not None else ""
        print(
            f"Epoch {epoch:3d}: Causal={causal:.3f} Trans={trans:.3f}{prec_str} "
            f"[{status_str:22s}] Score={score:+.3f} Wait={self.wait}/{self.patience}{warn_tag}"
        )

        # --- Early stopping checks ---
        if epoch < self.warmup_epochs:
            return

        if (self.best_epoch < 0
                and epoch >= self.warmup_epochs + self.max_no_valid_epochs):
            print(
                f"\nHARD STOP at epoch {epoch}: No valid model found in "
                f"{self.max_no_valid_epochs} epochs after warmup. "
                f"Architecture or data likely needs rework."
            )
            self._stopped_early = True
            self.model.stop_training = True
            self.tracker.summary()
            return

        if self.wait >= self.patience:
            self._stop_training(epoch, causal, trans)

    def _stop_training(self, epoch, causal, trans):
        self._stopped_early = True

        print(f"\n{'=' * 60}")
        print(f"EARLY STOPPING at epoch {epoch}")
        print(f"{'=' * 60}")
        print(
            f"Best valid model: epoch {self.best_epoch} "
            f"(score={self.best_valid_score:.3f})"
        )
        print(f"Final metrics: Causal={causal:.3f} Trans={trans:.3f}")
        print(f"\nFailure trajectory (last {len(self.failure_history)} entries):")
        for e, modes in self.failure_history:
            print(f"  Epoch {e:3d}: {'|'.join(modes)}")
        print(
            f"\nAccumulated failure modes: "
            f"{'|'.join(sorted(self.current_failure_modes)) or 'None'}"
        )
        print(f"Trend: {self.tracker.trend()}")

        if self.restore_best_weights and self.best_weights is not None:
            print(f"\nRestoring weights from epoch {self.best_epoch}")
            for w, bw in zip(self.model.weights, self.best_weights):
                w.assign(bw)
        else:
            print("\nWARNING: No valid checkpoint found. Model may be in degenerate state.")

        self.model.stop_training = True

    def on_train_end(self, logs=None):
        if not self._stopped_early:
            print(f"\n{'=' * 60}")
            print("TRAINING COMPLETED (all epochs)")
            print(f"{'=' * 60}")
            print(
                f"Best valid model: epoch {self.best_epoch} "
                f"(score={self.best_valid_score:.3f})"
            )
            if self.restore_best_weights and self.best_weights is not None:
                print("Restoring best weights...")
                for w, bw in zip(self.model.weights, self.best_weights):
                    w.assign(bw)
            else:
                print("WARNING: No valid checkpoint was ever recorded.")
            self.tracker.summary()
