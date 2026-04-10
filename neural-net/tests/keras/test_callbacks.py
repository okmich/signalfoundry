import numpy as np

from okmich_quant_neural_net.keras.callbacks import diagnose_model, DiagnosticTracker, RegimeEarlyStopping


# ---------------------------------------------------------------------------
# diagnose_model
# ---------------------------------------------------------------------------

class TestDiagnoseModel:

    def test_flickering(self):
        findings = diagnose_model(causal=0.40, trans=0.60)
        assert any("FLICKERING" in f for f in findings)

    def test_stuck(self):
        findings = diagnose_model(causal=0.80, trans=0.10)
        assert any("STUCK" in f for f in findings)

    def test_random(self):
        findings = diagnose_model(causal=0.40, trans=0.25)
        assert any("RANDOM" in f for f in findings)

    def test_healthy(self):
        findings = diagnose_model(causal=0.75, trans=0.60)
        assert any("HEALTHY" in f for f in findings)

    def test_unclear_fallback(self):
        # Metrics in a grey zone that doesn't match any named mode
        findings = diagnose_model(causal=0.60, trans=0.48)
        assert any("UNCLEAR" in f for f in findings)

    def test_hallucinating(self):
        findings = diagnose_model(causal=0.75, trans=0.80, precision=0.15)
        assert any("HALLUCINATING" in f for f in findings)

    def test_conservative(self):
        findings = diagnose_model(causal=0.75, trans=0.20, precision=0.90)
        assert any("CONSERVATIVE" in f for f in findings)

    def test_overfit_causal(self):
        findings = diagnose_model(
            causal=0.55, trans=0.40,
            train_causal=0.80,
        )
        assert any("OVERFITTING" in f for f in findings)

    def test_overfit_transitions(self):
        findings = diagnose_model(
            causal=0.65, trans=0.30,
            train_trans=0.65,
        )
        assert any("OVERFIT_TRANSITIONS" in f for f in findings)

    def test_no_acc_parameter(self):
        """diagnose_model must NOT accept an 'acc' positional argument."""
        import inspect
        sig = inspect.signature(diagnose_model)
        assert 'acc' not in sig.parameters, (
            "diagnose_model should not have an 'acc' parameter — "
            "plain accuracy was removed in favour of causal_regime_accuracy"
        )

    def test_returns_list(self):
        result = diagnose_model(causal=0.70, trans=0.55)
        assert isinstance(result, list)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# DiagnosticTracker
# ---------------------------------------------------------------------------

class TestDiagnosticTracker:

    def test_log_stores_entry(self):
        tracker = DiagnosticTracker()
        findings = tracker.log(epoch=0, causal=0.75, trans=0.60)
        assert len(tracker.history) == 1
        entry = tracker.history[0]
        assert entry['epoch'] == 0
        assert entry['metrics']['causal'] == 0.75
        assert entry['metrics']['trans'] == 0.60
        assert isinstance(findings, list)

    def test_trend_insufficient_data(self):
        tracker = DiagnosticTracker()
        tracker.log(epoch=0, causal=0.75, trans=0.60)
        assert tracker.trend(last_n=5) == "INSUFFICIENT_DATA"

    def test_trend_stable_healthy(self):
        tracker = DiagnosticTracker()
        for i in range(5):
            tracker.log(epoch=i, causal=0.75, trans=0.65)
        result = tracker.trend(last_n=5)
        assert "STABLE_HEALTHY" in result

    def test_trend_stalled(self):
        tracker = DiagnosticTracker()
        for i in range(5):
            tracker.log(epoch=i, causal=0.40, trans=0.25)  # all RANDOM
        result = tracker.trend(last_n=5)
        assert "STALLED" in result

    def test_trend_evolving(self):
        tracker = DiagnosticTracker()
        # Different metric values each epoch → different diagnoses
        tracker.log(epoch=0, causal=0.75, trans=0.65)  # HEALTHY
        tracker.log(epoch=1, causal=0.40, trans=0.25)  # RANDOM
        tracker.log(epoch=2, causal=0.75, trans=0.65)  # HEALTHY
        tracker.log(epoch=3, causal=0.40, trans=0.25)  # RANDOM
        tracker.log(epoch=4, causal=0.60, trans=0.48)  # UNCLEAR
        assert tracker.trend(last_n=5) == "EVOLVING"

    def test_summary_empty(self, capsys):
        tracker = DiagnosticTracker()
        tracker.summary()  # must not crash
        captured = capsys.readouterr()
        assert "No history" in captured.out

    def test_summary_populated(self, capsys):
        tracker = DiagnosticTracker()
        for i in range(3):
            tracker.log(epoch=i, causal=0.70, trans=0.55, precision=0.60)
        tracker.summary()  # must not crash
        captured = capsys.readouterr()
        assert "DIAGNOSTIC SUMMARY" in captured.out

    def test_max_history_respected(self):
        tracker = DiagnosticTracker(max_history=5)
        for i in range(10):
            tracker.log(epoch=i, causal=0.70, trans=0.55)
        assert len(tracker.history) == 5


# ---------------------------------------------------------------------------
# RegimeEarlyStopping — minimal mock model
# ---------------------------------------------------------------------------

class _MockModel:
    """Minimal stand-in for a Keras model inside a callback."""

    def __init__(self, n_weights=2):
        import tensorflow as tf
        self.weights = [
            tf.Variable(np.ones((3, 3), dtype=np.float32) * float(i + 1))
            for i in range(n_weights)
        ]
        self.stop_training = False


def _run_epoch(callback, epoch, causal, trans, prec=None):
    """Simulate one on_epoch_end call."""
    logs = {
        'val_causal_regime_accuracy': causal,
        'val_regime_transition_recall': trans,
    }
    if prec is not None:
        logs['val_regime_transition_precision'] = prec
    callback.on_epoch_end(epoch, logs)


class TestRegimeEarlyStopping:

    def _make_callback(self, **kwargs):
        cb = RegimeEarlyStopping(**kwargs)
        cb.set_model(_MockModel())
        return cb

    def test_no_max_acc_gap_parameter(self):
        """max_acc_gap must no longer be a constructor parameter."""
        import inspect
        sig = inspect.signature(RegimeEarlyStopping.__init__)
        assert 'max_acc_gap' not in sig.parameters, (
            "max_acc_gap was removed — plain accuracy is no longer tracked"
        )

    def test_healthy_label_when_gates_pass(self, capsys):
        cb = self._make_callback(min_causal=0.55, min_trans=0.30, warmup_epochs=0)
        _run_epoch(cb, 0, causal=0.70, trans=0.60)
        output = capsys.readouterr().out
        assert "HEALTHY" in output
        assert "BELOW_GATES" not in output

    def test_below_gates_label_when_causal_fails(self, capsys):
        cb = self._make_callback(min_causal=0.65, min_trans=0.30, warmup_epochs=0)
        _run_epoch(cb, 0, causal=0.50, trans=0.50)  # causal below threshold
        output = capsys.readouterr().out
        assert "BELOW_GATES" in output
        assert "causal<0.65" in output

    def test_below_gates_label_when_trans_fails(self, capsys):
        cb = self._make_callback(min_causal=0.55, min_trans=0.50, warmup_epochs=0)
        _run_epoch(cb, 0, causal=0.70, trans=0.30)  # trans below threshold
        output = capsys.readouterr().out
        assert "BELOW_GATES" in output
        assert "trans<0.5" in output

    def test_hard_failure_flickering_prevents_healthy(self, capsys):
        cb = self._make_callback(min_causal=0.55, min_trans=0.30, warmup_epochs=0)
        _run_epoch(cb, 0, causal=0.40, trans=0.60)  # FLICKERING
        output = capsys.readouterr().out
        assert "FLICKERING" in output
        assert "HEALTHY" not in output

    def test_hard_failure_stuck_prevents_healthy(self, capsys):
        cb = self._make_callback(min_causal=0.55, min_trans=0.50, warmup_epochs=0)
        _run_epoch(cb, 0, causal=0.80, trans=0.10)  # STUCK
        output = capsys.readouterr().out
        assert "STUCK" in output
        assert "HEALTHY" not in output

    def test_best_score_updated_on_valid_epoch(self):
        cb = self._make_callback(min_causal=0.55, min_trans=0.30, warmup_epochs=0)
        assert cb.best_valid_score == -float('inf')
        _run_epoch(cb, 0, causal=0.70, trans=0.60)
        expected = 0.4 * 0.70 + 0.6 * 0.60
        assert abs(cb.best_valid_score - expected) < 1e-4

    def test_wait_increments_when_gates_fail(self):
        cb = self._make_callback(min_causal=0.65, min_trans=0.50, warmup_epochs=0)
        _run_epoch(cb, 0, causal=0.70, trans=0.60)  # valid — resets wait
        _run_epoch(cb, 1, causal=0.50, trans=0.40)  # fails gates — wait++
        _run_epoch(cb, 2, causal=0.50, trans=0.40)  # fails gates — wait++
        assert cb.wait == 2

    def test_wait_resets_on_improvement(self):
        cb = self._make_callback(min_causal=0.55, min_trans=0.30, warmup_epochs=0)
        _run_epoch(cb, 0, causal=0.60, trans=0.40)  # valid, score A
        _run_epoch(cb, 1, causal=0.50, trans=0.30)  # fail → wait=1
        _run_epoch(cb, 2, causal=0.70, trans=0.60)  # better valid → wait=0
        assert cb.wait == 0
        assert cb.best_epoch == 2

    def test_stop_training_triggered_after_patience(self):
        cb = self._make_callback(
            patience=3, min_causal=0.65, min_trans=0.50, warmup_epochs=0
        )
        for i in range(4):
            _run_epoch(cb, i, causal=0.50, trans=0.30)  # all fail gates
        assert cb.model.stop_training is True

    def test_warmup_prevents_early_stop(self):
        cb = self._make_callback(
            patience=2, min_causal=0.65, min_trans=0.50, warmup_epochs=5
        )
        for i in range(4):
            _run_epoch(cb, i, causal=0.50, trans=0.30)  # all fail, but in warmup
        assert cb.model.stop_training is False

    def test_best_weights_restored_on_stop(self):
        cb = self._make_callback(
            patience=2, min_causal=0.55, min_trans=0.30,
            warmup_epochs=0, restore_best_weights=True
        )
        # First valid epoch sets best weights
        _run_epoch(cb, 0, causal=0.70, trans=0.60)
        saved = [w.numpy().copy() for w in cb.model.weights]

        # Modify model weights to simulate further training
        for w in cb.model.weights:
            w.assign(w * 99.0)

        # Trigger stop via patience
        _run_epoch(cb, 1, causal=0.40, trans=0.20)
        _run_epoch(cb, 2, causal=0.40, trans=0.20)
        _run_epoch(cb, 3, causal=0.40, trans=0.20)

        # Weights should be restored to epoch-0 values
        for w, s in zip(cb.model.weights, saved):
            np.testing.assert_allclose(w.numpy(), s, rtol=1e-5)
