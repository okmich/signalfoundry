import numpy as np
import pytest

from okmich_quant_ml.posterior_inference import AbstainMode, MarginGateInferer


def test_margin_gate_inferer_flat_mode_emits_abstain_label_when_gate_closed() -> None:
    probs = np.array(
        [
            [0.40, 0.35, 0.25],
            [0.80, 0.10, 0.10],
            [0.50, 0.45, 0.05],
        ],
        dtype=float,
    )
    inferer = MarginGateInferer(theta_top=0.70, theta_margin=0.20, abstain_mode=AbstainMode.FLAT, abstain_label=-1)

    labels = inferer.infer(probs)
    metadata = inferer.get_metadata()

    np.testing.assert_array_equal(labels, np.array([-1, 0, -1], dtype=np.int64))
    assert metadata["gate_open_count"] == 1
    assert metadata["abstained_count"] == 2
    assert metadata["abstain_mode"] == "flat"


def test_margin_gate_inferer_hold_last_mode_carries_most_recent_open_label() -> None:
    probs = np.array(
        [
            [0.45, 0.40, 0.15],
            [0.78, 0.12, 0.10],
            [0.55, 0.35, 0.10],
            [0.05, 0.85, 0.10],
            [0.30, 0.45, 0.25],
        ],
        dtype=float,
    )
    inferer = MarginGateInferer(theta_top=0.70, theta_margin=0.20, abstain_mode=AbstainMode.HOLD_LAST, abstain_label=2)

    labels = inferer.infer(probs)

    np.testing.assert_array_equal(labels, np.array([2, 0, 0, 1, 1], dtype=np.int64))


def test_margin_gate_inferer_uses_closed_interval_thresholds() -> None:
    probs = np.array([[0.80, 0.20]], dtype=float)
    inferer = MarginGateInferer(theta_top=0.80, theta_margin=0.60, abstain_mode=AbstainMode.FLAT, abstain_label=-1)

    labels = inferer.infer(probs)

    np.testing.assert_array_equal(labels, np.array([0], dtype=np.int64))


def test_margin_gate_inferer_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="theta_top"):
        MarginGateInferer(theta_top=1.10)
    with pytest.raises(ValueError, match="theta_margin"):
        MarginGateInferer(theta_margin=-0.10)
