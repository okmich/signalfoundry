import numpy as np
import pytest

from okmich_quant_ml.posterior_inference import (
    AbstainMode,
    ArgmaxInferer,
    CompositeGateInferer,
    EntropyGateInferer,
    MarginGateInferer,
    StabilityGateInferer,
)


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


def test_entropy_gate_inferer_flat_mode_opens_on_low_entropy() -> None:
    probs = np.array(
        [
            [0.95, 0.04, 0.01],
            [0.40, 0.30, 0.30],
            [0.90, 0.05, 0.05],
        ],
        dtype=float,
    )
    inferer = EntropyGateInferer(theta_entropy=0.50, abstain_mode=AbstainMode.FLAT, abstain_label=-1)

    labels = inferer.infer(probs)
    metadata = inferer.get_metadata()

    np.testing.assert_array_equal(labels, np.array([0, -1, 0], dtype=np.int64))
    assert metadata["gate_open_count"] == 2
    assert metadata["abstained_count"] == 1
    assert metadata["inferer"] == "EntropyGateInferer"


def test_entropy_gate_inferer_hold_last_mode_carries_label_through_abstain() -> None:
    probs = np.array(
        [
            [0.40, 0.35, 0.25],
            [0.95, 0.03, 0.02],
            [0.40, 0.35, 0.25],
            [0.02, 0.03, 0.95],
        ],
        dtype=float,
    )
    inferer = EntropyGateInferer(theta_entropy=0.40, abstain_mode=AbstainMode.HOLD_LAST, abstain_label=1)

    labels = inferer.infer(probs)

    np.testing.assert_array_equal(labels, np.array([1, 0, 0, 2], dtype=np.int64))


def test_entropy_gate_inferer_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="theta_entropy"):
        EntropyGateInferer(theta_entropy=1.50)


def test_composite_gate_inferer_opens_only_when_all_three_conditions_pass() -> None:
    probs = np.array(
        [
            [0.95, 0.04, 0.01],
            [0.50, 0.30, 0.20],
            [0.90, 0.05, 0.05],
            [0.80, 0.18, 0.02],
        ],
        dtype=float,
    )
    inferer = CompositeGateInferer(theta_top=0.70, theta_margin=0.50, theta_entropy=0.50,
                                   abstain_mode=AbstainMode.FLAT, abstain_label=-1)

    labels = inferer.infer(probs)
    metadata = inferer.get_metadata()

    np.testing.assert_array_equal(labels, np.array([0, -1, 0, -1], dtype=np.int64))
    assert metadata["gate_open_count"] == 2
    assert metadata["inferer"] == "CompositeGateInferer"


def test_composite_gate_inferer_is_no_stricter_than_individual_gates() -> None:
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(alpha=np.ones(3), size=200)
    theta_top, theta_margin, theta_entropy = 0.55, 0.20, 0.80

    composite = CompositeGateInferer(theta_top=theta_top, theta_margin=theta_margin, theta_entropy=theta_entropy,
                                     abstain_mode=AbstainMode.FLAT, abstain_label=-1)
    margin_only = MarginGateInferer(theta_top=theta_top, theta_margin=theta_margin,
                                    abstain_mode=AbstainMode.FLAT, abstain_label=-1)
    entropy_only = EntropyGateInferer(theta_entropy=theta_entropy,
                                      abstain_mode=AbstainMode.FLAT, abstain_label=-1)

    composite.infer(probs)
    margin_only.infer(probs)
    entropy_only.infer(probs)

    c_open = composite.get_metadata()["gate_open_count"]
    m_open = margin_only.get_metadata()["gate_open_count"]
    e_open = entropy_only.get_metadata()["gate_open_count"]

    assert c_open <= m_open
    assert c_open <= e_open


def test_composite_gate_inferer_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="theta_top"):
        CompositeGateInferer(theta_top=-0.10)
    with pytest.raises(ValueError, match="theta_margin"):
        CompositeGateInferer(theta_margin=1.10)
    with pytest.raises(ValueError, match="theta_entropy"):
        CompositeGateInferer(theta_entropy=1.10)


def test_composite_gate_relaxing_entropy_threshold_opens_more_rows() -> None:
    probs = np.array(
        [
            [0.80, 0.18, 0.02],
            [0.95, 0.03, 0.02],
        ],
        dtype=float,
    )
    strict = CompositeGateInferer(theta_top=0.70, theta_margin=0.50, theta_entropy=0.50,
                                  abstain_mode=AbstainMode.FLAT, abstain_label=-1)
    relaxed = CompositeGateInferer(theta_top=0.70, theta_margin=0.50, theta_entropy=0.80,
                                   abstain_mode=AbstainMode.FLAT, abstain_label=-1)

    strict.infer(probs)
    relaxed.infer(probs)

    assert strict.get_metadata()["gate_open_count"] < relaxed.get_metadata()["gate_open_count"]


def test_gating_inferers_accept_string_abstain_mode() -> None:
    probs = np.array([[0.95, 0.03, 0.02], [0.40, 0.30, 0.30]], dtype=float)

    margin_inferer = MarginGateInferer(theta_top=0.70, theta_margin=0.20, abstain_mode="flat", abstain_label=-1)
    entropy_inferer = EntropyGateInferer(theta_entropy=0.50, abstain_mode="flat", abstain_label=-1)
    composite_inferer = CompositeGateInferer(theta_top=0.70, theta_margin=0.20, theta_entropy=0.50,
                                             abstain_mode="hold_last", abstain_label=2)

    np.testing.assert_array_equal(margin_inferer.infer(probs), np.array([0, -1], dtype=np.int64))
    np.testing.assert_array_equal(entropy_inferer.infer(probs), np.array([0, -1], dtype=np.int64))
    np.testing.assert_array_equal(composite_inferer.infer(probs), np.array([0, 0], dtype=np.int64))

    assert margin_inferer.abstain_mode == AbstainMode.FLAT
    assert entropy_inferer.abstain_mode == AbstainMode.FLAT
    assert composite_inferer.abstain_mode == AbstainMode.HOLD_LAST


def test_gating_inferers_reject_malformed_posterior_inputs() -> None:
    margin_inferer = MarginGateInferer()
    entropy_inferer = EntropyGateInferer()
    composite_inferer = CompositeGateInferer()

    for inferer in (margin_inferer, entropy_inferer, composite_inferer):
        with pytest.raises(ValueError, match="shape"):
            inferer.infer(np.array([0.5, 0.5], dtype=float))
        with pytest.raises(ValueError, match="K >= 2"):
            inferer.infer(np.array([[1.0]], dtype=float))
        with pytest.raises(ValueError, match="NaN or Inf"):
            inferer.infer(np.array([[0.5, np.nan]], dtype=float))


def test_gating_inferers_metadata_survives_reinference() -> None:
    probs_first = np.array([[0.95, 0.03, 0.02]], dtype=float)
    probs_second = np.array([[0.40, 0.30, 0.30], [0.40, 0.30, 0.30]], dtype=float)

    inferer = MarginGateInferer(theta_top=0.70, theta_margin=0.20, abstain_mode=AbstainMode.FLAT, abstain_label=-1)

    inferer.infer(probs_first)
    assert inferer.get_metadata()["gate_open_count"] == 1
    inferer.infer(probs_second)
    assert inferer.get_metadata()["gate_open_count"] == 0
    assert inferer.get_metadata()["abstained_count"] == 2


def test_gating_inferers_handle_empty_input() -> None:
    empty = np.zeros((0, 3), dtype=float)

    for inferer in (MarginGateInferer(), EntropyGateInferer(), CompositeGateInferer()):
        labels = inferer.infer(empty)
        assert labels.shape == (0,)
        assert labels.dtype == np.int64


def test_stability_gate_inferer_opens_on_stable_argmax() -> None:
    probs = np.tile(np.array([0.6, 0.3, 0.1], dtype=float), (8, 1))
    inferer = StabilityGateInferer(theta_flip_rate=0.0, window=4, abstain_mode=AbstainMode.FLAT, abstain_label=-1)

    labels = inferer.infer(probs)
    metadata = inferer.get_metadata()

    np.testing.assert_array_equal(labels, np.zeros(8, dtype=np.int64))
    assert metadata["gate_open_count"] == 8
    assert metadata["gate_open_rate"] == pytest.approx(1.0)


def test_stability_gate_inferer_closes_on_alternating_argmax() -> None:
    probs = np.array([[0.6, 0.4], [0.4, 0.6]] * 10, dtype=float)
    inferer = StabilityGateInferer(theta_flip_rate=0.20, window=5, abstain_mode=AbstainMode.FLAT, abstain_label=-1)

    labels = inferer.infer(probs)
    metadata = inferer.get_metadata()

    # Steady-state flip rate saturates at 1.0, well above theta_flip_rate=0.20,
    # so only the expanding-window rows at the very start can have the gate open.
    assert metadata["gate_open_count"] <= 2
    assert metadata["abstained_count"] >= len(probs) - 2
    # Abstained rows carry the abstain label (-1) under FLAT mode.
    assert (labels == -1).sum() >= len(probs) - 2


def test_stability_gate_inferer_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="theta_flip_rate"):
        StabilityGateInferer(theta_flip_rate=1.5)
    with pytest.raises(ValueError, match="window must be >= 1"):
        StabilityGateInferer(window=0)


def test_stability_gate_inferer_handles_empty_input() -> None:
    labels = StabilityGateInferer().infer(np.zeros((0, 3), dtype=float))

    assert labels.shape == (0,)
    assert labels.dtype == np.int64


def test_argmax_inferer_returns_int64_and_rejects_nan() -> None:
    probs = np.array(
        [
            [0.10, 0.70, 0.20],
            [0.60, 0.25, 0.15],
        ],
        dtype=float,
    )

    labels = ArgmaxInferer().infer(probs)

    assert labels.dtype == np.int64
    np.testing.assert_array_equal(labels, np.array([1, 0], dtype=np.int64))

    with pytest.raises(ValueError, match="NaN or Inf"):
        ArgmaxInferer().infer(np.array([[0.5, np.nan]], dtype=float))
    with pytest.raises(ValueError, match="shape"):
        ArgmaxInferer().infer(np.array([0.5, 0.5], dtype=float))
