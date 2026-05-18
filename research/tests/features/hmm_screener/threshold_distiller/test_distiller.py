from __future__ import annotations

import numpy as np
import pytest

from okmich_quant_ml.hmm import InferenceMode

from okmich_quant_research.features.hmm_screener import (
    AxisType,
    ThresholdMethod,
    UnivariateHmmThresholdConfig,
    UnivariateHmmThresholdDistiller,
    build_hmm,
)


def _two_regime_feature(seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.normal(-2.0, 0.25, 260)
    right = rng.normal(2.0, 0.25, 260)
    return np.concatenate([left, right])


def _fit_two_state_gaussian(x: np.ndarray, random_state: int = 42):
    model = build_hmm("hmm_pmgnt", n_states=2, mm_n_components=1, random_state=random_state)
    model.fit(x.reshape(-1, 1))
    model.inference_mode = InferenceMode.FILTERING
    return model


def test_univariate_distiller_extracts_ordered_static_threshold() -> None:
    x = _two_regime_feature()
    model = _fit_two_state_gaussian(x)
    config = UnivariateHmmThresholdConfig(
        axis_type=AxisType.DIRECTION,
        threshold_method=ThresholdMethod.POSTERIOR_MAP_SWITCH,
    )
    result = UnivariateHmmThresholdDistiller(config).distill(model, x)

    assert len(result.thresholds) == 1
    assert -1.0 < result.threshold_values[0] < 1.0
    assert result.threshold_fidelity > 0.90
    assert result.adjusted_rand_index > 0.80
    assert result.state_summaries[0].feature_median < result.state_summaries[1].feature_median
    assert result.separability[0].center_distance_over_pooled_iqr > 5.0


def test_univariate_distiller_supports_emission_crossing_thresholds() -> None:
    x = _two_regime_feature()
    model = _fit_two_state_gaussian(x)
    config = UnivariateHmmThresholdConfig(
        axis_type=AxisType.VOLATILITY,
        threshold_method=ThresholdMethod.EMISSION_CROSSING,
    )
    result = UnivariateHmmThresholdDistiller(config).distill(model, x)

    assert len(result.thresholds) == 1
    assert result.thresholds[0].method == ThresholdMethod.EMISSION_CROSSING
    assert -1.0 < result.threshold_values[0] < 1.0


def test_univariate_distiller_supports_empirical_switch_quantile_thresholds() -> None:
    x = _two_regime_feature()
    model = _fit_two_state_gaussian(x)
    config = UnivariateHmmThresholdConfig(
        axis_type=AxisType.DIRECTION,
        threshold_method=ThresholdMethod.EMPIRICAL_SWITCH_QUANTILE,
    )
    result = UnivariateHmmThresholdDistiller(config).distill(model, x)

    assert len(result.thresholds) == 1
    assert result.thresholds[0].method == ThresholdMethod.EMPIRICAL_SWITCH_QUANTILE
    assert -1.0 < result.threshold_values[0] < 1.0


def test_univariate_distiller_rejects_multivariate_input() -> None:
    x = _two_regime_feature()
    model = _fit_two_state_gaussian(x)
    config = UnivariateHmmThresholdConfig(axis_type=AxisType.MOMENTUM)
    bad_x = np.zeros((100, 2))
    with pytest.raises(ValueError, match="1D array"):
        UnivariateHmmThresholdDistiller(config).distill(model, bad_x)