from __future__ import annotations

import numpy as np
import pytest

from okmich_quant_research.features.hmm_screener import (
    AxisType,
    EmissionFamily,
    ThresholdMethod,
    UnivariateHmmThresholdConfig,
    UnivariateHmmThresholdDistiller,
)


def _two_regime_feature(seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.normal(-2.0, 0.25, 260)
    right = rng.normal(2.0, 0.25, 260)
    return np.concatenate([left, right])


def test_univariate_distiller_extracts_ordered_static_threshold() -> None:
    config = UnivariateHmmThresholdConfig(
        axis_type=AxisType.DIRECTION,
        n_states_grid=(2,),
        emission_families=(EmissionFamily.GAUSSIAN,),
        threshold_method=ThresholdMethod.POSTERIOR_MAP_SWITCH,
        random_states=(42,),
    )
    result = UnivariateHmmThresholdDistiller(config).fit_distill(_two_regime_feature())

    assert result.selected_candidate.n_states == 2
    assert result.selected_candidate.emission_family == EmissionFamily.GAUSSIAN
    assert len(result.thresholds) == 1
    assert -1.0 < result.threshold_values[0] < 1.0
    assert result.threshold_fidelity > 0.90
    assert result.adjusted_rand_index > 0.80
    assert result.state_summaries[0].feature_median < result.state_summaries[1].feature_median
    assert result.separability[0].center_distance_over_pooled_iqr > 5.0


def test_univariate_distiller_supports_emission_crossing_thresholds() -> None:
    config = UnivariateHmmThresholdConfig(
        axis_type=AxisType.VOLATILITY,
        n_states_grid=(2,),
        emission_families=(EmissionFamily.GAUSSIAN,),
        threshold_method=ThresholdMethod.EMISSION_CROSSING,
        random_states=(42,),
    )
    result = UnivariateHmmThresholdDistiller(config).fit_distill(_two_regime_feature())

    assert len(result.thresholds) == 1
    assert result.thresholds[0].method == ThresholdMethod.EMISSION_CROSSING
    assert -1.0 < result.threshold_values[0] < 1.0


def test_univariate_distiller_supports_empirical_switch_quantile_thresholds() -> None:
    config = UnivariateHmmThresholdConfig(
        axis_type=AxisType.DIRECTION,
        n_states_grid=(2,),
        emission_families=(EmissionFamily.GAUSSIAN,),
        threshold_method=ThresholdMethod.EMPIRICAL_SWITCH_QUANTILE,
        random_states=(42,),
    )
    result = UnivariateHmmThresholdDistiller(config).fit_distill(_two_regime_feature())

    assert len(result.thresholds) == 1
    assert result.thresholds[0].method == ThresholdMethod.EMPIRICAL_SWITCH_QUANTILE
    assert -1.0 < result.threshold_values[0] < 1.0


def test_univariate_distiller_rejects_multivariate_input() -> None:
    config = UnivariateHmmThresholdConfig(axis_type=AxisType.MOMENTUM, n_states_grid=(2,))
    x = np.zeros((100, 2))
    with pytest.raises(ValueError, match="1D array"):
        UnivariateHmmThresholdDistiller(config).fit_distill(x)


def test_univariate_config_rejects_empty_grids() -> None:
    with pytest.raises(ValueError, match="n_states_grid"):
        UnivariateHmmThresholdConfig(axis_type=AxisType.EFFICIENCY, n_states_grid=())

    with pytest.raises(ValueError, match="emission_families"):
        UnivariateHmmThresholdConfig(axis_type=AxisType.EFFICIENCY, emission_families=())
