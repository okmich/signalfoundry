import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.hmm import DistType

from okmich_quant_research.posterior_inference.asymmetry import (
    HmmFitSpec,
    MarketAxis,
    PosteriorStream,
    WalkForwardWindow,
    frozen_artifact_posteriors,
    walk_forward_filtered_posteriors,
)


def _two_regime_frame(n: int = 700) -> pd.DataFrame:
    """Synthetic series with a clean 2-state volatility structure (alternating low/high-vol 50-bar blocks)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n, freq="5min")
    state = (np.arange(n) // 50) % 2
    vol = np.where(state == 0, 0.0005, 0.003)
    rets = rng.standard_normal(n) * vol
    close = 100.0 * np.exp(np.cumsum(rets))
    f_trend = pd.Series(rets).rolling(10).mean().bfill().to_numpy()
    f_vol = pd.Series(np.abs(rets)).rolling(10).mean().bfill().to_numpy()
    return pd.DataFrame({"close": close, "f_trend": f_trend, "f_vol": f_vol}, index=idx)


def test_walk_forward_filtered_posteriors_shapes_folds_and_simplex() -> None:
    data = _two_regime_frame(700)
    fit = HmmFitSpec(dist_type=DistType.NORMAL, n_states=2, random_state=0, max_iter=10)
    window = WalkForwardWindow(train=200, oos=100, step=100, lead_in=50)
    stream = walk_forward_filtered_posteriors(data, feature_columns=["f_trend", "f_vol"], fit=fit, window=window)

    assert isinstance(stream, PosteriorStream)
    # OOS at oos_lo in {200,300,400,500,600} -> 5 folds x 100 rows.
    assert stream.probs.shape == (500, 2)
    assert len(stream.index) == 500
    assert sorted(np.unique(stream.fold_ids).tolist()) == [0, 1, 2, 3, 4]
    assert stream.state_names == ["vol_0", "vol_1"]
    np.testing.assert_allclose(stream.probs.sum(axis=1), 1.0, atol=1e-6)
    # OOS rows are the contiguous block [200, 700) of the original index.
    assert stream.index[0] == data.index[200] and stream.index[-1] == data.index[699]


def test_rejects_overlapping_oos() -> None:
    data = _two_regime_frame(400)
    fit = HmmFitSpec(dist_type=DistType.NORMAL, n_states=2, max_iter=5)
    with pytest.raises(ValueError, match="overlap"):
        walk_forward_filtered_posteriors(data, feature_columns=["f_trend", "f_vol"], fit=fit,
                                         window=WalkForwardWindow(train=100, oos=100, step=50))


def test_non_volatility_identity_not_yet_supported() -> None:
    data = _two_regime_frame(400)
    fit = HmmFitSpec(dist_type=DistType.NORMAL, n_states=2, max_iter=5)
    with pytest.raises(NotImplementedError):
        walk_forward_filtered_posteriors(data, feature_columns=["f_trend", "f_vol"], fit=fit,
                                         window=WalkForwardWindow(train=100, oos=100), identity_axis=MarketAxis.TREND)


def test_requires_close_column() -> None:
    data = _two_regime_frame(400).drop(columns=["close"])
    fit = HmmFitSpec(dist_type=DistType.NORMAL, n_states=2, max_iter=5)
    with pytest.raises(ValueError, match="close"):
        walk_forward_filtered_posteriors(data, feature_columns=["f_trend", "f_vol"], fit=fit,
                                         window=WalkForwardWindow(train=100, oos=100))


def test_walk_forward_window_rejects_degenerate_geometry() -> None:
    with pytest.raises(ValueError, match="oos"):
        WalkForwardWindow(train=100, oos=0)            # default step would never advance -> was an infinite loop
    with pytest.raises(ValueError, match="train"):
        WalkForwardWindow(train=0, oos=50)
    with pytest.raises(ValueError, match="lead_in"):
        WalkForwardWindow(train=100, oos=50, lead_in=-1)
    with pytest.raises(ValueError, match="overlap"):
        WalkForwardWindow(train=100, oos=100, step=50)


def test_hmm_fit_spec_requires_two_states() -> None:
    with pytest.raises(ValueError, match="n_states"):
        HmmFitSpec(dist_type=DistType.NORMAL, n_states=1)
    with pytest.raises(ValueError, match="max_iter"):
        HmmFitSpec(dist_type=DistType.NORMAL, n_states=2, max_iter=0)
    with pytest.raises(ValueError, match="n_components"):
        HmmFitSpec(dist_type=DistType.NORMAL, n_states=2, is_mixture=True, n_components=1)


def test_frozen_artifact_posteriors_rejects_negative_lead_in() -> None:
    # lead_in is validated before the artifact is loaded, so this needs no real artifact on disk.
    data = _two_regime_frame(100)
    with pytest.raises(ValueError, match="lead_in"):
        frozen_artifact_posteriors("/nonexistent/artifact_dir", data, lead_in=-1)
