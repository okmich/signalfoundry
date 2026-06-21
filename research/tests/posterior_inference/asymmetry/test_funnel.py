import numpy as np
import pandas as pd

from okmich_quant_ml.hmm import DistType

from okmich_quant_research.posterior_inference.asymmetry import (
    CandidateResult,
    HmmFitSpec,
    MarketAxis,
    WalkForwardWindow,
    confirm_candidates,
)


def _two_regime_frame(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n, freq="5min")
    state = (np.arange(n) // 50) % 2
    vol = np.where(state == 0, 0.0005, 0.003)
    rets = rng.standard_normal(n) * vol
    close = 100.0 * np.exp(np.cumsum(rets))
    f_trend = pd.Series(rets).rolling(10).mean().bfill().to_numpy()
    f_vol = pd.Series(np.abs(rets)).rolling(10).mean().bfill().to_numpy()
    return pd.DataFrame({"close": close, "f_trend": f_trend, "f_vol": f_vol}, index=idx)


def test_confirm_candidates_runs_and_reports() -> None:
    data = _two_regime_frame(500)
    fit = HmmFitSpec(dist_type=DistType.NORMAL, n_states=2, random_state=0, max_iter=8)
    window = WalkForwardWindow(train=150, oos=75, step=75, lead_in=40)
    results = confirm_candidates([["f_trend", "f_vol"], ["f_vol"]], data, fit=fit, window=window,
                                 axes=[MarketAxis.VOLATILITY], horizons=[5], min_coverage=20.0)

    assert len(results) == 2
    assert all(isinstance(r, CandidateResult) for r in results)
    for r in results:
        assert r.error is None, r.error
        assert "volatility" in r.report.verdicts
        assert isinstance(r.confirmed_axes, list)


def test_confirm_candidates_captures_bad_subset() -> None:
    data = _two_regime_frame(500)
    fit = HmmFitSpec(dist_type=DistType.NORMAL, n_states=2, max_iter=5)
    window = WalkForwardWindow(train=150, oos=75)
    results = confirm_candidates([["does_not_exist"]], data, fit=fit, window=window,
                                 axes=[MarketAxis.VOLATILITY], horizons=[5])

    assert len(results) == 1
    assert results[0].report is None
    assert results[0].error is not None
    assert results[0].confirmed_axes == []
