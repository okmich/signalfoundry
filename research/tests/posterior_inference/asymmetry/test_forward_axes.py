import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.posterior_inference.asymmetry import MarketAxis, build_forward_outcomes, forward_axis_series


def _price_frame(T: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=T, freq="5min")
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(0.001 * rng.standard_normal(T)))
    return pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
                         "tick_volume": rng.integers(1, 100, T)}, index=idx)


def test_forward_axis_series_shapes_and_edges() -> None:
    prices = _price_frame(300)
    h = 10
    fwd, base = forward_axis_series(prices, MarketAxis.VOLATILITY, h)
    assert fwd.shape == (300,) and base.shape == (300,)
    assert np.isnan(fwd[-h:]).all()                       # forward window unavailable at the tail
    assert np.isnan(base[: h - 1]).all()                  # trailing window warm-up at the head
    assert np.all(fwd[np.isfinite(fwd)] >= 0)             # realized vol is non-negative


def test_forward_axis_trend_is_forward_log_return() -> None:
    prices = _price_frame(50)
    h = 5
    fwd, _ = forward_axis_series(prices, MarketAxis.TREND, h)
    log_close = np.log(prices["close"].to_numpy())
    assert fwd[0] == pytest.approx(log_close[h] - log_close[0])


def test_liquidity_requires_volume_column() -> None:
    prices = _price_frame(50).drop(columns=["tick_volume"])
    with pytest.raises(ValueError, match="LIQUIDITY"):
        forward_axis_series(prices, MarketAxis.LIQUIDITY, 5)


def test_rejects_bad_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        forward_axis_series(_price_frame(50), MarketAxis.TREND, 0)


def test_build_forward_outcomes_keys_and_lengths() -> None:
    prices = _price_frame(100)
    out = build_forward_outcomes(prices, [MarketAxis.TREND, MarketAxis.VOLATILITY], [4, 8])
    assert set(out) == {"trend_h4", "trend_h8", "volatility_h4", "volatility_h8"}
    assert all(fo.values.shape == (100,) for fo in out.values())
    assert out["trend_h4"].horizon == 4
