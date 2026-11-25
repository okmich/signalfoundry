"""Regressions for the 2026-11-24 review: Tayal refit atomicity, inputs and start-up; DST-correct trading days."""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.hmm.hierarchical import LegThreshold, MarketData
from okmich_quant_ml.hmm.hierarchical.flavours import FlowInput, TayalHHMM, block_ids


def _market(n: int = 6000, seed: int = 0, vol: float = 0.0008) -> MarketData:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol, n))), index=idx)
    return MarketData(close=close, volume=pd.Series(rng.poisson(50, n).astype(float), index=idx))


def _slice(md: MarketData, stop: int) -> MarketData:
    return MarketData(close=md.close.iloc[:stop], volume=md.volume.iloc[:stop])


def test_volume_rate_needs_only_close_and_volume():
    model = TayalHHMM(LegThreshold.legs_per_hour(1.0), flow=FlowInput.VOLUME_RATE, n_init=1, max_iter=10)
    model.fit(_market())                                  # no high / low
    assert model.encoder_.n_symbols == 18
    with pytest.raises(ValueError, match="high"):
        TayalHHMM(flow=FlowInput.SIGNED_VOLUME, n_init=1, max_iter=10).fit(_market())


def test_a_failed_refit_leaves_the_previous_fit_intact():
    md = _market()
    model = TayalHHMM(LegThreshold.legs_per_hour(1.0), n_init=1, max_iter=20).fit(md)
    theta, before = model.builder_.theta_, model.decode(md)
    model.threshold = LegThreshold.fixed(0.05)            # a leg size the short window cannot fill
    with pytest.raises(ValueError, match="legs on the training bars"):
        model.fit(_slice(md, 300))
    assert model.builder_.theta_ == theta
    pd.testing.assert_frame_equal(model.decode(md), before)


def test_a_window_with_no_confirmed_leg_yet():
    md = _market()
    model = TayalHHMM(LegThreshold.legs_per_hour(1.0), n_init=1, max_iter=20).fit(md)
    early = _slice(md, 3)
    frame = model.decode(early)
    assert frame.empty and list(frame.columns) == ["confirm_bar", "direction", "symbol", "leaf", "regime", "label",
                                                   "p_bull"]
    bars = model.bar_regime(early)
    assert len(bars) == 3 and bars.isna().all()


def test_an_empty_feed_decodes_to_nothing():
    md = _market()
    model = TayalHHMM(LegThreshold.legs_per_hour(1.0), n_init=1, max_iter=20).fit(md)
    empty = _slice(md, 0)
    assert model.decode(empty).empty
    assert model.bar_regime(empty).empty


@pytest.mark.parametrize("day", ["2026-03-08", "2026-11-01", "2026-06-15"])
def test_fx_day_rolls_at_17_new_york_across_dst(day):
    local = pd.date_range(pd.Timestamp(day) - pd.Timedelta("1D"), periods=72, freq="1h", tz="America/New_York")
    ids = block_ids(local.tz_convert("UTC"), tz="America/New_York", roll_hour=17)
    changes = np.flatnonzero(np.diff(ids) != 0) + 1
    assert len(changes) >= 2
    assert all(local[i].hour == 17 for i in changes)
