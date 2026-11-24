"""Tayal (2009) flavour: alphabet sizes, labelling, decoding modes, causality and a directional sanity check."""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.hmm import InferenceMode
from okmich_quant_ml.hmm.hierarchical import LegThreshold, MarketData
from okmich_quant_ml.hmm.hierarchical.flavours import FlowInput, RegimeLabel, TayalHHMM, TrendInput
from okmich_quant_ml.hmm.hierarchical.flavours.tayal import LEAVES, R_DOWN, R_UP


def _trending_market(n: int = 24000, seed: int = 0, drift: float = 0.0004, vol: float = 0.0008,
                     mean_dwell: int = 600) -> tuple[MarketData, np.ndarray]:
    """5-minute bars whose drift flips sign in regimes of ~``mean_dwell`` bars; returns the true sign per bar."""
    rng = np.random.default_rng(seed)
    sign = np.empty(n)
    s = 1.0
    for t in range(n):
        if rng.random() < 1.0 / mean_dwell:
            s = -s
        sign[t] = s
    idx = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(sign * drift + rng.normal(0, vol, n))), index=idx)
    volume = pd.Series(rng.poisson(60, n).astype(float), index=idx)
    return MarketData(close=close, high=close * 1.0003, low=close * 0.9997, volume=volume), sign


@pytest.fixture(scope="module")
def market():
    return _trending_market()


@pytest.fixture(scope="module")
def fitted(market):
    md, _ = market
    train = MarketData(close=md.close.iloc[:16000], high=md.high.iloc[:16000], low=md.low.iloc[:16000],
                       volume=md.volume.iloc[:16000])
    return TayalHHMM(LegThreshold.legs_per_hour(1.0), n_init=3, max_iter=200, random_state=1).fit(train)


def test_alphabet_sizes_follow_the_inputs(market):
    md, _ = market
    small = MarketData(close=md.close.iloc[:6000], high=md.high.iloc[:6000], low=md.low.iloc[:6000],
                       volume=md.volume.iloc[:6000])
    kwargs = dict(n_init=1, max_iter=20)
    assert TayalHHMM(flow=FlowInput.NONE, **kwargs).fit(small).encoder_.n_symbols == 6
    assert TayalHHMM(flow=FlowInput.VOLUME_RATE, **kwargs).fit(small).encoder_.n_symbols == 18
    assert TayalHHMM(trend=TrendInput.LEG_SIZE, flow=FlowInput.BOCPD, **kwargs).fit(small).encoder_.n_symbols == 18
    with pytest.raises(ValueError, match="volume"):
        TayalHHMM(flow=FlowInput.VOLUME_RATE, **kwargs).fit(MarketData(close=small.close))


def test_fit_labels_and_structure(fitted):
    assert set(fitted.labels_.values()) == {RegimeLabel.BULL, RegimeLabel.BEAR}
    bull = next(n for n, lab in fitted.labels_.items() if lab is RegimeLabel.BULL)
    other = R_UP if bull == R_DOWN else R_DOWN
    assert fitted.block_returns_[bull] >= fitted.block_returns_[other]
    p = fitted.model_.node_params(R_DOWN)
    assert p["entry"].tolist() == [1.0, 0.0] and p["exit"][1] == 0.0
    assert fitted.model_.topology.leaf_names == list(LEAVES)


def test_decode_frame_and_modes(fitted, market):
    md, _ = market
    frame = fitted.decode(md)
    assert list(frame.columns) == ["confirm_bar", "direction", "symbol", "leaf", "regime", "label", "p_bull"]
    assert frame["p_bull"].between(0, 1).all()
    assert set(frame["label"]) <= {"bull", "bear"}
    assert np.all(np.abs(np.diff(frame["direction"].to_numpy())) == 2)        # legs alternate
    test_start = md.close.index[16000]
    tail = fitted.decode(md, start=test_start)
    assert (tail.index >= test_start).all() and len(tail) < len(frame)
    pd.testing.assert_frame_equal(tail, frame[frame.index >= test_start])
    for mode in (InferenceMode.FILTERING, InferenceMode.VITERBI, InferenceMode.SMOOTHING):
        assert len(fitted.decode(md, mode=mode)) == len(frame)


def test_causal_decoding_is_prefix_invariant(fitted, market):
    md, _ = market
    full = fitted.decode(md, mode=InferenceMode.CAUSAL_VITERBI)
    cut = 20000
    part_md = MarketData(close=md.close.iloc[:cut], high=md.high.iloc[:cut], low=md.low.iloc[:cut],
                         volume=md.volume.iloc[:cut])
    part = fitted.decode(part_md, mode=InferenceMode.CAUSAL_VITERBI)
    pd.testing.assert_frame_equal(part, full.iloc[:len(part)])


def test_bar_regime_alignment(fitted, market):
    md, _ = market
    bars = fitted.bar_regime(md, delay=1)
    frame = fitted.decode(md)
    first = int(frame["confirm_bar"].iloc[0])
    assert bars.iloc[:first + 1].isna().all()
    k = 50
    row = frame.iloc[k]
    expected = 1.0 if row["label"] == "bull" else -1.0
    assert bars.iloc[int(row["confirm_bar"]) + 1] == expected


def test_regimes_read_the_planted_drift_out_of_sample(fitted, market):
    """On bars whose drift flips sign in long regimes, the causal bull/bear read agrees with the true drift sign out
    of sample well above chance (structure features see trends)."""
    md, sign = market
    bars = fitted.bar_regime(md, mode=InferenceMode.FILTERING, delay=1).to_numpy()
    test = np.arange(len(bars)) >= 16000
    ok = test & np.isfinite(bars)
    agreement = float(np.mean(bars[ok] == sign[ok]))
    assert agreement > 0.6
