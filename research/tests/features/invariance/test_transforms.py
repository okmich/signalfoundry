"""Transform correctness and the parity round-trip.

The transforms are EXACT in log-price space, so these tolerances are tight on purpose: measured on the
real corpus, a known-odd feature returns -0.99999 and a known-even one +0.99999. A loose tolerance here
would absorb a genuine bug rather than sampling noise.
"""
import numpy as np
import pandas as pd
import pytest

import okmich_quant_features.momentum as mom
import okmich_quant_features.path_structure as ps
from okmich_quant_features.timothymasters.single import trend as tm_trend
from okmich_quant_research.features.invariance import (SCALE_C, probe_invariance, reflect_ohlc,
                                                       rescale_ohlc, scale_exponent)
from okmich_quant_research.features.registry import Parity, ScaleClass

#: The transforms are exact, so anything looser than this is hiding a bug, not absorbing noise.
PARITY_TOL = 0.02


def _make_ohlc(T: int = 4000, seed: int = 7) -> pd.DataFrame:
    """FX-scale bars built from an intrabar random walk.

    High/low are path EXTREMA rather than symmetric cuffs around the close, so true-range features see
    a realistic bar geometry — which is what the reflection high/low swap has to get right.
    """
    rng = np.random.default_rng(seed)
    ticks = 12
    steps = rng.normal(0.0, 0.001 / np.sqrt(ticks), (T, ticks))
    steps[: T // 2] += 0.0008 / ticks                       # a bull half and a bear half, so a
    steps[T // 2:] -= 0.0008 / ticks                        # direction-aware feature has something to see
    intrabar = np.cumsum(steps, axis=1)
    start = np.r_[0.0, np.cumsum(intrabar[:, -1])[:-1]]
    paths = start[:, None] + intrabar
    return pd.DataFrame({"open": 100.0 * np.exp(start),
                         "high": 100.0 * np.exp(np.maximum(paths.max(axis=1), start)),
                         "low": 100.0 * np.exp(np.minimum(paths.min(axis=1), start)),
                         "close": 100.0 * np.exp(paths[:, -1])})


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    hi_np, lo_np = high.to_numpy(), low.to_numpy()
    return pd.DataFrame({
        "momentum.roc": mom.roc(close, window=14),
        "momentum.plus_di": mom.plus_di(high, low, close, period=14),
        "momentum.minus_di": mom.minus_di(high, low, close, period=14),
        "momentum.di_spread": mom.di_spread(high, low, close, period=14),
        "path_structure.efficiency_ratio": ps.efficiency_ratio(close, window=60),
        "timothymasters.trend.aroon_up": pd.Series(tm_trend.aroon_up(hi_np, lo_np, period=25), index=df.index),
        "timothymasters.trend.aroon_down": pd.Series(tm_trend.aroon_down(hi_np, lo_np, period=25), index=df.index),
        "timothymasters.trend.aroon_diff": pd.Series(tm_trend.aroon_diff(hi_np, lo_np, period=25), index=df.index),
    }, index=df.index)


@pytest.fixture(scope="module")
def raw():
    return _make_ohlc()


@pytest.fixture(scope="module")
def probe(raw):
    return probe_invariance(raw, _feature_engineering, symbol="SYNTH").set_index("feature")


# ── acceptance criterion 3: reflected frame integrity ─────────────────────────────────────────────

def test_reflected_frame_keeps_high_above_low(raw):
    """The high/low swap is load-bearing.

    ``p' = c0**2 / p`` is order-REVERSING, so without swapping, every reflected bar comes back with
    ``high < low``, every true-range feature silently returns garbage, and nothing raises. This is the
    single cheapest test that would catch a broken port.
    """
    reflected = reflect_ohlc(raw)
    assert (reflected["high"] >= reflected["low"]).all()


def test_reflected_frame_keeps_open_and_close_inside_the_bar(raw):
    reflected = reflect_ohlc(raw)
    for col in ("open", "close"):
        assert (reflected[col] >= reflected["low"]).all()
        assert (reflected[col] <= reflected["high"]).all()


def test_reflection_negates_every_log_return(raw):
    reflected = reflect_ohlc(raw)
    original_lr = np.diff(np.log(raw["close"].to_numpy()))
    reflected_lr = np.diff(np.log(reflected["close"].to_numpy()))
    np.testing.assert_allclose(reflected_lr, -original_lr, rtol=0, atol=1e-12)


def test_reflection_is_an_involution(raw):
    """Reflecting twice must return the original path exactly — it is its own inverse."""
    twice = reflect_ohlc(reflect_ohlc(raw))
    for col in ("open", "high", "low", "close"):
        np.testing.assert_allclose(twice[col].to_numpy(), raw[col].to_numpy(), rtol=1e-12)


def test_rescale_multiplies_log_returns_by_c(raw):
    rescaled = rescale_ohlc(raw, SCALE_C)
    original_lr = np.diff(np.log(raw["close"].to_numpy()))
    rescaled_lr = np.diff(np.log(rescaled["close"].to_numpy()))
    # Absolute, not relative: log-returns are ~1e-3, so a relative bar is dominated by the
    # exp/log round-trip's ULP noise on the few near-zero returns.
    np.testing.assert_allclose(rescaled_lr, SCALE_C * original_lr, rtol=0, atol=1e-12)


def test_rescale_is_order_preserving(raw):
    rescaled = rescale_ohlc(raw, SCALE_C)
    assert (rescaled["high"] >= rescaled["low"]).all()


def test_transforms_reject_a_frame_without_ohlc(raw):
    with pytest.raises(ValueError, match="missing column"):
        reflect_ohlc(raw.drop(columns=["high"]))


@pytest.mark.parametrize("transform", [reflect_ohlc, rescale_ohlc])
def test_transforms_reject_non_positive_prices(raw, transform):
    """Unguarded this does not raise: ``c0**2 / 0`` is inf and a negative base gives NaN, so the pool
    fills with garbage and every feature reports UNSCORED — which reads as "unmeasurable features"
    rather than "invalid input"."""
    bad = raw.copy()
    bad.iloc[10, bad.columns.get_loc("low")] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        transform(bad)


# ── acceptance criterion 1: parity round-trip ─────────────────────────────────────────────────────

def test_known_odd_feature_reflects_to_minus_one(probe):
    """``momentum.roc`` knows which way the market went, so reflection must flip its sign."""
    assert probe.at["momentum.roc", "refl_corr"] == pytest.approx(-1.0, abs=PARITY_TOL)
    assert probe.at["momentum.roc", "parity"] == Parity.ODD.value


def test_known_even_feature_reflects_to_plus_one(probe):
    """``path_structure.efficiency_ratio`` knows only how much, so reflection must leave it alone."""
    assert probe.at["path_structure.efficiency_ratio", "refl_corr"] == pytest.approx(1.0, abs=PARITY_TOL)
    assert probe.at["path_structure.efficiency_ratio", "parity"] == Parity.EVEN.value


def test_di_spread_is_odd_by_construction(probe):
    """Verifies the provenance claim on ``momentum.di_spread``'s registry stamp.

    Its stamp says "odd by construction" rather than naming a corpus, because the difference of a
    reflection-conjugate pair maps to its own negation as a matter of algebra. That is a proof, but a
    proof about code is worth executing.
    """
    assert probe.at["momentum.di_spread", "refl_corr"] == pytest.approx(-1.0, abs=PARITY_TOL)
    assert probe.at["momentum.di_spread", "parity"] == Parity.ODD.value


def test_aroon_diff_is_odd(probe):
    assert probe.at["timothymasters.trend.aroon_diff", "refl_corr"] == pytest.approx(-1.0, abs=PARITY_TOL)


# ── scale classification ──────────────────────────────────────────────────────────────────────────

def test_return_scale_feature_carries_scale(probe):
    """``roc`` is a return, so doubling every log-return doubles its spread: exponent ~ 1."""
    assert probe.at["momentum.roc", "scale_exp"] == pytest.approx(1.0, abs=0.05)
    assert probe.at["momentum.roc", "scale_class"] == ScaleClass.CARRYING.value


def test_normalised_feature_is_scale_free(probe):
    """``efficiency_ratio`` is a ratio of displacements, so rescaling cancels: exponent ~ 0."""
    assert probe.at["path_structure.efficiency_ratio", "scale_exp"] == pytest.approx(0.0, abs=0.05)
    assert probe.at["path_structure.efficiency_ratio", "scale_class"] == ScaleClass.FREE.value


def test_scale_exponent_of_a_pure_return_series(raw):
    lr = np.log(raw["close"]).diff()
    rescaled_lr = np.log(rescale_ohlc(raw, SCALE_C)["close"]).diff()
    assert scale_exponent(lr, rescaled_lr, SCALE_C) == pytest.approx(1.0, abs=1e-6)


# ── conjugate detection (the mechanism behind ONE_SIDED) ──────────────────────────────────────────

@pytest.mark.parametrize("feature,expected", [
    ("momentum.plus_di", "momentum.minus_di"),
    ("momentum.minus_di", "momentum.plus_di"),
    ("timothymasters.trend.aroon_up", "timothymasters.trend.aroon_down"),
    ("timothymasters.trend.aroon_down", "timothymasters.trend.aroon_up"),
])
def test_conjugate_search_pairs_the_one_sided_families(probe, feature, expected):
    """The reflected column matches its PARTNER, not itself — this is what ONE_SIDED detection rests on.

    Note what is deliberately NOT asserted here: that these features come out stamped ONE_SIDED. On a
    symmetric synthetic process +DI and -DI really are near-mirror images, so ``refl_corr`` measures
    about -0.97 and the classifier correctly calls them ODD. One-sidedness is a property of REAL price
    paths (measured -0.687 on 4 FX majors), where trending and asymmetric intrabar structure break the
    mirror. Manufacturing a fixture that reproduced -0.687 would be testing the fixture. The
    classification itself is pinned on the measured numbers in ``test_probe.py``, and the shipped
    verdicts in ``registry/test_axis.py``.
    """
    assert probe.at[feature, "conjugate"] == expected
    assert abs(probe.at[feature, "conjugate_corr"]) == pytest.approx(1.0, abs=PARITY_TOL)


def test_symmetric_features_have_no_conjugate(probe):
    for feature in ("momentum.roc", "path_structure.efficiency_ratio", "momentum.di_spread"):
        assert probe.at[feature, "conjugate"] == ""
