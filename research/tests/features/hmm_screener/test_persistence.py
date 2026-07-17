"""Stage-0b marginal-persistence diagnostic / opt-in filter.

Pins the contract and, deliberately, its LIMITS. The score is per-feature and marginal while the emission
is joint and shape-aware, so it must not remove by default:
  * white in the MEAN but structured in VARIANCE must be kept (screening on acf(x) alone would reject it);
  * gaps must not be spliced into fake persistence;
  * short samples must not be rejected (an autocorrelation estimate is ~1/sqrt(n) noisy);
  * covariance / tail-shape regimes are documented blind spots — pinned here so the limitation cannot be
    quietly forgotten and the default flipped to destructive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.features.hmm_screener import (
    HmmScreenerConfig,
    adjacent_pair_count,
    persistence_score,
    stage0b_persistence_filter,
)

FXPIG_M5_FLOOR = 0.15  # the calibrated (dataset-specific) opt-in value


def _white_noise(T: int = 4000, seed: int = 0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(size=T))


def _persistent(T: int = 4000, seed: int = 1, phi: float = 0.95) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + rng.normal()
    return pd.Series(x)


def _vol_clustering(T: int = 4000, seed: int = 2, phi: float = 0.99, vol_of_vol: float = 0.15) -> pd.Series:
    """Stochastic volatility: white in the mean, persistent in the variance.

    The shape of a real order-flow feature — sign unpredictable, magnitude clustered. Yields acf(x) ~ 0
    with acf(|x|) ~ 0.34, the same range as ``momentum.roc_velocity`` (0.21-0.35) this guard protects.
    """
    rng = np.random.default_rng(seed)
    h = np.zeros(T)
    for t in range(1, T):
        h[t] = phi * h[t - 1] + vol_of_vol * rng.normal()
    return pd.Series(rng.normal(size=T) * np.exp(h / 2))


# --------------------------------------------------------------------- scoring

def test_persistence_score_separates_noise_from_structure() -> None:
    assert persistence_score(_white_noise()) < 0.1
    assert persistence_score(_persistent()) > 0.8


def test_vol_clustering_feature_scores_above_the_floor_despite_white_mean() -> None:
    """False-positive guard: white in the mean, variance persists -> must survive the calibrated floor."""
    x = _vol_clustering()
    assert abs(x.autocorr(1)) < 0.1, "fixture should be white in the mean"
    assert persistence_score(x) > FXPIG_M5_FLOOR, "score must see the structure in |x| / x^2"


def test_gaps_are_not_spliced_into_fake_persistence() -> None:
    """[0, NaN, 1, NaN, 2, NaN, 3] has NO adjacent valid pair; dropna() would fake a perfect trend."""
    s = pd.Series([0.0, np.nan, 1.0, np.nan, 2.0, np.nan, 3.0])
    assert adjacent_pair_count(s) == 0
    assert np.isnan(persistence_score(s))
    # a gapped-but-real series still scores off its genuine adjacent pairs only
    base = _persistent(T=2000)
    gapped = base.copy()
    gapped[::3] = np.nan
    assert adjacent_pair_count(gapped) < len(base) - 1
    assert persistence_score(gapped) > 0.5


# --------------------------------------------------------------------- filtering

def test_diagnostic_is_the_default_and_removes_nothing() -> None:
    """Default must be non-destructive: the marginal test is blind to covariance / tail-shape regimes."""
    X = pd.DataFrame({"noise": _white_noise(), "signal": _persistent()})
    Xf, report = stage0b_persistence_filter(X, verbose=False)  # no min_persistence passed
    assert report.removed == []
    assert list(Xf.columns) == list(X.columns)
    assert HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2).min_persistence == 0.0


def test_opt_in_filter_removes_only_the_memoryless_feature() -> None:
    X = pd.DataFrame({"noise": _white_noise(), "signal": _persistent(), "vol_cluster": _vol_clustering()})
    Xf, report = stage0b_persistence_filter(X, min_persistence=FXPIG_M5_FLOOR, verbose=False)

    assert report.stage == "Stage0b_Persistence"
    assert report.removed == ["noise"]
    assert report.n_before == 3 and report.n_after == 2
    assert list(Xf.columns) == ["signal", "vol_cluster"]


def test_short_samples_are_never_rejected() -> None:
    """An acf estimate is ~1/sqrt(n) noisy: IID noise clears a 0.15 floor ~47% of the time at n=60,
    so the score cannot be acted on there. Such features are kept, not rejected."""
    X = pd.DataFrame({"short_noise": _white_noise(T=100)})
    Xf, report = stage0b_persistence_filter(X, min_persistence=FXPIG_M5_FLOOR, min_obs=600, verbose=False)
    assert report.removed == [], "must not reject on a sample too small to support the call"
    assert "short_noise" in Xf.columns
    # with the guard lowered to match the sample, rejection becomes permissible again
    _, report2 = stage0b_persistence_filter(X, min_persistence=0.9, min_obs=50, verbose=False)
    assert report2.removed == ["short_noise"]


def test_degenerate_columns_are_left_to_the_variance_filter() -> None:
    X = pd.DataFrame({"const": pd.Series(np.ones(700)), "signal": _persistent(T=700)})
    Xf, report = stage0b_persistence_filter(X, min_persistence=FXPIG_M5_FLOOR, verbose=False)
    assert "const" not in report.removed
    assert "const" in Xf.columns


# --------------------------------------------------------------------- documented blind spots

def test_blind_spot_covariance_regime_is_not_detectable_marginally() -> None:
    """Both marginals are white while their CORRELATION switches persistently. The score cannot see it,
    but a full-covariance emission can — which is why removal is opt-in, not the default."""
    rng = np.random.default_rng(0)
    T, dwell = 6000, 200
    state = np.repeat(np.arange(T // dwell) % 2, dwell)
    rho = np.where(state == 0, 0.9, -0.9)
    z1, z2 = rng.normal(size=T), rng.normal(size=T)
    a = pd.Series(z1)
    b = pd.Series(rho * z1 + np.sqrt(1 - rho ** 2) * z2)

    assert persistence_score(a) < 0.1 and persistence_score(b) < 0.1      # both look like noise
    assert abs(a.rolling(50).corr(b).autocorr(1)) > 0.9                   # yet the regime is strongly persistent


def test_blind_spot_tail_shape_regime_is_not_detectable_marginally() -> None:
    """Normal vs heavy-tailed at EQUAL variance, persistently. A Lambda emission fits a per-state tail
    parameter and can separate these; the marginal score (and even acf(x^4)) cannot."""
    rng = np.random.default_rng(0)
    T, dwell = 8000, 200
    state = np.repeat(np.arange(T // dwell) % 2, dwell)
    x = pd.Series(np.where(state == 0, rng.normal(0, 1, T), rng.standard_t(3, T) / np.sqrt(3.0)))

    assert abs(x[state == 0].var() - x[state == 1].var()) < 0.35          # variance ~matched
    assert x[state == 1].kurtosis() > 5 > x[state == 0].kurtosis()        # only the SHAPE differs
    assert persistence_score(x) < 0.1, "documented blind spot: marginal score cannot see a shape regime"


# --------------------------------------------------------------------- validation

@pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.5])
def test_helper_validates_its_own_threshold(bad: float) -> None:
    """The helper is publicly exported, so it cannot rely on HmmScreenerConfig for validation:
    NaN would silently disable every comparison, and >1 would remove every finite-scored feature."""
    X = pd.DataFrame({"signal": _persistent(T=700)})
    with pytest.raises(ValueError, match="min_persistence"):
        stage0b_persistence_filter(X, min_persistence=bad, verbose=False)


def test_helper_validates_min_obs() -> None:
    X = pd.DataFrame({"signal": _persistent(T=700)})
    with pytest.raises(ValueError, match="min_obs"):
        stage0b_persistence_filter(X, min_persistence=0.15, min_obs=1, verbose=False)


def test_config_validates_the_floor() -> None:
    with pytest.raises(ValueError, match="min_persistence"):
        HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2, min_persistence=1.5)


def test_rejected_baseline_feature_warns_and_is_excluded_from_every_subset() -> None:
    """A baseline feature must earn its place: under ABLATION every add-one subset is
    ``baseline + candidate``, so a rejected baseline feature would otherwise contaminate the search."""
    from okmich_quant_research.features.hmm_screener import HmmFeatureScreener, ScreenStrategy

    rng = np.random.default_rng(3)
    T = 900
    log_rets = rng.normal(0, 0.001, T)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    raw = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999,
                        "close": close, "tick_volume": rng.integers(50, 500, T).astype(float)},
                       index=pd.date_range("2024-01-01", periods=T, freq="5min"))

    def fe(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        lr = np.log(out["close"] / out["close"].shift(1))
        out["persistent_feat"] = lr.ewm(span=24, adjust=False).mean()
        out["coin_feat"] = pd.Series(rng.normal(size=len(out)), index=out.index)
        return out

    config = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2, data_size=T,
                               random_state=0, min_persistence=FXPIG_M5_FLOOR)
    screener = HmmFeatureScreener(config, raw, fe)

    with pytest.warns(UserWarning, match="did not earn their place"):
        result = screener.screen(["persistent_feat", "coin_feat"], strategy=ScreenStrategy.ABLATION,
                                 baseline=["persistent_feat", "coin_feat"])

    assert all("coin_feat" not in ev.features for ev in result.evaluations)
    reports = [r for r in result.stage_reports if r.stage == "Stage0b_Persistence"]
    assert reports and "coin_feat" in reports[0].removed
