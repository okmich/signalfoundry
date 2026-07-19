"""Stage-0c set-wise collinearity diagnostic / opt-in filter (pairwise near-duplicate gate).

Pins the contract and, deliberately, its LIMITS. It removes NEAR-DUPLICATES (a feature ~= a linear copy of
ANOTHER SINGLE feature) — scope-invariant, so safe as a global pre-filter — and nothing else:
  * orthogonal and weak-but-independent features are KEPT (it is a near-duplicate remover, not an
    orthogonaliser: forcing orthogonality would discard the weak-independent combinations the search is for);
  * a near-duplicate pair/cluster collapses to its MORE persistent member (the dwell-friendly choice);
  * MULTI-WAY dependencies where no pair is near-1 (sum = a + b) are KEPT — removing `sum` globally would
    destroy the valid {a, sum} / {b, sum} subsets; that is the per-subset fit's remit, not this stage's;
  * an underdetermined design (n < p) does NOT trigger spurious removals (no regression to saturate);
  * constant / short-overlap columns are left alone; removal is OPT-IN and default report-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.features.hmm_screener import (
    HmmScreenerConfig,
    nearest_duplicate_vif,
    persistence_score,
    stage0c_collinearity_filter,
)

SEVERE_VIF = 10.0  # textbook "severe multicollinearity" line; the FXPIG-M5 opt-in value (|r| ~ 0.949)


def _white_noise(T: int = 4000, seed: int = 0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(size=T))


def _persistent(T: int = 4000, seed: int = 1, phi: float = 0.95) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + rng.normal()
    return pd.Series(x)


def _with_correlation(a: pd.Series, r: float, seed: int) -> pd.Series:
    """A new series whose contemporaneous correlation with ``a`` is ~``r`` (so nearest-duplicate
    VIF ~1/(1-r^2)), built by mixing standardised ``a`` with fresh independent noise. The noise also
    dilutes its autocorrelation, so a highly-correlated partner is LESS persistent — used for the tie-break."""
    rng = np.random.default_rng(seed)
    az = (a - a.mean()) / a.std(ddof=0)
    noise = rng.normal(size=len(a))
    return pd.Series(r * az.values + np.sqrt(1.0 - r ** 2) * noise, index=a.index)


# --------------------------------------------------------------------- scoring

def test_orthogonal_features_score_near_one() -> None:
    X = pd.DataFrame({f"n{i}": _white_noise(seed=i) for i in range(4)})
    vifs = nearest_duplicate_vif(X)
    assert all(v < 1.5 for v in vifs.values()), f"independent features should sit at VIF ~1, got {vifs}"


def test_flags_a_near_duplicate_pair() -> None:
    a = _persistent(seed=1)
    X = pd.DataFrame({"a": a, "a_dupe": _with_correlation(a, 0.99, seed=7), "indep": _white_noise(seed=9)})
    vifs = nearest_duplicate_vif(X)
    assert vifs["a"] > 40 and vifs["a_dupe"] > 40, f"the ~duplicate pair should inflate, got {vifs}"
    assert vifs["indep"] < 1.5, f"the independent feature should stay ~1, got {vifs}"


def test_lone_and_constant_columns() -> None:
    assert nearest_duplicate_vif(pd.DataFrame({"only": _persistent(T=500)})) == {"only": 1.0}
    X = pd.DataFrame({"const": pd.Series(np.ones(500)), "signal": _persistent(T=500)})
    assert np.isnan(nearest_duplicate_vif(X)["const"]), "a no-variance column is the variance filter's remit"


# --------------------------------------------------------------------- filtering

def test_diagnostic_default_removes_nothing_but_still_scores() -> None:
    """Default must be non-destructive AND must still report scores, so a threshold can be calibrated
    from a run with no removals (the report-only mode must not be silent)."""
    a = _persistent(seed=1)
    X = pd.DataFrame({"a": a, "a_dupe": _with_correlation(a, 0.999, seed=2)})
    Xf, report = stage0c_collinearity_filter(X, verbose=False)  # no max_vif passed -> inf
    assert report.removed == []
    assert list(Xf.columns) == list(X.columns)
    scored = report.detail["nearest_duplicate_vif"]
    assert scored["a"] is not None and scored["a"] > SEVERE_VIF, "diagnostic mode must populate scores"
    assert HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2).max_vif == float("inf")


def test_opt_in_removes_only_the_redundant_feature() -> None:
    a = _persistent(seed=1)
    X = pd.DataFrame({
        "a": a,
        "a_dupe": _with_correlation(a, 0.99, seed=7),   # ~duplicate of a -> one of the pair goes
        "indep": _white_noise(seed=9),                  # orthogonal -> kept
        "weak": _with_correlation(a, 0.6, seed=5),      # VIF ~1.6 (<10) -> kept (not orthogonalised)
    })
    Xf, report = stage0c_collinearity_filter(X, max_vif=SEVERE_VIF, verbose=False)

    assert report.stage == "Stage0c_Collinearity"
    assert report.n_before == 4 and report.n_after == 3
    assert len(report.removed) == 1 and report.removed[0] in {"a", "a_dupe"}
    assert "indep" in Xf.columns and "weak" in Xf.columns
    survivors = nearest_duplicate_vif(Xf)
    assert max(v for v in survivors.values() if v == v) <= SEVERE_VIF


def test_tie_break_keeps_the_more_persistent_of_a_duplicate_pair() -> None:
    """A near-duplicate pair -> the tie-break decides. It must drop the LESS persistent member so the
    survivor is the more dwell-friendly one."""
    persistent = _persistent(T=8000, seed=1, phi=0.95)
    noisy_dupe = _with_correlation(persistent, 0.97, seed=3)  # corr high -> VIF>10; noise lowers its acf
    assert persistence_score(persistent) > persistence_score(noisy_dupe)  # precondition: the gap exists
    X = pd.DataFrame({"persistent": persistent, "noisy_dupe": noisy_dupe})
    Xf, report = stage0c_collinearity_filter(X, max_vif=SEVERE_VIF, verbose=False)
    assert report.removed == ["noisy_dupe"]
    assert list(Xf.columns) == ["persistent"]


def test_high_threshold_keeps_strongly_but_not_perfectly_correlated_features() -> None:
    """Not an orthogonaliser: at VIF>10 (|r|>~0.95) a corr~0.9 pair (VIF~5.3) must be KEPT."""
    a = _persistent(seed=1)
    X = pd.DataFrame({"a": a, "b_0p9": _with_correlation(a, 0.9, seed=4)})
    assert 3.0 < nearest_duplicate_vif(X)["a"] < 8.0, "corr~0.9 -> VIF ~5.3, safely under the ceiling"
    Xf, report = stage0c_collinearity_filter(X, max_vif=SEVERE_VIF, verbose=False)
    assert report.removed == []
    assert list(Xf.columns) == ["a", "b_0p9"]


def test_greedy_reduces_a_cluster_until_all_survivors_are_under_threshold() -> None:
    base = _persistent(seed=1)
    X = pd.DataFrame({
        "c1": _with_correlation(base, 0.99, seed=11),
        "c2": _with_correlation(base, 0.99, seed=12),
        "c3": _with_correlation(base, 0.99, seed=13),  # mutually ~collinear cluster of 3
        "indep": _white_noise(seed=99),
    })
    Xf, report = stage0c_collinearity_filter(X, max_vif=SEVERE_VIF, verbose=False)
    assert "indep" in Xf.columns
    assert len([c for c in Xf.columns if c.startswith("c")]) == 1, "cluster of 3 should collapse to 1"
    survivors = nearest_duplicate_vif(Xf)
    assert max(v for v in survivors.values() if v == v) <= SEVERE_VIF


def test_small_sample_is_never_acted_on() -> None:
    """Below min_obs jointly-present rows the correlation is untrusted, so nothing is removed."""
    a = _persistent(T=120, seed=1)
    X = pd.DataFrame({"a": a, "a_dupe": _with_correlation(a, 0.999, seed=2)})  # only 120 rows
    Xf, report = stage0c_collinearity_filter(X, max_vif=SEVERE_VIF, min_obs=200, verbose=False)
    assert report.removed == []
    assert list(Xf.columns) == ["a", "a_dupe"]


def test_constant_column_is_left_to_the_variance_filter() -> None:
    a = _persistent(T=600, seed=1)
    X = pd.DataFrame({"const": pd.Series(np.ones(600)), "a": a, "a_dupe": _with_correlation(a, 0.99, seed=2)})
    Xf, report = stage0c_collinearity_filter(X, max_vif=SEVERE_VIF, verbose=False)
    assert "const" not in report.removed
    assert "const" in Xf.columns


# --------------------------------------------------------------------- regression: the two P1 findings

def test_underdetermined_design_does_not_trigger_spurious_removals() -> None:
    """n < p: set-wise VIF saturates (R^2 -> 1, VIF -> cap) and would delete valid independent features.
    Pairwise correlation has no regression to saturate, so independent columns stay ~orthogonal and none
    are removed even with a permissive min_obs."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"f{i}": rng.normal(size=20) for i in range(22)})  # 20 rows, 22 independent features
    scores = nearest_duplicate_vif(X, min_obs=3)
    assert max(v for v in scores.values() if v == v) < 100.0, "no VIF should approach the duplicate cap"
    _, report = stage0c_collinearity_filter(X, max_vif=SEVERE_VIF, min_obs=3, verbose=False)
    assert report.removed == []


def test_multiway_dependency_is_kept_only_pairwise_duplicates_are_removed() -> None:
    """a, b independent and sum = a + b: the SET {a, b, sum} is perfectly collinear, but no PAIR is
    near-1 (corr(a, sum) = corr(b, sum) ~ 0.71). Removing `sum` globally would destroy the valid {a, sum}
    and {b, sum} subsets, so pairwise MUST keep all three."""
    a, b = _white_noise(seed=1), _white_noise(seed=2)
    X = pd.DataFrame({"a": a, "b": b, "sum": a + b})
    vifs = nearest_duplicate_vif(X)
    assert all(v < SEVERE_VIF for v in vifs.values()), f"no pair is a near-duplicate, got {vifs}"
    Xf, report = stage0c_collinearity_filter(X, max_vif=SEVERE_VIF, verbose=False)
    assert report.removed == []
    assert set(Xf.columns) == {"a", "b", "sum"}


# --------------------------------------------------------------------- documented blind spot

def test_blind_spot_regime_switching_covariance_is_not_flagged() -> None:
    """Two features whose CORRELATION switches +0.9/-0.9 by regime have ~0 average correlation, so the
    nearest-duplicate VIF is ~1 and both are (correctly) kept. Pinned to document that this sees only the
    static structure — which is why the threshold stays high and removal opt-in."""
    rng = np.random.default_rng(0)
    T, dwell = 6000, 200
    state = np.repeat(np.arange(T // dwell) % 2, dwell)
    rho = np.where(state == 0, 0.9, -0.9)
    z1, z2 = rng.normal(size=T), rng.normal(size=T)
    X = pd.DataFrame({"a": z1, "b": rho * z1 + np.sqrt(1 - rho ** 2) * z2})
    assert abs(X["a"].corr(X["b"])) < 0.2, "average correlation is ~0 despite the persistent regime"
    vifs = nearest_duplicate_vif(X)
    assert vifs["a"] < 1.5 and vifs["b"] < 1.5
    _, report = stage0c_collinearity_filter(X, max_vif=SEVERE_VIF, verbose=False)
    assert report.removed == []


# --------------------------------------------------------------------- validation

@pytest.mark.parametrize("bad", [float("nan"), 0.5, -1.0])
def test_helper_validates_its_own_threshold(bad: float) -> None:
    X = pd.DataFrame({"a": _persistent(T=500), "b": _persistent(T=500, seed=2)})
    with pytest.raises(ValueError, match="max_vif"):
        stage0c_collinearity_filter(X, max_vif=bad, verbose=False)


def test_helper_validates_min_obs() -> None:
    X = pd.DataFrame({"a": _persistent(T=500), "b": _persistent(T=500, seed=2)})
    with pytest.raises(ValueError, match="min_obs"):
        stage0c_collinearity_filter(X, max_vif=SEVERE_VIF, min_obs=1, verbose=False)


def test_config_validates_the_ceiling() -> None:
    with pytest.raises(ValueError, match="max_vif"):
        HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2, max_vif=0.5)


# --------------------------------------------------------------------- integration (the baseline concern)

def test_redundant_baseline_feature_warns_and_is_excluded_from_every_subset() -> None:
    """The baseline is privileged — under ABLATION every add-one subset is ``baseline + candidate`` — so a
    collinear baseline feature would otherwise contaminate the whole search. Stage 0c must catch it: the
    duplicate is dropped from every subset, the caller is warned, and the drop is recorded in the report."""
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
        persistent = lr.ewm(span=24, adjust=False).mean()
        out["persistent_feat"] = persistent
        out["dupe_feat"] = persistent + rng.normal(0, persistent.std() * 0.02, size=len(out))  # corr ~1
        return out

    config = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2, data_size=T,
                               random_state=0, min_persistence=0.0, max_vif=SEVERE_VIF)
    screener = HmmFeatureScreener(config, raw, fe)

    with pytest.warns(UserWarning, match="did not earn their place"):
        result = screener.screen(["persistent_feat", "dupe_feat"], strategy=ScreenStrategy.ABLATION,
                                 baseline=["persistent_feat", "dupe_feat"])

    assert all("dupe_feat" not in ev.features for ev in result.evaluations)
    reports = [r for r in result.stage_reports if r.stage == "Stage0c_Collinearity"]
    assert reports and "dupe_feat" in reports[0].removed
