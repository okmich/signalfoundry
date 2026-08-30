"""End-to-end smoke test for HmmFeatureScreener.

Uses small synthetic data so the test runs in a few seconds, and exercises:
- screener construction + Stage-0 variance pre-filter
- ablation subset enumeration
- HMM fit per subset
- direction-axis evaluator path
- Pareto classification across non-error subsets
- result DataFrame assembly
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.features.hmm_screener import (
    HmmFeatureScreener,
    HmmScreenerConfig,
    HmmScreenerResult,
    ParetoStatus,
    ScreenStrategy,
    SubsetEvaluation,
)


def _make_synthetic_ohlc(T: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic 1-asset OHLC frame with a two-half bull/bear structure."""
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, 0.001, T)
    log_rets[: T // 2] += 0.0008
    log_rets[T // 2 :] -= 0.0008
    close = 100.0 * np.exp(np.cumsum(log_rets))
    high = close * (1.0 + rng.uniform(0.0, 0.002, T))
    low = close * (1.0 - rng.uniform(0.0, 0.002, T))
    return pd.DataFrame({"close": close, "high": high, "low": low})


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    log_rets = np.log(df["close"] / df["close"].shift(1))
    df["log_rets_smooth_24"] = log_rets.ewm(span=24, adjust=False).mean()
    df["log_rets_smooth_48"] = log_rets.ewm(span=48, adjust=False).mean()
    return df


@pytest.mark.slow
def test_screener_end_to_end_direction_axis_returns_populated_result() -> None:
    raw = _make_synthetic_ohlc(T=2000)
    config = HmmScreenerConfig(
        signal_type="trend",
        algo="hmm_lambda",
        n_states=2,
        data_size=2000,
        horizons=(12, 60),
        random_state=42,
    )
    screener = HmmFeatureScreener(config, raw, _feature_engineering)
    result = screener.screen(["log_rets_smooth_24", "log_rets_smooth_48"],
                             strategy=ScreenStrategy.ABLATION)

    assert isinstance(result, HmmScreenerResult)
    assert isinstance(result.results_, pd.DataFrame)
    assert len(result.evaluations) >= 1
    # At least one subset should evaluate without raising an error.
    non_error = [e for e in result.evaluations if e.error is None]
    assert len(non_error) >= 1
    # Pareto statuses must be one of the four values.
    valid_statuses = {ParetoStatus.ASYMMETRY_CANDIDATE, ParetoStatus.TRAP, ParetoStatus.FRAGILE, ParetoStatus.DOMINATED}
    assert all(e.pareto_status in valid_statuses for e in result.evaluations)


def test_screener_rejects_missing_close_column_at_construction() -> None:
    # __init__ now validates raw_data has a 'close' column, raising immediately
    # instead of letting every per-subset evaluation fail.
    rng = np.random.default_rng(99)
    T = 200
    raw = pd.DataFrame({
        "high": 1.0 + rng.normal(0.0, 0.01, T),
        "low": 0.5 + rng.normal(0.0, 0.01, T),
    })

    def fe(df):
        return df.copy()

    config = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2, data_size=T)
    with pytest.raises(ValueError, match="'close'"):
        HmmFeatureScreener(config, raw, fe)


def test_screener_rejects_unknown_candidate_feature() -> None:
    raw = _make_synthetic_ohlc(T=200)

    def fe(df):
        return df.copy()  # produces no new columns

    config = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2, data_size=200)
    screener = HmmFeatureScreener(config, raw, fe)
    with pytest.raises(ValueError, match="feature_engineering did not produce"):
        screener.screen(["nonexistent_feature"])


def test_screener_ablation_generates_expected_subsets() -> None:
    raw = _make_synthetic_ohlc(T=200)
    config = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2, data_size=200)
    screener = HmmFeatureScreener(config, raw, _feature_engineering)

    # 2 candidates, baseline = [a]:
    #   baseline       = (a,)
    #   drop-one of a  = ()        -> filtered out (empty)
    #   add-one b      = (a, b)
    # So ablation should emit at least (a,) and (a, b).
    subsets = screener._generate_subsets(
        surviving=["log_rets_smooth_24", "log_rets_smooth_48"],
        strategy=ScreenStrategy.ABLATION,
        baseline=["log_rets_smooth_24"],
        max_subset_size=None,
    )
    assert ("log_rets_smooth_24",) in subsets
    assert tuple(sorted(["log_rets_smooth_24", "log_rets_smooth_48"])) in subsets


def test_screener_exhaustive_generates_all_subsets() -> None:
    raw = _make_synthetic_ohlc(T=200)
    config = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2, data_size=200)
    screener = HmmFeatureScreener(config, raw, _feature_engineering)

    subsets = screener._generate_subsets(
        surviving=["a", "b", "c"],
        strategy=ScreenStrategy.EXHAUSTIVE,
        baseline=None,
        max_subset_size=None,
    )
    # 2^3 - 1 = 7 non-empty subsets.
    assert len(subsets) == 7
    assert ("a",) in subsets
    assert ("a", "b") in subsets
    assert ("a", "b", "c") in subsets


def _make_subset_eval(features=("a",), axis_sep=1.0, secondary=2.0, honesty=0.1,
                       balance=2.0, error=None) -> SubsetEvaluation:
    """Helper for building synthetic SubsetEvaluation rows for classifier tests."""
    return SubsetEvaluation(
        features=tuple(features),
        n_features=len(features),
        axis_separation=axis_sep,
        secondary_robustness=secondary,
        secondary_label="n_significant_states",
        honesty=honesty,
        state_balance_ratio=balance,
        pareto_status=ParetoStatus.DOMINATED,
        warnings=(),
        raw_details={},
        elapsed_sec=0.0,
        error=error,
    )


def test_screener_classify_fragile_on_infinite_balance_ratio() -> None:
    raw = _make_synthetic_ohlc(T=200)
    config = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2, data_size=200)
    screener = HmmFeatureScreener(config, raw, _feature_engineering)

    # State collapse: one or more states never emitted -> balance ratio is +inf.
    evs = [_make_subset_eval(balance=float("inf"))]
    statuses = screener._classify(evs)
    assert statuses == [ParetoStatus.FRAGILE]


def test_screener_classify_fragile_on_insufficient_significant_states() -> None:
    raw = _make_synthetic_ohlc(T=200)
    config = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2,
                               data_size=200, min_significant_states=2)
    screener = HmmFeatureScreener(config, raw, _feature_engineering)

    # secondary_robustness=1.0 < min_significant_states=2 -> FRAGILE
    evs = [_make_subset_eval(secondary=1.0, balance=2.0)]
    statuses = screener._classify(evs)
    assert statuses == [ParetoStatus.FRAGILE]


def test_screener_classify_fragile_excluded_from_pareto_frontier() -> None:
    raw = _make_synthetic_ohlc(T=200)
    config = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2,
                               data_size=200, min_significant_states=2)
    screener = HmmFeatureScreener(config, raw, _feature_engineering)

    # Two subsets: one fragile (high sep but state collapse), one healthy (lower sep).
    # The fragile one should NOT count as dominating the healthy one.
    evs = [
        _make_subset_eval(features=("fragile",), axis_sep=2.0, secondary=1.0, honesty=0.05, balance=float("inf")),
        _make_subset_eval(features=("healthy",), axis_sep=1.0, secondary=2.0, honesty=0.05, balance=2.0),
    ]
    statuses = screener._classify(evs)
    assert statuses == [ParetoStatus.FRAGILE, ParetoStatus.ASYMMETRY_CANDIDATE]


def test_screener_classify_trap_supersedes_fragile_check_only_when_healthy() -> None:
    # Order of phases: structural FRAGILE wins over Pareto/TRAP. A subset that
    # would otherwise be a trap by honesty is classified FRAGILE if its state
    # structure is also degenerate.
    raw = _make_synthetic_ohlc(T=200)
    config = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2,
                               data_size=200, min_significant_states=2, honesty_trap_rate=0.4)
    screener = HmmFeatureScreener(config, raw, _feature_engineering)
    evs = [
        _make_subset_eval(features=("collapsed_and_overconf",), axis_sep=2.0, secondary=1.0,
                          honesty=0.8, balance=float("inf")),
    ]
    statuses = screener._classify(evs)
    assert statuses == [ParetoStatus.FRAGILE]


def test_screener_passthrough_collision_emits_warning_and_keeps_engineered_values() -> None:
    """If a candidate feature shares a name with a reserved passthrough column
    (close / open / high / low / tick_volume / volume), the engineered version
    wins on the join and a per-subset warning is attached so the override isn't silent.
    """
    raw = _make_synthetic_ohlc(T=400)

    def fe(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        log_rets = np.log(out["close"] / out["close"].shift(1))
        # Deliberately reuse a reserved passthrough name. The engineered values are
        # NOT the raw close — they're z-scored, which would clash with raw close if
        # the screener didn't deduplicate.
        out["close"] = (log_rets - log_rets.rolling(20).mean()) / log_rets.rolling(20).std()
        out["log_rets_smooth_24"] = log_rets.ewm(span=24, adjust=False).mean()
        return out

    # NB: Stage-0b persistence removal is off by default (min_persistence=0.0), which this test relies on.
    # The engineered "close" here is a z-score of iid synthetic log-returns (persistence score ~0.013), so
    # enabling the floor would legitimately drop it before subset generation and the collision could never
    # occur. This test is about passthrough-name collision, which is orthogonal to persistence.
    config = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2,
                               data_size=400, random_state=42)
    screener = HmmFeatureScreener(config, raw, fe)
    result = screener.screen(["close", "log_rets_smooth_24"], strategy=ScreenStrategy.ABLATION,
                             baseline=["close", "log_rets_smooth_24"])

    # At least one evaluation should attach the collision warning naming 'close'.
    collision_warnings_found = any(
        any("collide with reserved passthrough" in w and "close" in w for w in ev.warnings)
        for ev in result.evaluations
    )
    assert collision_warnings_found
    # And the screen still runs — no error attached purely due to the collision.
    assert any(ev.error is None for ev in result.evaluations)


def test_screener_result_fragile_property_and_repr() -> None:
    """`result.fragile` and `__repr__` both expose the FRAGILE count alongside keepers / traps."""
    raw = _make_synthetic_ohlc(T=200)
    config = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2, data_size=200)
    screener = HmmFeatureScreener(config, raw, _feature_engineering)

    # Synthetic evaluations: one keeper, one trap, one fragile, one dominated.
    evs = [
        _make_subset_eval(features=("keeper",), axis_sep=1.0, secondary=2.0, honesty=0.10, balance=2.0),
        _make_subset_eval(features=("trap",), axis_sep=2.0, secondary=2.0, honesty=0.50, balance=2.0),
        _make_subset_eval(features=("fragile",), axis_sep=0.5, secondary=1.0, honesty=0.05, balance=float("inf")),
    ]
    statuses = screener._classify(evs)
    evs = [screener._with_status(ev, status) for ev, status in zip(evs, statuses)]
    result = HmmScreenerResult(evaluations=evs, results_=screener._build_results_df(evs))

    # `fragile` property: returns the FRAGILE-classified subsets only.
    fragile_subsets = result.fragile
    assert len(fragile_subsets) == 1
    assert fragile_subsets[0].features == ("fragile",)

    # `__repr__` includes the fragile count.
    repr_text = repr(result)
    assert "1 fragile" in repr_text
    assert "candidates" in repr_text
    assert "traps" in repr_text


def test_subset_coherence_warnings_are_per_subset() -> None:
    """Off-axis / unregistered-feature warnings name only the subset(s) that actually contain the offending feature."""
    raw = _make_synthetic_ohlc(T=200)
    config = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=2, data_size=200)
    screener = HmmFeatureScreener(config, raw, _feature_engineering)

    # Features not in the registry get a "not in FeatureRegistry" warning ONLY on subsets that include them.
    warns_a = screener._validate_subset_coherence(("synthetic_feature_a",))
    warns_b = screener._validate_subset_coherence(("synthetic_feature_b",))
    warns_ab = screener._validate_subset_coherence(("synthetic_feature_a", "synthetic_feature_b"))

    assert any("synthetic_feature_a" in w for w in warns_a)
    assert not any("synthetic_feature_b" in w for w in warns_a)
    assert any("synthetic_feature_b" in w for w in warns_b)
    assert not any("synthetic_feature_a" in w for w in warns_b)
    assert any("synthetic_feature_a" in w for w in warns_ab)
    assert any("synthetic_feature_b" in w for w in warns_ab)
