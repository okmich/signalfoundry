import pytest

from okmich_quant_research.features.hmm_screener import HmmScreenerConfig, ScreenStrategy


def test_config_minimal_construction_succeeds() -> None:
    c = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=4)
    assert c.signal_type == "trend"
    assert c.algo == "hmm_lambda"
    assert c.n_states == 4
    assert c.allowed_signal_types is None
    assert c.effective_allowed_signal_types == frozenset({"trend"})


def test_config_effective_allowed_signal_types_strict_default() -> None:
    c = HmmScreenerConfig(signal_type="volatility", algo="hmm_lambda", n_states=3)
    assert c.effective_allowed_signal_types == frozenset({"volatility"})


def test_config_effective_allowed_signal_types_explicit_override() -> None:
    c = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=3,
                          allowed_signal_types=frozenset({"trend", "momentum"}))
    assert c.effective_allowed_signal_types == frozenset({"trend", "momentum"})


def test_config_rejects_unknown_signal_type() -> None:
    with pytest.raises(ValueError, match="signal_type"):
        HmmScreenerConfig(signal_type="not_a_real_type", algo="hmm_lambda", n_states=3)


def test_config_rejects_unknown_algo() -> None:
    with pytest.raises(ValueError, match="algo"):
        HmmScreenerConfig(signal_type="trend", algo="hmm_xyz", n_states=3)


def test_config_rejects_n_states_less_than_two() -> None:
    with pytest.raises(ValueError, match="n_states"):
        HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=1)


def test_config_rejects_out_of_range_honesty_threshold() -> None:
    with pytest.raises(ValueError, match="honesty_threshold"):
        HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=3, honesty_threshold=1.5)


def test_config_rejects_unknown_allowed_signal_type() -> None:
    with pytest.raises(ValueError, match="allowed_signal_types"):
        HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=3,
                          allowed_signal_types=frozenset({"trend", "made_up"}))


def test_screen_strategy_values() -> None:
    assert ScreenStrategy.ABLATION.value == "ablation"
    assert ScreenStrategy.EXHAUSTIVE.value == "exhaustive"


def test_config_min_significant_states_default_is_two() -> None:
    c = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=4)
    assert c.min_significant_states == 2


def test_config_max_balance_ratio_default_is_ten() -> None:
    c = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=4)
    assert c.max_balance_ratio == 10.0


def test_config_rejects_min_significant_states_below_one() -> None:
    with pytest.raises(ValueError, match="min_significant_states"):
        HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=4, min_significant_states=0)


def test_config_rejects_min_significant_states_above_n_states() -> None:
    with pytest.raises(ValueError, match="min_significant_states"):
        HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=3, min_significant_states=4)


def test_config_rejects_max_balance_ratio_below_one() -> None:
    with pytest.raises(ValueError, match="max_balance_ratio"):
        HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=4, max_balance_ratio=0.5)
