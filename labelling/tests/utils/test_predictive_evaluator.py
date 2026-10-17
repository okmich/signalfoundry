import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.utils.predictive_evaluator import FilterType, apply_filters, prepare_returns


@pytest.fixture
def prices():
    index = pd.date_range("2024-01-01", periods=50, freq="h")
    return pd.DataFrame({"close": 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 1e-3, 50)))}, index=index)


class TestPrepareReturns:
    def test_forward_return_spans_t_to_t_plus_h(self, prices):
        out = prepare_returns(prices, horizons=(3,))
        expected = np.log(prices["close"].iloc[3] / prices["close"].iloc[0])
        assert out["fwd_ret_3"].iloc[0] == pytest.approx(expected)
        assert out["fwd_ret_3"].iloc[-3:].isna().all()

    def test_same_bar_return_and_simple_returns(self, prices):
        out = prepare_returns(prices, log_returns=False, horizons=(1,))
        assert out["ret"].iloc[1] == pytest.approx(prices["close"].iloc[1] / prices["close"].iloc[0] - 1)
        assert out["fwd_ret_1"].iloc[0] == pytest.approx(out["ret"].iloc[1])

    def test_needs_a_datetime_index_or_date_column(self, prices):
        with pytest.raises(ValueError, match="DatetimeIndex"):
            prepare_returns(prices.reset_index(drop=True))


class TestApplyFilters:
    def test_categorical_and_threshold_filters_gate_the_signal(self):
        df = pd.DataFrame({"regime": [0, 1, 0, 1], "atr": [0.5, 0.5, 2.0, 0.5]})
        signal = pd.Series([1.0, -1.0, 1.0, 1.0])
        out = apply_filters(signal, df, {"regime": {"type": FilterType.CATEGORICAL, "allow_values": [0]},
                                         "atr": {"type": "threshold", "max": 1.0}})
        assert out.tolist() == [1.0, 0.0, 0.0, 0.0]

    def test_unknown_column_is_ignored(self):
        signal = pd.Series([1.0, -1.0])
        out = apply_filters(signal, pd.DataFrame({"x": [1, 2]}), {"missing": {"type": "categorical"}})
        assert out.tolist() == [1.0, -1.0]
