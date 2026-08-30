import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from okmich_quant_research.backtesting.signal_adapter import (
    positions_to_signals,
    signal_to_portfolio,
)


@pytest.fixture
def price_data() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = (rng.standard_normal(n).cumsum() + 100).clip(min=10)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=dates)


class TestPositionsToSignals:
    def test_edges_and_reversal(self):
        # 0 -> 1 (long entry), 1 -> -1 (same-bar long exit + short entry), -1 -> 0 (short exit)
        pos = pd.Series([0.0, 1.0, 1.0, -1.0, 0.0])
        entries, exits, short_entries, short_exits = positions_to_signals(pos)

        assert entries.tolist() == [False, True, False, False, False]
        assert exits.tolist() == [False, False, False, True, False]
        assert short_entries.tolist() == [False, False, False, True, False]
        assert short_exits.tolist() == [False, False, False, False, True]

    def test_first_bar_can_be_an_entry(self):
        assert positions_to_signals(pd.Series([1.0]))[0].iloc[0]  # entries[0] is True
        assert positions_to_signals(pd.Series([-1.0]))[2].iloc[0]  # short_entries[0] is True

    def test_nan_is_treated_as_flat(self):
        entries, exits, _, _ = positions_to_signals(pd.Series([np.nan, 1.0, np.nan]))
        assert entries.tolist() == [False, True, False]
        assert exits.tolist() == [False, False, True]  # 1 -> NaN(flat) closes the long


class TestSignalToPortfolio:
    def test_misaligned_index_raises_instead_of_silent_zero_trades(self, price_data: pd.DataFrame):
        # The classic trap: np.where wrapped without index= -> default RangeIndex, no overlap.
        def bad_signal(data: pd.DataFrame) -> pd.Series:
            return pd.Series(np.where(data["close"].values > data["close"].mean(), 1, -1))

        with pytest.raises(ValueError, match="does not overlap data.index"):
            signal_to_portfolio(price_data, bad_signal, freq="1h")

    def test_sparse_signal_index_is_allowed(self, price_data: pd.DataFrame):
        # A signal that only fires on a subset of bars is legitimate: reindex fills gaps flat.
        def sparse_signal(data: pd.DataFrame) -> pd.Series:
            idx = data.index[::2]
            return pd.Series(np.tile([1.0, -1.0], len(idx))[: len(idx)], index=idx)

        pf = signal_to_portfolio(price_data, sparse_signal, freq="1h")
        assert isinstance(pf, vbt.Portfolio)

    def test_missing_close_column_raises(self, price_data: pd.DataFrame):
        def sig(data: pd.DataFrame) -> pd.Series:
            return pd.Series(1.0, index=data.index)

        with pytest.raises(ValueError, match="close"):
            signal_to_portfolio(price_data.drop(columns=["close"]), sig, freq="1h")

    def test_non_series_return_raises(self, price_data: pd.DataFrame):
        with pytest.raises(TypeError, match="pandas Series"):
            signal_to_portfolio(price_data, lambda data: np.ones(len(data)), freq="1h")

    def test_aligned_signal_builds_portfolio(self, price_data: pd.DataFrame):
        def ma_signal(data: pd.DataFrame) -> pd.Series:
            fast = data["close"].rolling(3).mean()
            slow = data["close"].rolling(10).mean()
            return pd.Series(np.where(fast > slow, 1.0, -1.0), index=data.index)

        pf = signal_to_portfolio(price_data, ma_signal, freq="1h")
        assert isinstance(pf, vbt.Portfolio)
        assert len(pf.trades.records_readable) > 0
