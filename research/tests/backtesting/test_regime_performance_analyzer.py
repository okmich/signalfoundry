"""Tests for RegimePerformanceAnalyzer and the thin convenience methods
added to VectorBtBacktester / VectobtWalkForwardBacktester."""
import shutil
from typing import Tuple

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from okmich_quant_research.backtesting.regime_performance_analyzer import (
    RegimePerformanceAnalyzer,
)
from okmich_quant_research.backtesting.vectorbt_backtester import VectorBtBacktester


# ---------------------------------------------------------------------------
# helpers / shared fixtures
# ---------------------------------------------------------------------------

N_BARS = 200
RNG = np.random.default_rng(42)


def _make_price_data(n: int = N_BARS) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = (RNG.standard_normal(n).cumsum() + 100).clip(min=10)
    high = close * (1 + RNG.uniform(0, 0.01, n))
    low = close * (1 - RNG.uniform(0, 0.01, n))
    return pd.DataFrame(
        {
            "open": close * (1 + RNG.standard_normal(n) * 0.002),
            "high": high,
            "low": low,
            "close": close,
            "volume": RNG.integers(1_000, 10_000, n),
        },
        index=dates,
    )


@pytest.fixture(scope="module")
def sample_price_data() -> pd.DataFrame:
    return _make_price_data()


@pytest.fixture(scope="module")
def sample_regime_labels(sample_price_data: pd.DataFrame) -> pd.Series:
    """Three regimes cycling over the data."""
    n = len(sample_price_data)
    labels = np.tile([0, 1, 2], n // 3 + 1)[:n]
    return pd.Series(labels, index=sample_price_data.index, dtype=int)


@pytest.fixture(scope="module")
def sample_portfolio(sample_price_data: pd.DataFrame) -> vbt.Portfolio:
    """Simple MA-crossover portfolio for testing."""
    close = sample_price_data["close"]
    fast_ma = close.rolling(5).mean()
    slow_ma = close.rolling(20).mean()
    entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    return vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=10_000,
        fees=0.001,
        freq="D",
    )


@pytest.fixture(scope="module")
def analyzer(
    sample_portfolio: vbt.Portfolio,
    sample_regime_labels: pd.Series,
) -> RegimePerformanceAnalyzer:
    return RegimePerformanceAnalyzer(sample_portfolio, sample_regime_labels)


# ---------------------------------------------------------------------------
# construction / validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_creates_correctly(self, analyzer: RegimePerformanceAnalyzer):
        assert analyzer is not None
        assert len(analyzer._regimes) == 3

    def test_raises_on_no_overlap(self, sample_portfolio: vbt.Portfolio):
        future_labels = pd.Series(
            [0, 1, 2],
            index=pd.date_range("2099-01-01", periods=3, freq="D"),
            dtype=int,
        )
        with pytest.raises(ValueError, match="no overlap"):
            RegimePerformanceAnalyzer(sample_portfolio, future_labels)

    def test_regime_names_stored(
        self, sample_portfolio: vbt.Portfolio, sample_regime_labels: pd.Series
    ):
        names = {0: "trending", 1: "ranging", 2: "volatile"}
        rpa = RegimePerformanceAnalyzer(
            sample_portfolio, sample_regime_labels, regime_names=names
        )
        assert rpa._display_name(0) == "trending"
        assert rpa._display_name(99) == "99"


# ---------------------------------------------------------------------------
# regime_return_stats
# ---------------------------------------------------------------------------


class TestRegimeReturnStats:
    def test_shape(self, analyzer: RegimePerformanceAnalyzer):
        df = analyzer.regime_return_stats()
        assert df.shape[0] == 3, "one row per unique regime"
        expected_cols = {
            "n_bars", "total_return", "ann_return",
            "volatility", "sharpe", "sortino", "max_drawdown",
        }
        assert expected_cols.issubset(df.columns)

    def test_n_bars_coverage(self, analyzer: RegimePerformanceAnalyzer):
        df = analyzer.regime_return_stats()
        # All bars must be accounted for
        assert df["n_bars"].sum() == N_BARS

    def test_index_uses_regime_names(
        self, sample_portfolio: vbt.Portfolio, sample_regime_labels: pd.Series
    ):
        names = {0: "trending", 1: "ranging", 2: "volatile"}
        rpa = RegimePerformanceAnalyzer(
            sample_portfolio, sample_regime_labels, regime_names=names
        )
        df = rpa.regime_return_stats()
        assert "trending" in df.index
        assert "ranging" in df.index
        assert "volatile" in df.index


# ---------------------------------------------------------------------------
# regime_trade_stats
# ---------------------------------------------------------------------------


class TestRegimeTradeStats:
    def test_shape(self, analyzer: RegimePerformanceAnalyzer):
        df = analyzer.regime_trade_stats()
        assert df.shape[0] == 3, "one row per unique regime"
        expected_cols = {
            "n_trades", "win_rate", "avg_win", "avg_loss",
            "profit_factor", "expectancy", "avg_duration_bars",
        }
        assert expected_cols.issubset(df.columns)

    def test_total_trades_consistent(self, analyzer: RegimePerformanceAnalyzer):
        df = analyzer.regime_trade_stats()
        total_trades = len(analyzer._trades)
        assert df["n_trades"].sum() == total_trades

    def test_win_rate_range(self, analyzer: RegimePerformanceAnalyzer):
        df = analyzer.regime_trade_stats()
        valid = df["win_rate"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()


# ---------------------------------------------------------------------------
# regime_exposure
# ---------------------------------------------------------------------------


class TestRegimeExposure:
    def test_shape(self, analyzer: RegimePerformanceAnalyzer):
        df = analyzer.regime_exposure()
        assert df.shape[0] == 3
        expected_cols = {"n_bars", "pct_time", "n_bars_in_position", "pct_time_in_position"}
        assert expected_cols.issubset(df.columns)

    def test_n_bars_sums_to_total(self, analyzer: RegimePerformanceAnalyzer):
        df = analyzer.regime_exposure()
        assert df["n_bars"].sum() == N_BARS

    def test_pct_time_sums_to_one(self, analyzer: RegimePerformanceAnalyzer):
        df = analyzer.regime_exposure()
        assert abs(df["pct_time"].sum() - 1.0) < 1e-9

    def test_pct_time_in_position_range(self, analyzer: RegimePerformanceAnalyzer):
        df = analyzer.regime_exposure()
        valid = df["pct_time_in_position"].dropna()
        assert (valid >= 0).all() and (valid <= 1.0001).all()


# ---------------------------------------------------------------------------
# generate_full_report
# ---------------------------------------------------------------------------


class TestGenerateFullReport:
    def test_keys(self, analyzer: RegimePerformanceAnalyzer):
        report = analyzer.generate_full_report()
        assert set(report.keys()) == {"return_stats", "trade_stats", "exposure"}

    def test_all_are_dataframes(self, analyzer: RegimePerformanceAnalyzer):
        report = analyzer.generate_full_report()
        for v in report.values():
            assert isinstance(v, pd.DataFrame)


# ---------------------------------------------------------------------------
# plot methods (just check they run without raising)
# ---------------------------------------------------------------------------


class TestPlots:
    def test_plot_regime_summary_runs(self, analyzer: RegimePerformanceAnalyzer):
        fig = analyzer.plot_regime_summary()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_regime_returns_runs(self, analyzer: RegimePerformanceAnalyzer):
        fig = analyzer.plot_regime_returns()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_regime_trade_distribution_runs(self, analyzer: RegimePerformanceAnalyzer):
        fig = analyzer.plot_regime_trade_distribution()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_saves_to_file(self, analyzer: RegimePerformanceAnalyzer, tmp_path):
        out = tmp_path / "summary.png"
        fig = analyzer.plot_regime_summary(save_path=str(out))
        import matplotlib.pyplot as plt
        plt.close(fig)
        assert out.exists()


# ---------------------------------------------------------------------------
# VectorBtBacktester.analyze_by_regime
# ---------------------------------------------------------------------------

class _SimpleMACrossSignal:
    """Minimal signal compatible with VectorBtBacktester."""

    def __init__(self, fast: int = 5, slow: int = 20):
        self.fast = fast
        self.slow = slow

    def generate(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        close = data["close"]
        fa = close.rolling(self.fast).mean()
        sl = close.rolling(self.slow).mean()
        entries = (fa > sl) & (fa.shift(1) <= sl.shift(1))
        exits = (fa < sl) & (fa.shift(1) >= sl.shift(1))
        n = len(close)
        short_entries = pd.Series(False, index=close.index)
        short_exits = pd.Series(False, index=close.index)
        return entries, exits, short_entries, short_exits


class TestVectorBtBacktesterAnalyzeByRegime:
    def test_returns_analyzer(
        self,
        sample_price_data: pd.DataFrame,
        sample_regime_labels: pd.Series,
    ):
        bt = VectorBtBacktester(_SimpleMACrossSignal(), timeframe="1D")
        bt.run_backtest(sample_price_data, initial_capital=10_000, fees=0.001)
        rpa = bt.analyze_by_regime(sample_regime_labels)
        assert isinstance(rpa, RegimePerformanceAnalyzer)

    def test_raises_before_backtest(
        self,
        sample_regime_labels: pd.Series,
    ):
        bt = VectorBtBacktester(_SimpleMACrossSignal(), timeframe="1D")
        with pytest.raises(ValueError, match="Run backtest first"):
            bt.analyze_by_regime(sample_regime_labels)

    def test_regime_names_forwarded(
        self,
        sample_price_data: pd.DataFrame,
        sample_regime_labels: pd.Series,
    ):
        names = {0: "bull", 1: "bear", 2: "sideways"}
        bt = VectorBtBacktester(_SimpleMACrossSignal(), timeframe="1D")
        bt.run_backtest(sample_price_data, initial_capital=10_000)
        rpa = bt.analyze_by_regime(sample_regime_labels, regime_names=names)
        assert rpa.regime_names == names
