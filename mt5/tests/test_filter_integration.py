"""
Tests for filter factory (create_filter) and strategy integration.
"""

import pytest
from datetime import datetime, time
from unittest.mock import Mock, patch, MagicMock
from okmich_quant_core.config import StrategyConfig, FilterConfig
from okmich_quant_mt5.filters import create_filter, FilterChain


class TestFilterFactory:
    """Test create_filter factory function."""

    def test_create_empty_filter_chain(self):
        """Test creating filter chain with no filters."""
        config = StrategyConfig(
            name="TestStrategy",
            symbol="EURUSD",
            timeframe=5,
            magic=12345,
            filters=[],
        )

        chain = create_filter(config)
        assert isinstance(chain, FilterChain)
        assert len(chain.filters) == 0

    def test_create_datetime_filter(self):
        """Test creating datetime filter from config."""
        config = StrategyConfig(
            name="TestStrategy",
            symbol="EURUSD",
            timeframe=5,
            magic=12345,
            filters=[
                FilterConfig(
                    type="datetime",
                    params={
                        "name": "TradingHours",
                        "allowed_days": [0, 1, 2, 3, 4],
                        "allowed_time_ranges": [
                            {"from": "09:00", "to": "17:00"}
                        ],
                    },
                )
            ],
        )

        chain = create_filter(config)
        assert len(chain.filters) == 1
        assert chain.filters[0].name == "TradingHours"

        # Test that filter works
        context_pass = {"datetime": datetime(2025, 2, 12, 10, 0)}
        assert chain(context_pass) is True

        context_fail = {"datetime": datetime(2025, 2, 12, 20, 0)}
        assert chain(context_fail) is False

    def test_create_spread_filter(self):
        """Test creating spread filter from config."""
        config = StrategyConfig(
            name="TestStrategy",
            symbol="EURUSD",
            timeframe=5,
            magic=12345,
            filters=[
                FilterConfig(
                    type="spread",
                    params={"name": "SpreadCheck", "max_spread_points": 50},
                )
            ],
        )

        chain = create_filter(config)
        assert len(chain.filters) == 1
        assert chain.filters[0].name == "SpreadCheck"

        # Test that filter works
        context_pass = {"spread": 30}
        assert chain(context_pass) is True

        context_fail = {"spread": 100}
        assert chain(context_fail) is False

    def test_create_max_positions_filter(self):
        """Test creating max positions filter from config."""
        config = StrategyConfig(
            name="TestStrategy",
            symbol="EURUSD",
            timeframe=5,
            magic=12345,
            filters=[
                FilterConfig(
                    type="max_positions",
                    params={"name": "MaxPos", "max_positions": 3},
                )
            ],
        )

        chain = create_filter(config)
        assert len(chain.filters) == 1
        assert chain.filters[0].name == "MaxPos"

        # Test that filter works
        context_pass = {"open_positions": 2}
        assert chain(context_pass) is True

        context_fail = {"open_positions": 3}
        assert chain(context_fail) is False

    def test_create_multiple_filters(self):
        """Test creating chain with multiple filters."""
        config = StrategyConfig(
            name="TestStrategy",
            symbol="EURUSD",
            timeframe=5,
            magic=12345,
            filters=[
                FilterConfig(
                    type="datetime",
                    params={
                        "allowed_days": [0, 1, 2, 3, 4],
                        "allowed_time_ranges": [{"from": "09:00", "to": "17:00"}],
                    },
                ),
                FilterConfig(
                    type="spread",
                    params={"max_spread_points": 50},
                ),
                FilterConfig(
                    type="max_positions",
                    params={"max_positions": 3},
                ),
            ],
        )

        chain = create_filter(config)
        assert len(chain.filters) == 3

        # Test that all filters are applied
        context_all_pass = {
            "datetime": datetime(2025, 2, 12, 10, 0),
            "spread": 30,
            "open_positions": 2,
        }
        assert chain(context_all_pass) is True

        # Fail on datetime
        context_fail_dt = {
            "datetime": datetime(2025, 2, 15, 10, 0),  # Saturday
            "spread": 30,
            "open_positions": 2,
        }
        assert chain(context_fail_dt) is False

    def test_unknown_filter_type_raises_error(self):
        """Test that unknown filter type raises ValueError."""
        config = StrategyConfig(
            name="TestStrategy",
            symbol="EURUSD",
            timeframe=5,
            magic=12345,
            filters=[
                FilterConfig(
                    type="unknown_filter_type",
                    params={},
                )
            ],
        )

        with pytest.raises(ValueError, match="unknown filter type"):
            create_filter(config)

    def test_filter_without_name_uses_default(self):
        """Test that filters without custom names get default names."""
        config = StrategyConfig(
            name="TestStrategy",
            symbol="EURUSD",
            timeframe=5,
            magic=12345,
            filters=[
                FilterConfig(
                    type="spread",
                    params={"max_spread_points": 50},  # No name provided
                )
            ],
        )

        chain = create_filter(config)
        # Should use default name "SpreadFilter"
        assert chain.filters[0].name == "SpreadFilter"


@pytest.fixture
def mock_strategy_dependencies():
    """Mock all external dependencies for BaseMt5Strategy."""
    with patch("okmich_quant_mt5.strategy.fetch_symbol_info") as mock_symbol_info, \
         patch("okmich_quant_mt5.strategy.PriceBuffer") as mock_price_buffer, \
         patch("okmich_quant_mt5.strategy.get_position_manager") as mock_pm, \
         patch("okmich_quant_mt5.strategy.number_of_minutes_in_timeframe") as mock_tf:

        # Setup return values
        mock_symbol_info.return_value = {
            "symbol": "EURUSD",
            "point": 0.00001,
            "filling_mode": 1,
        }
        mock_tf.return_value = 5
        mock_pm.return_value = None  # No position manager

        yield {
            "symbol_info": mock_symbol_info,
            "price_buffer": mock_price_buffer,
            "position_manager": mock_pm,
            "timeframe": mock_tf,
        }


@pytest.fixture
def mock_signal():
    """Mock signal generator."""
    signal = Mock()
    return signal


class TestStrategyFilterIntegration:
    """Test filter integration with BaseMt5Strategy."""

    def test_strategy_initializes_filter_chain(self, mock_strategy_dependencies, mock_signal):
        """Test that strategy initializes filter chain from config."""
        from okmich_quant_mt5.strategy import BaseMt5Strategy

        config = StrategyConfig(
            name="TestStrategy",
            symbol="EURUSD",
            timeframe=5,
            magic=12345,
            filters=[
                FilterConfig(
                    type="spread",
                    params={"max_spread_points": 50},
                )
            ],
        )

        # Create a concrete strategy class for testing
        class TestStrategy(BaseMt5Strategy):
            def on_new_bar(self):
                pass

        strategy = TestStrategy(config, mock_signal)

        assert hasattr(strategy, "filter_chain")
        assert isinstance(strategy.filter_chain, FilterChain)
        assert len(strategy.filter_chain.filters) == 1

    def test_strategy_empty_filters(self, mock_strategy_dependencies, mock_signal):
        """Test strategy with no filters configured."""
        from okmich_quant_mt5.strategy import BaseMt5Strategy

        config = StrategyConfig(
            name="TestStrategy",
            symbol="EURUSD",
            timeframe=5,
            magic=12345,
            filters=[],  # Empty filters
        )

        class TestStrategy(BaseMt5Strategy):
            def on_new_bar(self):
                pass

        strategy = TestStrategy(config, mock_signal)

        assert hasattr(strategy, "filter_chain")
        assert len(strategy.filter_chain.filters) == 0

        # Empty chain should always pass
        context = {}
        assert strategy.filter_chain(context) is True


class TestGenericBasicStrategyFilterIntegration:
    """Test filter integration with GenericBasicStrategy.on_new_bar()."""

    @pytest.fixture
    def mock_all_mt5_functions(self):
        """Mock all MT5 functions used in GenericBasicStrategy."""
        with patch("okmich_quant_mt5.strategy.fetch_symbol_info") as mock_symbol_info, \
             patch("okmich_quant_mt5.strategy.PriceBuffer"), \
             patch("okmich_quant_mt5.strategy.get_position_manager"), \
             patch("okmich_quant_mt5.strategy.number_of_minutes_in_timeframe") as mock_tf, \
             patch("okmich_quant_mt5.strategy.get_positions") as mock_get_pos, \
             patch("okmich_quant_mt5.strategy.open_position") as mock_open_pos:

            mock_symbol_info.return_value = {
                "symbol": "EURUSD",
                "point": 0.00001,
                "filling_mode": 1,
            }
            mock_tf.return_value = 5
            mock_get_pos.return_value = []  # No open positions

            yield {
                "get_positions": mock_get_pos,
                "open_position": mock_open_pos,
            }

    def test_filters_block_trade_on_high_spread(self, mock_all_mt5_functions):
        """Test that high spread blocks trade entry."""
        from okmich_quant_mt5.strategy import GenericBasicStrategy
        import numpy as np

        config = StrategyConfig(
            name="TestStrategy",
            symbol="EURUSD",
            timeframe=5,
            magic=12345,
            filters=[
                FilterConfig(
                    type="spread",
                    params={"max_spread_points": 50},
                )
            ],
        )

        # Mock signal that returns buy signal
        mock_signal = Mock()
        mock_signal.generate.return_value = (
            np.array([0, 0, 1]),  # entries_long
            np.array([0, 0, 0]),  # exits_long
            np.array([0, 0, 0]),  # entries_short
            np.array([0, 0, 0]),  # exits_short
        )

        strategy = GenericBasicStrategy(config, mock_signal)

        # Mock fetch methods
        strategy.fetch_ohlcv = Mock(return_value=Mock())  # Mock OHLCV data
        strategy.fetch_latest_tick_info = Mock(return_value={
            "ask": 1.1000,
            "bid": 1.0999,
            "spread": 100  # High spread - should block
        })
        strategy.latest_run_dt = datetime(2025, 2, 12, 10, 0)

        # Execute strategy
        strategy.on_new_bar()

        # open_position should NOT be called due to filter blocking
        mock_all_mt5_functions["open_position"].assert_not_called()

    def test_filters_allow_trade_on_low_spread(self, mock_all_mt5_functions):
        """Test that low spread allows trade entry."""
        from okmich_quant_mt5.strategy import GenericBasicStrategy
        import numpy as np

        config = StrategyConfig(
            name="TestStrategy",
            symbol="EURUSD",
            timeframe=5,
            magic=12345,
            filters=[
                FilterConfig(
                    type="spread",
                    params={"max_spread_points": 50},
                )
            ],
        )

        # Mock signal that returns buy signal
        mock_signal = Mock()
        mock_signal.generate.return_value = (
            np.array([0, 0, 1]),  # entries_long
            np.array([0, 0, 0]),  # exits_long
            np.array([0, 0, 0]),  # entries_short
            np.array([0, 0, 0]),  # exits_short
        )

        strategy = GenericBasicStrategy(config, mock_signal)

        # Mock fetch methods
        strategy.fetch_ohlcv = Mock(return_value=Mock())
        strategy.fetch_latest_tick_info = Mock(return_value={
            "ask": 1.1000,
            "bid": 1.0999,
            "spread": 30  # Low spread - should allow
        })
        strategy.open_position = Mock(return_value=True)
        strategy.latest_run_dt = datetime(2025, 2, 12, 10, 0)

        # Execute strategy
        strategy.on_new_bar()

        # open_position SHOULD be called because filters passed
        strategy.open_position.assert_called_once()

    def test_filters_block_trade_on_weekend(self, mock_all_mt5_functions):
        """Test that weekend datetime blocks trade entry."""
        from okmich_quant_mt5.strategy import GenericBasicStrategy
        import numpy as np

        config = StrategyConfig(
            name="TestStrategy",
            symbol="EURUSD",
            timeframe=5,
            magic=12345,
            filters=[
                FilterConfig(
                    type="datetime",
                    params={
                        "allowed_days": [0, 1, 2, 3, 4],  # Mon-Fri
                        "allowed_time_ranges": [{"from": "09:00", "to": "17:00"}],
                    },
                )
            ],
        )

        mock_signal = Mock()
        mock_signal.generate.return_value = (
            np.array([0, 0, 1]),  # Buy signal
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
        )

        strategy = GenericBasicStrategy(config, mock_signal)

        strategy.fetch_ohlcv = Mock(return_value=Mock())
        strategy.fetch_latest_tick_info = Mock(return_value={
            "ask": 1.1000,
            "bid": 1.0999,
            "spread": 30
        })
        # Saturday - should block
        strategy.latest_run_dt = datetime(2025, 2, 15, 10, 0)

        strategy.on_new_bar()

        # Should not open position due to weekend filter
        mock_all_mt5_functions["open_position"].assert_not_called()

    def test_no_filters_always_allows_trade(self, mock_all_mt5_functions):
        """Test that strategy with no filters always allows trades."""
        from okmich_quant_mt5.strategy import GenericBasicStrategy
        import numpy as np

        config = StrategyConfig(
            name="TestStrategy",
            symbol="EURUSD",
            timeframe=5,
            magic=12345,
            filters=[],  # No filters
        )

        mock_signal = Mock()
        mock_signal.generate.return_value = (
            np.array([0, 0, 1]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
        )

        strategy = GenericBasicStrategy(config, mock_signal)

        strategy.fetch_ohlcv = Mock(return_value=Mock())
        strategy.fetch_latest_tick_info = Mock(return_value={
            "ask": 1.1000,
            "bid": 1.0999,
            "spread": 1000  # Even with huge spread, no filter to block
        })
        strategy.open_position = Mock(return_value=True)
        strategy.latest_run_dt = datetime(2025, 2, 15, 3, 0)  # Sunday 3 AM

        strategy.on_new_bar()

        # Should open position because no filters
        strategy.open_position.assert_called_once()