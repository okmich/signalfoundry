import MetaTrader5 as mt5

from okmich_quant_core.config import PositionManagerConfig, StrategyConfig


def with_strategy_config(
    position_manager_config: PositionManagerConfig,
) -> StrategyConfig:
    return StrategyConfig(
        name="TestSystem",
        symbol="EURUSD",
        timeframe=mt5.TIMEFRAME_H1,
        magic=12345,
        signal_params={},
        position_manager=position_manager_config,
    )
