"""Crypto config semantics and the config-time strategy-isolation rule."""
import json

import pytest
from pydantic import ValidationError

from okmich_quant_core import StrategyConfig
from okmich_quant_crypto import CryptoEventLoop, CryptoStrategyConfig, CryptoSystemConfig, CryptoVenueConfig
from okmich_quant_crypto.config import derive_log_symbol

from .conftest import RecordingStrategy, make_cfg, make_venue


def test_is_still_a_core_strategy_config():
    cfg = make_cfg()
    assert isinstance(cfg, StrategyConfig)


def test_symbol_derived_from_ccxt_symbol_is_core_safe():
    cfg = make_cfg()
    assert cfg.market_symbol == "BTC/USDT:USDT"
    assert cfg.symbol == "BTC/USDT-USDT" == derive_log_symbol("BTC/USDT:USDT")
    spot = make_cfg(market_symbol="BTC/USDT", market_type="spot")
    assert spot.symbol == "BTC/USDT"


def test_explicit_symbol_with_colon_is_rejected():
    with pytest.raises(ValidationError, match="reserved|core's log identity"):
        make_cfg(symbol="BTC/USDT:USDT")


def test_explicit_safe_symbol_is_kept():
    assert make_cfg(symbol="BTCUSDT.P").symbol == "BTCUSDT.P"


@pytest.mark.parametrize("overrides,msg", [
    ({"market_symbol": "BTC/USDT"}, "settle currency"),
    ({"market_type": "spot"}, "derivatives symbol"),
    ({"market_symbol": "BTC/USDT", "market_type": "spot", "leverage": 3}, "leverage is not valid"),
    ({"market_symbol": "BTC/USDT", "market_type": "spot", "sizing_unit": "contracts"}, "perpetuals only"),
    ({"max_number_of_open_positions": 2}, "max_number_of_open_positions must be 1"),
    ({"timeframe": "1w"}, "outside 1m..1d"),
    ({"timeframe": 5}, "string"),
    ({"leverage": 0}, "leverage must be > 0"),
    ({"close_grace_seconds": 10, "close_max_wait_seconds": 5}, "greater than close_grace_seconds"),
    ({"position_manager": {"type": "fixed_point", "sl": 100, "tp": 200}}, "point_size"),
    ({"stop_mode": "managed", "feed_mode": "poll", "managed_stop_poll_seconds": 10}, "managed_stop_poll_seconds"),
    ({"unexpected": 1}, "Extra inputs"),
])
def test_crypto_semantics_validators(overrides, msg):
    with pytest.raises(ValidationError, match=msg):
        make_cfg(**overrides)


def test_slow_managed_stops_allowed_when_explicit():
    cfg = make_cfg(stop_mode="managed", feed_mode="poll", managed_stop_poll_seconds=10, allow_slow_managed_stops=True)
    assert cfg.managed_stop_poll_seconds == 10


def test_point_manager_with_point_size_is_accepted():
    cfg = make_cfg(position_manager={"type": "fixed_point", "sl": 100, "tp": 200, "point_size": 0.1})
    assert cfg.position_manager.point_size == 0.1


def test_venue_requires_environment_and_known_exchange(tmp_path):
    with pytest.raises(ValidationError, match="environment"):
        CryptoVenueConfig(exchange_id="bybit")
    with pytest.raises(ValidationError, match="not a supported exchange"):
        make_venue(tmp_path, exchange_id="okx")
    with pytest.raises(ValidationError, match="environment variable NAME"):
        make_venue(tmp_path, api_key_env="sk-123 secret")
    assert make_venue(tmp_path, exchange_id="  ByBit ").exchange_id == "bybit"


def _system(tmp_path, strategies):
    return {"name": "sys", "venue": {"exchange_id": "bybit", "environment": "demo", "state_dir": str(tmp_path)},
            "strategies": strategies}


def _s(name, magic, symbol="BTC/USDT:USDT", market_type="linear_perp"):
    return {"name": name, "magic": magic, "market_symbol": symbol, "market_type": market_type, "timeframe": "5m"}


def test_isolation_one_strategy_per_market(tmp_path):
    with pytest.raises(ValidationError, match="nets them into one position"):
        CryptoSystemConfig(**_system(tmp_path, [_s("a", 1), _s("b", 2)]))


def test_spot_and_perp_on_the_same_base_are_separate_markets(tmp_path):
    cfg = CryptoSystemConfig(**_system(tmp_path, [_s("a", 1), _s("b", 2, "BTC/USDT", "spot")]))
    assert len(cfg.all_strategies()) == 2


def test_isolation_names_and_magics_unique(tmp_path):
    with pytest.raises(ValidationError, match="share magic"):
        CryptoSystemConfig(**_system(tmp_path, [_s("a", 1), _s("b", 1, "ETH/USDT:USDT")]))
    with pytest.raises(ValidationError, match="duplicate strategy name"):
        CryptoSystemConfig(**_system(tmp_path, [_s("a", 1), _s("a", 2, "ETH/USDT:USDT")]))


def test_strategy_xor_strategies(tmp_path):
    data = _system(tmp_path, [])
    with pytest.raises(ValidationError, match="Must provide"):
        CryptoSystemConfig(**data)
    data = _system(tmp_path, [_s("a", 1)])
    data["strategy"] = _s("b", 2, "ETH/USDT:USDT")
    with pytest.raises(ValidationError, match="Cannot provide both"):
        CryptoSystemConfig(**data)


def test_load_from_file(tmp_path):
    path = tmp_path / "system.json"
    path.write_text(json.dumps(_system(tmp_path, [_s("a", 1)])), encoding="utf-8")
    cfg = CryptoSystemConfig.load_from_file(path)
    assert isinstance(cfg.all_strategies()[0], CryptoStrategyConfig)


def test_event_loop_enforces_isolation_before_connecting(tmp_path):
    venue = make_venue(tmp_path)
    loop = CryptoEventLoop(venue)
    loop.add_strategy(RecordingStrategy(make_cfg(name="a", magic=1)))
    with pytest.raises(ValueError, match="nets them into one position"):
        loop.add_strategy(RecordingStrategy(make_cfg(name="b", magic=2)))
    assert loop.exchange is None  # nothing connected


#: A complete two-strategy system, as in signalfoundry-lab/examples/crypto_examples/system.bybit.demo.json.
FULL_SYSTEM = {
    "name": "btc_trend_bybit_demo",
    "venue": {"exchange_id": "bybit", "environment": "demo", "sub_account": "main", "api_key_env": "BYBIT_API_KEY",
              "secret_env": "BYBIT_API_SECRET", "margin_mode": None, "state_dir": ".crypto_state"},
    "strategies": [
        {"name": "btc_trend_5m", "market_symbol": "BTC/USDT:USDT", "market_type": "linear_perp", "timeframe": "5m",
         "magic": 5001, "bars_to_copy": 300, "feed_mode": "stream", "stop_mode": "auto", "stop_trigger": "last",
         "leverage": 3, "sizing_unit": "base_qty", "position_sizing": {"type": "risk_pct_of_equity", "risk_pct": 0.003},
         "position_manager": {"type": "fixed_atr_with_trailing", "sl": 2.0, "tp": 4.0, "trailing": 2.0,
                              "atr_period": 14},
         "filters": [{"type": "spread", "params": {"max_spread_pct": 0.0005}}], "signal_params": {}},
        {"name": "eth_spot_15m", "market_symbol": "ETH/USDT", "market_type": "spot", "timeframe": "15m", "magic": 5002,
         "bars_to_copy": 200, "feed_mode": "poll", "stop_mode": "auto", "sizing_unit": "quote_notional",
         "position_sizing": {"type": "fixed", "units": 50},
         "position_manager": {"type": "fixed_percent", "sl": 2.0, "tp": 4.0}, "signal_params": {}},
    ],
}


def test_full_system_config_is_valid(tmp_path):
    path = tmp_path / "system.json"
    path.write_text(json.dumps(FULL_SYSTEM), encoding="utf-8")
    cfg = CryptoSystemConfig.load_from_file(path)
    assert cfg.venue.exchange_id == "bybit" and cfg.venue.environment.value == "demo"
    perp, spot = cfg.all_strategies()
    assert perp.symbol == "BTC/USDT-USDT" and spot.symbol == "ETH/USDT"
