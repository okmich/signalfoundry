"""Shared fixtures for okmich_quant_crypto unit tests (no network)."""
from __future__ import annotations

import pytest

from okmich_quant_core import BaseSignal, RunnerIdentity
from okmich_quant_crypto import BaseCryptoStrategy, CryptoStrategyConfig, CryptoVenueConfig, VenueContext
from okmich_quant_crypto.venue import registry
from okmich_quant_crypto.venue.base import VenueProfile

from .fakes import MIN, T0, FakeClock, FakeExchange, RecordingLogger, RecordingNotifier


@pytest.fixture(autouse=True)
def _ops_log_base(tmp_path_factory, monkeypatch):
    """Keep the fail-closed default inference logger out of the production log root."""
    monkeypatch.setenv("OKMICH_QUANT_LOG_BASE", str(tmp_path_factory.mktemp("quant_logs")))


class FakeVenueProfile(VenueProfile):
    """The base profile under the test-only id ``fakex`` (the id of ``FakeExchange``)."""


@pytest.fixture(autouse=True)
def _fake_venue_supported(monkeypatch):
    """Put the test venue on the supported list for every test (and only for the test)."""
    monkeypatch.setitem(registry._SUPPORTED, "fakex", FakeVenueProfile)


@pytest.fixture
def clock():
    return FakeClock(T0 + 30 * 5 * MIN)


@pytest.fixture
def exchange(clock):
    return FakeExchange(clock)


@pytest.fixture
def venue(tmp_path):
    return make_venue(tmp_path)


def make_venue(tmp_path, **overrides) -> CryptoVenueConfig:
    data = {"exchange_id": "fakex", "environment": "demo", "state_dir": str(tmp_path / "state")}
    data.update(overrides)
    return CryptoVenueConfig(**data)


def make_cfg(**overrides) -> CryptoStrategyConfig:
    data = {"name": "s1", "market_symbol": "BTC/USDT:USDT", "market_type": "linear_perp", "timeframe": "5m",
            "magic": 42, "bars_to_copy": 5}
    data.update(overrides)
    return CryptoStrategyConfig(**data)


def seed_candles(exchange: FakeExchange, symbol: str, timeframe_ms: int, last_closed_open: int, count: int,
                 timeframe: str = "5m", start_price: float = 100.0) -> None:
    rows = []
    for i in range(count):
        ts = last_closed_open - (count - 1 - i) * timeframe_ms
        p = start_price + i
        rows.append([ts, p, p + 2, p - 2, p + 1, 10.0 + i])
    exchange.add_candles(symbol, timeframe, rows)


class LongEverySignal(BaseSignal):
    """Deterministic signal: long entry when the last close is above the last open, exit otherwise."""

    def generate(self, data, *args):
        import numpy as np
        up = (data["close"] > data["open"]).to_numpy()
        self.last_ctx = {"direction": 1 if up[-1] else 0, "bar_close": float(data["close"].iloc[-1]),
                         "features": {"close": float(data["close"].iloc[-1])}}
        return up.astype(int), (~up).astype(int), np.zeros(len(up), int), np.zeros(len(up), int)

    def get_signal_context(self):
        return getattr(self, "last_ctx", None)


class RecordingStrategy(BaseCryptoStrategy):
    """Records every cycle; optionally fails."""

    def __init__(self, cfg, signal=None, **kwargs):
        super().__init__(cfg, signal or LongEverySignal(), **kwargs)
        self.cycles: list = []
        self.fail = False

    async def on_new_bar(self):
        bars = self.fetch_price_bars()
        self.cycles.append((self.latest_run_dt, float(bars["close"].iloc[-1]) if len(bars) else None))
        if self.fail:
            raise RuntimeError("crypto boom")
        if self.signal_generator is not None:
            self.signal_generator.generate(bars)


def bind(strategy) -> None:
    strategy.bind_runner_identity(RunnerIdentity(runner_id="c-1", runner_start_token="tok", broker="fakex:demo",
                                                 account_id="key-abc", broker_session_id="s1"))


def make_strategy(cfg=None, *, cls=RecordingStrategy, notifier=None, max_consecutive_errors=5):
    rec = RecordingLogger()
    s = cls(cfg or make_cfg(), notifier=notifier, inference_logger=rec, max_consecutive_errors=max_consecutive_errors)
    bind(s)
    return s, rec


async def bootstrap(strategy, exchange, venue, clock, profile: VenueProfile | None = None):
    profile = profile or FakeVenueProfile(venue.exchange_id)
    ctx = VenueContext(exchange=exchange, profile=profile, venue=venue, clock=clock, sleep=clock.sleep)
    await strategy._bootstrap(ctx)
    return strategy


__all__ = ["make_venue", "make_cfg", "seed_candles", "LongEverySignal", "RecordingStrategy", "bind", "make_strategy",
           "bootstrap", "RecordingNotifier"]
