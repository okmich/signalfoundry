"""Crypto configuration, layered on core without changing it.

* :class:`CryptoStrategyConfig` subclasses ``core.StrategyConfig`` so ``BaseStrategy``, signals, filters and position
  managers accept it unchanged, and adds the per-strategy crypto fields.
* :class:`CryptoVenueConfig` holds what is shared by every strategy on one exchange account (credentials, environment,
  account-wide margin mode).
* :class:`CryptoSystemConfig` is the JSON root. It is NOT a ``core.SystemConfig``: that model requires a polled
  ``runloop`` the event-driven crypto runner does not have.

Symbols: core's log identity rejects ``:`` in a symbol (a reserved path character on Windows), but every CCXT perpetual
symbol has one (``BTC/USDT:USDT``). So the CCXT symbol lives in ``market_symbol`` and ``symbol`` - the token core logs
under - is derived from it with ``:`` replaced by ``-`` unless given explicitly.
"""
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from okmich_quant_core import PositionManagerType, StrategyConfig

from .enums import FeedMode, MarginMode, MarketType, PositionMode, SizingUnit, StopMode, StopTrigger, VenueEnvironment
from .timeframe_utils import timeframe_to_minutes
from .venue.registry import is_supported, supported_venues

#: Managed stops need prices between bars. Polling slower than this is too coarse to call a stop.
MAX_MANAGED_STOP_POLL_SECONDS = 5.0

_POINT_TYPES = {PositionManagerType.FIXED_POINT, PositionManagerType.FIXED_POINT_WITH_TRAILING,
                PositionManagerType.FIXED_POINT_WITH_BREAK_EVEN, PositionManagerType.DYNAMIC_POINT}
_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def derive_log_symbol(market_symbol: str) -> str:
    """The core-safe symbol token for a CCXT symbol: ``BTC/USDT:USDT`` -> ``BTC/USDT-USDT``."""
    return market_symbol.replace(":", "-")


class CryptoVenueConfig(BaseModel):
    """Account-level settings shared by every strategy on one exchange account."""
    model_config = ConfigDict(extra="forbid")

    exchange_id: str
    #: Required, no default: an accidental LIVE because a field was left out is not a failure mode worth allowing.
    environment: VenueEnvironment
    #: Label of the (sub-)account the API key belongs to. Part of the strategy-isolation key and the state file name.
    sub_account: str = "main"
    api_key_env: str = "CRYPTO_API_KEY"
    secret_env: str = "CRYPTO_API_SECRET"
    #: Only for venues that need an API passphrase (e.g. OKX).
    password_env: Optional[str] = None
    position_mode: PositionMode = PositionMode.ONE_WAY
    #: Applied at startup when set. On venues whose profile says the scope is ACCOUNT (Bybit unified account) this
    #: changes the WHOLE account; ``None`` leaves the account as it is.
    margin_mode: Optional[MarginMode] = None
    #: Milliseconds between requests for CCXT's rate limiter; ``None`` keeps CCXT's per-venue default.
    rate_limit_ms: Optional[int] = None
    #: Where lifecycle ids, managed stops and the spot ledger persist across restarts.
    state_dir: str = ".crypto_state"
    #: Extra options merged into the CCXT exchange's ``options`` (e.g. ``recvWindow``).
    ccxt_options: dict = Field(default_factory=dict)

    @field_validator("exchange_id")
    def _supported_exchange(cls, v: str) -> str:
        v = v.strip().lower()
        if not is_supported(v):
            raise ValueError(f"{v!r} is not a supported exchange (supported: {supported_venues()}); a new exchange "
                             f"is added on its own feature branch")
        return v

    @field_validator("api_key_env", "secret_env", "password_env")
    def _env_var_name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _ENV_VAR_RE.match(v):
            raise ValueError(f"{v!r} is not a valid environment variable NAME (put the name here, never the secret)")
        return v

    @field_validator("sub_account")
    def _sub_account(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("sub_account cannot be empty")
        return v

    @field_validator("rate_limit_ms")
    def _rate_limit(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("rate_limit_ms must be > 0")
        return v


class CryptoStrategyConfig(StrategyConfig):
    """``core.StrategyConfig`` plus the per-strategy crypto fields. Still an instance of ``StrategyConfig``."""

    #: CCXT unified symbol: ``BTC/USDT`` (spot) or ``BTC/USDT:USDT`` (USDT linear perpetual).
    market_symbol: str
    #: Narrowed from core's ``int | str``: a CCXT timeframe string, validated against the venue at startup.
    timeframe: str
    market_type: MarketType = MarketType.LINEAR_PERP
    feed_mode: FeedMode = FeedMode.STREAM
    stop_mode: StopMode = StopMode.AUTO
    stop_trigger: StopTrigger = StopTrigger.LAST
    leverage: Optional[float] = None
    sizing_unit: SizingUnit = SizingUnit.BASE_QTY
    #: POLL: seconds after a bar boundary before the first REST fetch of the closed candle.
    close_grace_seconds: float = 3.0
    #: Longest wait for a closed candle after its boundary. A bar later than this is stale: it updates the price buffer
    #: but does not run the strategy (acting on an old signal is worse than skipping it).
    close_max_wait_seconds: float = 30.0
    #: POLL: seconds between position / order polls.
    position_poll_seconds: float = 5.0
    #: Price poll interval for MANAGED stops when the feed is POLL (STREAM uses the WebSocket ticker).
    managed_stop_poll_seconds: float = 1.0
    #: Accept managed stops polled slower than ``MAX_MANAGED_STOP_POLL_SECONDS``.
    allow_slow_managed_stops: bool = False
    #: Seconds between REST reconciliations of positions / orders / fills in STREAM mode.
    reconcile_seconds: float = 60.0

    @model_validator(mode="before")
    @classmethod
    def _derive_symbol(cls, data):
        if isinstance(data, dict) and data.get("market_symbol") and not data.get("symbol"):
            data = {**data, "symbol": derive_log_symbol(str(data["market_symbol"]))}
        return data

    @field_validator("symbol")
    def _core_safe_symbol(cls, v: str) -> str:
        if ":" in v:
            raise ValueError(f"symbol {v!r} contains ':' which core's log identity rejects; put the CCXT symbol in "
                             f"market_symbol and leave symbol empty to derive {derive_log_symbol(v)!r}")
        return v

    @field_validator("timeframe")
    def _labelable_timeframe(cls, v: str) -> str:
        timeframe_to_minutes(v)
        return v

    @field_validator("leverage")
    def _positive_leverage(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("leverage must be > 0")
        return v

    @field_validator("close_grace_seconds", "position_poll_seconds", "managed_stop_poll_seconds", "reconcile_seconds")
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("interval settings must be >= 0")
        return v

    @model_validator(mode="after")
    def _crypto_semantics(self) -> "CryptoStrategyConfig":
        is_perp = self.market_type is MarketType.LINEAR_PERP
        if is_perp and ":" not in self.market_symbol:
            raise ValueError(f"{self.market_symbol!r}: a linear perpetual CCXT symbol carries its settle currency, "
                             f"e.g. 'BTC/USDT:USDT'")
        if not is_perp and ":" in self.market_symbol:
            raise ValueError(f"{self.market_symbol!r} is a derivatives symbol but market_type is {self.market_type}")
        if not is_perp and self.leverage is not None:
            raise ValueError("leverage is not valid for SPOT strategies")
        if not is_perp and self.sizing_unit is SizingUnit.CONTRACTS:
            raise ValueError("sizing_unit CONTRACTS applies to perpetuals only")
        if self.max_number_of_open_positions != 1:
            raise ValueError(
                "max_number_of_open_positions must be 1: a venue nets every fill on a symbol into ONE position (one "
                "entry price, one set of stops), so a second 'position' would only be a bigger first one.")
        if self.close_max_wait_seconds <= self.close_grace_seconds:
            raise ValueError("close_max_wait_seconds must be greater than close_grace_seconds")
        pm = self.position_manager
        if pm is not None and pm.type in _POINT_TYPES and not pm.point_size:
            raise ValueError(f"position manager {pm.type.value} needs an explicit point_size in quote price units "
                             f"(crypto has no MT5 'point'); percent or ATR managers are usually the better fit")
        if self.stop_mode is StopMode.MANAGED:
            check_managed_stop_latency(self)
        return self


def check_managed_stop_latency(cfg: CryptoStrategyConfig) -> None:
    """Reject a managed stop polled too slowly to deserve the name (POLL feed only; STREAM uses the ticker stream)."""
    if (cfg.feed_mode is FeedMode.POLL and cfg.managed_stop_poll_seconds > MAX_MANAGED_STOP_POLL_SECONDS
            and not cfg.allow_slow_managed_stops):
        raise ValueError(f"{cfg.name}: MANAGED stops with POLL feed and managed_stop_poll_seconds="
                         f"{cfg.managed_stop_poll_seconds} exceed {MAX_MANAGED_STOP_POLL_SECONDS}s; lower the "
                         f"interval or set allow_slow_managed_stops=true")


def isolation_key(venue: CryptoVenueConfig, strategy: CryptoStrategyConfig) -> tuple:
    """What two strategies must not share: one venue nets every fill on (account, market) into one position."""
    return (venue.exchange_id, venue.environment.value, venue.sub_account, strategy.market_type.value,
            strategy.market_symbol)


class CryptoSystemConfig(BaseModel):
    """JSON root for a crypto runner: one venue account, one or more strategies."""
    model_config = ConfigDict(extra="forbid")

    name: str
    venue: CryptoVenueConfig
    strategy: Optional[CryptoStrategyConfig] = None
    strategies: list[CryptoStrategyConfig] = Field(default_factory=list)

    @field_validator("name")
    def _name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("CryptoSystemConfig.name cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def _strategies_and_isolation(self) -> "CryptoSystemConfig":
        if self.strategy is None and not self.strategies:
            raise ValueError('Must provide either "strategy" or "strategies" (at least one)')
        if self.strategy is not None and self.strategies:
            raise ValueError('Cannot provide both "strategy" and "strategies" - choose one approach')
        check_isolation(self.venue, self.all_strategies())
        return self

    def all_strategies(self) -> list[CryptoStrategyConfig]:
        return [self.strategy] if self.strategy is not None else list(self.strategies)

    @classmethod
    def load_from_file(cls, file_path) -> "CryptoSystemConfig":
        with open(file_path, "r", encoding="utf-8") as fh:
            return cls.model_validate_json(fh.read())


def check_isolation(venue: CryptoVenueConfig, strategies: list[CryptoStrategyConfig]) -> None:
    """At most one strategy per (venue, environment, account, market type, symbol); magics unique.

    Names may repeat: the sleeves of a multi-trader share their strategy name (the runner-root log folder the Fleet
    Supervisor expects, as in MT5 / IB systems). State files are keyed by name AND symbol, and the isolation key above
    already forbids two strategies on one symbol, so a shared name never shares state.
    """
    seen_keys: dict[tuple, str] = {}
    seen_magics: dict[int, str] = {}
    for s in strategies:
        key = isolation_key(venue, s)
        if key in seen_keys:
            raise ValueError(f"strategies {seen_keys[key]!r} and {s.name!r} both trade {s.market_symbol} "
                             f"({s.market_type.value}) on {venue.exchange_id}/{venue.environment.value}/"
                             f"{venue.sub_account}; the venue nets them into one position, so magic cannot "
                             f"separate them")
        seen_keys[key] = s.name
        if s.magic in seen_magics:
            raise ValueError(f"strategies {seen_magics[s.magic]!r} and {s.name!r} share magic {s.magic}")
        seen_magics[s.magic] = s.name
