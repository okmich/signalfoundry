"""Read-only bar fetchers: ask the broker itself for missing, dropped or spiking bars (SPEC §3.1, §6.4, §7).

A ``FetchAdapter`` returns bars with **raw stamps** (the broker's own wall-clock digits, naive), exactly as the files in
``raw/`` store them. It never converts time: the overlap handshake (handshake.py) aligns whatever the broker serves.

``MT5FetchAdapter`` reaches the terminal the way ``fetch-mt5-data --broker <b> --account <a>`` does:
``okmich_quant_core.env_loader.EnvLoader`` reads ``.env.<b>.<a>`` from ``OKMICH_QUANT_ENV_DIR``, and
``MetaTrader5.initialize`` logs in with its TERMINAL_PATH / LOGIN_SERVER / LOGIN_ID / LOGIN_PASSWORD. Safety:
- ``MetaTrader5`` is imported lazily (on ``open``), so this package has no hard MT5 dependency.
- It does not import ``okmich_quant_mt5`` (live order path, its once-per-run tz decode). Bars keep the server's digits.
- Read calls only: initialize, account_info, last_error, symbol_info, symbol_select, copy_rates_range, shutdown.
- Data calls take positional arguments only: the binding silently ignores unknown keyword arguments. ``initialize``
  takes its documented keywords exactly as fetch-mt5-data passes them, and ``account_info`` then proves that the
  login, server and data folder match, so a silently ignored login can't fetch from another account.
- The env file's values are read and the process environment is restored, so credentials don't linger in os.environ.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import importlib
import io
import logging
import os
from pathlib import Path
from typing import Protocol

import pandas as pd

from .profile import BrokerProfile

log = logging.getLogger(__name__)

ENV_KEYS = ("TERMINAL_PATH", "LOGIN_SERVER", "LOGIN_ID", "LOGIN_PASSWORD")
#: MetaTrader5 last_error codes for IPC failures with the terminal (RES_E_INTERNAL_FAIL_*): retried, never counted.
MT5_TRANSIENT_CODES = frozenset({-10001, -10002, -10003, -10004, -10005})
#: MT5 codes that mean our request was wrong (invalid params, unsupported): a bug or a bad profile, not the broker.
MT5_REQUEST_ERRORS = frozenset({-2, -5, -7})
_MT5_TIMEFRAMES = {1: "TIMEFRAME_M1", 2: "TIMEFRAME_M2", 3: "TIMEFRAME_M3", 4: "TIMEFRAME_M4", 5: "TIMEFRAME_M5", 6: "TIMEFRAME_M6",
                   10: "TIMEFRAME_M10", 12: "TIMEFRAME_M12", 15: "TIMEFRAME_M15", 20: "TIMEFRAME_M20", 30: "TIMEFRAME_M30",
                   60: "TIMEFRAME_H1", 240: "TIMEFRAME_H4", 1440: "TIMEFRAME_D1"}


class FetchError(RuntimeError):
    """Base class for fetch failures."""


class TransientFetchError(FetchError):
    """The terminal is busy, disconnected or timed out. Retried within the run; never counts against a hole."""


class FetchUnavailableError(FetchError):
    """Fetching can't start or can't be trusted: no env file, login refused, wrong server, unknown symbol, bad request.
    The run stops before any ledger change."""


class FetchAdapter(Protocol):
    """Read-only source of a broker's bars in raw stamps."""

    name: str

    def open(self) -> None: ...

    def close(self) -> None: ...

    def fetch(self, symbol: str, start_raw: pd.Timestamp, end_raw: pd.Timestamp) -> pd.DataFrame:
        """Bars stamped in [start_raw, end_raw), naive raw index named ``raw``, columns as served (at least OHLC).
        An empty frame when the broker has none. ``symbol`` is the broker's name (the file stem)."""
        ...

    def served_from(self, symbol: str) -> pd.Timestamp | None:
        """The earliest raw stamp the source serves for ``symbol`` (history depth evidence), or None if unknown."""
        ...


def _raw_arg(ts: pd.Timestamp) -> dt.datetime:
    """A raw stamp as the binding's query argument: the server's digits as a UTC datetime, which the binding turns into
    the same epoch seconds MT5 uses for bar times (a naive datetime would be shifted by the host's local zone)."""
    return pd.Timestamp(ts).tz_localize("UTC").to_pydatetime()


def rates_frame(rates) -> pd.DataFrame:
    """A copy_rates_* structured array as a frame on naive raw stamps (epoch seconds = server wall-clock digits)."""
    df = pd.DataFrame(rates)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"], index=pd.DatetimeIndex([], name="raw", dtype="datetime64[ns]"))
    idx = pd.DatetimeIndex(pd.to_datetime(df.pop("time").to_numpy(), unit="s"), name="raw").as_unit("ns")
    df.index = idx
    return df


@contextlib.contextmanager
def _restored_environ():
    saved = dict(os.environ)
    try:
        yield
    finally:
        for k in set(os.environ) - set(saved):
            del os.environ[k]
        for k, v in saved.items():
            if os.environ.get(k) != v:
                os.environ[k] = v


def read_env_file(env_broker: str, env_account: str, env_dir: str | Path | None = None) -> dict[str, str]:
    """TERMINAL_PATH / LOGIN_SERVER / LOGIN_ID / LOGIN_PASSWORD from ``.env.<broker>.<account>``, read with EnvLoader the
    way fetch-mt5-data does, leaving the process environment as it was."""
    from okmich_quant_core.env_loader import EnvDirNotConfiguredError, EnvLoader
    out = io.StringIO()
    with _restored_environ(), contextlib.redirect_stdout(out):
        try:
            ok = EnvLoader(env_dir).load(env_broker, env_account)
        except EnvDirNotConfiguredError as e:
            raise FetchUnavailableError(str(e)) from None
        values = {k: os.environ.get(k, "") for k in ENV_KEYS}
    if not ok or not all(values.values()):
        said = " ".join(line.strip() for line in out.getvalue().splitlines() if "Warning" in line or "Error" in line)
        raise FetchUnavailableError(f"env file .env.{env_broker}.{env_account} not usable: {said or 'missing values'}")
    return values


class MT5FetchAdapter:
    """Refetch bars from a MetaTrader 5 terminal, read-only (see the module docstring for the safety rules)."""

    def __init__(self, env_broker: str, env_account: str, bar_minutes: int, expected_server: str | None = None,
                 env_dir: str | Path | None = None, module=None):
        if bar_minutes not in _MT5_TIMEFRAMES:
            raise ValueError(f"no MT5 timeframe for {bar_minutes}-minute bars")
        self.env_broker, self.env_account, self.bar_minutes = env_broker, env_account, bar_minutes
        self.expected_server, self.env_dir = expected_server, env_dir
        self._module, self._mt5, self._tf = module, None, None
        self._served_from: dict[str, pd.Timestamp | None] = {}
        self.calls = 0
        self.name = f"mt5:{env_broker}.{env_account}"

    @classmethod
    def for_profile(cls, profile: BrokerProfile, env_dir: str | Path | None = None, module=None) -> MT5FetchAdapter:
        """The profile's fetch source; the env file's server must be the profile's data folder (no mixing feeds)."""
        return cls(profile.fetch.env_broker, profile.fetch.env_account, profile.bar_minutes, profile.name, env_dir, module)

    def open(self) -> None:
        env = read_env_file(self.env_broker, self.env_account, self.env_dir)
        if self.expected_server and env["LOGIN_SERVER"] != self.expected_server:
            raise FetchUnavailableError(f".env.{self.env_broker}.{self.env_account} logs in to {env['LOGIN_SERVER']}, but the data "
                                        f"folder is {self.expected_server}: refusing to mix feeds")
        try:
            login = int(env["LOGIN_ID"])
        except ValueError:
            raise FetchUnavailableError(f".env.{self.env_broker}.{self.env_account}: LOGIN_ID is not a number") from None
        mt5 = self._module if self._module is not None else importlib.import_module("MetaTrader5")
        if not mt5.initialize(path=env["TERMINAL_PATH"], server=env["LOGIN_SERVER"], login=login, password=env["LOGIN_PASSWORD"]):
            code = mt5.last_error()
            raise FetchUnavailableError(f"MetaTrader5.initialize failed for {env['LOGIN_SERVER']}: {code}")
        acct = mt5.account_info()
        if acct is None or int(acct.login) != login or str(acct.server) != env["LOGIN_SERVER"]:
            got = (getattr(acct, "login", None), getattr(acct, "server", None))
            mt5.shutdown()
            raise FetchUnavailableError(f"terminal is logged in as {got}, not ({login}, {env['LOGIN_SERVER']})")
        self._mt5, self._tf = mt5, getattr(mt5, _MT5_TIMEFRAMES[self.bar_minutes])
        log.info("%s: connected to %s (login %s)", self.name, acct.server, acct.login)

    def close(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
            self._mt5 = None

    def __enter__(self) -> MT5FetchAdapter:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _require(self):
        if self._mt5 is None:
            raise FetchUnavailableError(f"{self.name}: not connected (call open() first)")
        return self._mt5

    def _error(self) -> tuple[int, str]:
        code = self._mt5.last_error()
        return (int(code[0]), str(code[1])) if isinstance(code, tuple) and len(code) == 2 else (0, str(code))

    def _fail(self, what: str) -> FetchError:
        """A failed data call: a bad request is ours to fix (stop); anything else may be the terminal (retry, not counted)."""
        num, msg = self._error()
        if num in MT5_REQUEST_ERRORS:
            return FetchUnavailableError(f"{what}: request refused ({num}, {msg})")
        return TransientFetchError(f"{what}: ({num}, {msg})")

    def _select(self, symbol: str) -> None:
        mt5 = self._require()
        info = mt5.symbol_info(symbol)
        if info is None:
            num, msg = self._error()
            if num in MT5_TRANSIENT_CODES:
                raise TransientFetchError(f"symbol_info({symbol}): ({num}, {msg})")
            raise FetchUnavailableError(f"{symbol} is not a symbol on this server: ({num}, {msg})")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise self._fail(f"symbol_select({symbol})")

    def fetch(self, symbol: str, start_raw: pd.Timestamp, end_raw: pd.Timestamp) -> pd.DataFrame:
        mt5 = self._require()
        self._select(symbol)
        self.calls += 1
        rates = mt5.copy_rates_range(symbol, self._tf, _raw_arg(start_raw), _raw_arg(end_raw))
        if rates is None:
            raise self._fail(f"copy_rates_range({symbol}, {start_raw}, {end_raw})")
        df = rates_frame(rates)
        return df[(df.index >= start_raw) & (df.index < end_raw)]

    def served_from(self, symbol: str) -> pd.Timestamp | None:
        if symbol not in self._served_from:
            mt5 = self._require()
            self._select(symbol)
            self.calls += 1
            now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D") + pd.Timedelta(days=2)    # past any server offset
            rates = mt5.copy_rates_range(symbol, self._tf, _raw_arg(pd.Timestamp("1990-01-01")), _raw_arg(now))
            self._served_from[symbol] = rates_frame(rates).index.min() if rates is not None and len(rates) else None
        return self._served_from[symbol]
