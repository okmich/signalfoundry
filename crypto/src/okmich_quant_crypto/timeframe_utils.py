"""CCXT timeframe helpers.

Core labels every bar by flooring Unix-epoch minutes to the timeframe (``BaseStrategy._derive_asof_bar_ts``). That
grid matches exchange candles for every timeframe up to one day, but not beyond: epoch weeks start on a Thursday while
exchange weeks start on Monday, and months have no fixed length. Timeframes above ``1d`` are therefore rejected.
"""
from datetime import datetime, timezone

from ccxt.base.exchange import Exchange

#: Longest timeframe whose candles line up with core's epoch-minute bar grid.
MAX_TIMEFRAME_MINUTES = 1440


def timeframe_to_seconds(timeframe: str) -> int:
    """Seconds in a CCXT timeframe string (``'5m'`` -> 300). Raises ``ValueError`` for anything unparseable."""
    if not isinstance(timeframe, str) or len(timeframe) < 2:
        raise ValueError(f"timeframe must be a CCXT timeframe string such as '5m' or '1h' (got {timeframe!r})")
    try:
        return int(Exchange.parse_timeframe(timeframe))
    except Exception as e:
        raise ValueError(f"unparseable CCXT timeframe {timeframe!r}: {e}") from e


def timeframe_to_minutes(timeframe: str) -> int:
    """Whole minutes in a CCXT timeframe, validated against the range core can label (1 minute .. 1 day)."""
    seconds = timeframe_to_seconds(timeframe)
    if seconds % 60 != 0:
        raise ValueError(f"timeframe {timeframe!r} is not a whole number of minutes; core labels bars in minutes")
    minutes = seconds // 60
    if minutes <= 0 or minutes > MAX_TIMEFRAME_MINUTES:
        raise ValueError(
            f"timeframe {timeframe!r} is outside 1m..1d. Longer candles do not line up with core's epoch-minute bar "
            f"labels (epoch weeks start on Thursday, exchange weeks on Monday; months vary in length).")
    if MAX_TIMEFRAME_MINUTES % minutes != 0:
        raise ValueError(f"timeframe {timeframe!r} does not divide a day evenly, so its candles cannot stay on a "
                         f"UTC grid")
    return minutes


def timeframe_to_ms(timeframe: str) -> int:
    return timeframe_to_minutes(timeframe) * 60_000


def validate_venue_timeframe(timeframe: str, supported: dict | None) -> int:
    """Check ``timeframe`` is offered by the venue (``exchange.timeframes``) and labelable by core. Returns minutes."""
    minutes = timeframe_to_minutes(timeframe)
    if supported is not None and timeframe not in supported:
        offered = sorted((tf for tf in supported if _labelable(tf)), key=timeframe_to_seconds)
        raise ValueError(f"timeframe {timeframe!r} is not offered by this venue; usable timeframes: {offered}")
    return minutes


def _labelable(timeframe: str) -> bool:
    try:
        timeframe_to_minutes(timeframe)
        return True
    except ValueError:
        return False


def bar_open_ms(ts_ms: int, tf_ms: int) -> int:
    """Open timestamp of the bar containing ``ts_ms``."""
    return (ts_ms // tf_ms) * tf_ms


def last_closed_bar_open_ms(now_ms: int, tf_ms: int) -> int:
    """Open timestamp of the most recent bar that has fully closed at ``now_ms``."""
    return bar_open_ms(now_ms, tf_ms) - tf_ms


def ms_to_utc(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def utc_now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)
