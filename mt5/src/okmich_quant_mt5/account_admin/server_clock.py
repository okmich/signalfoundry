"""The broker server clock, stated rather than guessed (ACCOUNT_ADMIN_SPEC §12.3 "Trigger").

MT5 reports every order, deal and tick time as SERVER wall-clock epoch seconds. Reading those as UTC shifts every age
and every day boundary by the broker's offset. Guessing the offset from the latest tick fails silently while the market
is closed, which is exactly when an Admin keeps running. So the Admin's config states the server clock the way this
codebase already does: an IANA zone plus a shift (IC Markets and most NY-close brokers: ``America/New_York`` + 7 h,
which follows US DST exactly). Live ticks are only used to catch a wrong statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

#: A converted tick this far in the future proves the stated clock is wrong (a stale tick only ever looks old).
FUTURE_TICK_TOLERANCE_S = 90.0


@dataclass(frozen=True)
class ServerClock:
    zone: ZoneInfo
    shift_h: float

    def to_utc(self, server_epoch: float) -> datetime:
        """Server wall-clock epoch digits -> the true UTC instant."""
        wall = datetime(1970, 1, 1) + timedelta(seconds=float(server_epoch))       # naive server wall-clock
        anchored = (wall - timedelta(hours=self.shift_h)).replace(tzinfo=self.zone)
        return anchored.astimezone(timezone.utc)

    def to_server_epoch(self, instant: datetime) -> int:
        """A UTC instant -> the server wall-clock epoch digits MT5 expects in history queries."""
        wall = instant.astimezone(self.zone).replace(tzinfo=None) + timedelta(hours=self.shift_h)
        return int((wall - datetime(1970, 1, 1)).total_seconds())

    def offset_h_at(self, instant: datetime) -> float:
        return (self.to_server_epoch(instant) - instant.timestamp()) / 3600.0

    def tick_contradicts(self, tick_server_epoch: float, now: datetime) -> str | None:
        """Why a tick proves this clock wrong, or ``None``. Only a tick that lands in the future can prove it: a
        stale tick lands in the past whatever the clock."""
        seen = self.to_utc(tick_server_epoch)
        ahead = (seen - now).total_seconds()
        if ahead > FUTURE_TICK_TOLERANCE_S:
            return (f"a tick converts to {seen.isoformat()}, {ahead:.0f}s in the future: the stated server clock "
                    f"({self.zone.key} {self.shift_h:+g}h) is behind the broker's")
        return None


def parse_broker_section(raw: Mapping[str, Any] | None, where: str = "broker") -> tuple[ServerClock | None, list[str]]:
    """Validate the MT5 ``broker`` section of the Admin config: ``server_clock_tz`` and ``server_clock_shift_h``."""
    if raw is None:
        return None, [f"{where}: missing; the MT5 Admin needs the broker server clock, e.g. "
                      f'{{"server_clock_tz": "America/New_York", "server_clock_shift_h": 7}}']
    if not isinstance(raw, Mapping):
        return None, [f"{where}: must be an object"]
    problems = [f"{where}.{k}: unknown key" for k in raw if k not in ("server_clock_tz", "server_clock_shift_h")]
    zone = None
    name = raw.get("server_clock_tz")
    try:
        zone = ZoneInfo(str(name)) if isinstance(name, str) and name.strip() else None
    except (ZoneInfoNotFoundError, ValueError):
        zone = None
    if zone is None:
        problems.append(f"{where}.server_clock_tz: must be an IANA time zone, got {name!r}")
    shift = raw.get("server_clock_shift_h")
    if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not -14 <= float(shift) <= 14 \
            or (float(shift) * 4) != int(float(shift) * 4):
        problems.append(f"{where}.server_clock_shift_h: must be hours in [-14, 14] in quarter-hour steps, got {shift!r}")
    if problems:
        return None, problems
    return ServerClock(zone, float(shift)), []
