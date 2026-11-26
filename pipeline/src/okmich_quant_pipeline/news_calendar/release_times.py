"""Release times per event type + DST-aware UTC conversion.

Fetchers return calendar dates only (the date listed on the agency's schedule). The orchestrators join those dates
with the tables below to produce a tz-aware UTC timestamp.

The tables are the source of truth for release times. Change a time here and both parquets pick it up on their next
build (``build.py``, ``economic_events.py``); no fetcher edits are needed.

- ``RELEASE_TIMES``: per event, the regular release time and the first release date it applies to, oldest first. The
  first date is the earliest the time has been checked, and ``to_utc`` refuses earlier dates rather than guess.
- ``EXCEPTIONS``: single releases that left the regular time.

Both builders compare every stamped row ForexFactory also lists against the instant it recorded
(``forexfactory.check_recorded_times``) and refuse to write on a disagreement. A new exception or a changed release
time therefore stops the next build instead of entering the parquet.
"""
from __future__ import annotations

import dataclasses as dc
import datetime as dt
from zoneinfo import ZoneInfo

from okmich_quant_pipeline.news_calendar._types import EventName

_NEW_YORK = "America/New_York"
_FRANKFURT = "Europe/Berlin"
_LONDON = "Europe/London"
_UTC = ZoneInfo("UTC")

# First release date ForexFactory's recorded instants were compared with this table (2011-2026, checked 2026-11-26).
_CHECKED_FROM = dt.date(2011, 1, 1)


@dc.dataclass(frozen=True)
class ReleaseTime:
    """A local wall-clock release time. ``tz`` is a zoneinfo key, so DST is applied when a date is localised."""

    hour: int
    minute: int
    tz: str

    def on(self, release_date: dt.date) -> dt.datetime:
        """This wall-clock time on ``release_date``, as a UTC datetime."""
        local = dt.datetime(release_date.year, release_date.month, release_date.day, self.hour, self.minute,
                            tzinfo=ZoneInfo(self.tz))
        return local.astimezone(_UTC)


RELEASE_TIMES: dict[EventName, tuple[tuple[dt.date, ReleaseTime], ...]] = {
    # United States — Fed. 14:00 statement and 14:30 press conference from the 2013-03-20 meeting (Fed release of
    # 2013-03-13); statements came at 12:30 or 14:15 before it.
    EventName.US_FOMC_STATEMENT: ((dt.date(2013, 3, 20), ReleaseTime(14, 0, _NEW_YORK)),),
    EventName.US_FOMC_PRESS_CONFERENCE: ((dt.date(2013, 3, 20), ReleaseTime(14, 30, _NEW_YORK)),),
    # United States — BLS, BEA and Census: 08:30 ET.
    EventName.US_NFP: ((_CHECKED_FROM, ReleaseTime(8, 30, _NEW_YORK)),),
    EventName.US_CPI: ((_CHECKED_FROM, ReleaseTime(8, 30, _NEW_YORK)),),
    EventName.US_PPI: ((_CHECKED_FROM, ReleaseTime(8, 30, _NEW_YORK)),),
    EventName.US_GDP: ((_CHECKED_FROM, ReleaseTime(8, 30, _NEW_YORK)),),
    EventName.US_PCE: ((_CHECKED_FROM, ReleaseTime(8, 30, _NEW_YORK)),),
    EventName.US_RETAIL_SALES: ((_CHECKED_FROM, ReleaseTime(8, 30, _NEW_YORK)),),
    # Euro area — ECB. Decision 13:45 and press conference 14:30 through the 2022-06-09 meeting; 14:15 and 14:45
    # from the 2022-07-21 meeting.
    EventName.EU_ECB_RATE_DECISION: ((_CHECKED_FROM, ReleaseTime(13, 45, _FRANKFURT)),
                                     (dt.date(2022, 7, 21), ReleaseTime(14, 15, _FRANKFURT))),
    EventName.EU_ECB_PRESS_CONFERENCE: ((_CHECKED_FROM, ReleaseTime(14, 30, _FRANKFURT)),
                                        (dt.date(2022, 7, 21), ReleaseTime(14, 45, _FRANKFURT))),
    # United Kingdom — BoE: 12:00 London.
    EventName.UK_BOE_RATE_DECISION: ((_CHECKED_FROM, ReleaseTime(12, 0, _LONDON)),),
}

# Releases that left the regular time. Each matches ForexFactory's recorded instant.
EXCEPTIONS: dict[tuple[EventName, dt.date], ReleaseTime] = {
    # BoE 2020: the 2020-03-11 emergency cut and the May, August and November Monetary Policy Reports came at 07:00;
    # the 2020-03-19 special meeting at 14:30.
    (EventName.UK_BOE_RATE_DECISION, dt.date(2020, 3, 11)): ReleaseTime(7, 0, _LONDON),
    (EventName.UK_BOE_RATE_DECISION, dt.date(2020, 3, 19)): ReleaseTime(14, 30, _LONDON),
    (EventName.UK_BOE_RATE_DECISION, dt.date(2020, 5, 7)): ReleaseTime(7, 0, _LONDON),
    (EventName.UK_BOE_RATE_DECISION, dt.date(2020, 8, 6)): ReleaseTime(7, 0, _LONDON),
    (EventName.UK_BOE_RATE_DECISION, dt.date(2020, 11, 5)): ReleaseTime(7, 0, _LONDON),
    # BEA put out Personal Income and Outlays at 10:00 on these days, after a GDP release at 08:30 (BEA embargo time on
    # the 2019-11-27 release, for example).
    **{(EventName.US_PCE, d): ReleaseTime(10, 0, _NEW_YORK) for d in (
        dt.date(2014, 12, 23), dt.date(2016, 2, 26), dt.date(2016, 12, 22), dt.date(2018, 12, 21),
        dt.date(2019, 11, 27), dt.date(2019, 12, 20), dt.date(2020, 11, 25), dt.date(2021, 11, 24),
        dt.date(2024, 11, 27), dt.date(2025, 4, 30), dt.date(2025, 12, 5), dt.date(2026, 1, 22))},
}


def release_time(release_date: dt.date, event: EventName) -> ReleaseTime:
    """The wall-clock time ``event`` was (or is scheduled to be) released on ``release_date``.

    Raises ``KeyError`` for an event without an entry, and ``ValueError`` for a date before the event's first checked
    release time.
    """
    if event not in RELEASE_TIMES:
        raise KeyError(f"No release-time convention for {event!r}")
    exception = EXCEPTIONS.get((event, release_date))
    if exception is not None:
        return exception
    eras = RELEASE_TIMES[event]
    if release_date < eras[0][0]:
        raise ValueError(f"{event.value} on {release_date}: no checked release time before {eras[0][0]}. Check the "
                         f"publisher's times for earlier years and add them to release_times.RELEASE_TIMES.")
    return next(t for start, t in reversed(eras) if release_date >= start)


def release_tz(event: EventName) -> str:
    """The zoneinfo key ``event`` is scheduled in; its ``release_date`` is a calendar date in this zone."""
    if event not in RELEASE_TIMES:
        raise KeyError(f"No release-time convention for {event!r}")
    return RELEASE_TIMES[event][0][1].tz


def local_date(instant: dt.datetime, event: EventName) -> dt.date:
    """The calendar date of a tz-aware ``instant`` in ``event``'s zone."""
    return instant.astimezone(ZoneInfo(release_tz(event))).date()


def to_utc(release_date: dt.date, event: EventName) -> dt.datetime:
    """Combine ``release_date`` with the event's release time on that date and return UTC.

    DST is handled by ``zoneinfo``. For example, 08:30 ET maps to 13:30 UTC in winter (EST, UTC−5) and 12:30 UTC in
    summer (EDT, UTC−4); 14:15 CET maps to 13:15 UTC in winter and 12:15 UTC in summer.
    """
    return release_time(release_date, event).on(release_date)
