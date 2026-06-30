"""Release-time conventions per event type + DST-aware UTC conversion.

Fetchers return calendar dates only (the date listed on the agency's
schedule). The orchestrator joins those dates with the convention table
below to produce a tz-aware UTC timestamp.

The table is the source of truth for release times — change a time here
and the parquet picks it up on the next ``build.py`` run, no fetcher
edits required.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from okmich_quant_pipeline.news_calendar._types import EventName

# (local hour, local minute, local timezone) per event. The timezone is a
# zoneinfo key so DST is applied automatically when we localise the date.
RELEASE_TIMES: dict[EventName, tuple[int, int, str]] = {
    # United States — Fed (Washington DC, US/Eastern)
    EventName.US_FOMC_STATEMENT: (14, 0, "America/New_York"),
    EventName.US_FOMC_PRESS_CONFERENCE: (14, 30, "America/New_York"),
    # United States — BLS (all releases 08:30 ET)
    EventName.US_NFP: (8, 30, "America/New_York"),
    EventName.US_CPI: (8, 30, "America/New_York"),
    EventName.US_PPI: (8, 30, "America/New_York"),
    # United States — BEA (08:30 ET)
    EventName.US_GDP: (8, 30, "America/New_York"),
    EventName.US_PCE: (8, 30, "America/New_York"),
    # United States — Census (08:30 ET)
    EventName.US_RETAIL_SALES: (8, 30, "America/New_York"),
    # Euro area — ECB (Frankfurt, CET/CEST)
    # Statement at 14:15 local; press conf at 14:45 local since 2022.
    EventName.EU_ECB_RATE_DECISION: (14, 15, "Europe/Berlin"),
    EventName.EU_ECB_PRESS_CONFERENCE: (14, 45, "Europe/Berlin"),
    # United Kingdom — BoE (London, GMT/BST)
    EventName.UK_BOE_RATE_DECISION: (12, 0, "Europe/London"),
}


def to_utc(release_date: dt.date, event: EventName) -> dt.datetime:
    """Combine ``release_date`` with the event's release time and return UTC.

    DST is handled by ``zoneinfo`` — e.g. 08:30 ET maps to 13:30 UTC in winter
    (EST, UTC−5) and 12:30 UTC in summer (EDT, UTC−4); 14:15 CET maps to 13:15
    UTC in winter and 12:15 UTC in summer (yes, the offset narrows in summer).
    """
    if event not in RELEASE_TIMES:
        raise KeyError(f"No release-time convention for {event!r}")
    hour, minute, tzname = RELEASE_TIMES[event]
    local = dt.datetime(release_date.year, release_date.month, release_date.day, hour, minute, tzinfo=ZoneInfo(tzname))
    return local.astimezone(ZoneInfo("UTC"))
