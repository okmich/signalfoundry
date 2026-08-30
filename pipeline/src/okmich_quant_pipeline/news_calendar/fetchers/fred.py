"""US data fetcher via ALFRED (Archival FRED, St. Louis Fed).

Why ALFRED instead of BLS/BEA/Census directly
---------------------------------------------
BLS blocks our IP at the CDN layer (403 on every ``/schedule/`` URL, including with browser-style headers). BEA and
Census don't have a single consolidated release-date page that's scrape-friendly. ALFRED solves both:
``alfred.stlouisfed.org/release?rid={N}`` lists every historical release date for a given indicator as
``data-release-date="YYYY-MM-DD"`` attributes on ``<li>`` elements.
ALFRED is the St. Louis Fed's archival-data service — it tracks vintage data, which means it knows the exact date every
release hit the wire.

ALFRED captures *all* release events including:
- Regular monthly releases
- Annual benchmark revisions (e.g., NFP's August preliminary benchmark)
- Holiday-shifted releases (e.g., NFP moved to Thursday before July 4)
- Post-government-shutdown combined releases (Oct/Nov 2025 NFP merged)

All of those move XAUUSD, so the mask treats them as one event class per release. No additional filtering needed.

Release-ID lookup (verified 2026-05-25)
---------------------------------------
- 9   = Advance Monthly Sales for Retail and Food Services -> US_RETAIL_SALES
- 10  = Consumer Price Index                                -> US_CPI
- 46  = Producer Price Index                                -> US_PPI
- 50  = Employment Situation                                -> US_NFP
- 53  = Gross Domestic Product                              -> US_GDP
- 54  = Personal Income and Outlays                         -> US_PCE
"""
from __future__ import annotations

import datetime as dt
import re

import pandas as pd

from okmich_quant_pipeline.http import get
from okmich_quant_pipeline.news_calendar._types import EventName, ImpactTier, Source

_RELEASE_URL = "https://alfred.stlouisfed.org/release?rid={rid}"

# Each ALFRED release ID maps to its canonical event name + the agency that
# publishes the underlying data. ALFRED is the delivery channel; the source
# field should reflect the publisher (BLS / BEA / CENSUS) so downstream
# consumers can group by agency.
RELEASE_MAP: dict[int, tuple[EventName, Source]] = {
    9: (EventName.US_RETAIL_SALES, Source.CENSUS),
    10: (EventName.US_CPI, Source.BLS),
    46: (EventName.US_PPI, Source.BLS),
    50: (EventName.US_NFP, Source.BLS),
    53: (EventName.US_GDP, Source.BEA),
    54: (EventName.US_PCE, Source.BEA),
}

# Matches: data-release-date="2024-01-05"
_DATE_ATTR_RE = re.compile(r'data-release-date="(20\d{2}-\d{2}-\d{2})"')


def _parse_release_dates(html: str, year_min: int, year_max: int) -> list[dt.date]:
    """Return the in-range release dates parsed from an ALFRED release page."""
    out: list[dt.date] = []
    for m in _DATE_ATTR_RE.finditer(html):
        d = dt.date.fromisoformat(m.group(1))
        if year_min <= d.year <= year_max:
            out.append(d)
    # ALFRED can emit the same date multiple times in its UI (e.g. once in the
    # main list, once in a "select" widget). Deduplicate.
    return sorted(set(out))


def _fetch_one(rid: int, event_name: EventName, source: Source, year_min: int, year_max: int) -> list[dict]:
    """Fetch one ALFRED release and return rows for the calendar."""
    resp = get(_RELEASE_URL.format(rid=rid))
    dates = _parse_release_dates(resp.text, year_min, year_max)
    if not dates:
        raise RuntimeError(
            f"FRED fetcher: parsed zero dates for rid={rid} ({event_name}) in "
            f"[{year_min}, {year_max}]. Page structure may have changed."
        )
    return [
        {
            "release_date": d,
            "event_name": event_name,
            "source": source,
            "impact_tier": ImpactTier.HIGH,
        }
        for d in dates
    ]


def fetch(year_min: int, year_max: int) -> pd.DataFrame:
    """Return a DataFrame of US high-impact release dates in ``[year_min, year_max]``.

    Iterates ``RELEASE_MAP`` and concatenates per-release frames.
    """
    rows: list[dict] = []
    for rid, (event_name, source) in RELEASE_MAP.items():
        rows.extend(_fetch_one(rid, event_name, source, year_min, year_max))
    df = pd.DataFrame(rows)
    return df.sort_values(["release_date", "event_name"]).reset_index(drop=True)


if __name__ == "__main__":
    df = fetch(2024, 2026)
    print(df.to_string(index=False))
    print(f"\n{len(df)} rows; events per type:")
    print(df.event_name.value_counts())
