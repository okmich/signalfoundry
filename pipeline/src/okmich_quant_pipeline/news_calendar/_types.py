"""Shared enums for the news-calendar package."""
from __future__ import annotations

import enum


class Source(enum.StrEnum):
    """Authoritative agency that publishes the release schedule."""

    FED = "federalreserve.gov"
    BLS = "bls.gov"
    BEA = "bea.gov"
    CENSUS = "census.gov"
    ECB = "ecb.europa.eu"
    BOE = "bankofengland.co.uk"


class EventName(enum.StrEnum):
    """Canonical name for each scheduled event type.

    Naming convention: ``{REGION}_{INDICATOR}[_{SUBTYPE}]``. The region prefix
    matters because CPI, GDP, etc. exist in multiple jurisdictions.
    """

    # United States — Fed
    US_FOMC_STATEMENT = "US_FOMC_STATEMENT"
    US_FOMC_PRESS_CONFERENCE = "US_FOMC_PRESS_CONFERENCE"
    # United States — BLS
    US_NFP = "US_NFP"
    US_CPI = "US_CPI"
    US_PPI = "US_PPI"
    # United States — BEA
    # GDP collapses Advance/Second/Third estimates into one event — the news
    # mask treats them identically (all 08:30 ET, all market-moving). A
    # downstream consumer that wants the distinction can re-derive it by
    # parsing release notes per date.
    US_GDP = "US_GDP"
    US_PCE = "US_PCE"
    # United States — Census
    US_RETAIL_SALES = "US_RETAIL_SALES"
    # Euro area — ECB
    EU_ECB_RATE_DECISION = "EU_ECB_RATE_DECISION"
    EU_ECB_PRESS_CONFERENCE = "EU_ECB_PRESS_CONFERENCE"
    # United Kingdom — BoE
    UK_BOE_RATE_DECISION = "UK_BOE_RATE_DECISION"


class ImpactTier(enum.IntEnum):
    """Coarse impact bucketing on XAUUSD.

    Week 1 only consumes ``HIGH``-tier events for the news mask; the field
    exists so downstream code can widen/narrow the mask without rebuilding
    the calendar.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3
