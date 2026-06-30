"""Series registry, conditioning channels, and pluggable availability policies.

Each macro series is sourced from FRED (single provider, stable public endpoint, no API key). The registry maps a stable
canonical name to its FRED id, its conditioning channel, and — critically — an **availability policy** describing
how long after the observation the value actually hits the wire.

Causal convention (no-lookahead)
--------------------------------
An observation may only be consumed by intraday bars from its ``available_from_utc`` onward. *How* that instant is
computed differs by series cadence/publisher, so it is a pluggable strategy rather than a hard-coded lag:

- ``BusinessDayLag`` — daily series released N business days later at a fixed UTC hour
  (VIX same evening = lag 0; credit/USD next business day = lag 1).
- ``CalendarDayLag`` — series whose release ignores business-day rolling (e.g. a weekly
  index published a fixed number of *calendar* days after the reference date).
- ``ExplicitRelease`` — irregular / event-driven series (rate decisions, CPI prints)
  whose source already carries the exact release timestamp; availability = that column.

The downstream asof-merge (``align.attach_exogenous``) only ever looks at ``available_from_utc`` + ``value``, so it is
cadence-agnostic: daily, weekly, monthly, or irregular series all attach through the same path. Adding a new series is
data-only: a ``SeriesSpec`` with the right policy — no engine change.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

# Default UTC hour-of-day stamped onto a lag-based release date. 22:00 clears the
# ~21:15 UTC US-close publish window (CBOE settlement, Fed H.10, ICE/Moody's EOD).
PUBLISH_HOUR_UTC = 22


class Channel(enum.StrEnum):
    """Macro conditioning channel a series feeds."""

    VOLATILITY = "volatility"
    RISK = "risk"
    USD = "usd"
    RATES = "rates"  # reserved for Phase-2 yields / policy-rate series


# --------------------------------------------------------------------------- #
# Availability policies
# --------------------------------------------------------------------------- #

@runtime_checkable
class AvailabilityPolicy(Protocol):
    """Computes the UTC instant from which each observation may be consumed."""

    def stamp(self, obs: pd.DataFrame) -> pd.Series:
        """Return a tz-aware (UTC) Series aligned to ``obs.index``."""
        ...


@dataclass(frozen=True)
class BusinessDayLag:
    """Release = obs date shifted ``lag`` *business* days, at ``hour_utc``.

    BusinessDay rolls any non-business observation date forward, which only ever makes
    the stamp later (more conservative), never earlier.
    """

    lag: int
    hour_utc: int = PUBLISH_HOUR_UTC

    def stamp(self, obs: pd.DataFrame) -> pd.Series:
        # BusinessDay(0) is intentional, not a no-op to optimize away: it also rolls any weekend
        # observation date forward to the next business day (the roll only ever makes the stamp
        # later, never earlier). The per-element offset is negligible at macro series sizes.
        shifted = obs["date"] + pd.tseries.offsets.BusinessDay(self.lag)
        stamped = shifted.dt.normalize() + pd.Timedelta(hours=self.hour_utc)
        return stamped.dt.tz_localize("UTC")


@dataclass(frozen=True)
class CalendarDayLag:
    """Release = obs date + ``lag`` *calendar* days, at ``hour_utc``.

    For series whose release cadence ignores business-day rolling (e.g. a weekly index
    published a fixed number of calendar days after the reference date).
    """

    lag: int
    hour_utc: int = PUBLISH_HOUR_UTC

    def stamp(self, obs: pd.DataFrame) -> pd.Series:
        stamped = obs["date"].dt.normalize() + pd.Timedelta(days=self.lag, hours=self.hour_utc)
        return stamped.dt.tz_localize("UTC")


@dataclass(frozen=True)
class ExplicitRelease:
    """Availability is supplied by the source as a real release-timestamp column.

    For irregular / event-driven series (rate decisions, CPI prints) where the exact
    release time is known and there is no fixed cadence to lag from. The named column
    must be tz-aware (or naive-UTC, which is localized to UTC).
    """

    column: str = "release_utc"

    def stamp(self, obs: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(obs[self.column], utc=True)


# --------------------------------------------------------------------------- #
# Series registry
# --------------------------------------------------------------------------- #

class MacroSeries(enum.StrEnum):
    """Stable canonical name for each macro series (decoupled from the FRED id)."""

    VIX = "VIX"
    VIX_3M = "VIX_3M"
    CREDIT_SPREAD = "CREDIT_SPREAD"
    USD_BROAD = "USD_BROAD"
    US_2Y = "US_2Y"
    US_10Y = "US_10Y"
    NFCI = "NFCI"


@dataclass(frozen=True)
class SeriesSpec:
    """FRED id + conditioning channel + availability policy for one macro series."""

    fred_id: str
    channel: Channel
    availability: AvailabilityPolicy
    description: str


SERIES: dict[MacroSeries, SeriesSpec] = {
    MacroSeries.VIX: SeriesSpec("VIXCLS", Channel.VOLATILITY, BusinessDayLag(0), "CBOE Volatility Index (VIX), daily close"),
    MacroSeries.VIX_3M: SeriesSpec("VXVCLS", Channel.VOLATILITY, BusinessDayLag(0), "CBOE S&P 500 3-Month Volatility Index (VIX3M)"),
    # Credit risk-on/off gauge. The natural choice — ICE BofA HY OAS (BAMLH0A0HYM2) — is
    # licence-capped to a rolling 3y window on FRED's anonymous CSV, too thin for a
    # stress-episode gauge. BAA10Y (Moody's Baa minus 10Y) is the canonical free daily
    # credit spread with full history; ~0.9 corr with HY OAS in stress. Restore true HY
    # OAS in Phase 2 via the keyed FRED API.
    MacroSeries.CREDIT_SPREAD: SeriesSpec("BAA10Y", Channel.RISK, BusinessDayLag(1), "Moody's Baa Corporate Yield minus 10Y Treasury (credit risk premium)"),
    MacroSeries.USD_BROAD: SeriesSpec("DTWEXBGS", Channel.USD, BusinessDayLag(1), "Nominal Broad US Dollar Index (Fed H.10)"),
    # Treasury yields (Fed H.15). H.15 posts ~16:15 ET same day; lag 1 is the conservative,
    # consistent choice (a day of staleness is immaterial for a slow rates conditioner — tighten
    # to 0 only if timeliness ever matters). The 2s10s curve is derived as a feature (US_10Y − US_2Y).
    MacroSeries.US_2Y: SeriesSpec("DGS2", Channel.RATES, BusinessDayLag(1), "US 2-Year Treasury Constant Maturity Yield (H.15)"),
    MacroSeries.US_10Y: SeriesSpec("DGS10", Channel.RATES, BusinessDayLag(1), "US 10-Year Treasury Constant Maturity Yield (H.15)"),
    # NFCI: weekly, dated by week-ending Friday, released the following Wednesday 08:30 ET.
    # +6 calendar days lands on Thursday — a one-day cushion past the nominal Wednesday release to
    # stay leak-safe across occasional holiday-shifted releases.
    MacroSeries.NFCI: SeriesSpec("NFCI", Channel.RISK, CalendarDayLag(6), "Chicago Fed National Financial Conditions Index (weekly)"),
}
