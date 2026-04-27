"""Bar size conversions and IB duration-string computation."""
import math
from datetime import datetime


BAR_SIZE_MINUTES: dict[str, int] = {
    "1 min": 1,
    "2 mins": 2,
    "5 mins": 5,
    "10 mins": 10,
    "15 mins": 15,
    "30 mins": 30,
    "1 hour": 60,
    "2 hours": 120,
    "4 hours": 240,
    "1 day": 1440,
}

# Conservative approximations of IB's per-bar-size max duration. The data fetcher
# always paginates anyway — these values govern chunk size, not correctness.
MAX_DURATION_DAYS: dict[str, int] = {
    "1 min": 1,
    "2 mins": 2,
    "5 mins": 7,
    "10 mins": 7,
    "15 mins": 7,
    "30 mins": 30,
    "1 hour": 30,
    "2 hours": 30,
    "4 hours": 30,
    "1 day": 365,
}


def bar_size_to_minutes(bar_size: str) -> int:
    if bar_size not in BAR_SIZE_MINUTES:
        raise ValueError(
            f"Unsupported bar size: {bar_size!r}. Supported: {list(BAR_SIZE_MINUTES)}"
        )
    return BAR_SIZE_MINUTES[bar_size]


def required_duration(bar_size: str, bars_needed: int, use_rth: bool,
                      safety_factor: float = 1.3) -> str:
    """Minimum IB duration string yielding at least bars_needed bars.

    Raises ValueError if computed duration exceeds IB's per-bar-size maximum
    (caller must paginate via fetch_historical_bars_paginated).
    """
    bar_minutes = bar_size_to_minutes(bar_size)
    minutes_per_day = 390 if use_rth else 1440

    if bar_minutes >= 1440:
        # bars_needed expressed in trading days; convert to calendar days.
        # The 365/252 ratio already accounts for weekends/holidays.
        days_needed = math.ceil(bars_needed * 365 / 252)
    else:
        bars_per_day = minutes_per_day / bar_minutes
        days_needed = math.ceil(bars_needed / bars_per_day * safety_factor)

    days_needed = max(days_needed, 1)
    max_days = MAX_DURATION_DAYS[bar_size]
    if days_needed > max_days:
        raise ValueError(
            f"Required duration ({days_needed} days) exceeds IB max ({max_days}) "
            f"for bar_size={bar_size!r}. Split into multiple reqHistoricalData calls."
        )

    if days_needed >= 365:
        return f"{math.ceil(days_needed / 365)} Y"
    if days_needed >= 30:
        return f"{math.ceil(days_needed / 30)} M"
    if days_needed >= 7:
        return f"{math.ceil(days_needed / 7)} W"
    return f"{days_needed} D"


def is_new_bar(bar_size: str, dt: datetime) -> bool:
    """Polled-mode fallback only. The event-driven default path uses BarAggregator."""
    minutes = bar_size_to_minutes(bar_size)
    total_minutes = int(dt.timestamp()) // 60
    return total_minutes % minutes == 0
