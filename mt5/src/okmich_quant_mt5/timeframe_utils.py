from datetime import timedelta, datetime

import MetaTrader5 as mt5

timeframe_minutes_dict = {
    mt5.TIMEFRAME_M1: 1,
    mt5.TIMEFRAME_M2: 2,
    mt5.TIMEFRAME_M3: 3,
    mt5.TIMEFRAME_M4: 4,
    mt5.TIMEFRAME_M5: 5,
    mt5.TIMEFRAME_M6: 6,
    mt5.TIMEFRAME_M10: 10,
    mt5.TIMEFRAME_M12: 12,
    mt5.TIMEFRAME_M15: 15,
    mt5.TIMEFRAME_M20: 20,
    mt5.TIMEFRAME_M30: 30,
    mt5.TIMEFRAME_H1: 60,
    mt5.TIMEFRAME_H2: 120,
    mt5.TIMEFRAME_H3: 180,
    mt5.TIMEFRAME_H4: 240,
    mt5.TIMEFRAME_H6: 360,
    mt5.TIMEFRAME_H8: 480,
    mt5.TIMEFRAME_H12: 720,
    mt5.TIMEFRAME_D1: 1440,
    mt5.TIMEFRAME_W1: 10080,
    mt5.TIMEFRAME_MN1: 43200,
}


def number_of_minutes_in_timeframe(timeframe) -> int:
    return timeframe_minutes_dict.get(timeframe, -1)


def is_timeframe_match(timeframe, dt: datetime) -> bool:
    minute = dt.minute
    second = dt.second

    # For all timeframes, a new candle starts when seconds are 0
    if second != 0:
        return False

    # Get the timeframe's minute periodicity
    period_minutes = number_of_minutes_in_timeframe(timeframe)
    if period_minutes == -1:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # For minute-based timeframes (M1 to H1)
    if period_minutes <= 60:
        return minute % period_minutes == 0

    # For hourly timeframes (H1 to H12)
    if 60 < period_minutes <= 720:
        return minute == 0 and dt.hour % (period_minutes // 60) == 0

    # For daily timeframe (D1)
    if period_minutes == 1440:
        return minute == 0 and dt.hour == 0

    # For weekly timeframe (W1)
    if period_minutes == 10080:
        return minute == 0 and dt.hour == 0 and dt.weekday() == 0  # Monday 00:00

    # For monthly timeframe (MN1)
    if period_minutes == 43200:
        return minute == 0 and dt.hour == 0 and dt.day == 1  # First day of month 00:00

    return False


def get_past_datetime(
    timeframe: int, dt: datetime, num_bars: int, week_start_day: int = 0
) -> datetime:
    if num_bars < 0:
        raise ValueError("Number of bars must be non-negative")

    period_minutes = number_of_minutes_in_timeframe(timeframe)
    if period_minutes == -1:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. Supported timeframes: {list(timeframe_minutes_dict.keys())}"
        )

    # For minute-based, hourly, and daily timeframes (M1 to D1)
    if period_minutes <= 1440:
        total_minutes = period_minutes * num_bars
        return dt - timedelta(minutes=total_minutes)

    # For weekly timeframe (W1)
    if period_minutes == 10080:
        return dt - timedelta(weeks=num_bars)

    # For monthly timeframe (MN1)
    if period_minutes == 43200:
        # Subtract months while handling year transitions
        year = dt.year
        month = dt.month - num_bars
        while month <= 0:
            year -= 1
            month += 12
        try:
            return dt.replace(
                year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0
            )
        except ValueError:
            month += 1
            if month > 12:
                year += 1
                month = 1
            return dt.replace(
                year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0
            )

    return dt
