from typing import Tuple

import numpy as np
import pandas as pd


class TemporalFeature:

    def __init__(self, data: pd.DataFrame):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")

        self.data = data
        self.index = data.index

    def hour_of_day(self) -> pd.Series:
        return pd.Series(self.index.hour, index=self.index, name="hour_of_day")

    def day_of_week(self) -> pd.Series:
        return pd.Series(self.index.dayofweek, index=self.index, name="day_of_week")

    def day_of_month(self) -> pd.Series:
        return pd.Series(self.index.day, index=self.index, name="day_of_month")

    def month(self) -> pd.Series:
        return pd.Series(self.index.month, index=self.index, name="month")

    def quarter(self) -> pd.Series:
        return pd.Series(self.index.quarter, index=self.index, name="quarter")

    def hour_of_day_cyclic(self) -> Tuple[pd.Series, pd.Series]:
        hour = self.index.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        return pd.Series(hour_sin, index=self.index, name="hour_sin"), pd.Series(
            hour_cos, index=self.index, name="hour_cos"
        )

    def day_of_week_cyclic(self) -> Tuple[pd.Series, pd.Series]:
        dow = self.index.dayofweek
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

        return pd.Series(dow_sin, index=self.index, name="dow_sin"), pd.Series(
            dow_cos, index=self.index, name="dow_cos"
        )

    def market_session(self, market: str = "US", input_tz: str = None) -> pd.Series:
        # Convert to market timezone if needed
        if market == "US":
            tz = "America/New_York"
            pre_market = (9, 0)  # 9:00 AM
            market_open = (9, 30)  # 9:30 AM
            market_close = (16, 0)  # 4:00 PM
            after_hours_end = (20, 0)  # 8:00 PM
        elif market == "EU":
            tz = "Europe/London"
            pre_market = (7, 0)
            market_open = (8, 0)
            market_close = (16, 30)
            after_hours_end = (20, 0)
        elif market == "ASIA":
            tz = "Asia/Tokyo"
            pre_market = (8, 0)
            market_open = (9, 0)
            market_close = (15, 0)
            after_hours_end = (18, 0)
        else:
            raise ValueError(f"Unsupported market: {market}")

        # Convert to market timezone so session hours are compared in local time.
        if self.index.tz is not None:
            local_time = self.index.tz_convert(tz)
        else:
            if input_tz is None:
                raise ValueError(
                    "Index has no timezone info. Pass input_tz='UTC' (or the actual "
                    "source timezone) so timestamps are interpreted unambiguously before "
                    "converting to the market timezone."
                )
            local_time = self.index.tz_localize(input_tz).tz_convert(tz)
        sessions = []
        for dt in local_time:
            hour, minute = dt.hour, dt.minute
            time_minutes = hour * 60 + minute

            pre_market_minutes = pre_market[0] * 60 + pre_market[1]
            market_open_minutes = market_open[0] * 60 + market_open[1]
            market_close_minutes = market_close[0] * 60 + market_close[1]
            after_hours_end_minutes = after_hours_end[0] * 60 + after_hours_end[1]

            if (
                time_minutes < pre_market_minutes
                or time_minutes >= after_hours_end_minutes
            ):
                sessions.append("closed")
            elif time_minutes < market_open_minutes:
                sessions.append("pre_market")
            elif time_minutes < market_close_minutes:
                sessions.append("open")
            else:
                sessions.append("after_hours")

        return pd.Series(sessions, index=self.index, name="market_session")

    def is_market_open(self, market: str = "US") -> pd.Series:
        sessions = self.market_session(market)
        return pd.Series(sessions == "open", index=self.index, name="is_market_open")
