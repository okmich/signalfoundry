"""
Signal file generation for ROC analysis.

Source / Attribution
--------------------
The signal-file format (``YYYYMMDD  value``) is the input format expected by Timothy Masters' C++ ROC.exe binary, described in:

    Timothy Masters, "Statistically Sound Indicators For Financial Market Prediction", Apress, 2013.

The ``SignalGenerator`` class is an original contribution of this project. It is provided to make it easy to export Python-computed indicator values in
the format accepted by Masters' reference binaries so that Python results can be cross-validated against the C++ implementation.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd


class SignalGenerator:
    """
    Export indicator values to Masters' signal-file format.

    Signal files have the format::

        YYYYMMDD  value
        20200103  0.523451
        20200106 -0.234120
        ...

    This format is consumed by Masters' C++ ``ROC.exe`` binary for cross-validation of Python ROC results against the reference C++ implementation.
    All methods are static — the class serves purely as a namespace.
    """

    @staticmethod
    def from_indicator(market_data: pd.DataFrame, indicator_func: Callable, output_path: Union[str, Path],
                       params: Optional[Dict] = None, date_column: Optional[str] = None) -> pd.DataFrame:
        """
        Compute an indicator and write its values to a signal file.

        Parameters
        ----------
        market_data : pd.DataFrame
            OHLCV data.  Must have either a datetime index or a column
            named *date_column*.
        indicator_func : Callable
            ``func(market_data, **params) -> pd.Series``
        output_path : str or Path
            Destination file path.
        params : dict, optional
            Keyword arguments forwarded to *indicator_func*.
        date_column : str, optional
            Name of the date column.  When ``None`` the DataFrame index is
            used.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``date`` (datetime) and ``signal``
            (float), NaN rows removed.
        """
        params = params or {}
        signals: pd.Series = indicator_func(market_data, **params)

        if date_column and date_column in market_data.columns:
            dates = pd.to_datetime(market_data[date_column])
        else:
            dates = pd.to_datetime(market_data.index)

        df = pd.DataFrame({"date": dates, "signal": signals.values}).dropna()
        SignalGenerator._write(df, output_path)
        return df[["date", "signal"]]

    @staticmethod
    def from_series(dates: pd.Series, signals: pd.Series, output_path: Union[str, Path]) -> None:
        """
        Write a pre-computed signal series to a signal file.

        Parameters
        ----------
        dates : pd.Series
            Dates (datetime or ``YYYYMMDD`` strings).
        signals : pd.Series
            Signal values aligned with *dates*.
        output_path : str or Path
            Destination file path.
        """
        df = pd.DataFrame(
            {"date": pd.to_datetime(dates), "signal": signals.values}
        ).dropna()
        SignalGenerator._write(df, output_path)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write(df: pd.DataFrame, output_path: Union[str, Path]) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            for _, row in df.iterrows():
                date_str = row["date"].strftime("%Y%m%d")
                fh.write(f"{date_str} {row['signal']:.6f}\n")
        print(
            f"Signal file written: {output_path}  "
            f"({len(df)} records, "
            f"{df['date'].min().date()} – {df['date'].max().date()})"
        )
