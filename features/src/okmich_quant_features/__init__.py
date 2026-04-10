import pandas as pd

from .fractional_diff import FractionalDifferentiator, fractional_differentiate_series, \
    get_optimal_fractional_differentiation_order
from .candle import candle_features


def zscore(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    if min_periods is None:
        min_periods = window // 2
    roll = series.rolling(window, min_periods=min_periods)
    return (series - roll.mean()) / (roll.std() + 1e-9)


def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes(include=['float64']).columns
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    ints = df.select_dtypes(include=['int64']).columns
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df


def optimize_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    return optimize_floats(optimize_ints(df))

