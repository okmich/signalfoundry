import pandas as pd

from .ma import ema, sma, tema, dema
from .momentum import log_returns


def create_price_based_features(df, returns_lags=(1, 3, 6, 12), col_name="close", rolling_period: int = 24,
                                ma_periods=(10, 20), ema_periods=None, dema_periods=None, tema_periods=None) -> pd.DataFrame:
    data = df.copy()
    # roc and returns
    for n in returns_lags:
        data[f"log_returns_{n}"] = log_returns(data[col_name], period=n)

    # ma
    if ma_periods:
        ma_short, ma_long = ma_periods
        data[f"ma_{ma_short}"] = sma(data[col_name].values, period=ma_short)
        data[f"ma_{ma_long}"] = sma(data[col_name].values, period=ma_long)
        data[f"ma_diff"] = data[f"ma_{ma_long}"] - data[f"ma_{ma_short}"]
    # ema
    if ema_periods:
        ema_short, ema_long = ema_periods
        data[f"ema_{ema_short}"] = ema(data[col_name].values, period=ema_short)
        data[f"ema_{ema_long}"] = ema(data[col_name].values, period=ema_long)
        data[f"ema_diff"] = data[f"ema_{ema_long}"] - data[f"ema_{ema_short}"]
    # dema
    if dema_periods:
        dema_short, dema_long = dema_periods
        data[f"dema_{dema_short}"] = dema(data[col_name].values, period=dema_short)
        data[f"dema_{dema_long}"] = dema(data[col_name].values, period=dema_long)
        data[f"dema_diff"] = data[f"dema_{dema_long}"] - data[f"dema_{dema_short}"]
    # tema
    if tema_periods:
        tema_short, tema_long = tema_periods
        data[f"tema_{tema_short}"] = tema(data[col_name].values, period=tema_short)
        data[f"tema_{tema_long}"] = tema(data[col_name].values, period=tema_long)
        data[f"tema_diff"] = data[f"tema_{tema_long}"] - data[f"tema_{tema_short}"]

    # higher moments
    data[f"skew_{rolling_period}"] = (
        data[f"log_returns_{returns_lags[0]}"].rolling(rolling_period).skew()
    )
    data[f"kurt_{rolling_period}"] = (
        data[f"log_returns_{returns_lags[0]}"].rolling(rolling_period).kurt()
    )

    return data.drop(columns=df.columns)
