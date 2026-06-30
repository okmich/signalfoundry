r"""Daily macro-regime data pipeline (FRED-sourced exogenous conditioners).

Fetches slow, causally-clean daily/weekly macro series from FRED and stamps each observation with the first instant an
intraday bar may consume it. These feed downstream intraday models as a conditioner / gate / sizer — NOT as a signal.

Store: per-series parquet + a ``_metadata.json`` metastore under a store directory (default ``E:\data_dump\macro_data\daily``),
mirroring the fetch-mt5-data convention. One idempotent command updates both data and metastore:

    fetch-macro-data            # incremental tail refresh of every series
    fetch-macro-data --full     # re-fetch full history

Series (channel): VIX, VIX_3M (volatility); CREDIT_SPREAD=BAA10Y, NFCI weekly (risk); USD_BROAD (usd); US_2Y, US_10Y (rates).
Derived: curve_2s10s. Scope: real instruments only — never synthetics.
"""
