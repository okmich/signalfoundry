"""FRED-sourced macro series fetchers.

``fred.fetch(series, start, end)`` returns a long-format DataFrame with one row
per observation date:

- ``date``               : ``datetime64[ns]`` — observation (trade) date, midnight-naive.
- ``series``             : ``str`` — canonical ``MacroSeries`` value.
- ``value``              : ``float`` — raw level.
- ``available_from_utc`` : ``datetime64[ns, UTC]`` — first instant an intraday
  bar may consume this value (see ``okmich_quant_pipeline.macro._types`` causal convention).
"""
