"""One module per source; each exports ``fetch(year_min, year_max)``.

Returned DataFrame schema (all fetchers):

- ``release_date``  : ``datetime.date`` — calendar date of the release.
- ``event_name``    : ``EventName`` — canonical event identifier.
- ``source``        : ``Source`` — publishing agency.
- ``impact_tier``   : ``ImpactTier`` — coarse impact bucket.

The orchestrator (``build.py``) joins these frames and applies release
times via ``release_times.to_utc`` to produce the final UTC parquet.
"""
