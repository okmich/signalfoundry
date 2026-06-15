"""Research-time posterior tooling (offline).

Sibling to ``okmich_quant_ml.posterior_inference`` (live/offline posterior *processing*). This package holds the
*research* layer that consumes those processing primitives plus forward-looking outcomes, and is **never** imported by a
live signal path. Currently: the asymmetry discovery + walk-forward validation stack under ``asymmetry``.
"""
