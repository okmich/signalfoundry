"""
Feature Registry
================
Queryable catalog of all features in okmich_quant_features.

Usage
-----
    from okmich_quant_research.features.registry import FeatureRegistry

    reg = FeatureRegistry()

    # Candidates for regime classification (HIGH or better)
    reg.candidates_for("regime", min_relevance="HIGH")

    # All directional features usable with only price data
    reg.directional().needs_only_price()

    # Summary table as a DataFrame
    reg.summary()
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from ._schema import (
    FeatureEntry,
    RELEVANCE_LEVELS,
    SIGNAL_TYPES,
    HORIZONS,
    MARKET_REGIMES,
    CRITICAL, HIGH, MEDIUM, LOW, NONE,
)
from ._catalog import CATALOG


class _FeatureView:
    """
    Chainable view over a list of FeatureEntry objects.
    All filter methods return a new _FeatureView so calls can be chained:
        reg.candidates_for("regime").directional().needs_only_price()
    """

    def __init__(self, entries: List[FeatureEntry]):
        self._entries = entries

    # ── Iterability ───────────────────────────────────────────────────────────

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"FeatureView({len(self._entries)} features)"

    def to_list(self) -> List[FeatureEntry]:
        return list(self._entries)

    def names(self) -> List[str]:
        """Return function names (short names, may have duplicates across modules)."""
        return [e.name for e in self._entries]

    def qualified_names(self) -> List[str]:
        """Return fully-qualified names: module.function_name."""
        return [e.qualified_name for e in self._entries]

    # ── Filters ───────────────────────────────────────────────────────────────

    def by_signal_type(self, signal_type: str) -> "_FeatureView":
        assert signal_type in SIGNAL_TYPES, f"Unknown signal_type {signal_type!r}"
        return _FeatureView([e for e in self._entries if e.signal_type == signal_type])

    def by_module(self, module: str) -> "_FeatureView":
        """Partial match — 'microstructure' matches all microstructure sub-modules."""
        return _FeatureView([e for e in self._entries if module in e.module])

    def by_horizon(self, horizon: str) -> "_FeatureView":
        assert horizon in HORIZONS, f"Unknown horizon {horizon!r}"
        return _FeatureView([e for e in self._entries if e.horizon == horizon])

    def works_in(self, regime: str) -> "_FeatureView":
        assert regime in MARKET_REGIMES, f"Unknown regime {regime!r}"
        return _FeatureView([e for e in self._entries if regime in e.works_best_in])

    def directional(self) -> "_FeatureView":
        """Features whose sign carries BUY/SELL meaning."""
        return _FeatureView([e for e in self._entries if e.directional])

    def causal_only(self) -> "_FeatureView":
        """Features that use only past/current bar data."""
        return _FeatureView([e for e in self._entries if e.causal])

    def needs_only_price(self) -> "_FeatureView":
        """Features requiring no volume, no spread, no benchmark."""
        return _FeatureView([
            e for e in self._entries
            if not e.needs_volume and not e.needs_spread and not e.needs_benchmark
        ])

    def filter(self, needs_volume: Optional[bool] = None, needs_spread: Optional[bool] = None,
               needs_benchmark: Optional[bool] = None, output_type: Optional[str] = None,
               directional: Optional[bool] = None) -> "_FeatureView":
        result = self._entries
        if needs_volume is not None:
            result = [e for e in result if e.needs_volume == needs_volume]
        if needs_spread is not None:
            result = [e for e in result if e.needs_spread == needs_spread]
        if needs_benchmark is not None:
            result = [e for e in result if e.needs_benchmark == needs_benchmark]
        if output_type is not None:
            result = [e for e in result if e.output_type == output_type]
        if directional is not None:
            result = [e for e in result if e.directional == directional]
        return _FeatureView(result)

    def candidates_for(self, task: str, min_relevance: str = HIGH) -> "_FeatureView":
        """
        Return features with relevance >= min_relevance for a given task.

        Parameters
        ----------
        task : str
            One of ``"regime"``, ``"return"``, or ``"direction"``.
        min_relevance : str
            Minimum relevance level (CRITICAL > HIGH > MEDIUM > LOW > NONE).
        """
        assert task in ("regime", "return", "direction"), (
            f"task must be 'regime', 'return', or 'direction', got {task!r}"
        )
        assert min_relevance in RELEVANCE_LEVELS, (
            f"min_relevance must be one of {RELEVANCE_LEVELS}"
        )
        cutoff = RELEVANCE_LEVELS.index(min_relevance)
        attr = {
            "regime":    "regime_relevance",
            "return":    "return_relevance",
            "direction": "direction_relevance",
        }[task]
        return _FeatureView([
            e for e in self._entries
            if RELEVANCE_LEVELS.index(getattr(e, attr)) <= cutoff
        ])

    # ── Output ────────────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame with one row per feature."""
        rows = []
        for e in self._entries:
            rows.append({
                "name":               e.name,
                "module":             e.module,
                "signal_type":        e.signal_type,
                "description":        e.description,
                "regime_relevance":   e.regime_relevance,
                "return_relevance":   e.return_relevance,
                "direction_relevance":e.direction_relevance,
                "horizon":            e.horizon,
                "directional":        e.directional,
                "causal":             e.causal,
                "output_type":        e.output_type,
                "needs_volume":       e.needs_volume,
                "needs_spread":       e.needs_spread,
                "needs_benchmark":    e.needs_benchmark,
                "works_best_in":      ", ".join(e.works_best_in),
                "notes":              e.notes,
            })
        return pd.DataFrame(rows)


class FeatureRegistry(_FeatureView):
    """
    Central registry of all feature-computing functions in okmich_quant_features.

    Provides a queryable, chainable interface over 270+ feature entries.

    Quick-start
    -----------
    >>> from okmich_quant_research.features.registry import FeatureRegistry
    >>> reg = FeatureRegistry()
    >>> len(reg)
    216

    >>> regime_candidates = reg.candidates_for("regime", min_relevance="HIGH")
    >>> price_only = reg.candidates_for("return").needs_only_price()

    >>> entry = reg.get("vpin")
    >>> entry.signal_type, entry.regime_relevance
    ('toxicity', 'CRITICAL')

    >>> df = reg.summary()
    >>> df[df.regime_relevance == "CRITICAL"]
    """

    def __init__(self):
        super().__init__(CATALOG)
        # Build lookup dicts
        self._by_qualified: dict[str, FeatureEntry] = {
            e.qualified_name: e for e in CATALOG
        }
        # Short name → list of entries (handles duplicates across modules)
        self._by_name: dict[str, list[FeatureEntry]] = {}
        for e in CATALOG:
            self._by_name.setdefault(e.name, []).append(e)

    def get(self, name: str) -> FeatureEntry:
        """
        Look up a feature by name.

        Accepts either a short name (``"vpin"``) or a qualified name
        (``"microstructure.order_flow.vpin"``).  Raises ``KeyError`` if not
        found; raises ``ValueError`` with disambiguation hint if the short
        name matches multiple modules.
        """
        if name in self._by_qualified:
            return self._by_qualified[name]
        matches = self._by_name.get(name, [])
        if not matches:
            raise KeyError(f"Feature {name!r} not found in registry.")
        if len(matches) > 1:
            options = [e.qualified_name for e in matches]
            raise ValueError(
                f"Ambiguous name {name!r} matches {len(matches)} entries. "
                f"Use a qualified name: {options}"
            )
        return matches[0]

    def all(self) -> list[FeatureEntry]:
        return list(self._entries)

    def __repr__(self) -> str:
        return f"FeatureRegistry({len(self._entries)} features)"