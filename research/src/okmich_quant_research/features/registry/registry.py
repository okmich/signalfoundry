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

from ._axis import Axis, is_eligible
from ._schema import (
    FeatureEntry,
    Parity,
    ScaleClass,
    RELEVANCE_LEVELS,
    SIGNAL_TYPES,
    HORIZONS,
    MARKET_REGIMES,
    CRITICAL, HIGH, MEDIUM, LOW, NONE,
)
from ._catalog import CATALOG


def _segments_elided(given: list[str], actual: list[str]) -> bool:
    """Do the ``given`` module segments appear in ``actual`` in order (gaps allowed)?

    Lets a caller write ``timothymasters.momentum`` for catalogued ``timothymasters.single.momentum``
    without the registry accepting an unrelated module that merely shares a function name. Order is
    required, so ``momentum.timothymasters`` does not match.
    """
    it = iter(actual)
    return all(seg in it for seg in given)


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
        """Features whose sign carries BUY/SELL meaning.

        NOTE this is the hand-DECLARED flag. For the measured answer use ``by_parity(Parity.ODD)`` —
        the two disagree: ``timothymasters.single.trend.aroon_up`` is declared non-directional yet measures
        ONE_SIDED (half of an odd pair), and ``momentum.plus_di``/``minus_di`` are declared directional
        yet neither is odd on its own.
        """
        return _FeatureView([e for e in self._entries if e.directional])

    def by_parity(self, *parities: Parity) -> "_FeatureView":
        """Features whose MEASURED parity is one of ``parities``. Unstamped features never match."""
        wanted = set(parities)
        return _FeatureView([e for e in self._entries if e.invariance is not None and e.invariance.parity in wanted])

    def by_scale_class(self, *scale_classes: ScaleClass) -> "_FeatureView":
        """Features whose MEASURED scale class is one of ``scale_classes``. Unstamped never match."""
        wanted = set(scale_classes)
        return _FeatureView([e for e in self._entries
                             if e.invariance is not None and e.invariance.scale_class in wanted])

    def eligible_for(self, axis: Axis) -> "_FeatureView":
        """Features admissible on ``axis``, decided by tag + measured invariance (``_axis.is_eligible``).

        There is no hand-maintained membership list behind this: an unmeasured feature is admitted on
        the tag gate alone, and a feature whose measured behaviour contradicts its name is rejected.
        """
        return _FeatureView([e for e in self._entries if is_eligible(e, axis)[0]])

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
                "parity":             e.invariance.parity.value if e.invariance else None,
                "scale_class":        e.invariance.scale_class.value if e.invariance else None,
                "conjugate":          e.invariance.conjugate if e.invariance else "",
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

        Accepts a short name (``"vpin"``), a qualified name
        (``"microstructure.order_flow.vpin"``), an unambiguous partially-qualified
        name (``"timothymasters.momentum.ppo"``), and any of those carrying a
        column selector (``"candle.candle_features@range"``).  Raises ``KeyError``
        if not found; raises ``ValueError`` with disambiguation hint if the name
        matches multiple modules.

        Use :meth:`resolve` when the column matters -- it is what decides eligibility
        on a multi-output entry.
        """
        return self.resolve(name)[0]

    def resolve(self, name: str) -> tuple[FeatureEntry, str | None]:
        """Resolve a feature name to ``(entry, column)``.

        The ``entry@column`` form is how the rest of this package addresses ONE output of a
        multi-output feature -- ``registry/_invariance_columns.csv`` is keyed exactly that way, and a
        screener subset names the column its recipe selected. Resolving it HERE is what lets the
        invariance gate judge the column that was actually used instead of skipping the feature
        entirely; a lookup that understood only the bare qualified name silently disabled that gate
        on every subset naming a column.

        The partial fallback accepts a name whose module path has segments ELIDED --
        ``timothymasters.momentum.ppo`` for catalogued ``timothymasters.single.momentum.ppo`` -- by
        requiring the function name to match exactly and the given module segments to appear in the
        catalogued module in order. It resolves only when exactly one entry matches, so it can never
        quietly pick a different feature than the caller meant. A bare short name is the same rule with
        no module segments to satisfy.
        """
        base, _, column = name.partition("@")
        column = column or None
        if base in self._by_qualified:
            return self._by_qualified[base], column
        *prefix, short = base.split(".")
        matches = [e for e in self._by_name.get(short, [])
                   if _segments_elided(prefix, e.module.split("."))]
        if not matches:
            raise KeyError(f"Feature {name!r} not found in registry.")
        if len(matches) > 1:
            options = [e.qualified_name for e in matches]
            raise ValueError(
                f"Ambiguous name {name!r} matches {len(matches)} entries. "
                f"Use a qualified name: {options}"
            )
        return matches[0], column

    def all(self) -> list[FeatureEntry]:
        return list(self._entries)

    def __repr__(self) -> str:
        return f"FeatureRegistry({len(self._entries)} features)"