"""Measured invariance stamps for catalogued features.

The stamps live in ``_invariance.csv`` rather than in ``_catalog.py`` on purpose. They are DATA — the
output of a measurement (``okmich_quant_research.features.invariance``) — not a hand-written judgement,
and re-measuring on a new corpus must be a data refresh, not a 1000-line code diff. Keeping them in a
file is also what makes ``measured_on`` provenance meaningful: every row says where it came from.

A feature ABSENT from the file is stamped ``None``, which means "never measured" — NOT "measured and
found neutral". ``registry._axis.is_eligible`` treats the two very differently: an unmeasured feature
passes the tag gate with a reason string saying so, while a ``MIXED`` feature is rejected outright.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ._schema import FeatureEntry, FeatureInvariance, Parity, ScaleClass

INVARIANCE_CSV = Path(__file__).with_name("_invariance.csv")
_EXPECTED_COLUMNS = ("feature", "parity", "scale_class", "conjugate", "measured_on")


def load_invariance_stamps(path: Path | None = None) -> dict[str, FeatureInvariance]:
    """Read ``_invariance.csv`` into ``{qualified_name: FeatureInvariance}``.

    Raises rather than degrading silently: a malformed stamp file would otherwise leave every feature
    unmeasured, which reads identically to "we never ran the probe" and would quietly re-open the
    declared-not-measured hole this whole layer exists to close.
    """
    src = path if path is not None else INVARIANCE_CSV
    stamps: dict[str, FeatureInvariance] = {}
    with open(src, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = set(_EXPECTED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{src.name} is missing column(s) {sorted(missing)}; got {reader.fieldnames}")
        for lineno, row in enumerate(reader, start=2):
            name = (row["feature"] or "").strip()
            if not name:
                raise ValueError(f"{src.name}:{lineno} has an empty 'feature'")
            if name in stamps:
                raise ValueError(f"{src.name}:{lineno} duplicates feature {name!r}")
            try:
                stamps[name] = FeatureInvariance(parity=Parity(row["parity"].strip()),
                                                 scale_class=ScaleClass(row["scale_class"].strip()),
                                                 conjugate=(row["conjugate"] or "").strip(),
                                                 measured_on=(row["measured_on"] or "").strip())
            except ValueError as exc:
                raise ValueError(f"{src.name}:{lineno} bad stamp for {name!r}: {exc}") from exc
    return stamps


def apply_invariance(catalog: list[FeatureEntry], stamps: dict[str, FeatureInvariance]) -> list[str]:
    """Attach stamps to catalog entries by qualified name; return the stamped names that matched nothing.

    Called once at ``_catalog`` import time, not per ``FeatureRegistry()`` construction — the catalog is a
    module-level singleton, so stamping it repeatedly would be a redundant global side effect.

    Unmatched names are RETURNED rather than raised on: the measurement corpus is a recipe pool, which
    legitimately contains columns that are not one-to-one with catalog entries. The caller decides how
    loud to be.
    """
    by_qualified = {e.qualified_name: e for e in catalog}
    for name, stamp in stamps.items():
        entry = by_qualified.get(name)
        if entry is not None:
            entry.invariance = stamp
    return sorted(set(stamps) - set(by_qualified))


INVARIANCE_STAMPS: dict[str, FeatureInvariance] = load_invariance_stamps()
