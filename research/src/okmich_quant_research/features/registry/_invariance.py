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

#: Per-COLUMN stamps for multi-output entries, keyed ``entry@column``. A companion to the per-entry
#: file, not a replacement: an entry whose columns AGREE keeps its ordinary single stamp and appears
#: nowhere here. Only the entries stamped ``Parity.HETEROGENEOUS`` need it, and for those it carries the
#: verdict the per-entry schema cannot express -- which of ``trend.bollinger_band``'s five outputs is
#: the odd one.
COLUMN_INVARIANCE_CSV = Path(__file__).with_name("_invariance_columns.csv")


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


def _check_conjugate_pairs(by_qualified: dict[str, FeatureEntry], stamps: dict[str, FeatureInvariance]) -> None:
    """Raise unless every APPLIED one-sided stamp pairs with a catalogued partner that points back.

    Scoped to stamps that landed on a catalog entry: a stamp matching nothing is a corpus artefact,
    already reported by ``apply_invariance``'s return value, and its conjugate is unreachable anyway.

    Once a one-sided stamp IS on a shipped entry, though, a partner that will not resolve, is not itself
    one-sided, or names a third feature MISDIRECTS ``hmm_screener._check_one_sided``. That guard clears a
    one-sided feature only when ``inv.conjugate`` is literally among the subset's columns, so a stale or
    wrong conjugate never matches and the guard fires on subsets that already hold BOTH halves — the
    exact case it exists to let through. Under ``OneSidedPolicy.EXCLUDE`` that marks a sound subset
    FRAGILE and keeps it off the Pareto frontier; under ``RAISE`` it aborts the screen outright. And the
    message it emits names a conjugate no pool can contain, so its own remedy ("include the conjugate")
    cannot be followed.

    A corrupted verdict, then, not a coverage gap — so it raises rather than joining
    UNSTAMPED_MEASUREMENTS in the returned list.
    """
    broken: list[str] = []
    for name, stamp in stamps.items():
        if stamp.parity is not Parity.ONE_SIDED or name not in by_qualified:
            continue
        partner = by_qualified.get(stamp.conjugate)
        if partner is None:
            broken.append(f"{name} -> conjugate {stamp.conjugate!r} is not a catalogued feature")
        elif partner.invariance is None:
            broken.append(f"{name} -> conjugate {stamp.conjugate!r} carries no invariance stamp")
        elif partner.invariance.parity is not Parity.ONE_SIDED:
            broken.append(f"{name} -> conjugate {stamp.conjugate!r} is stamped "
                          f"{partner.invariance.parity.value!r}, not {Parity.ONE_SIDED.value!r}")
        elif partner.invariance.conjugate != name:
            broken.append(f"{name} -> conjugate {stamp.conjugate!r} points back to "
                          f"{partner.invariance.conjugate!r}, not to {name!r}")
    if broken:
        raise ValueError(f"{INVARIANCE_CSV.name} has {len(broken)} unusable one-sided pair(s), which would "
                         f"misdirect the screener's one-sided guard onto a conjugate no subset can "
                         f"contain: {'; '.join(broken)}")


def apply_invariance(catalog: list[FeatureEntry], stamps: dict[str, FeatureInvariance]) -> list[str]:
    """Attach stamps to catalog entries by qualified name; return the stamped names that matched nothing.

    Called once at ``_catalog`` import time, not per ``FeatureRegistry()`` construction — the catalog is a
    module-level singleton, so stamping it repeatedly would be a redundant global side effect.

    Unmatched names are RETURNED rather than raised on: the measurement corpus is a recipe pool, which
    legitimately contains columns that are not one-to-one with catalog entries. The caller decides how
    loud to be. A broken one-sided PAIR is the opposite case and raises — see ``_check_conjugate_pairs``.
    """
    by_qualified = {e.qualified_name: e for e in catalog}
    for name, stamp in stamps.items():
        entry = by_qualified.get(name)
        if entry is not None:
            entry.invariance = stamp
    # After the apply loop, never inside it: a partner's stamp may not be attached yet when its own
    # half is reached, so checking mid-loop would report pairs broken in stamp-file order.
    _check_conjugate_pairs(by_qualified, stamps)
    return sorted(set(stamps) - set(by_qualified))


def load_column_stamps(path: Path | None = None) -> dict[str, dict[str, FeatureInvariance]]:
    """Read ``_invariance_columns.csv`` into ``{entry: {column: FeatureInvariance}}``.

    MISSING FILE IS FINE, malformed is not. The per-column file is an enrichment: without it
    ``is_eligible`` still gates correctly, it just cannot say HOW a heterogeneous entry's columns split.
    A malformed file, by contrast, would silently empty the mapping and turn every heterogeneous entry
    back into a bare "no per-entry verdict" -- losing exactly the detail the file exists to carry -- so
    it raises, for the same reason ``load_invariance_stamps`` does.
    """
    src = path if path is not None else COLUMN_INVARIANCE_CSV
    out: dict[str, dict[str, FeatureInvariance]] = {}
    if not src.exists():
        return out
    with open(src, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = set(_EXPECTED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{src.name} is missing column(s) {sorted(missing)}; got {reader.fieldnames}")
        for lineno, row in enumerate(reader, start=2):
            key = (row["feature"] or "").strip()
            if "@" not in key:
                raise ValueError(f"{src.name}:{lineno} key {key!r} is not 'entry@column'; the per-column "
                                 f"file must never hold a bare entry name — that belongs in "
                                 f"{INVARIANCE_CSV.name}")
            entry, column = key.split("@", 1)
            try:
                stamp = FeatureInvariance(parity=Parity(row["parity"].strip()),
                                          scale_class=ScaleClass(row["scale_class"].strip()),
                                          conjugate=(row["conjugate"] or "").strip(),
                                          measured_on=(row["measured_on"] or "").strip())
            except ValueError as exc:
                raise ValueError(f"{src.name}:{lineno} bad stamp for {key!r}: {exc}") from exc
            if column in out.setdefault(entry, {}):
                raise ValueError(f"{src.name}:{lineno} duplicates column {key!r}")
            out[entry][column] = stamp
    return out


INVARIANCE_STAMPS: dict[str, FeatureInvariance] = load_invariance_stamps()
COLUMN_STAMPS: dict[str, dict[str, FeatureInvariance]] = load_column_stamps()
