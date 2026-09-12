"""Screening axes — the partition a latent-state model is asked to separate.

WHY THIS IS NOT ``signal_type``
-------------------------------
``SIGNAL_TYPES`` (``_schema.py``) is a 12-value FEATURE TAG: a coarse "what family is this indicator
from" label attached by hand to every catalogued function. It was doing double duty as the screener's
axis key, and that conflation is exactly what broke: a feature's namespace was treated as evidence of
what it measures, and it is not. Measured on FXPIG-Server M5 (2026-09-01..03), only 9 of the 23
candidates in the trend pool were tagged ``trend.*``; the pool was won by ``momentum.minus_di``.

So: the tag taxonomy is left alone, and the axis becomes its own small concept. An axis draws on
several tags (``AXIS_SIGNAL_TYPES``), but membership is ultimately decided by MEASUREMENT — see
``is_eligible``.

WHY FOUR, AND WHY THEY ARE NOT FOUR OF A KIND
---------------------------------------------
Three axes are PRICE-PATH partitions derived from the invariance taxonomy in ``_schema.py``: reflection
splits odd (knows which way) from even (knows only how much), and rescaling splits scale-carrying from
scale-free.

                 scale-carrying                 scale-free
    odd          DIRECTIONAL (signed drift)     DIRECTIONAL (normalised direction)
    even         VOLATILITY                     PATH_STRUCTURE

``LIQUIDITY`` is NOT in that table. It is a different input SUBSTRATE — volume and order flow, not the
price path — so the reflection test cannot validate it: reflecting the price leaves a volume-driven
feature invariant, and the test carries no information about it. The asymmetry is deliberate. Do not
"fix" it by forcing a parity requirement onto ``LIQUIDITY``.

``momentum`` is absent, deliberately. Measured, 32 of 38 momentum candidates test ODD and 23 of 38 sit
at ``|r| >= 0.816`` against some trend candidate — the ``MAX_VIF = 3.0`` near-duplicate line. Worse, the
redundancy TRACKS informativeness (Spearman(nearest-trend |r|, descIC) = +0.538; redundant candidates
median descIC 0.024 vs 0.009 for independent ones), so momentum duplicates trend precisely where it
carries signal and is independent only where it is near-noise. There is no salvageable independent core,
so it is not an axis: it is trend at half the lookback. ``DIRECTIONAL`` replaces both.

An ``acceleration`` axis was built and killed (12 verified odd, second-order candidates, exhaustive
depth-3 screen on 6 majors): WEAK at best, and its joint with trend was indistinguishable from a
dwell-matched uninformative label. Do not re-add it.

DIRECTIONAL IS ONE AXIS, NOT TWO
--------------------------------
The ``odd`` row spans two sub-cells and they are NOT split. But a K=2 normal-emission HMM fitted on a
subset mixing scale-carrying and scale-free directional features can partition on move MAGNITUDE rather
than direction, and no current metric would reveal it — which is why the screener reports the winning
subset's sub-cell composition in ``raw_details["subset_cells"]``.

EVIDENCE LIMITS
---------------
The taxonomy was validated on 4 FX majors plus one index. Invariance is instrument-agnostic, so the
CLASSIFICATION should hold everywhere; axis USEFULNESS demonstrably does not — indices gave 7 symbols
and 7 different winning features on the trend axis, 6 of 7 at noise floor. ``volatility``,
``path_structure`` and ``liquidity`` have never been HMM-screened on this corpus: orthogonality means
they CAN carry independent information, not that they do.
"""
from __future__ import annotations

from enum import StrEnum

from ._schema import FeatureEntry, FeatureInvariance, Parity, ScaleClass, SIGNAL_TYPES


class Axis(StrEnum):
    """A screening axis: the partition an axis-specific HMM is asked to separate."""

    DIRECTIONAL = "directional"        # replaces trend + momentum
    PATH_STRUCTURE = "path_structure"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"


#: Price-path axes — the three the invariance taxonomy defines and can validate.
PRICE_PATH_AXES: frozenset[Axis] = frozenset({Axis.DIRECTIONAL, Axis.PATH_STRUCTURE, Axis.VOLATILITY})

#: Which FEATURE TAGS an axis may draw on. A coarse first gate only: passing it means the feature is
#: plausibly on-axis by family, not that it measures the right thing. ``is_eligible`` decides that.
#:
#: DIRECTIONAL's tag set was WIDENED 2026-09-04 on measurement, not on preference. With 220 of 313
#: catalogue entries stamped, the odd share per tag reads:
#:
#:     trend 96%   momentum 95%   price_structure 93%   order_flow 90%
#:     composite 44%   information 38%   volume_structure 31%   toxicity 14%
#:     regime 9%   liquidity 0%   volatility 0%
#:
#: ``price_structure`` and ``order_flow`` are more directional than ``regime``, which was already in
#: the set -- 57 features measured ODD were being held off the axis by their family label alone. The
#: fix belongs HERE and not in the catalogue: retagging ``volume.ad`` or ``microstructure.order_flow.cvd``
#: as "momentum" to move them would destroy the record that they read a volume substrate, and would
#: redo the precise conflation this module exists to end (see the docstring: a namespace is not
#: evidence of what a feature measures). VOLATILITY already draws on order_flow and volume_structure,
#: so an axis spanning several substrate families is the established pattern, not a new one.
#:
#: Widening the TAG gate does not widen the real gate: an even feature admitted here is still rejected
#: by the invariance rule below. The tags left out (composite/information/volume_structure/toxicity)
#: are genuinely mixed families; nothing stops a stamped odd member of one being admitted later on the
#: same evidence, but a coin-flip family is not "plausibly on-axis".
AXIS_SIGNAL_TYPES: dict[Axis, frozenset[str]] = {
    Axis.DIRECTIONAL: frozenset({"trend", "momentum", "regime", "price_structure", "order_flow"}),
    Axis.PATH_STRUCTURE: frozenset({"price_structure", "regime", "information"}),
    Axis.VOLATILITY: frozenset({"volatility", "regime", "volume_structure", "order_flow"}),
    # "regime" added 2026-09-05: three spread features that are squarely liquidity
    # (liquidity_resilience, spread_volatility_elasticity, spread_volume_correlation) carry the
    # regime tag, and the tag gate dropped them before stage-0 saw them. The other three axes
    # already draw on regime; LIQUIDITY not doing so was an inconsistency, not a decision.
    Axis.LIQUIDITY: frozenset({"liquidity", "order_flow", "volume_structure", "toxicity",
                               "information", "regime"}),
}

#: Primary forward horizon per axis, in bars. DIRECTIONAL is 18 because that is where it was measured
#: to peak: on the persisted labels the directional label's unconditional separation reaches nsep 0.091
#: at H=18 against 0.080 at H=12, with the per-symbol best ranging 9..36.
#:
#: The other three are 12, and that number carries no evidence at all — none of them has ever been
#: HMM-screened on this corpus, so there was nothing to calibrate against. Treat them as placeholders
#: and derive each from its own label before trusting a result that depends on the horizon.
AXIS_PRIMARY_HORIZON: dict[Axis, int] = {
    Axis.DIRECTIONAL: 18,
    Axis.PATH_STRUCTURE: 12,
    Axis.VOLATILITY: 12,
    Axis.LIQUIDITY: 12,
}

#: Parity/scale requirement per price-path axis. The single source of truth for eligibility — there is
#: no per-axis list of feature names anywhere, by design.
_AXIS_INVARIANCE_RULE: dict[Axis, tuple[frozenset[Parity], frozenset[ScaleClass] | None]] = {
    # ONE_SIDED is admitted but is NOT safe alone: see the screener's one-sided guard.
    Axis.DIRECTIONAL: (frozenset({Parity.ODD, Parity.ONE_SIDED}), None),
    Axis.VOLATILITY: (frozenset({Parity.EVEN}), frozenset({ScaleClass.CARRYING})),
    Axis.PATH_STRUCTURE: (frozenset({Parity.EVEN}), frozenset({ScaleClass.FREE})),
}


def _column_admissible(stamp: FeatureInvariance, axis: Axis) -> bool:
    """Does ONE column of a heterogeneous entry satisfy ``axis``'s invariance rule?

    Same rule the per-entry path applies, factored out so the two can never drift: a column admitted
    here must be admissible for exactly the reasons an ordinary single-column feature would be.
    """
    if stamp.parity in (Parity.MIXED, Parity.HETEROGENEOUS):
        return False
    if stamp.parity is Parity.UNSCORED or stamp.scale_class is ScaleClass.UNSCORED:
        return True
    parities, scales = _AXIS_INVARIANCE_RULE[axis]
    if stamp.parity not in parities:
        return False
    return scales is None or stamp.scale_class in scales


def _split_summary(cols: dict[str, FeatureInvariance]) -> str:
    """e.g. "3 odd, 2 even" -- how a heterogeneous entry's columns divide."""
    counts: dict[str, int] = {}
    for st in cols.values():
        counts[st.parity.value] = counts.get(st.parity.value, 0) + 1
    return ", ".join(f"{n} {p}" for p, n in sorted(counts.items(), key=lambda kv: -kv[1]))


def is_eligible(entry: FeatureEntry, axis: Axis) -> tuple[bool, str]:
    """Is ``entry`` admissible on ``axis``? Returns ``(ok, reason)``; ``reason`` is "" when ok and clean.

    Two gates. The TAG gate asks whether the feature's family is plausibly on-axis. The INVARIANCE gate
    asks what the feature actually measures, and it is the one that matters — it is what rejects a
    feature whose name says one thing and whose behaviour says another.

    The invariance gate applies only to the three PRICE-PATH axes; ``LIQUIDITY`` is tag-gated only (see
    the module docstring). An unstamped feature passes with a reason naming the gap, because "never
    measured" is not the same finding as "measured and wrong" and must not be silently treated as one.
    """
    if entry.signal_type not in AXIS_SIGNAL_TYPES[axis]:
        return False, (f"signal_type={entry.signal_type!r} is not drawn on by {axis.value} "
                       f"(tags: {sorted(AXIS_SIGNAL_TYPES[axis])})")

    if axis not in PRICE_PATH_AXES:
        return True, ""

    inv = entry.invariance
    if inv is None:
        return True, f"{entry.qualified_name} carries no invariance stamp; admitted on the tag gate alone"
    # A multi-output entry whose columns differ has no per-entry verdict, so the per-entry rule below
    # cannot be applied to it. Judge it on its COLUMNS instead: reject only when NOT ONE of them is
    # admissible here, and otherwise admit with an advisory saying how they split. Rejecting outright
    # would be a false negative -- trend.bollinger_band emits an odd column that a recipe may legitimately
    # select -- while admitting silently is what let a select-dependent stamp go unnoticed.
    if inv.parity is Parity.HETEROGENEOUS:
        # Imported HERE, not at module scope, and deliberately. ``test_no_hand_maintained_membership_list``
        # scans this module's namespace for any dict keyed by qualified feature names, because such a
        # list is exactly the hole this layer closed. COLUMN_STAMPS is measured data rather than a
        # hand-kept membership list, so it does not violate that rule's INTENT -- but binding it at
        # module scope would silence a guard worth keeping, and the guard is worth more than the import.
        from ._invariance import COLUMN_STAMPS

        cols = COLUMN_STAMPS.get(entry.qualified_name, {})
        if not cols:
            return True, (f"{entry.qualified_name} emits columns of differing invariance and carries no "
                          f"per-entry verdict; no per-column stamps are available to say how they split")
        ok_cols = sorted(c for c, st in cols.items() if _column_admissible(st, axis))
        if not ok_cols:
            return False, (f"{entry.qualified_name} emits {len(cols)} columns and NONE is admissible on "
                           f"{axis.value} ({_split_summary(cols)})")
        return True, (f"{entry.qualified_name} emits {len(cols)} columns of differing invariance "
                      f"({_split_summary(cols)}); {len(ok_cols)} admissible on {axis.value}: "
                      f"{ok_cols}. The verdict depends on which column is selected — name one")

    # MIXED is checked BEFORE unscored, deliberately. ``MIXED`` and ``UNSCORED`` are different findings
    # and ``_schema`` keeps them as separate values for exactly this reason: UNSCORED means the
    # measurement was attempted and is not trustworthy, while MIXED means it SUCCEEDED and the answer
    # is "this feature confounds direction with magnitude". A definite parity verdict does not become
    # provisional because the SCALE test happened to degenerate -- scale only chooses between
    # VOLATILITY and PATH_STRUCTURE among even features, and says nothing about a mixed one.
    # Ordered the other way round, ``path_structure._velocity_path.velocity_consistency``
    # (parity=mixed, scale_class=unscored) was admitted to DIRECTIONAL "on the tag gate alone" while
    # the codebase already knew it belongs to no price-path axis.
    if inv.parity is Parity.MIXED:
        return False, (f"{entry.qualified_name} tests MIXED — it confounds direction with magnitude, so "
                       f"it belongs to no price-path axis. That is a defect in the feature, not a "
                       f"taxonomy gap")
    if inv.parity is Parity.UNSCORED or inv.scale_class is ScaleClass.UNSCORED:
        return True, (f"{entry.qualified_name} invariance is UNSCORED (degenerate or too few "
                      f"observations to measure); admitted on the tag gate alone")

    parities, scales = _AXIS_INVARIANCE_RULE[axis]
    if inv.parity not in parities:
        return False, (f"{entry.qualified_name} tests {inv.parity.value}; {axis.value} requires "
                       f"{sorted(p.value for p in parities)}")
    if scales is not None and inv.scale_class not in scales:
        return False, (f"{entry.qualified_name} tests {inv.parity.value}/{inv.scale_class.value}; "
                       f"{axis.value} requires {sorted(s.value for s in scales)}")

    if axis is Axis.DIRECTIONAL and inv.parity is Parity.ONE_SIDED:
        return True, (f"{entry.qualified_name} is ONE-SIDED (conjugate {inv.conjugate!r}): its low state "
                      f"pools 'the other way' WITH 'no move at all', so a K=2 split on it alone is not "
                      f"an up/down partition. Pair it with its conjugate or use the canonical spread")
    return True, ""


# Guard the tables against drift: every axis must have a tag set, a horizon and (for price-path axes) a
# rule, and every tag named must be a real signal_type.
assert set(AXIS_SIGNAL_TYPES) == set(Axis), "AXIS_SIGNAL_TYPES must cover every Axis"
assert set(AXIS_PRIMARY_HORIZON) == set(Axis), "AXIS_PRIMARY_HORIZON must cover every Axis"
assert set(_AXIS_INVARIANCE_RULE) == set(PRICE_PATH_AXES), "_AXIS_INVARIANCE_RULE must cover the price-path axes"
assert not set().union(*AXIS_SIGNAL_TYPES.values()) - set(SIGNAL_TYPES), "AXIS_SIGNAL_TYPES names an unknown tag"
