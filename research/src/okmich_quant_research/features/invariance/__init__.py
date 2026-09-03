"""Feature invariance — deciding what a feature measures by measuring it, not by reading its name.

THE PROBLEM
-----------
The catalogue holds ~300 indicator functions. Before you can use one you need to know what it actually measures: does it
tell you WHICH WAY the market is going, how VIOLENT the move is, or what SHAPE the path had? Until this package existed
the only evidence for that was the function's name and a hand-typed label, and nobody had ever checked.

THE TRICK: DRIVE THE ROUTE BACKWARDS
------------------------------------
Imagine a mystery gauge bolted to a car. You cannot open it. Does it measure VELOCITY (direction and
speed) or just SPEED?

There is an experiment that settles it. Drive the exact same route, backwards.

  * the gauge now reads the NEGATIVE of what it read before  -> it knows direction (a velocity gauge)
  * the gauge reads EXACTLY THE SAME                         -> it never knew direction (a speed gauge)

That is the whole idea. "Driving the route backwards" is :func:`reflect_ohlc`: rebuild the price path with every
log-return negated, so every up-tick becomes an equal down-tick. Re-run the indicator on the flipped path and correlate
it against the original. No theory, no model — run it twice, compare.

Measured on FXPIG-Server M5:

    indicator                          reading on the flipped chart     verdict
    momentum.roc                       -1.000 x original                knows direction   (ODD)
    path_structure.efficiency_ratio    +1.000 x original                magnitude only    (EVEN)

The SECOND experiment (:func:`rescale_ohlc`) drives the same route twice as fast: multiply every
log-return by ``c``. If the indicator's spread scales with ``c`` it measures actual size; if it is
unchanged it is a ratio. That is what separates VOLATILITY from PATH_STRUCTURE.

Crossed, the two experiments define the three price-path axes in ``registry._axis``.

THE NASTY CASE, AND WHY IT MATTERS
----------------------------------
Now imagine two gauges: ACCELERATOR pedal pressure and BRAKE pedal pressure.

Accelerator high means accelerating forward. Accelerator low means braking OR sitting still — and from
that one gauge you cannot tell those apart. Drive the route backwards and something odd happens: the
accelerator gauge does not flip to minus-itself, it starts reading whatever the BRAKE gauge read
before. It turned into the OTHER gauge.

That is ``momentum.minus_di`` and ``momentum.plus_di``, measured:

    momentum.minus_di    flipped-chart reading -0.687    but matches plus_di at 1.000

Why it matters, concretely. The trend axis was screened by asking a K=2 HMM to split the market in
two, and ``minus_di`` won on 11 of 14 FX symbols. Those two states read naturally as "up regime / down
regime". They were not. ``minus_di`` is the brake pedal: its high state is "moving down hard", its low
state pools "moving up" WITH "not moving at all". So the split was "strong down move" vs "everything
else", and every separation number in that corpus looked healthy while measuring the wrong thing.
Nothing else in the codebase could have caught it — only the conjugate search here distinguishes
"half of a pair" from "genuinely confounded".

The fix is the obvious one: accelerator MINUS brake, a single number that is positive going forward,
negative going backward and zero when neither. That is ``momentum.di_spread``, which flips cleanly to
-1.000.

WHAT THIS PACKAGE IS
--------------------
The machine that runs those two experiments over a whole pool of indicators and writes down the
verdicts. The verdicts it has already produced live in ``registry/_invariance.csv`` — 97 indicators
graded. The screener reads that file to decide which indicators are admissible on which axis, instead
of trusting their names (``registry.is_eligible``).

WHEN YOU ACTUALLY NEED TO RUN IT
--------------------------------
Most of the time you do not: the verdicts ship with the code. It earns its keep at four moments.

  1. A NEW INDICATOR is added. You cannot know what it measures by reading it — that is the whole
     lesson. Probe it and stamp it, or it is admitted on its namespace alone.
  2. A POOL THE PROBE NEVER COVERED is screened. Only 97 of the 313 catalogue entries are graded, so
     most axes still carry unmeasured candidates; the screener names them once per run.
  3. AN EXISTING INDICATOR'S CODE CHANGES. A stamp is a claim about behaviour, so editing the
     behaviour can leave the stamp quietly stale.
  4. A DIFFERENT INSTRUMENT CLASS is taken on and the classification should be verified rather than
     assumed (the shipped stamps are 4 FX majors plus one index).

Think of it as the calibration rig for a set of instruments: not used daily, but the thing you reach
for the day someone asks whether a gauge really measures what its label says.

    >>> import pandas as pd
    >>> from okmich_quant_research.features.invariance import (
    ...     aggregate_stamps, probe_invariance, stamp_summary, unscored, write_stamps_csv)
    >>> from okmich_quant_research.features.registry._invariance import INVARIANCE_CSV
    >>> frames = [probe_invariance(raw, feature_engineering, symbol=sym) for sym, raw in corpus.items()]
    >>> stamps = aggregate_stamps(pd.concat(frames), measured_on="FXPIG M5, 6 symbols, 80k bars, ...")
    >>> stamp_summary(stamps)          # eyeball before committing
    >>> unscored(stamps)               # what could NOT be measured -- a defect list, not a filler cell
    >>> write_stamps_csv(stamps, INVARIANCE_CSV)

``feature_engineering`` is the same ``Callable[[pd.DataFrame], pd.DataFrame]`` ``HmmFeatureScreener``
takes, so any pool you can screen you can probe, with no adapter.

A SECOND USE
------------
:func:`nearest_neighbour_redundancy` asks whether two pools are really separate axes at all. It is
what killed the momentum axis: 23 of 38 momentum candidates sat within the ``MAX_VIF = 3.0``
near-duplicate line of some trend candidate. Worth running over any two pools before spending fits on
treating them as independent.

See ``probe`` for the method's limits, and ``registry._axis`` for what the verdicts are used to decide.
"""
from ._classify import (MIN_OBS, PARITY_BAND, SCALE_C, SCALE_CARRY, SCALE_FREE, VIF3_R, best_match,
                        classify_cell, classify_parity, classify_scale, cross_correlations, iqr,
                        scale_exponent)
from ._transforms import OHLC, reflect_ohlc, rescale_ohlc
from .probe import (PROBE_COLUMNS, STAMP_COLUMNS, aggregate_stamps, nearest_neighbour_redundancy,
                    probe_invariance, stamp_summary, unscored, write_stamps_csv)

__all__ = [
    # transforms
    "OHLC",
    "reflect_ohlc",
    "rescale_ohlc",
    # classification
    "classify_parity",
    "classify_scale",
    "classify_cell",
    "scale_exponent",
    "iqr",
    "cross_correlations",
    "best_match",
    "PARITY_BAND",
    "SCALE_C",
    "SCALE_CARRY",
    "SCALE_FREE",
    "MIN_OBS",
    "VIF3_R",
    # probe
    "probe_invariance",
    "aggregate_stamps",
    "stamp_summary",
    "write_stamps_csv",
    "nearest_neighbour_redundancy",
    "unscored",
    "PROBE_COLUMNS",
    "STAMP_COLUMNS",
]
