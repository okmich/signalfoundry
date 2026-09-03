from dataclasses import dataclass, field
from enum import StrEnum
from typing import List

# ── Signal type taxonomy ──────────────────────────────────────────────────────
SIGNAL_TYPES = {
    "toxicity":         "Order flow toxicity and adverse selection metrics",
    "order_flow":       "Volume and order flow imbalance measures",
    "liquidity":        "Transaction cost, spread, and depth measures",
    "volatility":       "Price variance and risk measures",
    "momentum":         "Price persistence and rate-of-change",
    "regime":           "Market state and structural change detection",
    "price_structure":  "Intrabar geometry and price path characteristics",
    "volume_structure": "Volume distribution, concentration, and timing",
    "information":      "Information asymmetry and entropy measures",
    "composite":        "Multi-component meta-features",
    "trend":            "Directional bias and trend quality",
    "temporal":         "Calendar and session-based time features",
}

# ── Relevance levels ──────────────────────────────────────────────────────────
CRITICAL = "CRITICAL"
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"
NONE = "NONE"

RELEVANCE_LEVELS = (CRITICAL, HIGH, MEDIUM, LOW, NONE)

# ── Horizon labels ────────────────────────────────────────────────────────────
H_INTRADAY = "intraday"   # < 1 session / single bar
H_SHORT    = "short"      # 1–5 bars
H_MEDIUM   = "medium"     # 5–20 bars
H_LONG     = "long"       # 20+ bars
H_ANY      = "any"        # horizon-agnostic

HORIZONS = (H_INTRADAY, H_SHORT, H_MEDIUM, H_LONG, H_ANY)

# ── Market regime labels ──────────────────────────────────────────────────────
R_TRENDING = "trending"
R_RANGING  = "ranging"
R_VOLATILE = "volatile"
R_LOW_VOL  = "low_vol"
R_CRISIS   = "crisis"

MARKET_REGIMES = (R_TRENDING, R_RANGING, R_VOLATILE, R_LOW_VOL, R_CRISIS)


# ── Measured invariance taxonomy ──────────────────────────────────────────────
# Two exact, model-free transforms of the price path decide what a feature actually measures. They are
# applied in LOG-PRICE space about the first close, so they are exact on log-returns rather than
# approximate:
#
#   reflect:  p' = c0**2 / p          # every log-return negated; DECREASING, so high/low MUST swap
#   rescale:  p' = c0 * (p/c0)**c     # every log-deviation multiplied by c; order-preserving
#
# Parity comes from reflection, scale class from rescaling. Crossed, they define the three PRICE-PATH
# axes (see registry._axis):
#
#                 scale-carrying        scale-free
#     odd         DIRECTIONAL           DIRECTIONAL
#     even        VOLATILITY            PATH_STRUCTURE
#
# These stamps are MEASURED, never declared -- see okmich_quant_research.features.invariance, which
# produces them, and registry/_invariance.csv, which carries them.


class Parity(StrEnum):
    """Behaviour of a feature under reflection of the price path."""

    ODD = "odd"              # corr(f_reflected, f) ~ -1: knows WHICH WAY
    EVEN = "even"            # corr(f_reflected, f) ~ +1: knows only HOW MUCH
    ONE_SIDED = "one-sided"  # f_reflected matches a DIFFERENT column ~ +1: half of an odd pair
    MIXED = "mixed"          # none of the above: confounds direction with magnitude -- a defect
    UNSCORED = "unscored"    # measurement attempted but not trustworthy (degenerate / too few obs)


class ScaleClass(StrEnum):
    """Behaviour of a feature under rescaling of the price path's log-deviations."""

    CARRYING = "scale-carrying"  # exponent ~ 1: the feature's spread tracks move size
    FREE = "scale-free"          # exponent ~ 0: normalised, size-invariant
    PARTIAL = "partial"          # between the two bands
    UNSCORED = "unscored"


@dataclass(frozen=True)
class FeatureInvariance:
    """A measured invariance stamp for one feature.

    ``conjugate`` is the reflection partner and is set IFF ``parity`` is ``ONE_SIDED`` -- e.g.
    ``momentum.minus_di`` carries ``conjugate="momentum.plus_di"``. ``measured_on`` records the corpus,
    bar count and date the stamp came from, because a stamp without provenance is just another
    assertion.
    """

    parity: Parity
    scale_class: ScaleClass
    conjugate: str = ""
    measured_on: str = ""

    def __post_init__(self):
        if (self.parity is Parity.ONE_SIDED) != bool(self.conjugate):
            raise ValueError(
                f"conjugate must be set iff parity is ONE_SIDED; got parity={self.parity!r}, "
                f"conjugate={self.conjugate!r}"
            )


@dataclass
class FeatureEntry:
    """Metadata record for a single feature function."""

    name: str                       # Python function name (importable)
    module: str                     # dotted module path relative to okmich_quant_features
    signal_type: str                # one of SIGNAL_TYPES
    description: str                # one-line plain-English description

    regime_relevance:    str = MEDIUM  # usefulness for regime classification
    return_relevance:    str = MEDIUM  # usefulness for predicting future returns
    direction_relevance: str = LOW     # usefulness for directional (BUY/SELL) signal

    horizon: str = H_ANY                                    # best prediction horizon
    works_best_in: List[str] = field(default_factory=list)  # regime context tags

    directional: bool = False       # True if sign carries BUY/SELL meaning
    causal: bool = True             # True if uses only past/current bar data
    output_type: str = "series"     # series | dataframe | scalar | array

    needs_spread:    bool = False   # requires bid-ask spread input
    needs_volume:    bool = False   # requires volume input
    needs_benchmark: bool = False   # requires peer/benchmark data

    notes: str = ""

    # Measured invariance stamp; ``None`` means never measured (NOT "measured and found neutral").
    # Populated at import time from registry/_invariance.csv -- see registry._invariance.
    invariance: FeatureInvariance | None = None

    def __post_init__(self):
        assert self.signal_type in SIGNAL_TYPES, (
            f"Unknown signal_type {self.signal_type!r}. Valid: {list(SIGNAL_TYPES)}"
        )
        for attr in ("regime_relevance", "return_relevance", "direction_relevance"):
            val = getattr(self, attr)
            assert val in RELEVANCE_LEVELS, (
                f"Bad relevance {val!r} on {attr}. Valid: {RELEVANCE_LEVELS}"
            )
        assert self.horizon in HORIZONS, (
            f"Unknown horizon {self.horizon!r}. Valid: {HORIZONS}"
        )

    @property
    def qualified_name(self) -> str:
        """Return fully-qualified name: module.function_name."""
        return f"{self.module}.{self.name}"