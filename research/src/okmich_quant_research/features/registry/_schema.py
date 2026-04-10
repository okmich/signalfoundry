from dataclasses import dataclass, field
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