"""Pure barrier geometry — shared TBM primitive.

Lives in `features` as the only common leaf in the dependency tree (no project
deps, no heavy IO/ML baggage). Both `okmich_quant_ml.tbm` and
`okmich_quant_labelling.tbm` import from here.

No pandas, no numpy, no allocations — these functions sit on the live hot path.

Volatility contract: `volatility` is a unitless return-volatility (e.g., EWM std
of log returns). Barriers are computed multiplicatively against entry price:

    upper = entry_price * (1 + pt_multiplier * volatility)
    lower = entry_price * (1 - sl_multiplier * volatility)

Both `compute_barrier_levels` and downstream consumers (`get_labels`,
`PositionMonitor`) MUST use this same contract.
"""

from enum import Enum
from math import isfinite
from typing import Optional


class BarrierHit(str, Enum):
    UPPER = "upper"
    LOWER = "lower"
    VERTICAL = "vertical"


def compute_barrier_levels(entry_price: float, volatility: float, pt_multiplier: float,
                           sl_multiplier: float) -> dict:
    if not isfinite(entry_price) or entry_price <= 0:
        raise ValueError(f"entry_price must be finite and > 0, got {entry_price}")
    if not isfinite(volatility) or volatility <= 0:
        raise ValueError(f"volatility must be finite and > 0, got {volatility}")
    if not isfinite(pt_multiplier) or pt_multiplier < 0:
        raise ValueError(f"pt_multiplier must be finite and >= 0, got {pt_multiplier}")
    if not isfinite(sl_multiplier) or sl_multiplier < 0:
        raise ValueError(f"sl_multiplier must be finite and >= 0, got {sl_multiplier}")
    if sl_multiplier > 0 and sl_multiplier * volatility >= 1.0:
        raise ValueError(
            f"sl_multiplier * volatility = {sl_multiplier * volatility:.4f} >= 1 would produce "
            f"a non-positive lower barrier; reduce sl_multiplier or pick a different volatility series"
        )

    upper = entry_price * (1.0 + pt_multiplier * volatility) if pt_multiplier > 0 else None
    lower = entry_price * (1.0 - sl_multiplier * volatility) if sl_multiplier > 0 else None
    return {"upper": upper, "lower": lower}


def check_barrier_touch(price: float, upper: Optional[float], lower: Optional[float]) -> int:
    """Return 1 if upper hit, -1 if lower hit, 0 otherwise.

    Gap-through convention: upper is evaluated first. If price gaps through both
    barriers in a single bar, upper wins.
    """
    if upper is not None and price >= upper:
        return 1
    if lower is not None and price <= lower:
        return -1
    return 0
