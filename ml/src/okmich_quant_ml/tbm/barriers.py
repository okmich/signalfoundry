"""Re-export of `okmich_quant_features.tbm.barriers` for backward compatibility.

The geometry primitive moved to `features` so both `ml` and `labelling` can
share it without one pulling the other's heavy dependency tree.
"""

from okmich_quant_features.tbm.barriers import (
    BarrierHit,
    check_barrier_touch,
    compute_barrier_levels,
)

__all__ = ["BarrierHit", "check_barrier_touch", "compute_barrier_levels"]
