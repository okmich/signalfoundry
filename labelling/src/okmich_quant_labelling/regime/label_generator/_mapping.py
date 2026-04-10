"""
Yardstick → mapping-function dispatcher.

Converts raw HMM / oracle state integers to semantic yardstick labels by
calling the appropriate map_label_to_* or map_regime_to_* function from
okmich_quant_labelling.utils.label_util.

Each mapping function takes a DataFrame that contains a state/regime column
and returns a dict {state: label}.  apply_yardstick_mapping() handles the
per-yardstick differences in column-kwarg names transparently.
"""

from typing import Any, Dict, Optional, Union

import pandas as pd

from okmich_quant_labelling.regime.threshold_optimizer import MarketPropertyType
from okmich_quant_labelling.utils.label_util import (
    map_label_to_momentum_score,
    map_label_to_trend_direction,
    map_regime_to_path_structure_score,
    map_regime_to_volatility_score,
)

# -----------------------------------------------------------------------
# Dispatch tables
# -----------------------------------------------------------------------

_MAPPING_FN = {
    MarketPropertyType.DIRECTION: map_label_to_trend_direction,
    MarketPropertyType.MOMENTUM: map_label_to_momentum_score,
    MarketPropertyType.VOLATILITY: map_regime_to_volatility_score,
    MarketPropertyType.PATH_STRUCTURE: map_regime_to_path_structure_score,
}

# The state column kwarg name differs across functions
_STATE_COL_KWARG: Dict[MarketPropertyType, str] = {
    MarketPropertyType.DIRECTION: "state_col",
    MarketPropertyType.MOMENTUM: "regime_col",
    MarketPropertyType.VOLATILITY: "regime_col",
    MarketPropertyType.PATH_STRUCTURE: "regime_col",
}

# The return column kwarg name (where applicable)
_RETURN_COL_KWARG: Dict[MarketPropertyType, Optional[str]] = {
    MarketPropertyType.DIRECTION: "return_col",
    MarketPropertyType.MOMENTUM: "ret_col",
    MarketPropertyType.VOLATILITY: None,       # uses vol_proxy_col, not return_col
    MarketPropertyType.PATH_STRUCTURE: None,   # uses price columns directly
}


# -----------------------------------------------------------------------
# Public helper
# -----------------------------------------------------------------------

def apply_yardstick_mapping(
    df: pd.DataFrame,
    states: pd.Series,
    yardstick: Union[str, MarketPropertyType],
    state_col: str = "_state",
    return_col: Optional[str] = None,
    **extra_kwargs,
) -> pd.Series:
    """
    Apply the appropriate map_label_to_* function for the given yardstick.

    Parameters
    ----------
    df : pd.DataFrame
        Training fold DataFrame.  Must include all columns required by the
        mapping function (e.g. return_col for DIRECTION / MOMENTUM, or
        high/low/close for PATH_STRUCTURE with default method).
    states : pd.Series
        Raw state integers (from HMM or oracle), aligned to df.index.
    yardstick : str or MarketPropertyType
        Target yardstick.
    state_col : str, default='_state'
        Temporary column name used to insert states into df.
    return_col : str, optional
        Name of the log-return column in df.  Translated to the correct
        kwarg name for the target yardstick automatically:
        - DIRECTION  → return_col
        - MOMENTUM   → ret_col
        - VOLATILITY / PATH_STRUCTURE → ignored (no return column needed).
        Pass None to omit (some mapping functions have sensible defaults).
    **extra_kwargs
        Additional keyword arguments forwarded directly to the underlying
        map_label_to_* function (e.g. method, min_samples, min_sharpe).

    Returns
    -------
    pd.Series
        Labels aligned to df.index.  Dtype and value range depend on the
        yardstick (e.g. {-1, 0, 1} for DIRECTION after conservative mapping).
        NaN entries from states are preserved.
    """
    yardstick = MarketPropertyType(yardstick)

    map_fn = _MAPPING_FN.get(yardstick)
    if map_fn is None:
        raise ValueError(
            f"No mapping function registered for yardstick '{yardstick.value}'. "
            f"Supported: {[y.value for y in _MAPPING_FN]}"
        )

    # Insert states as a temporary column
    tmp = df.copy()
    tmp[state_col] = states.values

    # Translate return_col → yardstick-specific kwarg name
    if return_col is not None:
        ret_kwarg = _RETURN_COL_KWARG.get(yardstick)
        if ret_kwarg is not None:
            extra_kwargs.setdefault(ret_kwarg, return_col)

    # Call mapping function: {state_kwarg: state_col, ...extra}
    state_kwarg = _STATE_COL_KWARG[yardstick]
    state_mapping: Dict[Any, int] = map_fn(
        tmp, **{state_kwarg: state_col}, **extra_kwargs
    )

    # Apply mapping to produce the label series
    return states.map(state_mapping).rename("regime_label")
