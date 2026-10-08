"""Bind a registry ``FeatureEntry`` to a callable that takes ONE OHLCV frame, and reduce its output to per-bar columns.

``FeatureEntry`` is pure metadata: a function's ``name``, ``module`` and output type, no callable and no default
parameters. The invariance probe and the causality audit both need ``Callable[[pd.DataFrame], ...]``, so this module
closes the gap mechanically: introspect the signature, resolve each parameter NAME against the frame, call the
function. The resolution table below is the whole trick and is deliberately a visible table rather than ``getattr``
heuristics — a wrong binding produces a plausible-looking number, not an error.

Ported from ``signalfoundry-lab/research/features/invariant_test`` (2026-10-08) so that the library can test its
own catalogue.

THREE THINGS IT REFUSES TO BIND, AND WHY EACH IS A CATEGORY RATHER THAN A GAP
-----------------------------------------------------------------------------
1. CROSS-MARKET (``timothymasters.multi_market``, ``timothymasters.cross_market``, ``peer_*``, ``market_df``, ...).
   These take a SECOND instrument; refused by module as well as by parameter name, because ``multi_market`` spells
   its list-of-markets argument ``closes``, identical to a single-series ``closes``.
2. EVENT-INDEXED (the directional-change event parsers and features consuming their output): one row per event,
   not per bar, so there is no shared bar index to compare on.
3. NOT A FEATURE: weight generators, barrier helpers and parameter searches that live in the catalogue.
"""
from __future__ import annotations

import importlib
import inspect
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

#: Whole modules that are cross-sectional by construction. Refused before the signature is read.
CROSS_MARKET_MODULES = frozenset({"timothymasters.multi_market", "timothymasters.cross_market"})

#: Parameter names that require a second instrument.
CROSS_MARKET_PARAMS = frozenset({
    "close1", "close2", "high1", "low1", "high2", "low2", "open1", "open2",
    "peer_high", "peer_low", "peer_close", "peer_volume",
    "market_df", "sector_df", "universe_dfs", "benchmark", "benchmark_df",
    "benchmark_spread", "benchmark_mid",
})

#: Parameters that are another feature's EVENT STREAM, not a price path.
EVENT_STREAM_PARAMS = frozenset({"trends", "trends_s", "trends_b", "events", "dc_events"})

#: Qualified names that are helpers/searches rather than features of a price path.
NOT_A_FEATURE = frozenset({
    "fractional_diff.get_weights",
    "fractional_diff.get_optimal_fractional_differentiation_order",
    "tbm.compute_barrier_levels",
    "tbm.check_barrier_touch",
    "stats_optimal_search.optimal_autocorrelation_param_search",
    "stats_optimal_search.optimal_variance_ratio_param_search",
    "path_structure._zigzag_density.find_threshold_for_reversals",
    "directional_change.parse_dc_events",
    "directional_change.parse_dual_dc",
})


@dataclass(frozen=True)
class Binding:
    """The result of trying to make ``entry`` callable. ``fn`` is None iff ``blocker`` is set."""

    qualified_name: str
    fn: Callable[[pd.DataFrame], Any] | None
    blocker: str = ""
    category: str = ""          # cross-market | event-indexed | not-a-feature | unbound | import
    resolved_module: str = ""
    required: tuple[str, ...] = ()
    overrides_used: dict[str, Any] = field(default_factory=dict)


# Parameter resolution, grouped by what the parameter MEANS (the catalogue spells "close" six ways). ``data``
# resolves to CLOSE, not to the frame; ``d`` is a differencing ORDER, not a frame.
_CLOSE = ("close", "closes", "close_prices", "prices", "price_series", "series", "values", "px", "price", "data")
_HIGH = ("high", "highs", "high_prices")
_LOW = ("low", "lows", "low_prices")
_OPEN = ("open_", "open", "opens", "open_prices")
_VOLUME = ("volume", "volumes", "volume_series", "tick_volume")
_FRAME = ("df", "ohlcv", "bars", "frame")
_RETURNS = ("returns", "log_returns", "return_series", "returns_series", "rets")


def _series_resolvers(df: pd.DataFrame) -> dict[str, Callable[[], Any]]:
    close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]
    # MT5 feeds carry tick_volume, not traded volume; it is passed through as the volume substrate.
    vol = df["tick_volume"] if "tick_volume" in df.columns else df.get("volume")
    log_rets = np.log(close / close.shift(1))
    mid = (high + low) / 2.0
    spread = df["spread"].astype(float) if "spread" in df.columns else pd.Series(np.nan, index=df.index)

    out: dict[str, Callable[[], Any]] = {}
    for group, val in ((_CLOSE, close), (_HIGH, high), (_LOW, low), (_OPEN, open_), (_VOLUME, vol), (_FRAME, df),
                       (_RETURNS, log_rets)):
        for n in group:
            out[n] = lambda v=val: v
    out["spread"] = lambda s=spread: s
    out["mid_price"] = lambda m=mid: m
    out["volatility_series"] = lambda r=log_rets: r.rolling(20).std()
    return out


#: Required numeric knobs with no default in the signature. 20 bars is used uniformly so the choice is visible.
NUMERIC_DEFAULTS: dict[str, Any] = {
    "period": 20, "window": 20, "lookback": 20, "window_size": 20, "n": 20, "span": 20,
    "q": 2, "length": 20, "periods": 20, "num_periods": 20, "bars": 20, "d": 0.4,
}

#: Fixed per-feature overrides — measured necessities. The weekly tick-volume phase needs ~6 months of 5m bars;
#: the daily phase is used so the feature is not all-NaN on short inputs.
FIXED_OVERRIDES: dict[str, dict[str, Any]] = {
    "volume.tick_volume_phase_zscore": {"phase": "daily"},
}


def _ma_types() -> tuple:
    """``norm_moving_average`` dispatches on the ENUM member; VWAP is excluded (needs a volume argument)."""
    try:
        from okmich_quant_features.trend.normalized_ma import MovingAverageType as T
        return tuple(m for m in T if m.value != "vwap")
    except Exception:
        return ()


#: Dispatchers whose single stamp is only meaningful if every type agrees; probed once per type.
DISPATCH_SWEEPS: dict[str, tuple[str, tuple[Any, ...]]] = {
    "trend.normalized_ma.norm_moving_average": ("ma_type", _ma_types()),
}


def metastore_overrides(server: str, timeframe: int, symbol: str, metastore_file: str | None = None
                        ) -> dict[str, dict[str, Any]]:
    """Per-feature kwargs from a symbol metastore, for features whose library default never fires on an instrument
    (e.g. ``zigzag_density``'s 2% threshold on an FX major). Opt-in: needs ``metastore_file`` or the
    ``SYMBOL_METASTORE_FILE`` environment variable; returns {} when neither is available."""
    path = metastore_file or os.environ.get("SYMBOL_METASTORE_FILE")
    if not path:
        return {}
    try:
        os.environ.setdefault("SYMBOL_METASTORE_FILE", path)
        from okmich_quant_utils.symbol_metastore import SymbolMetastore
        props = SymbolMetastore().get_symbol_properties(server, timeframe, symbol) or {}
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    zz = props.get("zigzag_density_params") or {}
    if "threshold" in zz:
        out["path_structure.zigzag_density"] = {k: v for k, v in zz.items() if k in ("threshold", "window", "align")}
    ctl = props.get("bi_ctl_label") or {}
    if "omega" in ctl:
        out["trend.continuous_trend_labeling"] = {"omega": float(ctl["omega"])}
        out["trend.continuous_trend.ctl_trend_features"] = {"omega": float(ctl["omega"])}
    tp = props.get("trend_persistence_params") or {}
    if tp:
        keep = ("window", "smooth", "zscore_norm")
        out["trend.trend_persistence_labeling"] = {k: v for k, v in tp.items() if k in keep}
    fd = props.get("frac_diff") or {}
    if "optimal_d" in fd:
        out["fractional_diff.fractional_differentiate_series"] = {"d": float(fd["optimal_d"]),
                                                                  "window_size": int(fd.get("opt_window_size", 100))}
    return out


#: DC reversal threshold as a multiple of the median bar range — only needs to fire often enough.
DC_THETA_SCALE = 2.0


def derived_overrides(raw: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Data-derived kwargs, computed ONCE on the original frame and then held FIXED across any variant of it
    (reflected, rescaled, truncated). Re-deriving per variant would change the builder, not just the input."""
    theta = DC_THETA_SCALE * float(np.nanmedian(((raw["high"] - raw["low"]) / raw["close"]).to_numpy()))
    if not np.isfinite(theta) or theta <= 0.0:
        return {}
    return {"directional_change.dc_live_features": {"theta": theta}, "directional_change.idc_parse": {"theta": theta}}


def build_binding(qualified_name: str, module: str, name: str, df_probe: pd.DataFrame,
                  overrides: dict[str, Any] | None = None) -> Binding:
    """Try to make ``module.name`` callable as ``f(frame) -> Series|DataFrame|tuple``.

    ``df_probe`` resolves parameter NAMES only; the returned callable re-resolves against whatever frame it is handed.
    """
    ov = dict(FIXED_OVERRIDES.get(qualified_name, {}))
    ov.update(overrides or {})
    if module in CROSS_MARKET_MODULES:
        return Binding(qualified_name, None, f"{module} is cross-sectional by construction", "cross-market", module)
    if qualified_name in NOT_A_FEATURE:
        return Binding(qualified_name, None, "helper/search/event-parser, not a function of a price path",
                       "not-a-feature", module)
    try:
        mod = importlib.import_module("okmich_quant_features." + module)
    except Exception as ex:
        return Binding(qualified_name, None, f"import failed: {type(ex).__name__}: {ex}", "import", module)
    fn = getattr(mod, name, None)
    if fn is None:
        return Binding(qualified_name, None, f"{name!r} not found in {module}", "import", module)
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as ex:
        return Binding(qualified_name, None, f"no signature: {ex}", "unbound", module)

    required = tuple(p.name for p in sig.parameters.values()
                     if p.default is inspect.Parameter.empty
                     and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))
    if cross := [p for p in required if p in CROSS_MARKET_PARAMS]:
        return Binding(qualified_name, None, f"needs a second instrument ({', '.join(cross)})", "cross-market",
                       module, required)
    if ev := [p for p in required if p in EVENT_STREAM_PARAMS]:
        return Binding(qualified_name, None, f"consumes an event stream ({', '.join(ev)}); event-indexed",
                       "event-indexed", module, required)

    resolvers = _series_resolvers(df_probe)
    sweep_param = DISPATCH_SWEEPS.get(qualified_name, (None,))[0]
    unknown = [p for p in required if p not in resolvers and p not in NUMERIC_DEFAULTS and p not in ov
               and p != sweep_param]
    if unknown:
        return Binding(qualified_name, None, f"unbound parameter(s): {', '.join(unknown)}", "unbound", module, required)

    def call(frame: pd.DataFrame, _fn=fn, _req=required, _ov=dict(ov)):
        res = _series_resolvers(frame)
        kwargs = dict(_ov)
        for p in _req:
            if p in _ov:
                continue
            kwargs[p] = res[p]() if p in res else NUMERIC_DEFAULTS[p]
        return _fn(**kwargs)

    return Binding(qualified_name, call, "", "", module, required, dict(ov))


def numeric_columns(out, name: str, n_bars: int, depth: int = 0) -> tuple[dict[str, pd.Series], str]:
    """Reduce a feature's return value to named PER-BAR numeric series (``name`` or ``name@column``).

    A scalar or an event-indexed output (length != bars) yields nothing, with the reason. One level of tuple/dict
    unpacking only: deeper nesting is a function's internals, not more features.
    """
    if isinstance(out, (tuple, dict)) and depth >= 1:
        return {}, "nested container below the top level -- internals, not features"
    if isinstance(out, list):
        # A per-bar list (e.g. discretize_volume's assignments) is a series in disguise; anything else is not.
        try:
            out = np.asarray(out, dtype=float)
        except (TypeError, ValueError):
            return {}, "list is not numeric"
    if isinstance(out, tuple):
        cols: dict[str, pd.Series] = {}
        for i, v in enumerate(out):
            sub, _ = numeric_columns(v, f"{name}@{i}", n_bars, depth + 1)
            cols.update(sub)
        return cols, "" if cols else "tuple held no per-bar numeric series"
    if isinstance(out, dict):
        cols = {}
        for k, v in out.items():
            sub, _ = numeric_columns(v, f"{name}@{k}", n_bars, depth + 1)
            cols.update(sub)
        return cols, "" if cols else "dict held no per-bar numeric series"
    if isinstance(out, pd.Series):
        if not pd.api.types.is_numeric_dtype(out):
            return {}, "non-numeric Series"
        if len(out) != n_bars:
            return {}, f"EVENT-INDEXED ({len(out)} rows vs {n_bars} bars)"
        return {name: out.astype(float)}, ""
    if isinstance(out, pd.DataFrame):
        if len(out) != n_bars:
            return {}, f"EVENT-INDEXED ({len(out)} rows vs {n_bars} bars)"
        cols = {f"{name}@{c}": out[c].astype(float) for c in out.columns if pd.api.types.is_numeric_dtype(out[c])}
        return cols, "" if cols else "DataFrame had no numeric column"
    if isinstance(out, np.ndarray):
        if out.dtype == object:
            return {}, "object-dtype ndarray"
        if out.ndim == 1:
            if len(out) != n_bars:
                return {}, f"EVENT-INDEXED ({len(out)} rows vs {n_bars} bars)"
            return {name: pd.Series(out, dtype=float)}, ""
        if out.ndim == 2 and out.shape[0] == n_bars:
            return {f"{name}@{i}": pd.Series(out[:, i], dtype=float) for i in range(out.shape[1])}, ""
        return {}, f"ndarray shape {out.shape} is not per-bar"
    if isinstance(out, (int, float, np.integer, np.floating)):
        return {}, "returns a SCALAR -- no series"
    return {}, f"unhandled return type {type(out).__name__}"


def call_feature(b: Binding, raw: pd.DataFrame) -> tuple[dict[str, pd.Series], str]:
    """Call the binding (sweeping a dispatcher's type argument when it has one) and reduce to per-bar columns."""
    qn = b.qualified_name
    sweep = DISPATCH_SWEEPS.get(qn)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if sweep is None:
            return numeric_columns(b.fn(raw.copy()), qn, len(raw))
        param, values = sweep
        name = qn.rsplit(".", 1)[1]
        cols, why = {}, ""
        for val in values:
            nb = build_binding(qn, b.resolved_module, name, raw, overrides={param: val})
            if nb.fn is None:
                why = why or nb.blocker
                continue
            sub, w = numeric_columns(nb.fn(raw.copy()), f"{qn}@{val}", len(raw))
            cols.update(sub)
            why = why or w
        return cols, "" if cols else why
