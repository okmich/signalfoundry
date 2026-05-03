# Changelog — okmich-quant-labelling

## 2.2.0

Production-correctness sprint. Backwards-incompatible **semantic** changes; the
public API surface (function names, package layout) is the same as 2.1.0 but
results will shift on existing pipelines that depended on the previous semantics.

### Breaking semantic changes

- **`get_labels` exit price = barrier price on horizontal hits** (was bar close).
  Eliminates the wick-touch / close-far-from-barrier pathology where `label`
  and `ret` could disagree in sign. Vertical exits still use `close[t1_iloc]`.
- **`PositionMonitor.on_bar` exit price = barrier price on horizontal hits**
  (was the `price` argument). Vertical exits still use `price`.
- **Same-bar tie-break default flipped to `WORST_CASE`** (was upper-first).
  When both upper and lower are touched in the same bar, the label is `-1`
  by default rather than `+1`. Restore the prior behavior with
  `same_bar_policy=BarrierTiePolicy.UPPER_FIRST`. Applies to both `get_labels`
  and `PositionMonitor`.

### New strict checks

- **`get_labels`** rejects `volatility` whose `attrs['vol_kind']` is not
  `'return'` (set by `get_daily_vol`). Prevents the silent-but-wrong scenario
  of feeding price-unit ATR or annualized Parkinson into `get_labels`.
- **`compute_barrier_levels`** rejects `sl_multiplier * volatility >= 1`
  (would produce a non-positive lower barrier).
- **`train_meta_model`** rejects partition-style sklearn splitters (KFold,
  StratifiedKFold, ShuffleSplit, RepeatedKFold, etc.) by default. Override with
  `allow_unsafe_splitters=True`. `TimeSeriesSplit` and iterables-of-pairs
  (e.g., from `purged_walk_forward_cv`) are always permitted.
- **`train_meta_model`** raises by default when CV-fold timestamps are missing
  from `features.index` (was: silently dropped). Override with
  `strict_timestamp_resolution=False`.
- **`train_meta_model`** rejects NaN/non-finite `sample_weight` after reindex
  to `features.index`.

### Other

- `scikit-learn` now declared explicitly in `pyproject.toml` deps (was
  transitively pulled via `features`).
- `BarrierTiePolicy` enum added in `okmich_quant_labelling.tbm.labeling` and
  `okmich_quant_ml.tbm.position_monitor`.
- `VolKind` enum added in `okmich_quant_labelling.tbm.volatility`. Each
  estimator tags its output `pd.Series` via `series.attrs['vol_kind']`.

### Migration guide

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: volatility has vol_kind='price'` from `get_labels` | Passing `get_atr_vol`/`get_std_vol` output directly | Use `get_daily_vol` (RETURN kind), or convert price-unit vol to per-bar return-vol manually before calling |
| `ValueError: volatility has vol_kind='annualized'` | Passing `get_parkinson_vol`/`get_garman_klass_vol`, or `get_daily_vol(annualize=True)` | Pass `get_daily_vol(annualize=False)`, or divide annualized vol by `sqrt(annualization_factor * bars_per_day)` |
| Existing label distribution shifts toward more `-1` | Same-bar policy default changed to WORST_CASE | Pass `same_bar_policy=BarrierTiePolicy.UPPER_FIRST` to restore 2.1 behaviour |
| Backtest equity curves shift | Exit price now barrier price on horizontal hits | Expected; this is the more realistic and consistent fill convention |
| `ValueError: sl_multiplier * volatility = 1.05 >= 1` | High-vol regime with large `sl` | Reduce `sl`, or accept the implicit "no stop" by setting `sl=0` |
| `ValueError: KFold is a partition-style splitter ...` from `train_meta_model` | Passing a non-time-aware sklearn splitter | Pass `purged_walk_forward_cv(...)` or `TimeSeriesSplit`, or set `allow_unsafe_splitters=True` if you really know your data is non-temporal |
| `KeyError: 'train' split: N/M timestamps not in feature index` | CV folds reference timestamps dropped during feature NaN cleanup | Either re-build CV from final feature index, or pass `strict_timestamp_resolution=False` |

---

## 2.1.0

- `barriers.py` relocated from `okmich_quant_ml.tbm` to `okmich_quant_features.tbm`
  to break `labelling -> ml` dependency edge. `okmich_quant_ml.tbm.barriers` keeps
  re-exports for back-compat.
- Various validation hardening (P1+P2 from production review).

---

## 2.0.0

Hard break from the legacy `okmich_quant_labelling.prediction.tbm` API. The old
`apply_tbm` / `tbm_from_signals` / `TBMConfig` / `BarrierConfig` /
`VolatilityConfig` / `optimize_tbm_*` functions are deleted. The replacement is
`okmich_quant_labelling.tbm.*` with the López de Prado spec semantics:

- **Log returns** in `ret` (was: simple returns).
- **Path-directional labels** (was: side-aware). Side handling moved to the
  meta-labelling layer.
- **Per-event `t1`** (was: global `max_holding_bars`).
- **Unitless return-volatility contract** (was: price-unit volatility).
- Adds CUSUM event sampling, purged walk-forward CV, meta-labelling, and a
  side-aware live `PositionMonitor`.

No compatibility shim is provided. The semantic contract differences make a
shim worse than nothing — it would silently misrepresent its inputs. Migrate
explicitly by reading the new module docstrings.
