# Macro feature engineering + no-lookahead asof-merge

Status: **IMPLEMENTED** — part of the now-**closed** macro data-asset layer (see the main spec's Status
section). 122 macro + news_calendar tests passing; verified end-to-end on real EURUSD 5m. The same
`attach_exogenous` merge serves the macro series, the event-timing features, and the event surprise.
Package: `okmich_quant_pipeline.macro` (+ `news_calendar`). Stores refreshed via `fetch-macro-data` /
`fetch-news-calendar` / `fetch-economic-events`.

---

## 1. Objective

Turn the raw daily macro parquet into **conditioning features**, then broadcast them onto
intraday bars **without lookahead**, producing `macro_*` columns the intraday model conditions
on (as a sizer/gate, not a signal). Built **generic across cadence** — daily, weekly, monthly,
or irregular event-driven series all flow through the same path.

Modules (pure / IO-free except `fetchers/`, `update.py`, `metastore.py`, `reader.py`, `attach.py`):
- `_types.py` — series registry + **pluggable availability policies**.
- `fetchers/fred.py` — FRED fredgraph fetch (no API key).
- `features.py` — raw → daily features (`compute_macro_features`), via composable recipes.
- `align.py` — the cadence-agnostic no-lookahead merge (`attach_exogenous`).
- `reader.py` — `load_macro` (concat the per-series store), `load_macro_features`.
- `metastore.py` / `update.py` — `_metadata.json` metastore + the `fetch-macro-data` incremental updater.
- `attach.py` — wrapper `attach_macro_to_dataset`.
- `pipeline/tests/macro/test_macro.py` — 27 tests.

---

## 2. Placement (as built)

The macro package lives **inside `okmich_quant_pipeline`, alongside `dataset_builder`**. The attach
runs as a **post-step on the `DatasetBuilder` output** (`attach.attach_macro_to_dataset`), kept
out of `dataset_builder.py` itself so macro-vs-no-macro stays a reversible `± columns` ablation and
the live-feeding builder is untouched. Because macro is now in the same package, folding it in as a
`_add_macro_features()` step is a trivial in-package option with no dependency concerns — deferred
until macro proves out. (Earlier this lived in lab and the placement was driven by a lab → source
dependency constraint, now dissolved by the move.)

---

## 3. Genericity — the two seams that make new data drop-in

### (a) Availability policy (`_types.py`) — *when* an observation becomes public
A per-series strategy computes `available_from_utc`, so series with different publish behaviour
need no engine change:
- `BusinessDayLag(lag, hour_utc=22)` — daily series released N **business** days later (VIX=0, credit/USD=1).
- `CalendarDayLag(lag, hour_utc)` — release ignores business-day rolling (e.g. a weekly index N calendar days out).
- `ExplicitRelease(column)` — irregular/event-driven (rate decisions, CPI); availability = a real release-timestamp column.

Adding a series = a `SeriesSpec(fred_id, channel, availability, description)` entry. That's it.

### (b) Cadence-agnostic merge (`align.py`) — the merge only reads `available_from_utc` + `value`
No assumption that features share a cadence or schedule. A single forward-fill reconciles any
mix (proven by `test_mixed_cadence_daily_plus_weekly`, `test_weekly_cadence_attaches`,
`test_irregular_explicit_release_attaches`).

---

## 4. `features.py` — daily feature engineering

`compute_macro_features(raw, recipes=DEFAULT_RECIPES) -> long [date, feature, value, available_from_utc]`.

- **Computed on observation cadence, never the broadcast intraday series.** A 20-point window is
  20 *observations* (20 trading days daily / 20 weeks weekly). Windows are observation-based, so
  transforms are cadence-agnostic.
- **Composable recipes.** `FeatureRecipe(name, sources, fn)` + generic transforms
  (`level`, `zscore`, `change`, `log_return`, `ratio`). Each feature owns its own window (bound
  into `fn`) — no global window imposed. Adding a feature = append a recipe.
- **Per-feature availability = latest over its source series** (a cross-series feature can't be
  used until every leg is public; `test_cross_series_availability_is_latest_of_inputs`).

`DEFAULT_RECIPES` (18): `vix_level`, `vix_z20`, `vix_chg5`, `vixts_ratio`, `vixts_z20`,
`credit_level`, `credit_z20`, `credit_chg5`, `usd_ret5`, `usd_z20`, `us2y_level`, `us2y_chg5`,
`us10y_level`, `us10y_chg5`, `curve_2s10s`, `curve_2s10s_z20`, `nfci_level`, `nfci_chg4`.
(Factor-reduction/PCA deferred.)

---

## 5. `align.py` — the no-lookahead merge

`attach_exogenous(bars, features, *, prefix="macro_") -> bars + {prefix}{feature} columns`.

Algorithm:
1. Validate `bars.index` is a sorted, UTC-tz-aware `DatetimeIndex` (else `ValueError`).
2. Pivot features to wide, indexed by `available_from_utc` (one column per feature). Different
   release times → interleaved NaNs across rows.
3. `ffill` down columns — each feature carries its last-known value forward to every later
   release instant. Causal (past → future only); this is what lets one merge serve mixed cadences.
4. **Normalize both merge keys to `ns`** (parquet bar indices are often `datetime64[us]`; stamps
   are `[ns]` — `merge_asof` rejects mismatched resolution), then one
   `merge_asof(direction="backward")`: each bar gets the most recent value with
   `available_from_utc <= bar_ts`. Never a future observation.

Worked example (pinned by `test_no_lookahead` / `test_heterogeneous_availability_worked_example`):
a bar at `D+1 10:00 UTC` gets VIX(D) and CREDIT(D−1), never VIX(D+1)/CREDIT(D) (public only at D+1 22:00).

Edge cases: pre-history / warmup → NaN; weekend/holiday gaps bridged by ffill; tz-naive or
non-UTC or unsorted bars → `ValueError`; mixed datetime resolution → normalized.

---

## 6. Consume path

- `reader.load_macro_features(path)` = `load_macro` + `compute_macro_features`.
- `attach.attach_macro_to_dataset(dataset, macro_path, *, drop_warmup=True)` — loads features,
  `attach_exogenous`, optionally drops `macro_*` warmup-NaN rows (source `_trim` runs before this
  attach and keys off `feat_/tm_/candle_/temporal_`, so it won't drop `macro_*` warmup itself).

---

## 7. Tests (`tests/macro/test_macro.py` — 34 passing)

No-lookahead (centerpiece), boundary `<=` at release instant, heterogeneous availability,
weekend ffill, namespace/shape, empty-features no-op, tz/sort/dtype guards, **mixed datetime
resolution**, weekly cadence, mixed daily+weekly cadence, irregular `ExplicitRelease`, the three
availability policies, z-score vs reference + warmup, default-recipe columns, cross-series
availability = latest-of-inputs, end-to-end attach on synthetic, plus store-integrity regressions
(revision-absorption order, metastore corrupt-vs-OSError, empty-fetch guard, malformed-CSV guard,
retry validation).

**Real-data check:** 37,120 EURUSD 5m bars → 10 macro cols, 0% NaN over the slice; spot bar
2024-04-01 09:55 UTC resolves to VIX(2024-03-28) across the Good-Friday/weekend gap, provably not
Apr-1's VIX (released 22:00).

---

## 8. Validation (downstream, separate PR)

Ablation in `research/regime_gate_walkforward/`: base vs base + macro, **per-year net-of-cost**.
Macro enters as a **continuous sizer**, not a binary gate. No per-year lift ⇒ characterized
detector, not a system; stop there.

---

## 9. Out of scope (this PR)

Factor reduction/PCA; daily macro-HMM posteriors (Path B); changes to source `dataset_builder.py`.
(The keyed-API first-print vintage machinery is now **built** — `fetchers/alfred.py` + a per-series
`FredSource`/`vintage` discriminator — though no production series is vintaged yet; see the spec's
Vintage note for why NFCI stayed on CSV and HY-OAS is opt-in.)

**Update:** the event channel is **built**, in two shapes. *Timing* (`minutes_to_next` /
`minutes_since_last` / `blackout`, `news_calendar/features.py` → `attach_events_to_dataset`) is
forward/symmetric, computed **per-bar**, and deliberately does **not** use this asof-merge. *Surprise*
(`economic_events.py` → `attach_surprise_to_dataset`) IS backward-looking, so it rides exactly this
`attach_exogenous` path: a per-release standardized `surprise` stamped `ExplicitRelease` at the release
instant, ffilled to each later bar — the canonical use of this merge for an irregular event-driven series.
