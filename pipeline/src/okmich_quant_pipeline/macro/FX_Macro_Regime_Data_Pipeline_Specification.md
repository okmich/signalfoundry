# Macro Regime Data Pipeline Specification for Intraday FX Trading

> **Revision note (2026-06-23).** Reframed from a *supervised regime classifier* to a
> *slow-moving conditioner / gate / sizer* layer that plugs into the existing HMM +
> posterior-inference + CTL stack. The headline change: we do **not** train a model to
> emit "Risk-On / USD-Dominance" probabilities (there is no ground truth for those
> labels — such a model either becomes an arbitrary rule-labeller dressed as ML, or it
> silently collapses into a forward-return predictor, which is the intraday model's job).
> Instead, macro enters as an exogenous **gate/sizer input**, validated the same way every
> other gate in this repo is validated: per-year net-of-cost lift in the walk-forward harness.

---

## Objective

Build a daily macro-data pipeline that produces **exogenous conditioning features** for the
downstream intraday models (5-minute and 15-minute FX, Gold, and — with caveats — Bitcoin
and index CFDs).

The purpose is **not** to predict the next candle, and **not** to emit a labelled
"market environment" class. The purpose is to supply a small set of slow, causally-clean
exogenous variables that the intraday regime/sizing logic can condition on — primarily
through the **volatility channel** and the **event channel** (see below).

**Core thesis:** macro is a *conditioner*, not a *signal*. This mirrors the conclusions
already reached on the existing tracks — `vol-sizer = whipsaw-protection`,
`CTL = directional bias source only; tools handle timing / sizing / risk`. Macro is just
another gate input feeding that same axis-2 sizing/gating scaffold.

---

## Where macro actually pays at 5m/15m (and where it doesn't)

At intraday horizons the dominant variance is microstructure / session / intraday-vol, not
slow daily macro. Macro's marginal value concentrates in exactly two channels:

### 1. The volatility channel — highest ROI
`VIX → next-day realized intraday volatility` is one of the most robust, causal, freely
available relationships in markets. **This drives sizing, not direction:** scale down or
flatten on vol-shock days; allow the mean-reversion oscillator edge more room on calm days.

### 2. The event channel — highest ROI, and already half-built
Scheduled macro releases (NFP / CPI / FOMC / ECB / BoE) produce real, exogenous intraday
spikes that price-only features cannot see in advance. The calendar infrastructure for this
**already exists** in `utilities/news_calendar/fetchers/` (ALFRED-vintaged FRED, ForexFactory,
FOMC). The remaining work is feature wiring, not data sourcing:
- `surprise = (actual − forecast) / σ_historical`
- `minutes_to_next_high_impact_event`
- event-importance / blackout flags

### What pays *less* than first assumed
The **directional** macro-price content (DXY trend, 2s10s slope) at a 5m/15m horizon. Keep
these as a *slow USD bias for real FX/metal only* and do not expect much standalone edge.

### Collinearity warning
The seven "Tier 1/2" series are heavily collinear — DXY is a EUR/JPY/GBP/CAD basket,
SPX↔VIX ≈ −0.8, and the "currency strength index" is DXY re-derived. There are really
**~3–4 independent factors** here (USD, risk appetite, rates, growth/commodity), not 7.
Run a factor reduction (PCA / simple residualization) before believing 10 new columns add
10 new axes.

---

## Scope discipline (hard constraint)

Macro features apply **only to real instruments**: EURUSD, XAUUSD, SP500, BTC.

- **Synthetics get nothing.** Deriv `Volatility 100 Index` and similar are RNG-driven; macro
  is meaningless and any apparent edge there is spurious. Do not merge macro onto synthetic
  symbols.
- **BTC is weak/unstable** w.r.t. macro. Its real exogenous drivers are perp funding /
  basis (see additions table). Treat BTC macro-sensitivity as low-confidence.

---

## The #1 correctness risk: timestamp alignment / no-lookahead

The repo currently has **no asof-merge / daily→intraday broadcast layer and no exogenous-
feature alignment logic.** Building one correctly is the single most important task in this
spec. The trap:

- DXY / SPX / VIX / yield "daily closes" are **US-session-anchored (~21:00 UTC)**. Yahoo /
  Stooq give a *date*, not a timestamp. FRED `DGS2` / `DGS10` publish with ~1 business-day
  lag and are *as-of the prior day*.
- FX intraday "day D" begins ~Sunday 21:00 UTC and runs 24h. Stamping a same-date macro close
  onto same-date intraday bars **leaks the future** — London-session bars of day D would be
  "seeing" a US close that has not happened yet.

**Alignment rule (mandatory):**
1. Align every macro series by its actual *publish* timestamp, not its label date.
2. A macro feature is usable only on intraday bars **strictly after** its release timestamp.
3. When the publish timestamp is uncertain, **lag a full calendar day**.
4. Use **ALFRED (archival/vintage) FRED**, never regular FRED — regular FRED returns *revised*
   numbers (lookahead). The existing `utilities/news_calendar/fetchers/fred.py` already does
   this correctly; the macro fetcher must follow the same discipline.

The asof-merge must be a **backward** merge (intraday bar ← most recent already-published
macro value), forward-filled within the day.

---

## Data series

### Built — current data layer (7 series → 18 features) — **BUILT**

All series are sourced from **FRED** (single provider, stable public `fredgraph.csv` endpoint,
**no API key**) — Yahoo was dropped to avoid a fragile screen-scrape. Implemented as the
`okmich_quant_pipeline.macro` package (signalfoundry/pipeline); **per-series** parquet + a
`_metadata.json` metastore under `E:\data_dump\macro_data\daily\`, refreshed with the
`fetch-macro-data` command (incremental tail refresh; `--full` re-fetches history). The registry
is data-only: a series is one `SeriesSpec` (FRED id + channel + availability policy); a feature
is one `FeatureRecipe`. Availability is a pluggable policy — `BusinessDayLag`, `CalendarDayLag`,
or `ExplicitRelease` — so daily and weekly (and future irregular) series share one merge path.

| Canonical | FRED id | Channel | Availability | Notes |
|---|---|---|---|---|
| `VIX` | `VIXCLS` | Volatility | BusinessDayLag(0) | Next-day realized-vol proxy. Close known same evening. |
| `VIX_3M` | `VXVCLS` | Volatility | BusinessDayLag(0) | Term structure (`VIX/VIX3M`) = stress-transition flag. |
| `CREDIT_SPREAD` | `BAA10Y` | Risk | BusinessDayLag(1) | Moody's Baa − 10Y. **Substitute for ICE HY-OAS** (`BAMLH0A0HYM2`), licence-capped to a rolling 3y window on FRED's anonymous CSV. BAA10Y is the free daily credit spread, full history, ~0.9 corr with HY-OAS in stress. |
| `USD_BROAD` | `DTWEXBGS` | USD | BusinessDayLag(1) | Fed H.10 broad index. Better USD factor than DXY (EUR-heavy). |
| `US_2Y` | `DGS2` | Rates | BusinessDayLag(1) | 2Y Treasury (H.15). Rate-expectations proxy. |
| `US_10Y` | `DGS10` | Rates | BusinessDayLag(1) | 10Y Treasury (H.15). Long-term growth/inflation. |
| `NFCI` | `NFCI` | Risk | CalendarDayLag(6) | Chicago Fed financial conditions. **Weekly** (week-ending Fri, released next Wed; +6d→Thu cushion). |

Derived feature (no series): `curve_2s10s` = `US_10Y − US_2Y` (+ `curve_2s10s_z20` = inversion regime).

**18 features** (`features.DEFAULT_RECIPES`): vix `level/z20/chg5`, `vixts_ratio/z20`, credit
`level/z20/chg5`, usd `ret5/z20`, us2y `level/chg5`, us10y `level/chg5`, `curve_2s10s` + `_z20`,
nfci `level/chg4`. All windows are observation-based (cadence-correct), computed on the daily/weekly
series — never on the broadcast intraday frame.

**Vintage note.** The data layer uses FRED's *latest-vintage* CSV plus a conservative publish stamp
(`available_from_utc`). VIX/VIX3M are market closes (never revised); BAA10Y/DTWEXBGS/DGS/NFCI carry
only minor revisions and the publish lag dominates. The keyed-API **first-print vintage path is
built** (`fetchers/alfred.py` + a per-series `FredSource`/`vintage` discriminator), but **no
production series is vintaged yet**: NFCI's ALFRED archive only starts 2011-05-27 (vintaging it would
drop ~2010–2011), so it stayed on the CSV path. The machinery exists for the surprise-channel actuals
(where first-print is mandatory). **ICE HY-OAS could not be restored to full history** — `BAMLH0A0HYM2`
is licence-capped to a rolling ~3y window *even with the key* (the cap is on the data, not the
anonymous CSV); it is added as an **opt-in** `HY_OAS` series (`HY_OAS_RECIPES`, kept out of
`DEFAULT_RECIPES` so its 2023+-only coverage can't NaN-truncate longer datasets).

### Planned data additions (not yet built)

| Series / feature | Source | Purpose |
|---|---|---|
| Economic calendar surprise + blackout | *already fetched* (`news_calendar/`) | `surprise=(actual−forecast)/σ`, minutes-to-event, blackout — via the `ExplicitRelease` policy. Highest info-per-bar; no new sourcing. |
| DXY | Stooq / Yahoo `DX-Y.NYB` | Broad USD strength (redundant-ish with DTWEXBGS; keep one). |
| S&P 500 | Yahoo `^GSPC` / Stooq | Risk appetite (already on disk as a Deriv instrument). |
| Gold | Yahoo `GC=F` | Safe-haven / USD relationship (also a traded instrument). |
| Crude Oil (WTI) | Yahoo `CL=F` | Growth/commodity factor; CAD regimes. |
| Point-in-time vintages | FRED **API key** (ALFRED) | True vintages for revised series + restore real ICE HY-OAS. |

### Other free data worth adding (ranked by marginal value)

| # | Source (free) | Adds | Why over the base set |
|---|---|---|---|
| 1 | Credit spread — FRED `BAA10Y` (HY-OAS `BAMLH0A0HYM2` is licence-capped; restore via keyed API later) | Risk on/off | **Built.** Less collinear than VIX, credit leads. |
| 2 | Calendar surprise + blackout — *already fetched* | event timing | *Planned.* Highest intraday ROI, near-zero new cost. |
| 3 | VIX term structure — FRED `VIXCLS` / `VXVCLS` | risk regime | **Built** (`vixts_ratio`). Independent of VIX level; earlier stress transitions. |
| 4 | NFCI — FRED `NFCI`, weekly | financial conditions | **Built.** Slow, orthogonal risk axis. |
| 5 | Instrument's own overnight gap + trailing realized vol | expected day vol | Already on disk; at 5m/15m often *beats* macro. Cheapest. |
| 6 | Trade-weighted broad USD — FRED `DTWEXBGS` | cleaner USD | Better than DXY for non-EUR pairs. |
| 7 | BTC perp funding / basis — exchange APIs (Binance/Bybit) | crypto positioning | The *actual* exogenous driver for BTC. |
| 8 | COT positioning — CFTC, weekly | crowding extremes | Slow positioning mean-reversion (FX, gold). |

**Explicitly skip:** Citi CESI (paid — build your own surprise index from #2), Google
Trends / social sentiment (noise), copper standalone (loads on the same growth factor as oil).

---

## Feature engineering

Per series (daily), then asof-merged onto intraday under the alignment rule:
- Daily return, 5-day return, 20-day return
- Rolling volatility (e.g. 20-day)
- 20-day z-score
- For yields: daily change, weekly change, momentum, curve spread + spread change
- For VIX: change, % change, vol-of-vol, level z-score, **term-structure ratio** (`VIX/VIX3M`)
- For credit: OAS level, OAS change, OAS z-score
- For calendar: `surprise = (actual − forecast)/σ`, `minutes_to_next_high_impact`, importance tier

Then **factor-reduce** the price-macro block (PCA or residualization) before feeding the
intraday model, to avoid feeding 4 collinear copies of "the USD factor."

---

## How it plugs into the existing stack

Two consumption paths. Do **Path A** first; graduate to **Path B** only if A shows lift.

### Path A — exogenous columns (start here) — **BUILT**
1. **Fetcher:** `okmich_quant_pipeline.macro` package (`fetchers/`, `_types.py`, `metastore.py`,
   `update.py`, `reader.py`) over the shared `okmich_quant_pipeline.http` / `._io` utilities.
   Per-series parquet + `_metadata.json` metastore at
   `E:\data_dump\macro_data\daily\`, refreshed via `fetch-macro-data` (incremental) / `--full`.
   *The fetcher code is the deliverable.*
2. **Daily features:** `features.compute_macro_features` (z-scores, term-structure ratio,
   changes), computed on daily cadence.
3. **Asof-merge: `align.attach_exogenous`** (cadence-agnostic backward asof-merge with ffill) +
   the `attach.attach_macro_to_dataset` wrapper attach `macro_*` columns onto a `DatasetBuilder`
   output. The macro package now lives **inside `okmich_quant_pipeline`, alongside
   `dataset_builder`**, so the earlier lab → source backwards-dependency concern is moot: the
   attach stays a standalone post-step (keeping the macro-vs-no-macro ablation a reversible
   `± columns`), and folding it into `dataset_builder` as `_add_macro_features()` is now a trivial
   in-package option. See `ASOF_MERGE_SPEC.md`.
4. **Screen, don't assume.** Let the existing HMM screener
   (`research/.../features/hmm_screener/`) test whether each macro feature adds
   axis separation; it already Pareto-classifies on `(axis_separation, honesty)`. A macro
   feature that doesn't lift honesty does not ship. No new validation philosophy required.

### Path B — daily macro-HMM posteriors (elegant; later)
Fit an HMM/GMM on the **daily** macro feature panel → daily `γ_t` posteriors `(T_daily, K)`.
That is exactly the shape the existing `posterior_inference` package consumes
(`okmich_quant_ml/posterior_inference/`): run it through `MarginGateInferer` /
`StabilityGateInferer`, then asof-merge the resulting **daily regime posterior** down to
intraday (with the shift). This reuses the entire regime stack — macro becomes "just another
posterior source," consistent with the source-agnostic posterior-eval design.

### The one genuinely new piece of infrastructure
The **no-lookahead backward asof-merge / broadcast layer**. Build it once, test it hard
(unit tests asserting no intraday bar sees an unpublished macro value), and reuse everywhere.

---

## Validation protocol

- Prove macro-as-gate in `research/regime_gate_walkforward/`.
- Metric: **per-year net-of-cost** lift of macro-gated *sizing* vs ungated baseline. Lead
  with per-year, never aggregate (aggregate hides regime dependence and is flattered by good
  periods).
- Decision rule: if macro gating does not lift per-year net, it is a *characterized detector*,
  not a system — log it as a regime-track artifact and stop, per the regime-vs-signal
  two-track discipline.
- Leakage guard: the walk-forward must fit any macro-HMM **inside-fold**, with the
  asof-merge applied so no fold sees future macro releases.

---

## Storage design

### `macro_daily` (raw, vintaged)
`date`, `release_timestamp_utc`, `series_id`, `value`
(long format keyed by series + publish timestamp — preserves vintage for ALFRED series.)

### `macro_features` (derived, daily)
`date`, plus engineered columns: `vix_level`, `vix_ts_ratio`, `hy_oas`, `hy_oas_z20`,
`usd_broad_ret_5d`, `usd_broad_z20`, `dxy_*`, `spx_*`, `us2y_change`, `yield_curve`,
`yield_curve_change`, factor-reduced `macro_pc1..pc3`, …

### `economic_events` (already produced by `news_calendar/`)
`timestamp`, `event_name`, `country`, `actual`, `forecast`, `previous`, `surprise`,
`importance`.

---

## Status & roadmap

**Built (data layer):** 7 FRED series → 18 features, leak-free cadence-agnostic attach
(`okmich_quant_pipeline.macro`), scoped to real instruments (EURUSD / XAUUSD / SP500 / BTC) — never
synthetics. Plus: the **feature store** (`store.py` / `build-macro-features` + coverage/gap/staleness
`report.py`); the **`news_calendar` package** migrated in-package (calendar asset built); and the
**event-timing features** (`news_calendar/features.compute_event_features` →
`attach.attach_events_to_dataset`: `minutes_to_next` / `minutes_since_last` / `blackout`).

This workstream is **data-centric**: the goal is a clean, broad, correctly time-aligned exogenous
data asset. Strategy validation (sizer/gate in `regime_gate_walkforward`, per-year net-of-cost) and
the daily macro-HMM (Path B) are the eventual *consumers* of this data — deferred until explicitly
taken up, not part of building the layer.

**Done since first draft:** (#1) feature store + coverage reports — **built**. The `news_calendar`
fetchers — **migrated in-package** (were lab-side). The **event-timing** half of the event channel
(`minutes_to_next` / `minutes_since_last` / `blackout`) — **built**. (#2) keyed-API **vintage
machinery** (`fred_key.py`, `fetchers/alfred.py`, `FredSource`/`vintage` dispatch) — **built**.

**Remaining (data-centric):**
2. **Point-in-time vintages** — machinery **built**, but the two intended wins both hit data limits
   (see Vintage note): ICE HY-OAS stays ~3y licence-capped *with the key* (added opt-in, not in
   defaults); NFCI first-print archive only starts 2011, so it stayed on CSV. The keyed first-print
   path is ready and is the **prerequisite for the surprise feature** (its actuals *must* be first-print).
3. **Commodity/equity breadth** — DXY / S&P 500 / Gold / Oil; BTC perp funding/basis. *(Not cleanly
   on FRED — WTI is `DCOILWTICO`, but DXY / Gold spot need another source, reintroducing the Yahoo
   dependency we dropped; resolve sourcing before adding.)*
4. **Event channel — surprise** *(the gated half)* — `surprise = (first-print actual − consensus)/σ`.
   **Depends on #2** (vintaged actuals) **and** on consensus/forecast sourcing (the open fork — partly
   free from the ForexFactory blob we already parse for ECB/BoE; US consensus is the hard piece).
   Architecture: the *actual* lives once as a vintaged macro series via the `ExplicitRelease`
   availability policy (calendar supplies release timestamps); `surprise` is a derived join, not a
   stored duplicate.

**Mechanism note:** `ExplicitRelease` + the backward asof-merge is the path for *surprise* (a
backward-looking series). The event-*timing* features are forward/symmetric, so they are computed
**per-bar** against the calendar (`compute_event_features`), not via the asof-merge — these are two
different paths the earlier draft conflated.

**Never:** the supervised 4-probability regime classifier (no ground truth); macro on synthetics.

---

## Final recommendation

For a retail intraday FX/metal trader, the highest information-per-unit-effort macro inputs
are, in order:

1. **VIX (level + term structure)** — sizing via the volatility channel. *Built.*
2. **Economic-calendar event channel** — *timing features (minutes-to-event / blackout) built*; the
   higher-value **surprise** add is gated on vintaged actuals (#2) + consensus sourcing.
3. **Credit spread (`BAA10Y`)** — the cleanest free daily risk-on/off gauge. *Built* (real ICE HY-OAS pending API key).
4. **Broad trade-weighted USD** — slow directional bias for real FX/metal. *Built.*

Three of the four are built; the **event channel** is the highest-value remaining data add. All
are **exogenous conditioner inputs** under strict no-lookahead alignment, scoped to real
instruments — kept as a data asset, not yet validated as a strategy (that's a deliberate later step).
