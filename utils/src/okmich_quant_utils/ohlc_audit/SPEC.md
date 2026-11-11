# OHLC audit + remediation tool: requirements

Status: APPROVED 2026-11-11 (§11). P1 + P2 DONE and hardened (§12, §13). P3 DONE 2026-11-11 (§14). P4 BUILT 2026-11-11
(§15); its live dry run on the FXPIG terminal waits for the user. P5 next.
PROMOTED 2026-11-11 from the lab to `okmich_quant_utils.ohlc_audit`, with the `raw-data-audit` console script (§16).
Earlier sections keep the lab paths (`utilities/ohlc_audit/...`, `python -m utilities.ohlc_audit`) as history.

Replaces `utilities/ohlc_gap_audit.py`. Motivated by the 2026-11-11 audit of the FXPIG-Server 5m export (25 symbols).
That tool passed 7 symbols that each carry the 2026-10-29 splice, and three of the seven (AUDCAD, AUDJPY, USDJPY)
also carry mis-clocked or interleaved windows. It also failed 18 symbols, mostly on normal exchange breaks.

## 1. What went wrong

### 1.1 The current audit tool

| # | Defect | Effect seen on 2026-11-11 |
|---|---|---|
| 1 | Reads the index's `+02:00` label as the real zone. FXPIG stamps are NY + 7/9/6/7 h. | Its 17:00–01:00 "UTC" break window lands on NY afternoons. FX outages get filed as `daily_break`/`holiday` (14 FX pairs: 18 h on 2023-07-04/05 filed as a holiday). |
| 2 | One FX/gold rule set for all asset classes. | Index breaks and exchange holidays are flagged UNEXPLAINED (JPN225 188, HKG50 51). Crypto maintenance is flagged too. |
| 3 | Only checks the spacing between timestamps. | Misses 1h-shifted duplicates (10-29 splice, all symbols), interleaves (09-08, 09-09), whole weeks on the wrong clock (AUD pairs, USDCAD), and Saturday bars (AUDUSD/EURCAD). |
| 4 | No clock model, no list of bad windows. | ≥7 studies keep their own `true_clock`. None has the 10-29 seam or the per-symbol exceptions. |
| 5 | Detects only, never repairs. | Each study re-derives its own cuts by hand. |

### 1.2 Ingest: why `raw/` gets corrupted (found while scoping v2)

The SF fetcher (`okmich_quant_mt5.utils.mt5_data_fetcher`) works like this:

- It resumes from `_metadata.json` `last_datetime` and re-fetches an overlap.
- It merges on **timestamp only** (`keep="last"`).
- Bars are stamped by `functions.mt5._resolve_broker_tz`. That function decodes the server offset **once per run**
  from the latest tick, and accepts any value within ±14 h.

That leaves two failure modes:

- **The broker's clock moves between runs** (seams at 2024-06-05, 2026-08-30, 2026-10-29). The overlap bars return
  under different stamps, both copies survive the merge, and we get the repeated blocks and two-level interleaves found
  on 2026-11-11.
- **A run starts while the market is closed.** The decode reads a stale tick and locks a wrong offset for the whole
  run. This is a plausible cause of the AUDUSD/EURCAD Saturday tail, which is Friday data re-stamped ~13.5 h late
  (hypothesis, not proven).

So cleaning `raw/` once is not enough; every future refresh must pass through the same guard (§6.6).

`functions.mt5` is on the live order path (`okmich_quant_mt5.strategy` imports it). This tool does not modify it and
does not use its stamping. The stale-tick decode risk also affects live bar stamping. It is reported to the user, not
fixed here.

## 2. Goals and non-goals

**Goals**

- G1. Detect every defect class found on 2026-11-11, with zero false alarms on structural closures.
- G2. Remediate, then promote the result **over `raw/`**, so that `raw/` can be trusted as clean. The pre-promotion
  original is always snapshotted first.
- G3. Fill holes and replace bad bars **from the broker itself**. A hole is permanent only after 3 failed attempts.
- G4. Broker-agnostic core. FXPIG is the first profile.
- G5. Single source of truth for broker clocks and known bad windows.
- G6. Every refresh goes through the tool, so `raw/` stays clean.
- G7. Protect pre-registered holdouts (§8).

**Non-goals (v1)**

- Synthetic bars. No forward-fill or interpolation into `raw/`. Holes are filled only with the broker's own bars.
- Backfilling from a different broker. Mixing feeds creates the interleave defect this tool removes.
- Editing prices by formula. A spike is replaced only by the broker's re-fetched bar (§6.5).
- Any change to code on a live order path (`okmich_quant_mt5.functions`, `strategy`, `broker_session`, systems).

## 3. Architecture

```
            ┌──────────── refresh (§6.6) ────────────┐
broker MT5 ─┤ fetch tail ─► overlap handshake ───────┼─► staging/<broker>/<tf>/
            └────────────────────────────────────────┘          ▲
raw/<broker>/<tf>/ ── copy ─────────────────────────────────────┘
                                                                │
staging ─► [4] detect ─► [6.1–6.3] drop corrupt windows ─► [6.4–6.5] refetch holes / dropped windows / spikes
                                                                │   (handshake + verify; hole ledger)
                                                                ▼
                                   re-audit: 0 CRITICAL? ── no ─► stop, raw/ untouched, report
                                                                │ yes
                                   snapshot raw/ ─► atomic promote staging → raw/ ─► _audit/ sidecars + manifest
```

### 3.1 Broker-agnostic core (G4)

The core knows only these interfaces. Each broker supplies implementations.

| Interface | Role | v1 implementations |
|---|---|---|
| `BrokerProfile` | Symbols → asset class, clock model, sessions, holidays, maintenance, fetch source, holdout cutoffs, registry path | `profiles/fxpig.py` |
| `ClockModel` | `to_utc(raw_index, symbol)`, `expected_raw_events(symbol, date)` (weekly open/close, daily break) | `EraShiftClock` (NY + shift eras, plus per-symbol overrides) |
| `FetchAdapter` | `fetch(symbol, start_raw, end_raw) -> DataFrame` in raw stamps, read-only | `MT5FetchAdapter` |
| `SessionTemplate` | Expected-bar grid per asset class, in exchange-local time; brokers override hours | FX, METAL, US_INDEX, EU_INDEX, ASIA_INDEX, CRYPTO |

Planned later, not built in v1: `PerDayOffsetClock` (TopOne's per-day FX clock), `UtcClock` (exchange crypto), plus
Binance and IB adapters. P6 adds a second profile, audit-only, to prove the core needs no changes.

### 3.2 FXPIG profile (data, versioned in git)

- **Clock eras** (raw = NY local + shift):
  - +7 h before 2024-06-05.
  - +9 h to 2026-08-29.
  - +6 h from the 2026-08-30 week.
  - +7 h from the 2026-10-29 splice.
- **Per-symbol overrides**, each marked `confirmed` or `proposed`:
  - USDCAD, week of 2026-08-23: +6 h.
  - AUD pairs, 2026-09-14..18 and 09-29..10-16: +9 h. 09-09..11 is mixed.
- **Sessions and holidays:**
  - Session templates are derived once from the data, reviewed, then frozen. Broker CFD hours differ from the
    exchange's.
  - Holidays come from the `holidays` package (installed): XNYS, XETR, XHKG, XJPX. FX closures (Dec 25, Jan 1) are
    listed in the profile.
  - A US early-close rule covers metals and US-index CFDs.
  - Declared crypto maintenance windows.
- **Fetch source:** resolved exactly as `fetch-mt5-data --broker fxify --account demo` does.
  - `okmich_quant_core.env_loader.EnvLoader().load("fxify", "demo")` reads `.env.fxify.demo` from the folder named
    by `OKMICH_QUANT_ENV_DIR`. There is no hardcoded folder.
  - That file supplies `TERMINAL_PATH`, `LOGIN_SERVER`, `LOGIN_ID` and `LOGIN_PASSWORD`.
  - The profile stores only the pair (env_broker=`fxify`, env_account=`demo`), never a path or credentials.
- **Registry** `profiles/fxpig_bad_windows.csv`. Append-only, seeded from the 2026-11-11 list.

### 3.3 Session calendar

- `expected_grid(symbol, start, end)` gives the expected 5m slots, each tagged OPEN, CLOSED_WEEKEND, CLOSED_BREAK,
  HOLIDAY, EARLY_CLOSE or MAINTENANCE.
- Slots present in fewer than 90% of weeks are tagged THIN: no-tick bars are possible there.

## 4. Detectors

Each detector returns typed `Finding` rows:
kind (`FindingKind` StrEnum), severity, symbols, start/end in raw and UTC, n_bars, evidence, detector, proposed action,
and `broker_wide` (≥3 symbols overlapping).

| ID | Detector | Rule (defaults; configurable) | Severity |
|---|---|---|---|
| D1 | Timestamp integrity | Duplicate, non-monotonic, off-grid or future stamps | CRITICAL |
| D2 | OHLC integrity | high/low/open/close violations, NaN/inf, non-positive prices | CRITICAL |
| D3 | Missing bars | Expected OPEN slot without a bar, grouped into runs: OUTAGE (≥30 min MAJOR, else MINOR), THIN_NO_TICK (MINOR), HOLE (≥1 day, MAJOR) | MAJOR / MINOR |
| D4 | Out-of-session bars | A bar in a closed slot (e.g. Saturday for non-crypto) | CRITICAL |
| D5 | Overfull period | Bars per week/day above the template maximum | CRITICAL |
| D6 | Repeated block | ≥4 consecutive non-flat bars identical (OHLC + tick_volume) to the bars L back, L ∈ 1..288 | CRITICAL |
| D7 | Two-feed interleave | \|Δ₁\| > 6× robust local scale, \|Δ₂\| < 0.3\|Δ₁\|, alternating signs. ≥3 hits within 6 bars. | CRITICAL |
| D8a | Clock alignment (returns) | Per symbol-week, the lag in ±72 bars that maximises correlation with a peer basket. Flag if \|lag\| ≥ 3, the gain is > 0.05 and the peer correlation is ≥ 0.15. FX: signed returns of same-currency pairs. Metals, US indices, crypto: rank-correlation of \|r\| against class peers. | CRITICAL |
| D8b | Clock alignment (event stamps) | Weekly open/close and daily-break start compared with `expected_raw_events`. Covers peerless symbols (DE30, HKG50, JPN225). | CRITICAL |
| D9 | Spikes | A reverting jump > 25× the local median range, or a wick > 40×. Same move on ≥3 same-class symbols within ±1 bar → MARKET_EVENT (INFO), otherwise SPIKE_SUSPECT (MINOR). | MINOR / INFO |
| D10 | Stale or flat | Unchanged close for > N bars in an active slot; tick_volume == 0; zero-range bars | MINOR |
| D11 | Spread validity | Spread ≤ 0, or implausible against a rolling median | MINOR |
| D12 | History rewrite | Diff against the previous snapshot: bars that changed value or stamp, or disappeared | CRITICAL |

Structural closures are emitted as INFO, so the report shows that they were recognised.

## 5. Report

- **Per-symbol verdict**, with coverage (real bars / expected bars):
  - CLEAN: no CRITICAL or MAJOR findings.
  - USABLE_AFTER_REMEDIATION: every CRITICAL finding is remediable.
  - REVIEW: anything else.
- Findings table, one row per finding.
- Broker-wide event table: findings shared by ≥3 symbols.
- Remediation summary: rows dropped, filled, replaced and restamped per symbol; open and permanent holes.
- Diff since the last run: new, resolved and changed findings.
- Quality timeline: a symbol × week heatmap of the worst severity.
- A console summary.

## 6. Remediation

The pipeline runs in a fixed order, is deterministic and idempotent, and works in `staging/`. `raw/` changes only at
§6.8.

### 6.1 Dedupe

Exact duplicate stamps: keep the policy-chosen copy and log the rest.

### 6.2 Clock

- Every bar must follow the profile's clock: the eras plus confirmed overrides.
- Bars from a window **detected** as mis-clocked, with no confirmed override, are **dropped**. The resulting hole goes
  to refetch (§6.4), which brings the broker's bars back through the overlap handshake.
- A restamp is applied only when a confirmed override says so.

### 6.3 Drop

- Drop registry windows and detected CRITICAL windows (D4–D8, D12).
- For repeated blocks, drop the later copy.
- Detections not yet in the registry are applied too, and reported as `unregistered` for review.

### 6.4 Refetch: holes and dropped windows

**What to fetch.** Every expected-but-missing run except structural ones (holiday, break, early close, maintenance),
plus every window dropped in §6.2–6.3.

**How.** One `FetchAdapter.fetch` per contiguous window, with a margin of ≥12 trusted bars on each side.

**Overlap handshake.** This is required; it is exactly the check the SF fetcher lacks.

1. Match the fetched margin bars to the stored trusted bars by exact OHLC equality.
2. If the margins agree at a single constant offset k on ≥80% of margin bars on **both** sides, restamp the fetched
   bars by −k. This absorbs a broker clock change.
3. If no consistent offset exists, or the margins disagree on price, the attempt is **REJECTED_VERIFY**.

**Verify.** Splice the fetched bars into a copy of the series, then re-run D1, D4–D8 on the splice window ± 1 day.
Any CRITICAL finding means REJECTED_VERIFY. Otherwise the window is FILLED or PARTIAL.

**Hole ledger** (`_audit/holes.parquet`, persistent across runs). One row per hole:
symbol, start/end, class, first_seen, attempts, last_attempt, last_result, status (OPEN, FILLED, PARTIAL, PERMANENT,
STRUCTURAL).

**Failure counting (G3):**
- EMPTY (the broker returns no bars) and REJECTED_VERIFY each count as one failed attempt.
- At most one counted attempt per hole per run, and counted attempts are at least 24 h apart, so a broker-side
  history repair has time to land.
- Transient errors do not count: terminal not connected, IPC error, timeout. They are retried within the run with
  backoff.
- **3 failed attempts → PERMANENT.** Permanent holes are never re-fetched unless forced (`--retry-permanent`).

The ledger records the server's earliest available bar as evidence. Many 2023 holes may predate FXPIG's history depth:
the pre-2024-06-05 bars came from the merged FXIFY-old history (`5-history`, no longer on disk).

### 6.5 Spikes

- MARKET_EVENT (the same move across symbols, e.g. JPY interventions): never touched.
- SPIKE_SUSPECT: refetch the bar with margins, through the same handshake and ledger.
  - If the broker's bar differs and no longer spikes, replace it, and log the old values in the manifest.
  - If it is identical, keep it and flag SPIKE_CONFIRMED. This is what a live system saw.
  - 3 identical refetches → PERMANENT. The flag stays.

### 6.6 Refresh: ingest through the tool (G6)

`refresh` replaces the FXIFY line in `market-data-integration.bat` for profiled brokers:

1. Copy `raw/` to `staging/`.
2. Fetch the tail from `last_datetime` minus a margin.
3. Overlap handshake (§6.4).
4. Run §6.1–6.5 on the touched range.
5. Promote (§6.8).

The SF fetcher keeps serving unprofiled brokers. Offering the handshake to SF is a separate, later decision.

### 6.7 Quality flags (sidecar)

- One `QualityFlag` IntFlag per bar, stored in `_audit/quality/<symbol>.parquet`. The `raw/` schema is unchanged.
- Flags: AFTER_GAP, NEAR_DROPPED, RESTAMPED, REFETCHED, SPIKE_SUSPECT, SPIKE_CONFIRMED, MARKET_EVENT, SPREAD_INVALID,
  ROLLOVER_WINDOW (FX 21–24 UTC bid-bar artefact), THIN_SLOT, PERMANENT_HOLE_ADJACENT.
- Flags are hygiene masks, not features. Flags built from full-sample statistics must not feed a model.

### 6.8 Verify, snapshot, promote

1. **Re-audit staging.** 0 CRITICAL is required. On failure, `raw/` is untouched and the run reports why.
2. **Snapshot first.** Copy `raw/<broker>/<tf>/` to `snapshots/<broker>/<tf>/<export_ts>/`, hash-verified, before
   any write to `raw/`. Snapshots are kept indefinitely (~100 MB each).
3. **Promote file by file, atomically** (temp + move, as the SF fetcher does). Update `_metadata.json` (bars_count,
   last_datetime, last_audit) so the SF fetcher stays compatible.
4. **Write `_audit/`** in the broker folder, not next to the symbol parquets, so `glob("*.parquet")` loaders are
   unaffected:
   - `findings.parquet`, `gaps.parquet`, `holes.parquet`, `quality/`.
   - `manifest.json`: tool and profile versions, snapshot id, input/output hashes, every action taken.
5. **"Eventually" means a gate.** Promotion stays disabled until P5. The first promotion needs the user's sign-off on
   a dry-run diff.

### 6.9 Consequences for studies

- Every reader of `raw/` gets cleaned data from the first promotion on.
- Past results stay reproducible from the snapshot recorded in each manifest.
- **Horse race P6** (pre-registered, scores bars after 2026-10-29 at 2027-02-01 / 2027-11-01): before the first
  checkpoint, its §14 must record that `raw/` is cleaned by this tool, at version X, by mechanical rules frozen before
  scoring. The tool does not edit that spec.
- Several study-local `true_clock` copies lack the 10-29 era (§1.1 #4). They read post-10-29 bars 1 h off today,
  regardless of this tool.

## 7. API and CLI

```python
from utilities.ohlc_audit import audit_broker, remediate_broker, refresh_broker, load_clean

report = audit_broker("FXPIG-Server", timeframe=Timeframe.M5)
manifest = remediate_broker("FXPIG-Server", timeframe=Timeframe.M5, policy=RemediationPolicy(), promote=False)
manifest = refresh_broker("FXPIG-Server", timeframe=Timeframe.M5)
df = load_clean("EURUSD", broker="FXPIG-Server", timeframe=Timeframe.M5)  # true-UTC index + quality flags
```

```
python -m utilities.ohlc_audit audit     --broker FXPIG-Server --tf 5 [--symbols ...] [--json]
python -m utilities.ohlc_audit remediate --broker FXPIG-Server --tf 5 [--no-fetch] [--promote] [--dry-run]
python -m utilities.ohlc_audit refresh   --broker FXPIG-Server --tf 5
python -m utilities.ohlc_audit diff      --broker FXPIG-Server --tf 5
python -m utilities.ohlc_audit holes     --broker FXPIG-Server --tf 5 [--retry-permanent]
```

- `utilities/ohlc_gap_audit.py` becomes a thin shim. It has no callers today.
- CLAUDE.md conventions apply: StrEnum/IntFlag for every fixed set, one-line signatures under 120 chars.
- **MT5 safety:**
  - `MT5FetchAdapter` connects to the terminal the same way `fetch-mt5-data` does, using
    `initialize(path, server, login, password)` from the env file, so it is no more invasive than today's fetch.
  - It calls the `MetaTrader5` binding directly and uses only read calls: `initialize`, `symbol_select`,
    `copy_rates_range`, `copy_rates_from_pos`, `shutdown`.
  - It does not import `okmich_quant_mt5.functions`, so it avoids the live-path stamping and its stale-tick decode.
    The raw server epoch digits are taken as-is and aligned by the handshake.
  - Positional arguments only, because the binding silently ignores unknown kwargs.

## 8. Holdout protection

- The profile carries `holdout_after` per study scope. FXPIG: 2026-10-29.
- After the cutoff, detection, refetch and remediation still run; the rules are mechanical and frozen before scoring.
- The report shows only windows and kinds there: no correlations, magnitudes or vol statistics.
- The clock validation script never reads post-cutoff bars.

## 9. Acceptance criteria

| # | Test | Pass condition |
|---|---|---|
| A1 | Synthetic injections: gap, ±3 h week, interleave, repeated block, Saturday tail, spike, stale run, duplicate stamp | 100% found with the right kind; 0 findings on a clean synthetic series with sessions and holidays |
| A2 | Golden regression on the 2026-11-11 export | Every seed-registry window reported as CRITICAL or MAJOR |
| A3 | Structural closures (HKG50 holidays and HKEX weather closures, DE30 Christmas/Easter, 2025-01-09, 2025-11-28 CME outage, 2026-09-07 early close, metals early closes) | INFO only. Crypto weekend downtime is irregular, so it is NOT structural: it is reported as an outage and refetched (amended in P1). |
| A4 | Clock | Cash-open spike at 09:30 NY in each pre-cutoff era (the +60 min control moves it); D8a lag 0 on every FX symbol-week after remediation |
| A5 | Handshake, with a fake adapter | A +1 h-shifted fetch is aligned. A fetch with price disagreement on the margins is rejected. A fetch that would create an interleave is rejected. |
| A6 | Ledger | EMPTY on 3 runs ≥24 h apart → PERMANENT; transient errors don't count; FILLED holes leave the open set |
| A7 | Spikes | An injected single-symbol spike is replaced when the adapter returns a clean bar, and flagged SPIKE_CONFIRMED when it returns the same bar; MARKET_EVENT is untouched |
| A8 | Promotion safety | The snapshot exists and hash-matches before the first write; a kill mid-promotion leaves each file fully old or fully new; failed re-audit → `raw/` byte-identical |
| A9 | Remediated output | No duplicates, monotonic, no out-of-session bars, weekly counts ≤ template maximum, re-audit = 0 CRITICAL |
| A10 | Idempotency | Running remediate twice with no new broker data gives identical hashes |
| A11 | Performance | Audit + remediate without fetch: < 5 min for 25 × ~260k bars. Fetch time is reported separately. |
| A12 | Agnosticism (P6) | A second profile runs audit-only with zero changes to core modules |

## 10. Phases and gates

| Phase | Scope | Gate |
|---|---|---|
| P1 | Core interfaces, FXPIG profile, `EraShiftClock`, `expected_grid`, registry seed, shim | A4 clock validation; templates reproduce the A3 closures |
| P2 | Detectors D1–D11, findings, report | A1 + A2 + A3 |
| P3 | Remediation in staging without fetch (§6.1–6.3, 6.7), snapshots, manifest, dry-run diff, D12 | A8 (staging only) + A9 + A10 + A11 |
| P4 | `MT5FetchAdapter`, handshake, verify, hole ledger, spike replacement | A5 + A6 + A7; one live dry run on FXPIG, results reviewed with the user |
| P5 | Promotion to `raw/`, `refresh`, `diff`/`holes` commands, bat-line switch | User signs off the first promotion diff; A8 on real data |
| P6 | Second broker profile (audit-only), top-down review, promotion-to-SF decision | A12; user decision |

## 11. Decisions

### 11.1 Confirmed 2026-11-11

| # | Decision |
|---|---|
| 1 | Clean data eventually overwrites `raw/` (staging → verify → snapshot → promote, gated at P5) |
| 2 | Mis-clocked windows: drop by default; restamp only via a confirmed override (refetch then refills them) |
| 3 | Holes: refetch from the same broker; permanent after 3 failed attempts |
| 4 | Broker-agnostic core; FXPIG first |
| 5 | Holdout: blind mode (§8) |
| 6 | Snapshots: keep one per export (they are also the pre-promotion backup) |
| 7 | Spikes: flag; replace with the broker's re-fetched bar when it differs |
| — | Calendar source (frozen templates + `holidays`) and `load_clean` for new studies: accepted defaults |

### 11.2 Answered 2026-11-11 (were open in v2)

| # | Decision |
|---|---|
| Q1 | The cleaned `raw/` keeps **broker-native era stamps**, normalised so every symbol follows the era table, with the label unchanged. True UTC comes from `to_utc` / `load_clean`. |
| Q2 | FXPIG is fetched by **the tool's `refresh`**, with the handshake. It replaces the FXIFY line in `market-data-integration.bat` at P5. |
| Q3 | The terminal is resolved the same way as `fetch-mt5-data`: `--broker fxify --account demo` → `$OKMICH_QUANT_ENV_DIR/.env.fxify.demo` (§3.2, §7). |

## 12. Phase log

### P1: profile, clock, calendar, registry (DONE 2026-11-11)

Built: `types.py`, `clock.py`, `calendar.py`, `profile.py`, `registry.py`, `profiles/fxpig.py` and
`profiles/fxpig_bad_windows.csv` (22 rows), `validation/` (clock_check, derive_sessions, template_check) and 30 unit tests
(`python -m pytest utilities/ohlc_audit/tests`). The `ohlc_gap_audit.py` shim moves to P2, when there is an
`audit_symbol` to forward to.

**Design changes from §3:**

- **The clock is an offset schedule in raw time, not a tz localisation.** raw - UTC = era shift (or override) + NY's
  UTC offset. The DST switch defaults to NY's own transition read on the raw clock, and `DstSwitch` records measured
  per-symbol moments.
  - `to_utc` never returns NaT.
  - `unrepresentable_utc()` lists UTC hours that no raw stamp can reach (fall-back folds, backward seams). These are
    not gaps.
- **Slot states** are OPEN, THIN, CLOSED, HOLIDAY, EARLY_CLOSE and MAINTENANCE.
  - CLOSED covers weekend, overnight and lunch, and stays strict.
  - THIN is declared per template (FX rollover 17:00-18:00 NY). Empirical THIN stays a P2 detector input.
  - Holiday windows are anchor pairs (day offset, optional business-day step, time, zone). They relabel OPEN and THIN
    slots only, and bars inside them are tolerated.

**Gate A4 (clock), pre-cutoff, `validation/clock_check.py`:**

- The cash-open spike sits at 09:30 NY in eras 1-3 for US500, DJ30, USTEC and RUS2000 (2.5-4.5x the 09:00 level). The
  +60 min control moves it to 10:30. In DST-mismatch weeks DJ30, USTEC and RUS2000 also peak at 09:30; US500 peaks at
  09:50.
- BTC against Binance (true UTC): lag 0 in 150/150 weeks, median correlation 0.999. ETH: only 5 weeks could be
  tested (the ETH Binance export on disk covers 2026-09-08 -> 10-06), lag 0 in all 5 (0.995). ETH's clock rests on
  BTC's (same FXPIG feed). *Corrected 2026-11-11: the first run logged weeks with no reference as lag 0 ("140/140").*
- FX opens Sun 17:00 and closes Fri 16:55 NY in all 4 eras, era 4 included (timestamps only).
- Duplicate or reversed UTC stamps outside registry DROP windows: 0 on all 25 symbols.

**Gate (templates), `validation/template_check.py`:**

- Coverage of OPEN slots is 99.0-99.9%. The AUD pairs (97.4%) and USDCAD (99.25%) are lower only because their
  proposed-override weeks are excluded.
- 0 bars in CLOSED slots and 0 off-grid stamps.
- The remaining unregistered missing runs of 30 min or more are one-off outages that P2 reports and P4 refetches:
  - Crypto weekend downtime: 6.
  - HKG50 around the 2024-03-10 DST change: 5.
  - RUS2000: 1.
  - EURJPY: 1.

**FXPIG facts established in P1:**

- **Clock eras** NY+7 / +9 (2024-06-05) / +6 (2026-08-30) / +7 (2026-10-29 20:45), confirmed per era.
- **Crypto clock changes:**
  - The crypto feed applies US DST at its own moment. Measured: 2024-03-10 06:40, 2024-11-02 04:45, 2025-11-01 20:00
    and 2026-03-08 07:00 raw. 2026-11-01 is unmeasured (Binance export ends 2026-10-06).
  - Crypto switched to NY+6 at raw 2026-08-29 18:05, about 6 h before the era boundary. This is a CONFIRMED override.
- **Weekly schedules.** Every non-crypto symbol trades on a New York schedule:
  - FX: Sun 17:00 to Fri 17:00.
  - Metals: Sun 18:00 to Fri 16:50, with a 17:00-18:00 break.
  - The 7 index CFDs, DE30/HKG50/JPN225 included: Sun 18:00 to Fri 17:00, with a 17:00-18:00 break.
- **Holiday calendars:**
  - JPN225 follows CME, not Japanese holidays.
  - DE30 closes for Xetra holidays from 22:00 Berlin on the previous trading day to 00:15 UTC on the next, and stops
    at 22:00 Berlin on US short days.
  - HKG50 closes for HKEX holidays and weather closures from 03:00 HKT after the previous trading day to 09:15 HKT on
    the next. On US short days it stops early (16:30 HKT to 17:00 NY), with no fixed time.
- **Replay found by the lag-288 repeat scan:** raw 2026-09-08 16:20 -> 09-09 07:25. All 25 symbols replay bars from
  13.5 h earlier, 3,556 exact copies. The registry DROP covers 16:20 -> 10:00 and replaces the two shorter interleave
  windows. The 2026-11-11 audit list (and `vol_forecast_horse_race/check_corrupt_windows.py`) masked only the two
  edges.
- **The repeat scan up to lag 576 finds only three events:** the replay, AUD on 09-29 (3 h) and the 10-29 splice (1 h).

### P2: detectors, findings, report (DONE 2026-11-11)

Built:
- `findings.py`, `symbol_data.py` (the per-symbol inputs, built once), `detectors/` (integrity D1-D2, coverage D3-D5,
  content D6/D7/D9-D11, alignment D8a-D8b, reference D13), `audit.py` (`audit_symbol`, `audit_broker`), `report.py`
  and the `python -m utilities.ohlc_audit audit` CLI.
- `utilities/ohlc_gap_audit.py` is now a shim: `--path` is resolved to broker and symbol, then the single-symbol audit
  runs.
- 12 new tests, 42 in total.

**Design choices made while building (beyond §4):**

- **D1** also emits DST_FOLD (MAJOR, inspect) for bars in a raw hour the server clock passed through twice
  (`clock.ambiguous_raw`). A bar there may hold either pass, or both merged; only a reference can tell which.
- **D5 overfull** is a cross-check with action NONE: D1/D4 name the offending bars.
- **D7 alternation** is MINOR per symbol, raised to MAJOR (broker-wide) at 5 or more symbols within 30 min. A 2-symbol
  class (BTC + ETH) whipsawing together is a market move, not a feed event.
- **D8a peers:**
  - FX uses pairs sharing the quote currency. Same-base pairs are added only when fewer than two share the quote,
    because same-base pairs shared the 2026-09 AUD defect.
  - Hour lags from -12 h to +12 h, with 60% agreement of ALL peers with data (abstentions count against).
  - Metals and indices: |return| ranks against class peers, week by week. HKG50/JPN225 rely on D8b.
- **D8b:** the week's close sits a whole number of hours off the calendar, and its open shifts the same amount, up to
  +60 min more for a thin open. Weeks with holiday or early-close slots at either end are skipped.
- **D9:** same-class spikes within +-1 bar on min(3, class size) symbols become MARKET_EVENT (INFO).
- **D13 (new): reference check** against an independent true-UTC series, declared per symbol in the profile (FXPIG:
  BTC/ETH vs Binance).
  - Rolling median deviation above max(3x its prior level, 15 bp) is flagged.
  - At least 50 bp or at least 12 bars is CRITICAL (drop). Smaller is MAJOR (inspect). 1-2 bar blips are dropped.
  - A weekly whole-hour lag against the reference is CLOCK_MISALIGNED.
- **Robust scales** use prior-day medians, not rolling medians (speed).
- **Verdicts:**
  - REVIEW: any unregistered CRITICAL/MAJOR finding of a judgment kind (clock, bad prices, repeated block, overlap,
    OHLC, overfull, broker-wide interleave).
  - USABLE_AFTER_REMEDIATION: otherwise, if there are any CRITICAL/MAJOR findings.
  - CLEAN: none.
- **Broker-wide events** are clusters of MAJOR/CRITICAL findings of one kind on 3 or more symbols. Minor no-tick gaps
  are left out.
- **Blind mode:** findings after `holdout_after_raw` keep their window and kind; the report blanks their evidence.

**Gates:**

- **A1 (synthetic broker):** 0 findings when clean. All 9 injections are found with the right kind and window: outage,
  repeated block, Saturday bars, stale run, a week shifted +3 h (D8b + D8a), interleave, spike, duplicate stamp. D13
  flags a 100 bp divergence as CRITICAL.
- **A2 (`validation/golden.py`):** detectors alone hit 27/27 gated registry and override windows. Outage rows count any
  OUTAGE/HOLE detection, since 10-25 min outages are MINOR by design.
- **A3:** 0 unregistered CRITICAL/MAJOR findings inside recognised structural closures.
- **Runtime:** about 1.5 min for 25 x ~260k bars.

**FXPIG state after P2:**

- 25 symbols usable after remediation, 0 for review.
- 18 unregistered CRITICAL/MAJOR findings: 16 outages (refetch candidates: crypto weekend downtime, the HKG50 night of
  2024-03-10/11, RUS2000, EURJPY) and the 2 unmeasured crypto folds of 2026-11-01.
- Registry now has 27 rows:
  - Crypto folds 2024-11-02 / 2025-11-01.
  - Stale BTC during FOMC 2024-05-01.
  - Two inspect rows where FXPIG BTC is smoother than Binance with normal tick flow.
  - 2026-03-08 corrected to 06:40-07:35.
- Not yet built: "diff since the last run" and the quality heatmap (P4/P5 per §10). The CLI writes reports to
  `<data dir>/_audit/reports/<run time>/` by default; P2 runs used the scratchpad.

### P2 addendum: other bar schemas (2026-11-11)

Extra columns are never an issue.
- The audit requires only a DatetimeIndex and numeric open/high/low/close (`symbol_data.check_bar_schema`).
- It checks `tick_volume` and `spread` only when present.
- Everything else is carried untouched: IB `volume` / `average` (VWAP) / `barCount`, CCXT `volume`, and any custom
  column.
- IB's `volume = -1` / `average = -1` ("not available" on FX midpoint bars) is deliberately not a finding.

Added:
- **`ZoneClock(tz)`** for sources whose stamps are real wall time in one zone. CCXT (Binance/Bybit) and IB store UTC,
  so the clock is the identity.
- **`SymbolSpec.positive_prices`**, default True. Set it False for series that legitimately sit at or below 0
  (premium index, basis). It turns off D2 NON_POSITIVE and switches returns to plain differences.
- **`NotABarSeries`**: a clear error for non-bar datasets, such as CCXT `funding_rate` / `open_interest` or a
  non-datetime index.

Checked by `tests/test_audit_schemas.py` (51 tests now in total):
- IB-style frames, including -1 sentinels, are clean.
- CCXT trade / mark / premium klines are clean.
- The positive_prices setting matters.
- Extra columns leave the findings identical.
- Non-bar datasets are rejected.

Real data, `validation/schema_check.py`:
- The Binance BTC/ETH ohlcv, mark, index and premium-index klines audit without error. BTC: three clean, the premium
  index usable (one 30+ min outage).
- funding_rate and open_interest are rejected with NotABarSeries.
- IB parquets are not on this machine (fetch-ib-data writes to E:\).
- The ETH Binance export on disk covers only 2026-09-08 -> 10-06.

## 13. Promotion design (decided 2026-11-11; amends §3, §5, §6.8, §7, §8)

**Run inputs, never stored in a profile.** A `BrokerProfile` holds broker data only: clock, sessions, symbols, fetch
source, and reference paths relative to the raw root.
- **Market-data root:** `--data-dir`, else env `OKMICH_QUANT_MARKET_DATA_DIR` (the folder containing `raw/`), else
  fail. The argument wins, so a run aimed at a copy never falls back to the real tree.
- **Bad-window registry:** a CLI input (`--registry`). It is no longer a profile field. Without it every finding is
  unregistered.
- **Blind-after cutoff:** a CLI input (`--blind-after`). It is no longer a profile field (amends §8). The FXPIG
  validation tools default it to the horse-race cutoff.
- **Reports:** go to `--out`, or a per-run folder under the OS temp dir (amends §5).

**The market-data tree stays clean.** Nothing the tool produces is written inside it: no reports, no `_audit/`.
- This amends §6.4/6.7/6.8. The hole ledger, quality flags, manifest, staging and snapshots move to a separate state
  folder given as an input.
- **Decide at P3:** its name and default, e.g. `--state-dir` / an env var.

**CLI = `raw-data-audit`** (console script `okmich_quant_utils.ohlc_audit.cli:main`).
- `raw-data-audit audit` is built.
- remediate / refresh / diff / holes arrive with P3-P5.
- Exit codes: 0 clean or usable, 1 review, 2 usage or configuration error.

**Robustness work done:**
- `types.py` was renamed `enums.py`, because it shadowed the stdlib `types` when run from the package folder.
- A one-call public API in `__init__`, and a README.
- ruff clean (E, F, W, B, UP, SIM; line length 120).
- Edge-case tests: empty frame rejected; single bar, unsorted, tz-aware, NaN rows and all-closed frames handled.
- A determinism test (identical findings across runs).
- CLI tests: exit codes; nothing written into the data tree.
- 67 tests in total.
- Real-data regression after the refactor is unchanged: A2 27/27, A3 pass, 25/25 usable, 18 unregistered
  CRITICAL/MAJOR findings.
- **MT5 refetch adapter (P4):** lives in the package with a lazy `MetaTrader5` import (decided).

## 14. P3: remediation into the state folder (DONE 2026-11-11)

Built:
- `state.py`: `--state-dir`, else `OKMICH_QUANT_DATA_STATE_DIR`, else fail. The folder must exist and must not sit
  inside the market-data tree. Layout per broker/tf: `staging/`, `quality/`, `runs/<time>/`, `snapshots/<time>/`.
- `remediate.py`: `remediate_broker`.
- `snapshot.py`: `snapshot_broker` and `history_diff`, which is D12.
- CLI `remediate` / `snapshot` / `diff`.
- 8 new tests, 75 in total.

**Rules, refining §6:**

- **Rows only:** rows are selected from the original frame, never edited. Columns, dtypes, the index name and the tz
  label stay as in raw.
- **Order and duplicates:** sort by raw stamp (out-of-order bars keep their stamps), then drop duplicate stamps,
  keeping the first.
- **Dropped automatically, listed or not (mechanical):** off-grid or future stamps, clock overlaps, broken OHLC, NaN or
  non-positive prices, out-of-session bars, copied blocks.
- **Dropped:** registry DROP windows, and PROPOSED override windows (refetched in P4).
- **Held, never dropped (decided 2026-11-11):** unlisted judgment findings (D8a clock, D8b stamp drift, D13 bad
  prices). The bars stay, are flagged HELD_FOR_REVIEW, and the symbol status is REVIEW until the registry records a
  decision.
- **Confirmed clock overrides are not restamped in the file.** A file can't carry two clocks across an era boundary,
  so the clock applies them at read time.
- **Spread is never edited (decided 2026-11-11; amends §6.6).** Size is no evidence of invalidity: a blown-out spread
  is the market (silver).
  - Negative spread is SPREAD_INVALID.
  - Zero is SPREAD_ZERO (INFO; real on raw accounts, or a recording quirk).
  - The extreme-spread rule is gone from D11.
- **Quality flags** (`QualityFlag`, sidecar `quality/<file>.parquet`): AFTER_GAP, NEAR_DROPPED, SPIKE_SUSPECT,
  MARKET_EVENT, SPREAD_INVALID, SPREAD_ZERO, THIN_SLOT, STALE_RUN, ZERO_VOLUME, DST_FOLD, INTERLEAVE_SUSPECT,
  REFERENCE_DIVERGENCE, HELD_FOR_REVIEW. RESTAMPED and REFETCHED are reserved for P4.
- **Verify:** re-audit the staged copies with the same registry.
  - VERIFIED: 0 CRITICAL detector findings outside held windows.
  - FAILED_VERIFY: otherwise.
  - REVIEW: when findings are held.

**Gates on the real FXPIG export (`raw-data-audit remediate --registry fxpig_bad_windows.csv`):**

- **A9:** 25/25 VERIFIED, 0 held, 0 failed. The staged copies are monotonic, have no duplicates, no out-of-session
  bars, and 0 CRITICAL findings.
  - Typical removal is about 1,500 bars per symbol: the first week 1,283, the replay 212, the splice about 40.
  - AUD pairs lose about 7,800 (proposed override weeks), USDCAD about 3,000, crypto about 460.
- **A10:** two runs give byte-identical staging files (sha256 per symbol).
- **A8 (staging part):** the input hashes in the manifest match `raw/` after the run, so raw is untouched.
- **A11:** audit + remediate + verify for 25 x ~260k bars in 3m07s.
- **D12:** snapshot then diff gives 0 rewrites on unchanged files. A synthetic test confirms that rewritten and
  vanished bars are found, and that bars appended after the snapshot are not.

## 15. P4: refetch from the broker (BUILT 2026-11-11; live dry run pending)

Built:
- `fetch.py`: the `FetchAdapter` protocol and `MT5FetchAdapter`. Errors: `TransientFetchError` (retried, never counted)
  and `FetchUnavailableError` (stops the run before any ledger change).
- `handshake.py`, `ledger.py` (`HoleLedger`, `Target`, `Outcome`), `fills.py` (fill store, `compose`), `refetch.py`
  (targets, chunked fetch, verify), `policy.py` (MECHANICAL / JUDGMENT / VERIFY_REJECT).
- CLI: `remediate --fetch [--retry-permanent] [--min-hole-minutes N] [--env-dir DIR]` and `holes`.
- 24 new tests, 99 in total.

**Decisions taken in P4 (amend §6.4, §6.5, §6.7, §7, §14):**

1. **Fetch is opt-in** (`--fetch`; amends §7's `--no-fetch`). Nothing connects to a terminal unless asked. P5's
   `refresh` fetches by definition.
2. **Targets:**
   - DROPPED: every hole overlapping a window remediation dropped. Always asked.
   - MISSING: other raw-file holes of at least `--min-hole-minutes` (default 30, the MAJOR line). This amends §6.4
     "every expected-but-missing run". On the 2026-11-11 export, 3,634 of the 3,717 raw holes are under 30 min and
     3,514 of those are single bars, mostly metals and JPN225 in quiet hours: no-tick slots. Asking for them costs
     thousands of terminal calls a run, and they would all end PERMANENT. They keep AFTER_GAP.
   - SPIKE: single-symbol spike suspects. MARKET_EVENT is never asked.
3. **Fill store** (`<state>/<broker>/<tf>/fills/<file>.parquet`). Staging is rebuilt from `raw/` every run, so the
   verified broker bars persist there, in the raw schema plus provenance columns, until P5 promotes them.
   - The store only grows.
   - A fill applies where raw has no bar.
   - A replacement applies only while raw still holds the logged old bar, so a later export wins.
   - Runs without `--fetch` splice the store in too, which keeps reruns byte-identical.
4. **Rows (amends §14):** every staged row is either a raw row or a verified broker row from the store. Nothing is
   edited by formula.
5. **Handshake:**
   - The fetch range is padded by the 12 h maximum shift, so a moved broker clock still returns the margins.
   - Offsets are whole bars with |k| <= 12 h. Ties go to the smallest |k|.
   - Stamps are compared as ns integers after normalisation. The first real-data run caught the bug: parquet loads in us.
6. **Verify:**
   - D1, D2, D4-D8 and D13 run on the splice +-8 days, against peers sliced the same way (for D8a).
   - A *new* finding of a VERIFY_REJECT kind, at any severity, within a day of the target rejects that target.
     VERIFY_REJECT is: mechanical, judgment, duplicate, non-monotonic, overfull, interleave.
   - A replaced spike must no longer be a D9 suspect.
   - **Unchanged is not a fix.** Fetched bars identical in stamp and OHLC to bars remediation dropped are refused
     before verification. The broker still holds the defect, so serving it again is no new evidence. If nothing else
     is left, the attempt is CONFIRMED and counted, like a spike served unchanged.
     - Found by the replay check below: verification alone let 28 of 153 dropped windows back in. These were the
       10-29 splice tail on 16 symbols, the crypto DST-fold hours, and 2 days of the AUDCAD +9 h week, where D8a
       abstained.
     - A window the registry drops too widely therefore stays out. Narrow the registry row to recover good bars.
7. **Ledger:**
   - Statuses: OPEN, PARTIAL, FILLED, PERMANENT, RESOLVED.
     - STRUCTURAL is dropped: structural closures never become targets.
     - RESOLVED is new: a target no longer found, because the export now has the bars.
   - PARTIAL (progress) doesn't count. After a partial fill, the residual targets match their row by containment.
   - An attempt inside the 24 h cool-down is still made (it may fill) but doesn't count.
   - A spike CONFIRMED counts. After 3, the spike is PERMANENT with SPIKE_CONFIRMED kept.
8. **Terminal behaviour:**
   - A fetch holding under half the stored bars of its range is asked again twice, 2 s apart, because MT5 downloads
     history lazily.
   - Transient errors: 3 tries with 2 s / 8 s backoff. After 3 failed fetches in a row the run stops fetching (exit 1).
   - When a fetch returns nothing at all, the ledger records the earliest bar the terminal serves (history depth,
     "Max bars in chart").
9. **MT5 safety (refines §7):**
   - Calls: `initialize`, `account_info`, `last_error`, `symbol_info`, `symbol_select`, `copy_rates_range`, `shutdown`.
   - `initialize` takes fetch-mt5-data's four keywords. `account_info` must then show the env file's login and
     server, and the server must equal the profile's data folder (no mixing feeds).
   - Query bounds are raw digits on a UTC datetime. The MetaTrader5 docs say a naive datetime is read in the host's
     zone, and this host is UTC+2.
   - The env file is read with EnvLoader, and `os.environ` is restored afterwards.
10. **Flags:**
    - REFETCHED and RESTAMPED (offset != 0) on store rows.
    - SPIKE_CONFIRMED (bit 15) on spikes the broker served unchanged.
    - PERMANENT_HOLE_ADJACENT (§6.7) is not built: AFTER_GAP marks the bar, and the ledger says whether the hole is
      permanent.
11. **Lint note corrected:** the package has always been linted with `--ignore E501,SIM108`. Signatures follow the
    120-character rule; other lines may run longer. §13's "line length 120" overstated it.

**Offline check on the real FXPIG export (2026-11-11).** A fake broker served the raw files back unchanged: the worst
case, a broker whose history still holds every defect. Dry run, nothing written.
- **275 targets, 172 fetch calls:**
  - 153 dropped windows: all CONFIRMED (served unchanged, refused).
  - 70 holes of 30 min or more: all EMPTY.
  - 52 spikes: all CONFIRMED (kept, SPIKE_CONFIRMED).
  - Nothing was put back, and 25/25 symbols are VERIFIED with P3's row counts.
- **Without the identity rule,** verification alone rejected 125 of the 153 dropped windows:
  - every replay-window target;
  - the USDCAD week and most AUD clock weeks, by D8a, D6, D7 and D13.
  - 28 slipped through (see decision 6).
- **This check found a bug first:** stamps were compared across us/ns units, so every handshake failed. It is fixed,
  with a regression test (an adapter returning a us index with a tz label).
- **Time:** 256 s in total, 47 s of it refetch work, on a loaded machine. On the same machine at that moment, the
  pre-P4 tests ran about 3x slower than earlier in the day.
- **Not done: the live dry run** (`remediate --fetch --dry-run`) against the FXIFY terminal. It waits for the user.
  - It also checks, read-only, how the binding reads naive query datetimes on this UTC+2 host.
  - SF `functions.mt5._to_server_naive` passes naive ones on the live path. Report only, don't fix.

## 16. Promotion to signalfoundry (2026-11-11)

The user asked to promote before P5 and to set up the CLI in signalfoundry. This amends the earlier plan, which was to
move the package once it was complete.
- **Package:** `utils/src/okmich_quant_utils/ohlc_audit/`, with this SPEC and the README.
- **Tests:** `utils/tests/ohlc_audit/`. Imports were rewritten, and sibling test helpers are imported relatively.
- **`utils/pyproject.toml`:**
  - Adds `holidays==0.94`, which was already in `uv.lock` as a transitive dependency. The lock gains only the edge.
  - Adds `[project.scripts] raw-data-audit = "okmich_quant_utils.ohlc_audit.cli:main"`.
- **Install:** `uv pip install -e utils --no-deps` re-registered the editable package and created `raw-data-audit.exe`
  in the venv. There was no `uv sync`.
- **Lab cleanup (the user's decision, 2026-11-11):** nothing of the tool stays in the lab.
  - Deleted: the lab copy, the `validation/` scripts, the deprecated `ohlc_gap_audit.py` shim and the seed registry
    CSV. Lab commits `d1fca45` and `54c6ba7` keep the early versions.
  - The FXPIG registry is to be rebuilt with the tool (README "Building a registry").
  - The vol horse race is unaffected: `vfhr/data.py` keeps its own frozen copy of the DROP rows (`AUDIT_DROPS`). Its
    SPEC still cites the old CSV path as provenance.
  - The lab `market-data-integration.bat` keeps `raw-data-audit` sample lines.
- **P5** (promotion into `raw/`, `refresh`) is built here in signalfoundry.

## 17. Code review fixes (2026-11-11)

Three reviews of the promoted package found nineteen defects (R1-R19). All were reproduced on the code, fixed, and given
regression tests in `utils/tests/ohlc_audit/test_audit_review_fixes.py`, 121 tests in total. R14 and R16 were holes in
the R2 and R5 fixes of the first round.

| # | Defect | Fix |
|---|---|---|
| R1 | Clock overlaps were judged in stored row order, so two swapped valid rows deleted one of them. | D1 judges overlaps in raw-stamp order. Order problems stay NON_MONOTONIC, and remediation sorts them. |
| R2 | Mechanical drops used finding windows (stamp + bar), so an off-grid 08:20:30 also deleted 08:25:00. | `Finding.rows` records the defective rows. Mechanical defects drop exactly those rows; only registry rows and proposed overrides drop by window. Duplicates keep a defect-free copy when there is one. |
| R3 | Any overlap registered a finding: a 5-minute row silenced a week-long clock finding. | Registered only when compatible rows cover all of it but one bar. Outage / hole rows register only missing-bar findings. A partial overlap is noted in the evidence, and the finding is held. |
| R4 | Verification ignored any finding that overlapped one of the same kind, even when it got worse (25 bp MAJOR to 953 bp CRITICAL). | `worse_findings`: a finding is rejected when it is new, more severe, has more bars, or covers time outside the existing ones. |
| R5 | The hold rule required `Action.DROP`, and MAJOR BAD_PRICES uses NONE, so remediation said VERIFIED where the audit said REVIEW. | Unregistered CRITICAL/MAJOR judgment findings are held whatever their action. A MAJOR review finding that survives remediation also makes the symbol REVIEW. |
| R6 | Blinding checked only a finding's start, so a window crossing the cutoff showed evidence from post-cutoff bars. | A finding is blind when its window reaches past the cutoff. The same rule applies to refetch attempts. |
| R7 | Refetch targets came from the staged span, so a dropped head or tail vanished. | Holes are found on the original audit span. A target without 12 trusted bars on one side is recorded NO_MARGIN, not counted, and not asked. |
| R8 | Dropping every bar raised NotABarSeries and aborted the broker run. | Per-symbol status EMPTY: no staging is written, and the other symbols carry on. P5 must never promote an EMPTY symbol. |
| R9 | NaT timestamps passed the schema check and survived into staging, which then wasn't in order. | NaT rows are quarantined in `SymbolData`, and D1 MISSING_STAMP (CRITICAL, mechanical) drops them by row. |
| R10 | The handshake tested only the most-voted offset. 100%/75% beat 83%/83%, though only the latter met both thresholds. | Every voted offset is scored. No shift wins when it qualifies, else the single qualifying offset. Several qualifying offsets are ambiguous: reject, because the margins can't tell the clock (e.g. flat, repeated bars). |
| R11 | Spike clustering chained: spikes on three symbols at 01:00, 01:15 and 01:30 became one protected MARKET_EVENT. | Each spike needs its own same-class peers within +-1 bar. No chaining. |
| R12 | Bars older than the snapshot (a backfill) counted as inserted, i.e. a rewrite. | Only the snapshot's span is compared, at both ends. |
| R13 | A requested symbol missing from the snapshot was skipped, so a default diff against a subset snapshot passed silently. | `SnapshotCoverageError` (exit 2) names the missing symbols and the ones the snapshot holds. |
| R14 | `assess` dropped every registered finding before counting CRITICALs, so a registered CLOCK_OVERLAP or OVERFULL passed VERIFIED. | A CRITICAL integrity (non-judgment) finding fails verification whatever the registry or held windows say. Registry decisions and held windows excuse judgment findings only. |
| R15 | Verification ignored magnitude: 59.8 bp to 953.1 bp over the same 101 bars (both CRITICAL) was accepted as REPLACED. | `worse_findings` also compares magnitude per kind (`MAGNITUDE`: bp off reference, interleave size, clock lag, stamp drift). Any increase rejects, with no tolerance (the user's ruling, 2026-11-12): a rejected refetch counts as a failed attempt even when an interleave's size only drifts as bars are added. |
| R16 | Overlap detection flagged only the first copy of a duplicated stamp, and the planner kept the unflagged copy. | An overlap belongs to the timestamp: every row at an overlapping stamp is flagged and dropped. |
| R17 | The original span converted `last_raw + bar`, which crossed a forward clock seam and cut 2 h off the tail. | The span is the UTC extrema of the observed stamps, plus one bar in UTC. |
| R18 | Market-event matching compared every spike with every other in its class: O(n^2), 27 s for 2,000 spikes. | `audit.market_event_spikes`: sorted with a two-sided window, O(n log n). Classes with too few spiking symbols are skipped. 20,000 spikes run well under 2 s in the test. |
| R19 | An EMPTY run left the previous run's staging and quality files, which readers could still pick up. | An EMPTY symbol's old staging and quality files are removed (write runs only). |

**Follow-up from a real-data dry run** (FXPIG, no registry):
- R2's row-exact drops left BTCUSDT bars at the 2026-03-08 DST switch that don't overlap, but sit 52.7 bp off Binance.
  The old window drop had removed them silently.
- They surfaced as a CRITICAL BAD_PRICES after remediation, labelled FAILED_VERIFY. That label means a defect slipped
  through the mechanical fixes.
- The status rule (`remediate.assess`) now says:
  - a judgment finding exposed by remediation is REVIEW;
  - a registered judgment finding (e.g. an `inspect` decision) never fails verification;
  - FAILED_VERIFY is kept for CRITICAL non-judgment findings, registered or not (R14).
- Result with no registry: 6 VERIFIED, 19 REVIEW, 0 FAILED_VERIFY.
  - Held: BTC/ETH against Binance, an AUDJPY clock shift, and broker-wide replay/splice interleaves.
  - Without the R5 fix those interleaves would have passed as VERIFIED.

**Behaviour changes worth knowing:**
- Registry rows must now cover the whole defect window. A row drawn short leaves the finding held: widen the row to
  the detector's window.
- A symbol whose audit verdict is REVIEW comes out of remediation REVIEW unless the cause was mechanical and removed.
- Mechanical rules drop only the rows they prove wrong. Bars that are only judged wrong wait for a registry decision.

