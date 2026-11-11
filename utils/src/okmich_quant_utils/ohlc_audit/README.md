# ohlc_audit: raw bar-data audit (`raw-data-audit`)

Audits a broker's raw OHLC bar parquets and reports what is wrong with them:
- gaps, and bars where the market is closed;
- wrong clocks (seams, per-symbol shifts, DST folds);
- copied, replayed or interleaved feeds;
- broken prices, spikes, stale runs, bad spread;
- divergence from an independent reference.

Every finding has a kind, a severity, a window in both raw and UTC time, and the remediation action it implies. The
design and its decisions are in [SPEC.md](SPEC.md).

- **Built:** P1-P4 (detection, report, remediation into a staging copy, snapshots, history diff, refetch from the
  broker through the overlap handshake with a hole ledger).
- **Not built yet:** promotion into `raw/` and `refresh` (P5).

Part of `okmich-quant-utils`, promoted from the lab on 2026-11-11. The console script `raw-data-audit` is installed
with the package. The FXPIG registry is being rebuilt with the tool (see "Building a registry").

## Run it

```
raw-data-audit audit     --broker FXPIG-Server [--registry CSV] [--symbols ...] [--blind-after T] [--out DIR | --no-write] [--json]
raw-data-audit remediate --broker FXPIG-Server [--registry CSV] [--symbols ...] [--blind-after T] [--dry-run]
                         [--fetch [--retry-permanent] [--min-hole-minutes N] [--env-dir DIR]]
raw-data-audit holes     --broker FXPIG-Server [--symbols ...] [--all]
raw-data-audit snapshot  --broker FXPIG-Server [--symbols ...]
raw-data-audit diff      --broker FXPIG-Server [--snapshot NAME] [--symbols ...]
```

`python -m okmich_quant_utils.ohlc_audit <command> ...` is the same. All commands take `--data-dir`, `--tf`,
`--symbols` and `--quiet`; the stateful ones (`remediate`, `holes`, `snapshot`, `diff`) also take `--state-dir`.

| Input | Meaning |
|---|---|
| `--data-dir` | Market-data root, the folder that contains `raw/`. Else env `OKMICH_QUANT_MARKET_DATA_DIR`, else exit 2. The argument wins. |
| `--state-dir` | Tool state: staging copies, quality flags, run manifests, snapshots. Else env `OKMICH_QUANT_DATA_STATE_DIR`, else exit 2. Must exist, and must sit outside the market-data tree. |
| `--registry` | Bad-window CSV: your curated keep/drop decisions. Optional; without it every finding is unregistered. |
| `--blind-after` | Raw stamp after which findings show only window and kind (holdout protection). A finding whose window reaches past it is blind too. |
| `--out` | `audit` report folder. Default: `<OS temp>/raw-data-audit/<broker>/<run time>/`. |

**Nothing is ever written into the market-data tree.**

**`remediate`** builds `<state>/<broker>/<tf>/staging/<file>.parquet`. Columns, dtypes, the index name and its tz
label are exactly as in raw. Every row is a raw row or a verified broker row from the fill store; nothing is edited.
- Sorts by stamp; rows that are merely out of order are kept. Duplicate stamps keep one copy, the first without a
  defect.
- Drops registry DROP windows and PROPOSED clock-override windows (refetched later): every row stamped inside.
- Drops mechanical defects **by exact row**, never by window: rows without a timestamp, copied blocks, out-of-session
  bars, broken OHLC, off-grid or future stamps, clock overlaps. A bad 08:20:30 stamp never takes 08:25:00 with it.
- **Holds** unregistered judgment findings (clock shifts, stamp drift, bad prices against a reference), whatever their
  action: the bars are kept and the symbol is REVIEW until the registry records a decision.
- Writes per-bar `QualityFlag`s to `quality/`, and a manifest plus before/after findings to `runs/<run time>/`.
- Re-audits the staged copy:
  - FAILED_VERIFY: a CRITICAL integrity finding remains (overlap, overfull, broken OHLC...), registered or not, held
    window or not. Registry decisions record judgments; they never excuse broken data.
  - REVIEW: held findings, or an unregistered judgment finding (CRITICAL/MAJOR) or MAJOR review finding still present
    outside the held windows, e.g. a broker-wide interleave, or bars exposed once mechanically bad neighbours were
    removed.
  - VERIFIED: neither.
  - EMPTY: every bar was dropped. Nothing is staged for the symbol; a previous run's staging and quality files for it are
    removed. The rest of the run carries on.
- The same input gives byte-identical output. `--dry-run` writes nothing.

**`remediate --fetch`** (P4) also asks the broker itself for bars, read-only, through its MT5 terminal. It connects
the way `fetch-mt5-data --broker <b> --account <a>` does: the profile names the env file `.env.<b>.<a>` in
`OKMICH_QUANT_ENV_DIR` (or `--env-dir`). Without `--fetch` nothing connects anywhere.
- **Targets:**
  - every hole that overlaps a dropped window;
  - every other raw-file hole of `--min-hole-minutes` or more (default 30; shorter gaps are no-tick slots and keep
    their AFTER_GAP flag);
  - every single-symbol spike suspect. Market events are never touched.
  - Holes are found on the original file's span, so a dropped head or tail is a target too. Such a target has no
    trusted bars on one side; it is recorded as NO_MARGIN and not asked (a one-sided handshake can't prove the clock).
- **Overlap handshake:** each target is fetched with 12 trusted bars on each side. An offset qualifies when it
  reproduces at least 80% of both margins by exact OHLC; every offset the margins vote for is checked.
  - No shift wins when it qualifies.
  - Otherwise a single qualifying offset restamps the fetched bars (this absorbs a broker clock change).
  - Several qualifying offsets are ambiguous, and so is no qualifying offset: the attempt is rejected.
- **Unchanged is not a fix:** fetched bars identical to bars remediation dropped are refused. The broker still holds
  the defect. If nothing else is left, the attempt counts as CONFIRMED.
- **Verify:** the splice is re-audited over ±8 days, with D1, D2, D4-D8 and D13. Any new or worse defect within a day of
  the target rejects it: new, more severe, deeper (bp off the reference, interleave size, clock lag), or reaching
  further than the finding already there.
- **Spikes:**
  - If the broker now serves a different bar that no longer spikes, the bar is replaced. The old values are logged in
    the manifest.
  - If it serves the same bar, the bar is kept and flagged SPIKE_CONFIRMED (what a live system saw).
- **Winners** go to the fill store `<state>/<broker>/<tf>/fills/`. Every later run splices them in, with or without
  `--fetch`, until P5 promotes them.
- **Hole ledger** `<state>/<broker>/<tf>/holes.parquet`:
  - These results count as failed attempts: EMPTY, REJECTED_VERIFY and CONFIRMED. At most one counts per run, and
    counted attempts are at least 24 h apart.
  - 3 counted failures make a target PERMANENT. It is then skipped unless `--retry-permanent` is given.
  - Terminal, IPC and timeout errors never count. They are retried with backoff; after 3 failed fetches in a row the
    run stops fetching. NO_MARGIN doesn't count either.
- `--fetch --dry-run` fetches and reports, but writes nothing (no ledger, no fill store, no staging).

**`holes`** prints the ledger: targets by kind and status, then the open, partial and permanent ones.

**`snapshot`** copies `raw/<broker>/<tf>` into `snapshots/<time>/`, hash-verified.

**`diff`** (D12) reports bars that changed, vanished or were inserted since a snapshot: an export that rewrote
history.
- Only the snapshot's own span is compared. Newer data and an older backfill are not rewrites.
- Every symbol asked for (default: all of them) must be in the snapshot, else exit 2, naming the missing ones. A
  snapshot of a subset is compared with `--symbols` and that subset.

**Outputs of `audit`:** a console summary, plus `findings.csv`, `summary.csv`, `events.csv` and `summary.json`.

**Exit status:**
- `0`: all clean / usable / verified, no rewrite.
- `1`: something needs review.
- `2`: a usage or configuration error.

## Python API

```python
from okmich_quant_utils.ohlc_audit import audit_broker, audit_symbol, get_profile

rep = audit_broker("FXPIG-Server", data_dir=r"D:\data_dump\market_data", registry_path="fxpig_bad_windows.csv")
rep.summary_frame(); rep.findings_frame(); rep.events
rep = audit_broker(profile=my_profile, frames={"EURUSD": df})      # bars in memory, no data root needed
```

## Concepts

**Broker profile** (`profiles/<broker>.py`). Broker data only, no run settings:
- **Clock: raw stamp to UTC.**
  - `EraShiftClock`: MT5 "NY close" servers. Raw = NY + shift, with eras, per-symbol overrides, and measured DST
    switch moments.
  - `ZoneClock`: stamps already in one real zone; UTC for CCXT and IB.
- **Session templates** per symbol group: the weekly schedule plus holiday rules (anchor windows over `holidays`
  calendars), and declared THIN ranges.
- **Symbol specs:** asset class, session, and `positive_prices` (False for basis or premium series).
- **Optional reference series** for D13, as paths relative to `<data root>/raw`.

**Bar schema.** Only a DatetimeIndex and numeric `open/high/low/close` are required. MT5 `tick_volume`/`spread` are
checked when present. Spread is never judged by size: a blown-out spread is the market. Negative spread is invalid;
zero is flagged for information. IB `volume`/`average`/`barCount` (including their `-1` sentinels) and CCXT `volume` are carried
untouched. Non-bar datasets (funding rate, open interest) raise `NotABarSeries`.

**Slot states** (from `expected_grid`):
- `OPEN`: a bar is expected.
- `THIN`: absence is minor.
- `CLOSED`: a bar is out of session.
- `HOLIDAY` / `EARLY_CLOSE`: absence expected, bar tolerated.

**Detectors** (SPEC §4):
- D1 stamps, clock overlaps, DST folds.
- D2 OHLC integrity.
- D3 missing bars, and structural closures as INFO.
- D4 out of session; D5 overfull weeks.
- D6 copied blocks; D7 alternation.
- D8a peer clock alignment; D8b weekly stamp drift.
- D9 spikes / market events (a spike is a market event when enough same-class symbols spike within one bar of it);
  D10 stale runs / zero volume; D11 spread.
- D13 reference check.

**Severity:**
- CRITICAL: data present but wrong.
- MAJOR: 30 min or more missing inside a session, or a broker-wide anomaly.
- MINOR, INFO.

**Verdict:**
- `review`: an unregistered CRITICAL/MAJOR finding of a judgment kind.
- `usable_after_remediation`: other CRITICAL/MAJOR findings only.
- `clean`: none.

**Registry CSV.** One row per known window:
`id, symbols (ALL or A;B), from_raw, to_raw (raw stamps, end exclusive), kind, action (drop|refetch|inspect), evidence,
found_on, export_date`.
- A finding is "registered" when compatible rows cover all of it but at most one bar. A partial overlap is no decision
  about the rest: the finding stays unregistered (held for review) and names the rows that cover part of it.
- `outage` / `hole` rows record missing bars, so they register only missing-bar findings, never a clock or price one.
- `inspect` rows record a decision to keep bars.

## Adding a broker

1. Write `profiles/<broker>.py`: symbols, clock, sessions, fetch source. Register it in `profiles/__init__.py`.
2. Derive the weekly schedules from the bars and freeze them in the profile. A good template leaves almost no bars in
   CLOSED slots and covers the OPEN slots.
3. Validate the clock:
   - US cash-open moves land at 09:30 New York in every era;
   - a true-UTC reference, where one exists, shows lag 0.
4. Build the broker's registry (below).
5. Gate: with the registry, the audit leaves no unregistered CRITICAL/MAJOR finding, and structural closures are INFO
   only.

The FXPIG validation scripts of 2026-11-11 were lab-only and are deleted. Lab commit `d1fca45` keeps their first
versions.

## Building a registry

The registry records **decisions**, not detections. The detectors find the windows: on the 2026-11-11 export they
flagged all 27 windows of the first FXPIG registry. Mechanical defects need no row, because they are dropped anyway.

1. `raw-data-audit audit --broker <b> --out <dir>`, then read `findings.csv`.
2. For each unregistered CRITICAL/MAJOR finding, decide and add a row:
   - `drop`: the bars are wrong (clock shift, bad prices against a reference, an unverifiable stretch);
   - `refetch`: bars are missing and should be asked for again;
   - `inspect`: you looked, and the bars stay.
   Put the evidence in the `evidence` column.
3. Re-run `audit --registry <csv>` until no unregistered CRITICAL/MAJOR finding is left, then
   `remediate --registry <csv> --dry-run`.

Keep the CSV under version control, outside the market-data tree. The first FXPIG registry was deleted on 2026-11-11
for a rebuild with this workflow. The vol horse race keeps its own frozen copy of the DROP rows in code.

**MT5 safety (`fetch.py`):**
- `MetaTrader5` is imported only when a fetch opens; nothing in this package imports `okmich_quant_mt5`.
- Calls used: `initialize`, `account_info`, `last_error`, `symbol_info`, `symbol_select`, `copy_rates_range`,
  `shutdown`.
- Data calls are positional. `initialize` gets the same four keywords fetch-mt5-data passes. `account_info` must then
  show the env file's login and server, and the server must be the profile's data folder, so no other feed can mix in.
- Query bounds go in as raw digits on a UTC datetime, and bars come back as the server's own digits: no tz decode.
- The env file is read with `EnvLoader`, and the process environment is restored afterwards.

## Tests

`python -m pytest utils/tests/ohlc_audit` covers:
- clock;
- calendar;
- registry;
- synthetic injections A1 (and the determinism check);
- schemas;
- run inputs and edge cases: empty, single bar, unsorted, tz-aware, NaN rows, all bars closed;
- remediation (A9, A10);
- refetch with a fake broker: handshake and verify (A5), ledger (A6), spikes (A7). A fake `MetaTrader5` module checks
  the adapter's call surface, its env handling and its error mapping.

## Packaging

- `utils/pyproject.toml` declares the `holidays` dependency (0.94) and the `raw-data-audit` console script. After
  pulling, re-register the editable package so the script exists:
  `uv pip install --python .venv/Scripts/python.exe -e utils --no-deps` (no `uv sync`, which prunes other packages).
- The MT5 adapter imports `MetaTrader5` lazily, so utils gains no hard MT5 dependency. It uses
  `okmich_quant_core.env_loader`, which utils already depends on.
- Python >= 3.12 (StrEnum). Package imports are relative only.
- Lint: `ruff check --select E,F,W,B,UP,SIM --ignore E501,SIM108`. Signatures follow the 120-character rule; other
  long lines are tolerated.
