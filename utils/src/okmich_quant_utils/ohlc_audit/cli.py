"""``raw-data-audit``: one front door to the raw bar-data audit (console script of okmich-quant-utils).

    raw-data-audit audit     --broker FXPIG-Server [--registry CSV] [--symbols ...] [--blind-after T] [--out DIR | --no-write] [--json]
    raw-data-audit remediate --broker FXPIG-Server [--registry CSV] [--symbols ...] [--dry-run]
                             [--fetch [--retry-permanent] [--min-hole-minutes N] [--env-dir DIR]]
    raw-data-audit holes     --broker FXPIG-Server [--symbols ...] [--all]
    raw-data-audit snapshot  --broker FXPIG-Server [--symbols ...]
    raw-data-audit diff      --broker FXPIG-Server [--snapshot NAME] [--symbols ...]

Inputs:
- Market-data root (the folder containing raw/): ``--data-dir``, else OKMICH_QUANT_MARKET_DATA_DIR, else fail.
- Tool state (staging, quality flags, runs, snapshots): ``--state-dir``, else OKMICH_QUANT_DATA_STATE_DIR, else fail.
  Must sit outside the market-data tree.
- Bad-window registry: ``--registry``. Without it every finding is unregistered.
- ``--fetch`` connects read-only to the broker's MT5 terminal the way ``fetch-mt5-data`` does (the profile's
  ``.env.<broker>.<account>`` in OKMICH_QUANT_ENV_DIR, or ``--env-dir``) and refetches holes, dropped windows and
  spike suspects. Without it nothing connects anywhere. ``--fetch --dry-run`` fetches and reports but writes nothing.
Nothing is ever written into the market-data tree. ``audit`` reports go to ``--out`` or an OS temp folder.

Exit status:
- 0: all clean / usable / verified, no history rewrite.
- 1: something needs review (a review verdict, a symbol held or failing verification, a history rewrite, a fetch run
  stopped by terminal errors).
- 2: a usage or configuration error.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import report as rpt
from .audit import audit_broker
from .enums import HoleStatus, RemediationStatus, Timeframe, Verdict
from .fetch import FetchError, MT5FetchAdapter
from .ledger import HoleLedger
from .profile import MarketDataDirNotConfiguredError
from .profiles import PROFILES, get_profile
from .refetch import RefetchConfig
from .remediate import remediate_broker
from .snapshot import history_diff, snapshot_broker
from .state import StateDirNotConfiguredError, StateLayout, resolve_state_dir
from .symbol_data import NotABarSeries

PROG = "raw-data-audit"
_ERRORS = (MarketDataDirNotConfiguredError, StateDirNotConfiguredError, FetchError, FileNotFoundError, FileExistsError, NotABarSeries, KeyError,
           ValueError)
_FETCH_ONLY = ("retry_permanent", "min_hole_minutes", "env_dir")


def _common(p: argparse.ArgumentParser, state: bool, registry: bool) -> None:
    p.add_argument("--broker", required=True, choices=list(PROFILES))
    p.add_argument("--tf", type=int, default=int(Timeframe.M5), choices=[int(t) for t in Timeframe], help="bar minutes")
    p.add_argument("--data-dir", type=Path, help="market-data root containing raw/ (default: $OKMICH_QUANT_MARKET_DATA_DIR)")
    if state:
        p.add_argument("--state-dir", type=Path, help="tool state folder, outside the data tree (default: $OKMICH_QUANT_DATA_STATE_DIR)")
    if registry:
        p.add_argument("--registry", type=Path, help="bad-window registry CSV (curated keep/drop decisions)")
        p.add_argument("--blind-after", type=pd.Timestamp, help="raw stamp after which findings show windows and kinds only")
    p.add_argument("--symbols", nargs="*", help="only these symbols (peer checks then use only these)")
    p.add_argument("--quiet", action="store_true", help="no progress lines on stderr")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=PROG, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="audit a broker's raw bar parquets and report findings (read-only)")
    _common(a, state=False, registry=True)
    a.add_argument("--out", type=Path, help="report folder (default: a per-run folder under the OS temp dir)")
    a.add_argument("--no-write", action="store_true", help="print only, write no files")
    a.add_argument("--json", action="store_true", help="print the JSON summary instead of the console report")
    a.add_argument("--max-rows", type=int, default=40, help="rows per console table")
    r = sub.add_parser("remediate", help="build remediated copies in the state folder (raw/ is never touched)")
    _common(r, state=True, registry=True)
    r.add_argument("--dry-run", action="store_true", help="plan and verify, print the diff, write nothing")
    r.add_argument("--fetch", action="store_true", help="refetch holes, dropped windows and spike suspects from the broker's MT5 terminal")
    r.add_argument("--retry-permanent", action="store_true", help="with --fetch: ask again for targets the ledger marks permanent")
    r.add_argument("--min-hole-minutes", type=int, help="with --fetch: raw-file holes shorter than this are not refetched (default 30)")
    r.add_argument("--env-dir", type=Path, help="with --fetch: folder holding .env.<broker>.<account> (default: $OKMICH_QUANT_ENV_DIR)")
    h = sub.add_parser("holes", help="the hole ledger: refetch targets and their state (read-only)")
    _common(h, state=True, registry=False)
    h.add_argument("--all", action="store_true", help="also list filled and resolved targets")
    s = sub.add_parser("snapshot", help="hash-verified copy of the broker's raw files into the state folder")
    _common(s, state=True, registry=False)
    d = sub.add_parser("diff", help="D12: did the raw files rewrite history since a snapshot?")
    _common(d, state=True, registry=False)
    d.add_argument("--snapshot", help="snapshot folder name (default: the latest)")
    return ap


def _audit(args) -> int:
    rep = audit_broker(args.broker, Timeframe(args.tf), args.symbols, data_dir=args.data_dir, registry_path=args.registry,
                       blind_after=args.blind_after)
    print(json.dumps(rpt.to_json(rep), indent=2, default=str) if args.json else rpt.format_console(rep, args.max_rows))
    if not args.no_write:
        paths = rpt.write(rep, args.out or rpt.default_out_dir(rep))
        print(f"\nreport files: {paths['findings'].parent}", file=sys.stderr)
    return 1 if any(r.verdict is Verdict.REVIEW for r in rep.symbols.values()) else 0


def _remediate(args) -> int:
    given = [f"--{a.replace('_', '-')}" for a in _FETCH_ONLY if getattr(args, a) not in (None, False)]
    if given and not args.fetch:
        raise ValueError(f"{', '.join(given)} only apply with --fetch")
    profile = get_profile(args.broker, args.data_dir)
    adapter, cfg = None, None
    if args.fetch:
        cfg = RefetchConfig(min_hole_minutes=args.min_hole_minutes if args.min_hole_minutes is not None else 30,
                            retry_permanent=args.retry_permanent)
        adapter = MT5FetchAdapter.for_profile(profile, args.env_dir)
        adapter.open()
    try:
        run = remediate_broker(profile=profile, timeframe=Timeframe(args.tf), symbols=args.symbols, state_dir=args.state_dir,
                               registry_path=args.registry, blind_after=args.blind_after, write=not args.dry_run, adapter=adapter, refetch=cfg)
    finally:
        if adapter is not None:
            adapter.close()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(f"REMEDIATION {'DRY RUN ' if args.dry_run else ''}{run.broker} {run.bar_minutes}m  run {run.run_at:%Y-%m-%d %H:%M} UTC  "
          f"registry: {run.registry or 'none'}")
    print(run.summary_frame().to_string(index=False))
    for r in run.symbols.values():
        for h in r.held:
            print(f"  HELD {r.symbol}: {h['kind']} {h['start_raw']} -> {h['end_raw']} ({h['detector']}) {h['evidence']}")
        if r.remaining_critical:
            print(f"  FAILED VERIFY {r.symbol}: {r.remaining_critical}")
    counts = pd.Series([r.status.value for r in run.symbols.values()]).value_counts().to_dict()
    print("status: " + ", ".join(f"{s.value}={counts.get(s.value, 0)}" for s in RemediationStatus))
    aborted = _print_refetch(run)
    if run.written:
        print(f"\nstaging + manifest: {run.run_dir}", file=sys.stderr)
    return 0 if all(r.status is RemediationStatus.VERIFIED for r in run.symbols.values()) and not aborted else 1


def _print_refetch(run, max_rows: int = 40) -> bool:
    """Print the refetch outcome; True if fetching stopped early on terminal errors."""
    rf = run.refetch
    if rf is None:
        return False
    note = "" if run.written else " (dry run: ledger and fill store unchanged)"
    print(f"\nREFETCH via {run.fetch_source}: {len(rf.seen)} targets, {len(rf.skipped)} skipped as permanent, {rf.calls} terminal calls, "
          f"{rf.seconds:.0f}s{note}")
    if rf.aborted:
        print(f"  STOPPED: {rf.aborted}")
    att = run.attempts_frame()
    if att.empty:
        return bool(rf.aborted)
    print(att.groupby(["kind", "result"]).agg(targets=("id", "size"), bars=("filled", "sum"), counted=("counted", "sum")).to_string())
    rest = att[~att["result"].isin(["filled", "replaced"])]
    if len(rest):
        cols = ["symbol", "kind", "cause", "start_raw", "end_raw", "expected", "filled", "result", "attempts", "status", "evidence"]
        print(f"\nnot filled ({len(rest)}; first {min(len(rest), max_rows)}):")
        print(rest[cols].head(max_rows).to_string(index=False))
    return bool(rf.aborted)


def _layout(args) -> tuple:
    profile = get_profile(args.broker, args.data_dir)
    if profile.bar_minutes != args.tf:
        raise ValueError(f"{profile.name} profile is for {profile.bar_minutes}-minute bars, not {args.tf}")
    return profile, StateLayout(resolve_state_dir(args.state_dir, profile.raw_root.parent), profile.name, profile.bar_minutes)


def _holes(args) -> int:
    profile, layout = _layout(args)
    df = HoleLedger.load(layout.ledger).frame()
    if args.symbols:
        df = df[df["symbol"].isin(args.symbols)]
    if df.empty:
        print(f"no hole ledger rows under {layout.base} (run `{PROG} remediate --fetch` first)")
        return 0
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(f"HOLE LEDGER {profile.name} {profile.bar_minutes}m: {len(df)} targets")
    print(pd.crosstab(df["kind"], df["status"]).to_string())
    show = df if args.all else df[df["status"].isin([HoleStatus.OPEN.value, HoleStatus.PARTIAL.value, HoleStatus.PERMANENT.value])]
    cols = ["symbol", "kind", "cause", "start_raw", "end_raw", "expected", "filled", "attempts", "last_attempt", "last_result", "status",
            "evidence"]
    if len(show):
        print(show[cols].to_string(index=False))
    return 0


def _snapshot(args) -> int:
    profile, layout = _layout(args)
    layout.ensure()
    print(f"snapshot: {snapshot_broker(profile, layout, args.symbols)}")
    return 0


def _diff(args) -> int:
    profile, layout = _layout(args)
    snap = layout.snapshots / args.snapshot if args.snapshot else layout.latest_snapshot()
    if snap is None or not (snap / "snapshot.json").is_file():
        raise FileNotFoundError(f"no snapshot under {layout.snapshots} (run `{PROG} snapshot` first)")
    found = history_diff(profile, snap, args.symbols)
    print(f"D12 history diff vs snapshot {snap.name}: {len(found)} rewritten window(s)")
    if found:
        pd.set_option("display.width", 250)
        print(pd.DataFrame([f.as_row() for f in found])[["symbol", "start_raw", "end_raw", "n_bars", "evidence"]].to_string(index=False))
    return 1 if found else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format=f"{PROG}: %(message)s", stream=sys.stderr, force=True)
    if getattr(args, "registry", None) is not None and not args.registry.is_file():
        print(f"{PROG}: --registry {args.registry} is not a file", file=sys.stderr)
        return 2
    try:
        return {"audit": _audit, "remediate": _remediate, "holes": _holes, "snapshot": _snapshot, "diff": _diff}[args.cmd](args)
    except _ERRORS as e:
        print(f"{PROG}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
