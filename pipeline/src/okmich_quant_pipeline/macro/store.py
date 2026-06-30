r"""build-macro-features — materialize the macro feature store.

Reads the raw per-series macro store (``reader.load_macro``), computes the conditioning features
(``features.compute_macro_features``), and persists the long feature frame to a feature-store
directory as a single parquet — written atomically (temp + fsync + replace). A coverage / gap /
staleness report (``report.py``) is written alongside it.

Features-only by design: the no-lookahead broadcast onto intraday bars stays a reversible
post-step at consume time (``attach.attach_macro_to_dataset``), so the macro-vs-no-macro ablation
remains a ``± columns`` toggle rather than being baked into the store.

    build-macro-features                                  # default macro store -> default feature store
    build-macro-features --macro-store <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

from okmich_quant_pipeline.macro._io import atomic_write_parquet
from okmich_quant_pipeline.macro.features import DEFAULT_RECIPES, FeatureRecipe, compute_macro_features
from okmich_quant_pipeline.macro.reader import load_macro
from okmich_quant_pipeline.macro.report import MacroReport, build_report, write_report
from okmich_quant_pipeline.macro.update import DEFAULT_STORE

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_STORE = r"E:\data_dump\feature_data\macro"
FEATURES_FILENAME = "macro_features.parquet"


def build_feature_store(macro_store_dir: Path | str, out_dir: Path | str, *,
                        recipes: tuple[FeatureRecipe, ...] = DEFAULT_RECIPES,
                        asof: dt.date | None = None) -> MacroReport:
    """Materialize the macro feature store and its coverage report.

    Loads the raw per-series store once, engineers features, writes the long feature frame to
    ``{out_dir}/macro_features.parquet`` atomically, and writes the coverage/gap/staleness report
    next to it. Returns the report (the parquet lands at ``{out_dir}/macro_features.parquet``).

    ``asof`` is the reference "today" for staleness (defaults to ``dt.date.today()`` at call
    time); pass it explicitly in tests for determinism.
    """
    out_dir = Path(out_dir)
    raw = load_macro(macro_store_dir)
    features = compute_macro_features(raw, recipes)

    out_path = out_dir / FEATURES_FILENAME
    atomic_write_parquet(features, out_path)

    report = build_report(raw, features, asof=asof or dt.date.today())
    write_report(report, out_dir)

    logger.info(f"Wrote {len(features)} feature rows ({features['feature'].nunique()} features) -> {out_path}")
    for line in report.summary_lines():
        logger.info(f"  {line}")
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    parser = argparse.ArgumentParser(description="Materialize the macro feature store (features parquet + coverage report).")
    parser.add_argument("--macro-store", default=DEFAULT_STORE, help=f"Raw macro store directory (default: {DEFAULT_STORE})")
    parser.add_argument("--out", default=DEFAULT_FEATURE_STORE, help=f"Feature store output directory (default: {DEFAULT_FEATURE_STORE})")
    args = parser.parse_args()

    report = build_feature_store(args.macro_store, args.out)
    sys.exit(1 if report.has_stale else 0)


if __name__ == "__main__":
    main()
