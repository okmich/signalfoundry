"""
CLI entry point for running research experiments.

Installed as the `run-experiment` command (see pyproject.toml).
Also runnable as a module: python -m okmich_quant_research.cli.run_experiment

Usage:
------
    # Run a config by path (absolute or relative to CWD)
    run-experiment path/to/config.yaml

    # Synthetic data (for quick testing)
    run-experiment path/to/config.yaml --synthetic --samples 5000

    # Fast mode (smaller walk-forward windows)
    run-experiment path/to/config.yaml --fast

    # Load data from a specific parquet file
    run-experiment path/to/config.yaml --data-path /data/US500.parquet

    # List all yaml files in a directory (default: CWD)
    run-experiment --list
    run-experiment --list --config-dir playground/research/models_research_configs

Examples:
---------
    run-experiment playground/research/models_research_configs/smoke_test_sklearn.yaml
    run-experiment playground/research/models_research_configs/smoke_test_sklearn.yaml --synthetic
    run-experiment playground/research/models_research_configs/clustering_trend.yaml --fast
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from okmich_quant_research.models.experiment_runner import ExperimentRunner
from okmich_quant_research.models.utils.config_parser import ConfigParser


def generate_synthetic_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="5min")

    returns = np.random.randn(n_samples) * 0.0005 + 0.00001
    price = 5000 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": price + np.random.randn(n_samples) * 0.5,
            "high": price + np.abs(np.random.randn(n_samples)) * 1.0,
            "low": price - np.abs(np.random.randn(n_samples)) * 1.0,
            "close": price,
            "volume": np.random.randint(1000, 50000, n_samples),
        },
        index=dates,
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def list_configs(config_dir: Path):
    """List YAML config files in the given directory."""
    yaml_files = sorted(config_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"No YAML configs found in: {config_dir}")
        return

    print(f"\nConfigs in: {config_dir}")
    print("-" * 60)
    for yaml_file in yaml_files:
        try:
            config = ConfigParser(str(yaml_file))
            model_type = config.get_model_type()
            research_type = config.get_research_type()
            print(f"  {yaml_file.name}")
            print(f"      type: {model_type}  |  research: {research_type}")
        except Exception as e:
            print(f"  {yaml_file.name}  (parse error: {e})")
        print()


def run_experiment(config_path: Path, data_path: str = None, synthetic: bool = False, n_samples: int = 10000,
                   override_walk_forward: dict = None):
    """Run a single experiment from a YAML config."""

    if not config_path.exists():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)

    print("=" * 80)
    print(f"Config: {config_path}")
    print("=" * 80)

    config = ConfigParser(str(config_path))
    config_dict = config.to_dict()

    if override_walk_forward:
        config_dict.setdefault("walk_forward", {}).update(override_walk_forward)

    # Disable EDA for non-interactive CLI runs
    config_dict.get("feature_engineering", {}).get("eda", {}).pop("enabled", None)
    if "eda" in config_dict.get("feature_engineering", {}):
        config_dict["feature_engineering"]["eda"]["enabled"] = False

    config = ConfigParser(config_dict=config_dict)

    # Load data
    if synthetic:
        print(f"\nUsing synthetic data ({n_samples} samples)")
        df = generate_synthetic_data(n_samples)
    elif data_path:
        print(f"\nLoading data from: {data_path}")
        df = pd.read_parquet(data_path)
        max_samples = config.get_max_samples()
        if max_samples:
            df = df[-max_samples:]
    else:
        print("\nLoading data from config...")
        data_cfg = config.get_data_config()
        source_folder = data_cfg.get("source_folder")
        symbol = config.get_symbol()
        timeframe = config.get_timeframe()
        max_samples = config.get_max_samples()

        parquet_path = Path(source_folder) / timeframe / f"{symbol}.parquet"
        if not parquet_path.exists():
            print(f"Error: data file not found: {parquet_path}")
            print("Tip: use --synthetic or --data-path to supply data directly.")
            sys.exit(1)

        df = pd.read_parquet(parquet_path)
        if max_samples:
            df = df[-max_samples:]

    if df.empty:
        print("Error: dataset is empty after loading/slicing. Check the source file and --max-samples.")
        sys.exit(1)

    print(f"Data: {df.shape[0]:,} rows  |  {df.index[0]} -> {df.index[-1]}")

    runner = ExperimentRunner(config=config)
    result = runner.run_with_data(df)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Models trained : {len(result.trained_models)}")
    print(f"Features used  : {len(result.selected_features)}")
    if result.rankings:
        print("\nRankings:")
        for r in result.rankings[:5]:
            print(f"  {r.rank}. {r.model_name}: {r.composite_score:.4f}")
    print(f"\nOutput: {result.output_dir}")

    return result


def main():
    parser = argparse.ArgumentParser(
        prog="run-experiment",
        description="Run a research experiment from a YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "config",
        nargs="?",
        help="Path to YAML config file (absolute or relative to CWD)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List YAML configs in --config-dir (default: CWD)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Directory to list configs from (used with --list)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to a parquet file (overrides config data source)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic OHLCV data instead of loading from disk",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of synthetic samples (default: 10000)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Override walk-forward to small windows for quick testing",
    )

    args = parser.parse_args()

    if args.list:
        config_dir = Path(args.config_dir) if args.config_dir else Path.cwd()
        list_configs(config_dir)
        return

    if not args.config:
        parser.print_help()
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    override_wf = None
    if args.fast:
        override_wf = {"train_period": 2000, "test_period": 400, "step_period": 400}

    run_experiment(config_path=config_path, data_path=args.data_path,
                   synthetic=args.synthetic, n_samples=args.samples,
                   override_walk_forward=override_wf)


if __name__ == "__main__":
    main()
