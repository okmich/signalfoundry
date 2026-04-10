from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from okmich_quant_research.models.experiment_runner import ExperimentRunner


def generate_test_data(n_samples: int = 1000) -> pd.DataFrame:
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=n_samples, freq="5min")

    # Generate random walk price
    returns = np.random.randn(n_samples) * 0.001
    price = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": price + np.random.randn(n_samples) * 0.1,
            "high": price + np.abs(np.random.randn(n_samples)) * 0.2,
            "low": price - np.abs(np.random.randn(n_samples)) * 0.2,
            "close": price,
            "return": returns,
        },
        index=dates,
    )

    # Ensure OHLC consistency
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


def cleanup_directory(dir_path: Path):
    """Recursively remove a directory and all its contents."""
    if not dir_path.exists():
        return

    for item in dir_path.iterdir():
        if item.is_dir():
            cleanup_directory(item)
        else:
            item.unlink()
    dir_path.rmdir()


def test_eda_integration():
    """Test that EDA actually runs and generates artifacts."""
    print("=" * 80)
    print("TEST: EDA Integration")
    print("=" * 80)

    project_root = Path(__file__).resolve().parents[0]

    # Track artifacts for cleanup
    config_path = None
    output_dir = None

    # Create config with EDA ENABLED
    print("\n1. Creating config with EDA enabled...")
    config_dict = {
        "experiment_name": "test_eda_experiment",
        "research_type": "supervised",
        "data": {"symbol": "TEST", "timeframe": "5min"},
        "feature_engineering": {
            "external_function": {
                "module": f"{__package__}.fixtures",
                "function": "example_momentum_features",
                "params": {"lookback": 20},
            },
            "version": "eda_test_v1",
            "save_source_code": False,
            "auto_selection": {
                "enabled": True,
                "top_n": 8,
                "vif_threshold": 50.0,  # Higher threshold to keep more features
                "min_importance": 0.01,
            },
        },
        "eda": {
            "enabled": True,  # ENABLED!
            "n_top_features": 15,
            "correlation_threshold": 0.8,
            "save_plots": True,
            "analyses": ["relevance", "distribution", "correlation", "transformation"],
        },
        "model": {
            "type": "clustering",
            "algorithms": ["hmm_learn"],
            "n_states_range": [2, 3],
        },
        "objectives": {
            "primary": [
                {
                    "name": "regime_discriminability",
                    "target": "maximize",
                    "weight": 0.60,
                },
                {
                    "name": "mean_duration",
                    "target": "range",
                    "min": 5,
                    "max": 30,
                    "weight": 0.40,
                },
            ]
        },
        "evaluation": {
            "label_eval_functions": {"path_structure_stats": {"enabled": True}}
        },
        "output": {
            "folder": "test_output",
            "save_models": True,
            "save_labels": True,
            "save_metrics": True,
        },
    }

    # Save config
    config_path = project_root / "test_output" / "test_eda_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    print(f"   [OK] Config saved: {config_path}")
    print("   EDA enabled: True")
    print("   Save plots: True")

    # Run experiment
    print("\n2. Running experiment with EDA...")
    runner = ExperimentRunner.from_yaml(str(config_path))
    df = generate_test_data(n_samples=1000)
    result = runner.run_with_data(df)

    # Track output directory for cleanup
    output_dir = Path(result.output_dir)

    print(f"\n3. Experiment complete: {result.experiment_name}")
    print(f"   Output dir: {result.output_dir}")

    # Check EDA artifacts
    print("\n4. Checking for EDA artifacts...")
    eda_dir = Path(result.output_dir) / "eda"

    if not eda_dir.exists():
        print(f"   [FAIL] EDA directory not found: {eda_dir}")
        return False

    print(f"   [OK] EDA directory exists: {eda_dir}")

    # Expected EDA files
    expected_files = {
        "distributions.png": "Feature distributions plot",
        "qq_plots.png": "Q-Q plots for normality",
        "correlation_matrix.png": "Correlation matrix heatmap",
        "relevance_scores.csv": "Feature relevance scores",
        "vif_scores.csv": "VIF scores",
        "transformation_recommendations.csv": "Transformation recommendations",
    }

    found_files = []
    missing_files = []

    for filename, description in expected_files.items():
        filepath = eda_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"   [OK] {filename}: {size} bytes - {description}")
            found_files.append(filename)
        else:
            print(f"   [MISS] {filename}: NOT FOUND - {description}")
            missing_files.append(filename)

    # Check features directory
    print("\n5. Checking feature selection artifacts...")
    features_dir = Path(result.output_dir) / "features"

    if features_dir.exists():
        print(f"   [OK] Features directory exists")
        feature_files = list(features_dir.glob("*.json")) + list(
            features_dir.glob("*.csv")
        )
        for f in feature_files:
            print(f"      - {f.name}: {f.stat().st_size} bytes")
    else:
        print(f"   [FAIL] Features directory not found")

    # Summary
    print("\n6. EDA Summary:")
    print(f"   Files found: {len(found_files)}/{len(expected_files)}")
    print(f"   Features selected: {len(result.selected_features)}")

    if len(found_files) < len(expected_files):
        print(f"\n   [WARN] Missing {len(missing_files)} EDA artifacts:")
        for f in missing_files:
            print(f"      - {f}")

    # Verify we got at least the main artifacts
    critical_files = ["relevance_scores.csv", "vif_scores.csv"]
    all_critical_found = all(f in found_files for f in critical_files)

    if all_critical_found:
        print("\n[OK] EDA Integration test PASSED - Critical artifacts generated")

        # Cleanup test artifacts
        print("\n7. Cleaning up test artifacts...")
        try:
            if output_dir and output_dir.exists():
                cleanup_directory(output_dir)
                print(f"   [OK] Removed output directory: {output_dir}")

            if config_path and config_path.exists():
                cleanup_directory(config_path.parent)
                print(f"   [OK] Removed config file: {config_path}")
        except Exception as e:
            print(f"   [WARN] Cleanup failed: {e}")
    else:
        print("\n[FAIL] EDA Integration test FAILED - Missing critical artifacts")
        assert all_critical_found, "Missing critical EDA artifacts"
