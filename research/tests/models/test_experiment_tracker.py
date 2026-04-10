from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from okmich_quant_research.models.experiment_runner import ExperimentRunner
from okmich_quant_research.models.experiment_tracker import (
    ExperimentTracker,
    load_experiment,
)
from okmich_quant_research.models.report_generator import (
    ReportGenerator,
    generate_report,
)


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


def cleanup_experiments_folder(tracker: ExperimentTracker = None):
    experiments_dir = tracker.experiments_root.parent
    if experiments_dir.exists():
        cleanup_directory(experiments_dir)
        return experiments_dir
    return None


def generate_test_data_with_features(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic data with features for testing."""
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

    # Generate synthetic features
    df["rsi"] = (
        50 + 20 * np.sin(np.arange(n_samples) * 0.1) + np.random.randn(n_samples) * 5
    )
    df["atr"] = np.abs(df["high"] - df["low"]).rolling(14).mean().bfill()
    df["momentum"] = df["close"].pct_change(10).fillna(0)
    df["volatility"] = df["return"].rolling(20).std().bfill()
    df["volume_ratio"] = 1.0 + np.random.randn(n_samples) * 0.2

    return df


def run_test_experiment() -> tuple:
    """Run a test experiment and return (result, experiment_id, config_path)."""
    print("=" * 80)
    print("Running test experiment for Phase 4 testing...")
    print("=" * 80)

    # Get project root (4 levels up from this test file)
    project_root = Path(__file__).resolve().parents[0]

    # Create minimal config
    config_dict = {
        "experiment_name": "test_phase4_experiment",
        "research_type": "supervised",
        "data": {"symbol": "TEST", "timeframe": "5min"},
        "feature_engineering": {
            "external_function": {
                "module": f"{__package__}.fixtures",
                "function": "example_momentum_features",
                "params": {"lookback": 20},
            },
            "version": "phase4_test_v1",
            "save_source_code": False,
        },
        "auto_selection": {"enabled": True, "top_n": 10, "vif_threshold": 10.0},
        "eda": {"enabled": False},
        "model": {
            "type": "clustering",
            "algorithms": ["hmm_pmgnt"],
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
            "folder": None,
            "save_models": True,
            "save_labels": True,
            "save_metrics": True,
        },
    }

    # Save config
    config_path = project_root / "test_output" / "test_phase4_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    # Run experiment
    runner = ExperimentRunner.from_yaml(str(config_path))
    df = generate_test_data_with_features(n_samples=1000)
    result = runner.run_with_data(df)

    # Get experiment ID
    experiment_id = Path(result.output_dir).name

    return result, experiment_id, config_path


def test_experiment_tracker():
    """Test experiment tracking functionality."""
    print("\n" + "=" * 80)
    print("TEST: ExperimentTracker")
    print("=" * 80)

    # Run experiment
    print("\n1. Running test experiment...")
    result, experiment_id, config_path = run_test_experiment()
    output_dir = Path(result.output_dir)
    print(f"   [OK] Experiment completed: {experiment_id}")

    # Initialize tracker
    print("\n2. Initializing experiment tracker...")
    tracker = ExperimentTracker()
    print("   [OK] Tracker initialized")

    # Save experiment
    print("\n3. Saving experiment to tracker...")
    saved_id = tracker.save_experiment(
        result,
        tags=["test", "phase4", "algorithms"],
        notes="Test experiment for Phase 4 validation",
    )
    assert saved_id == experiment_id, "Experiment ID mismatch"
    print(f"   [OK] Experiment saved: {saved_id}")

    # List experiments
    print("\n4. Listing experiments...")
    experiments_df = tracker.list_experiments(limit=5)
    print(f"   [OK] Found {len(experiments_df)} experiments")
    print(experiments_df[["experiment_id", "best_model", "best_score"]])

    # Load experiment
    print("\n5. Loading experiment...")
    loaded_result = tracker.load_experiment(experiment_id)
    print(f"   [OK] Loaded experiment: {loaded_result.experiment_name}")
    best_model = loaded_result.get_best_model()
    if best_model:
        print(f"   Best model: {best_model.model_name}")
    else:
        print(f"   Best model: None")
    print(f"   Models loaded: {len(loaded_result.trained_models)}")

    # Verify loaded data
    print("\n6. Verifying loaded data...")
    assert loaded_result.experiment_name == result.experiment_name
    assert len(loaded_result.trained_models) == len(result.trained_models)
    assert len(loaded_result.rankings) == len(result.rankings)

    # Compare best models (handle None case)
    loaded_best = loaded_result.get_best_model()
    result_best = result.get_best_model()
    if loaded_best is not None and result_best is not None:
        assert loaded_best.model_name == result_best.model_name
    else:
        assert loaded_best == result_best  # Both should be None
    print("   [OK] All data verified")

    # Test comparison
    print("\n7. Testing experiment comparison...")
    comparison_df = tracker.compare_experiments([experiment_id])
    print(f"   [OK] Comparison generated: {len(comparison_df)} experiments")
    print(comparison_df[["name", "best_model", "best_score"]])

    # Test convenience function
    print("\n8. Testing load_experiment convenience function...")
    loaded_result2 = load_experiment(experiment_id)
    assert loaded_result2.experiment_name == result.experiment_name
    print(f"   [OK] Convenience function works")

    print("\n[OK] ExperimentTracker test PASSED")

    # Cleanup test artifacts
    print("\n9. Cleaning up test artifacts...")
    try:
        if output_dir and output_dir.exists():
            cleanup_directory(output_dir)
            print(f"   [OK] Removed output directory: {output_dir}")

        if config_path and config_path.exists():
            cleanup_directory(config_path.parent)
            print(f"   [OK] Removed config file: {config_path.parent}")

        # Also cleanup the experiments folder created by ExperimentTracker
        experiments_dir = cleanup_experiments_folder(tracker)
        if experiments_dir:
            print(f"   [OK] Removed experiments directory: {experiments_dir}")
    except Exception as e:
        print(f"   [WARN] Cleanup failed: {e}")


@pytest.fixture(scope="module")
def result():
    """Fixture that provides a test experiment result."""
    result, experiment_id, config_path = run_test_experiment()
    yield result

    # Cleanup after all tests using this fixture
    try:
        output_dir = Path(result.output_dir)
        if output_dir.exists():
            cleanup_directory(output_dir)
        if config_path.exists():
            cleanup_directory(config_path.parent)
        cleanup_experiments_folder()
    except Exception:
        pass


def test_report_generator(result):
    """Test report generation."""
    print("\n" + "=" * 80)
    print("TEST: ReportGenerator")
    print("=" * 80)

    # Initialize generator
    print("\n1. Initializing report generator...")
    generator = ReportGenerator()
    print("   [OK] Generator initialized")

    # Generate report (default location)
    print("\n2. Generating HTML report (default location)...")
    report_path = generator.generate_report(result)
    print(f"   [OK] Report generated: {report_path}")

    # Verify report exists
    print("\n3. Verifying report file...")
    report_file = Path(report_path)
    assert report_file.exists(), "Report file not found"
    print(f"   [OK] Report file exists: {report_file.stat().st_size} bytes")

    # Read and check content
    print("\n4. Checking report content...")
    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert result.experiment_name in content, "Experiment name not in report"
    best_model = result.get_best_model()
    if best_model is not None:
        assert best_model.model_name in content, "Best model name not in report"
        assert "Model Rankings" in content, "Rankings section missing"
        assert "Regime Statistics" in content, "Regime stats section missing"
    else:
        print("   [WARN] No best model available (no models trained)")
    print("   [OK] Report content validated")

    # Test convenience function
    print("\n5. Testing generate_report convenience function...")
    report_path2 = generate_report(
        result, output_path=str(Path(result.output_dir) / "report2.html")
    )
    assert Path(report_path2).exists()
    print(f"   [OK] Convenience function works: {report_path2}")

    print("\n[OK] ReportGenerator test PASSED")

    # Cleanup report files
    print("\n6. Cleaning up report artifacts...")
    try:
        if Path(report_path).exists():
            Path(report_path).unlink()
            print(f"   [OK] Removed report: {report_path}")
        if Path(report_path2).exists():
            Path(report_path2).unlink()
            print(f"   [OK] Removed report2: {report_path2}")
    except Exception as e:
        print(f"   [WARN] Cleanup failed: {e}")


def test_integration(result):
    """Test integration of tracker + report generator."""
    print("\n" + "=" * 80)
    print("TEST: Integration (Tracker + Report)")
    print("=" * 80)

    # Initialize tracker
    print("\n1. Initializing tracker...")
    tracker = ExperimentTracker()

    # Save the experiment from fixture
    experiment_id = Path(result.output_dir).name
    tracker.save_experiment(result, tags=["integration-test"])
    print(f"   [OK] Saved experiment: {experiment_id}")

    # Load experiment
    print("\n2. Loading experiment...")
    loaded_result = tracker.load_experiment(experiment_id)
    print(f"   [OK] Loaded: {loaded_result.experiment_name}")

    # Generate report for loaded experiment
    print("\n3. Generating report for loaded experiment...")
    report_path = generate_report(loaded_result)
    print(f"   [OK] Report generated: {report_path}")

    # Verify report
    assert Path(report_path).exists()
    print("   [OK] Report verified")

    print("\n[OK] Integration test PASSED")

    # Cleanup report
    print("\n4. Cleaning up report artifact...")
    try:
        if Path(report_path).exists():
            Path(report_path).unlink()
            print(f"   [OK] Removed report: {report_path}")

        # Cleanup experiments folder
        experiments_dir = cleanup_experiments_folder(tracker)
        if experiments_dir:
            print(f"   [OK] Removed experiments directory: {experiments_dir}")
    except Exception as e:
        print(f"   [WARN] Cleanup failed: {e}")
