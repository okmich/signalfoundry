from pathlib import Path

import numpy as np
import pandas as pd

from okmich_quant_research.models.utils import ConfigParser
from okmich_quant_research.models.utils.feature_logging import (
    FeatureFunctionLogger,
    load_feature_function,
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


def test_feature_logging():
    """Test feature function logging."""
    print("=" * 80)
    print("TEST: Feature Function Logging")
    print("=" * 80)

    # Get project root (4 levels up from this test file)
    project_root = Path(__file__).resolve().parents[0]

    # Initialize logger
    logger = FeatureFunctionLogger()

    # Log external function
    print("\n1. Logging external feature function...")
    metadata = logger.log_function(
        module=f"{__package__}.fixtures",
        function="combined_features_v1",
        params={"lookback": 20, "include_volume": True},
        version="combined_v1",
        save_source=True,
    )

    print(f"   [OK] Module: {metadata['module']}")
    print(f"   [OK] Function: {metadata['function']}")
    print(f"   [OK] Version: {metadata['version']}")
    print(f"   [OK] Timestamp: {metadata['timestamp']}")
    print(f"   [OK] Code Hash: {metadata.get('code_hash', 'N/A')}")

    # Save metadata
    print("\n2. Saving metadata...")
    test_dir = project_root / "test_output"
    test_dir.mkdir(exist_ok=True)

    metadata_path = test_dir / "feature_metadata.json"
    logger.save_metadata(str(metadata_path))
    print(f"   [OK] Saved to: {metadata_path}")

    # Save source code
    print("\n3. Saving source code snapshot...")
    source_path = test_dir / "feature_engineering_code.py"
    logger.save_source_code(str(source_path))
    print(f"   [OK] Saved to: {source_path}")

    # Load function back
    print("\n4. Loading function from metadata...")
    loaded_func = load_feature_function(str(metadata_path))
    print(f"   [OK] Loaded function: {loaded_func.__name__}")

    # Test function execution
    print("\n5. Testing function execution...")
    test_df = pd.DataFrame(
        {
            "open": np.random.randn(100).cumsum() + 100,
            "high": np.random.randn(100).cumsum() + 101,
            "low": np.random.randn(100).cumsum() + 99,
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100),
        }
    )

    features = loaded_func(test_df, lookback=20, include_volume=True)
    print(f"   [OK] Generated {len(features.columns)} features")
    print(f"   [OK] Feature names: {list(features.columns[:5])}...")

    print("\n[OK] Feature logging test PASSED")

    # Cleanup test artifacts
    print("\n6. Cleaning up test artifacts...")
    try:
        if test_dir.exists():
            cleanup_directory(test_dir)
            print(f"   [OK] Removed test directory: {test_dir}")
    except Exception as e:
        print(f"   [WARN] Cleanup failed: {e}")


def test_config_parser():
    """Test config parsing."""
    print("\n" + "=" * 80)
    print("TEST: Config Parser")
    print("=" * 80)

    # Get project root (4 levels up from this test file)
    project_root = Path(__file__).resolve().parents[0]

    # Test each config file playground/
    config_dir = project_root / "configs"
    config_files = [
        "hmm_directional.yaml",
        "clustering_volatility.yaml",
        "path_structure.yaml",
    ]

    for config_file in config_files:
        print(f"\n{config_file}:")
        print("-" * 80)

        config_path = config_dir / config_file
        parser = ConfigParser(str(config_path))

        # Test basic getters
        print(f"   Experiment name: {parser.get_experiment_name()}")
        print(f"   Research type: {parser.get_research_type()}")
        print(f"   Symbol: {parser.get_symbol()}")
        print(f"   Timeframe: {parser.get_timeframe()}")
        print(f"   Model type: {parser.get_model_type()}")

        # Test feature engineering config
        fe_config = parser.get_feature_engineering_config()
        if "external_function" in fe_config:
            ext_func = parser.get_external_function_config()
            print(f"   Feature function: {ext_func['module']}.{ext_func['function']}")
            print(f"   Feature params: {ext_func['params']}")

        print(f"   Feature version: {parser.get_feature_version()}")
        print(f"   Save source code: {parser.should_save_source_code()}")

        # Test auto selection
        if parser.is_auto_selection_enabled():
            print(f"   Auto selection: enabled")
            print(f"     - Top N: {parser.get_top_n_features()}")
            print(f"     - VIF threshold: {parser.get_vif_threshold()}")

        # Test EDA
        if parser.is_eda_enabled():
            print(f"   EDA: enabled")
            print(f"     - N top features: {parser.get_eda_n_top_features()}")

        # Test objectives
        objectives = parser.get_primary_objectives()
        print(f"   Primary objectives: {len(objectives)}")
        for obj in objectives[:2]:  # Show first 2
            print(f"     - {obj['name']}: weight={obj.get('weight', 'N/A')}")

        # Test evaluation
        eval_funcs = parser.get_label_eval_functions()
        print(f"   Eval functions: {len(eval_funcs)}")

        # Test unsupervised mapping
        if parser.is_unsupervised_label_mapping_enabled():
            print(f"   Unsupervised mapping: enabled")
            mapping_params = parser.get_unsupervised_mapping_params()
            print(f"     - Params: {mapping_params}")

        # Test output config
        print(f"   Output folder: {parser.get_output_folder()}")
        print(f"   Save models: {parser.should_save_models()}")
        print(f"   Generate report: {parser.should_generate_report()}")

        # Test constraints
        constraints = parser.get_constraints()
        if constraints:
            print(f"   Constraints: {len(constraints)} defined")
            print(
                f"     - Min observations: {parser.get_min_observations_per_regime()}"
            )

        print(f"   [OK] Config parsed successfully")

    print("\n[OK] Config parser test PASSED")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RESEARCH UTILITIES TEST SUITE")
    print("=" * 80)

    results = []

    try:
        results.append(("Feature Logging", test_feature_logging()))
    except Exception as e:
        print(f"\n[FAIL] Feature logging test FAILED: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Feature Logging", False))

    try:
        results.append(("Config Parser", test_config_parser()))
    except Exception as e:
        print(f"\n[FAIL] Config parser test FAILED: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Config Parser", False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)
    print("\n" + ("=" * 80))
    if all_passed:
        print("ALL TESTS PASSED [OK]")
    else:
        print("SOME TESTS FAILED [FAIL]")
    print("=" * 80)

    return all_passed
