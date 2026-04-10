from typing import Dict, Any

import numpy as np
import pandas as pd


def compare_trading_data(
    df_demo: pd.DataFrame,
    df_live: pd.DataFrame,
    date_from: str,
    date_to: str,
    tolerance: float = 1e-6,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    spread_col: str = "spread",
    volume_col: str = "tick_volume",
    timestamp_col: str = "time",
) -> Dict[str, Any]:
    """
    Compare demo and live trading data to ensure data quality consistency.

    Parameters:
    -----------
    df_demo : pd.DataFrame
        Demo account trading data
    df_live : pd.DataFrame
        Live account trading data
    date_from : str
        Start date for comparison (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
    date_to : str
        End date for comparison (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
    timestamp_col : str
        Name of the timestamp column (default: 'timestamp')
    tolerance : float
        Numerical tolerance for floating point comparisons (default: 1e-6)

    Returns:
    --------
    Dict containing comparison results and any discrepancies found
    """

    results = {"passed": True, "issues": [], "summary": {}, "details": {}}

    # Convert date strings to datetime
    date_from_dt = pd.to_datetime(date_from)
    date_to_dt = pd.to_datetime(date_to)

    # Create copies and ensure timestamp column is datetime
    df_demo = df_demo.copy()
    df_live = df_live.copy()

    if timestamp_col not in df_demo.columns or timestamp_col not in df_live.columns:
        results["passed"] = False
        results["issues"].append(
            f"Timestamp column '{timestamp_col}' not found in one or both dataframes"
        )
        return results

    df_demo[timestamp_col] = pd.to_datetime(df_demo[timestamp_col])
    df_live[timestamp_col] = pd.to_datetime(df_live[timestamp_col])

    # Filter by date range
    df_demo_filtered = (
        df_demo[
            (df_demo[timestamp_col] >= date_from_dt)
            & (df_demo[timestamp_col] <= date_to_dt)
        ]
        .sort_values(timestamp_col)
        .reset_index(drop=True)
    )

    df_live_filtered = (
        df_live[
            (df_live[timestamp_col] >= date_from_dt)
            & (df_live[timestamp_col] <= date_to_dt)
        ]
        .sort_values(timestamp_col)
        .reset_index(drop=True)
    )

    # 1. Check number of records
    n_demo = len(df_demo_filtered)
    n_live = len(df_live_filtered)
    results["summary"]["demo_records"] = n_demo
    results["summary"]["live_records"] = n_live

    if n_demo != n_live:
        results["passed"] = False
        results["issues"].append(
            f"Record count mismatch: Demo={n_demo}, Live={n_live}, Difference={abs(n_demo - n_live)}"
        )

    # 2. Check if both dataframes are empty
    if n_demo == 0 and n_live == 0:
        results["issues"].append("Both dataframes are empty for the given date range")
        return results
    elif n_demo == 0 or n_live == 0:
        results["passed"] = False
        results["issues"].append(
            f"One dataframe is empty: Demo={n_demo} records, Live={n_live} records"
        )
        return results

    # 3. Check timestamp alignment
    timestamp_match = df_demo_filtered[timestamp_col].equals(
        df_live_filtered[timestamp_col]
    )
    results["summary"]["timestamps_match"] = timestamp_match

    if not timestamp_match:
        results["passed"] = False
        # Find missing timestamps
        demo_ts = set(df_demo_filtered[timestamp_col])
        live_ts = set(df_live_filtered[timestamp_col])

        missing_in_live = demo_ts - live_ts
        missing_in_demo = live_ts - demo_ts

        if missing_in_live:
            results["issues"].append(
                f"Timestamps in demo but not in live: {len(missing_in_live)} records"
            )
            results["details"]["missing_in_live"] = sorted(list(missing_in_live))[
                :10
            ]  # First 10

        if missing_in_demo:
            results["issues"].append(
                f"Timestamps in live but not in demo: {len(missing_in_demo)} records"
            )
            results["details"]["missing_in_demo"] = sorted(list(missing_in_demo))[
                :10
            ]  # First 10

    # 4. Check for required trading columns
    required_cols = [open_col, high_col, low_col, close_col, spread_col, volume_col]
    demo_cols = [col.lower() for col in df_demo_filtered.columns]
    live_cols = [col.lower() for col in df_live_filtered.columns]

    # Create column mapping (case-insensitive)
    demo_col_map = {col.lower(): col for col in df_demo_filtered.columns}
    live_col_map = {col.lower(): col for col in df_live_filtered.columns}

    missing_demo = [col for col in required_cols if col not in demo_cols]
    missing_live = [col for col in required_cols if col not in live_cols]

    if missing_demo or missing_live:
        results["passed"] = False
        if missing_demo:
            results["issues"].append(f"Missing columns in demo: {missing_demo}")
        if missing_live:
            results["issues"].append(f"Missing columns in live: {missing_live}")

    # 5. Compare price and volume data (for common columns)
    common_cols = [
        col for col in required_cols if col in demo_cols and col in live_cols
    ]

    for col in common_cols:
        demo_col_name = demo_col_map[col]
        live_col_name = live_col_map[col]

        col_results = {
            "identical": False,
            "max_diff": None,
            "mean_diff": None,
            "mismatches": 0,
        }

        # Only compare rows with matching timestamps
        if timestamp_match:
            demo_values = df_demo_filtered[demo_col_name].values
            live_values = df_live_filtered[demo_col_name].values

            # Check for exact match first
            if np.array_equal(demo_values, live_values):
                col_results["identical"] = True
            else:
                # Calculate differences
                diff = np.abs(demo_values - live_values)
                col_results["max_diff"] = float(np.max(diff))
                col_results["mean_diff"] = float(np.mean(diff))
                col_results["mismatches"] = int(np.sum(diff > tolerance))

                if col_results["mismatches"] > 0:
                    results["passed"] = False
                    results["issues"].append(
                        f"{col.upper()} mismatch: {col_results['mismatches']} records differ "
                        f"(max diff: {col_results['max_diff']:.6f}, mean diff: {col_results['mean_diff']:.6f})"
                    )

                    # Store sample of mismatches
                    mismatch_idx = np.where(diff > tolerance)[0][:5]  # First 5
                    if len(mismatch_idx) > 0:
                        col_results["sample_mismatches"] = []
                        for idx in mismatch_idx:
                            col_results["sample_mismatches"].append(
                                {
                                    "timestamp": str(
                                        df_demo_filtered.iloc[idx][timestamp_col]
                                    ),
                                    "demo": float(demo_values[idx]),
                                    "live": float(live_values[idx]),
                                    "diff": float(diff[idx]),
                                }
                            )
        else:
            # If timestamps don't match, merge and compare
            demo_subset = df_demo_filtered[[timestamp_col, demo_col_name]].copy()
            live_subset = df_live_filtered[[timestamp_col, live_col_name]].copy()

            merged = pd.merge(
                demo_subset,
                live_subset,
                on=timestamp_col,
                how="inner",
                suffixes=("_demo", "_live"),
            )

            if len(merged) > 0:
                demo_vals = merged[f"{demo_col_name}_demo"].values
                live_vals = merged[f"{live_col_name}_live"].values

                diff = np.abs(demo_vals - live_vals)
                col_results["compared_records"] = len(merged)
                col_results["max_diff"] = float(np.max(diff))
                col_results["mean_diff"] = float(np.mean(diff))
                col_results["mismatches"] = int(np.sum(diff > tolerance))

                if col_results["mismatches"] > 0:
                    results["passed"] = False
                    results["issues"].append(
                        f"{col.upper()} mismatch in {len(merged)} common records: "
                        f"{col_results['mismatches']} differ (max: {col_results['max_diff']:.6f})"
                    )

        results["details"][col] = col_results

    # 6. Check OHLC consistency within each dataset
    ohlc_cols = ["open", "high", "low", "close"]
    if all(col in common_cols for col in ohlc_cols):
        for df_name, df in [("demo", df_demo_filtered), ("live", df_live_filtered)]:
            o_col = demo_col_map["open"] if df_name == "demo" else live_col_map["open"]
            h_col = demo_col_map["high"] if df_name == "demo" else live_col_map["high"]
            l_col = demo_col_map["low"] if df_name == "demo" else live_col_map["low"]
            c_col = (
                demo_col_map["close"] if df_name == "demo" else live_col_map["close"]
            )

            # High should be >= Open, Low, Close
            invalid_high = (
                (df[h_col] < df[o_col])
                | (df[h_col] < df[l_col])
                | (df[h_col] < df[c_col])
            )
            # Low should be <= Open, High, Close
            invalid_low = (
                (df[l_col] > df[o_col])
                | (df[l_col] > df[h_col])
                | (df[l_col] > df[c_col])
            )

            n_invalid = invalid_high.sum() + invalid_low.sum()
            if n_invalid > 0:
                results["passed"] = False
                results["issues"].append(
                    f"OHLC consistency violation in {df_name}: {n_invalid} records with invalid high/low"
                )

    # 7. Check for missing/null values
    for df_name, df in [("demo", df_demo_filtered), ("live", df_live_filtered)]:
        for col in common_cols:
            col_name = demo_col_map[col] if df_name == "demo" else live_col_map[col]
            null_count = df[col_name].isnull().sum()
            if null_count > 0:
                results["passed"] = False
                results["issues"].append(
                    f"Null values in {df_name} {col}: {null_count} records"
                )

    # 8. Check for duplicate timestamps
    for df_name, df in [("demo", df_demo_filtered), ("live", df_live_filtered)]:
        dup_count = df[timestamp_col].duplicated().sum()
        if dup_count > 0:
            results["passed"] = False
            results["issues"].append(
                f"Duplicate timestamps in {df_name}: {dup_count} duplicates"
            )

    # 9. Check data continuity (gaps in timestamps)
    for df_name, df in [("demo", df_demo_filtered), ("live", df_live_filtered)]:
        if len(df) > 1:
            time_diffs = df[timestamp_col].diff().dropna()
            if len(time_diffs) > 0:
                expected_interval = (
                    time_diffs.mode()[0]
                    if len(time_diffs.mode()) > 0
                    else time_diffs.median()
                )
                gaps = time_diffs[time_diffs > expected_interval * 1.5]
                if len(gaps) > 0:
                    results["issues"].append(
                        f"Data gaps detected in {df_name}: {len(gaps)} gaps larger than expected"
                    )
                    results["details"][f"{df_name}_gaps"] = int(len(gaps))

    # Final summary
    if results["passed"]:
        results["summary"][
            "message"
        ] = "✓ All checks passed - data quality is identical"
    else:
        results["summary"][
            "message"
        ] = f"✗ Data quality issues found - {len(results['issues'])} issues detected"

    return results


def print_comparison_report(results: Dict[str, Any]) -> None:
    """
    Print a formatted report of the comparison results.

    Parameters:
    -----------
    results : Dict
        Results dictionary from compare_trading_data function
    """
    print("\n" + "=" * 70)
    print("TRADING DATA QUALITY COMPARISON REPORT")
    print("=" * 70)

    print(f"\nStatus: {results['summary']['message']}")
    print(f"\nRecords Compared:")
    print(f"  Demo: {results['summary'].get('demo_records', 'N/A')}")
    print(f"  Live: {results['summary'].get('live_records', 'N/A')}")
    print(f"  Timestamps Match: {results['summary'].get('timestamps_match', 'N/A')}")

    if results["issues"]:
        print(f"\n⚠ ISSUES FOUND ({len(results['issues'])}):")
        for i, issue in enumerate(results["issues"], 1):
            print(f"  {i}. {issue}")

    if results["details"]:
        print(f"\nDETAILED COLUMN COMPARISON:")
        for col, details in results["details"].items():
            if isinstance(details, dict) and "identical" in details:
                print(f"\n  {col.upper()}:")
                if details["identical"]:
                    print(f"    ✓ Identical")
                else:
                    print(f"    Max Difference: {details.get('max_diff', 'N/A')}")
                    print(f"    Mean Difference: {details.get('mean_diff', 'N/A')}")
                    print(f"    Mismatches: {details.get('mismatches', 'N/A')}")

                    if "sample_mismatches" in details:
                        print(f"    Sample Mismatches:")
                        for sample in details["sample_mismatches"]:
                            print(
                                f"      {sample['timestamp']}: Demo={sample['demo']}, "
                                f"Live={sample['live']}, Diff={sample['diff']}"
                            )

    print("\n" + "=" * 70 + "\n")
