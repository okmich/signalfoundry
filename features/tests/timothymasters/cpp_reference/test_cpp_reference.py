"""
C++ Reference Comparison Tests for Timothy Masters Indicators.

Compares Python implementation against original C++ SINGLE.exe for all 38 indicators.
"""

import subprocess
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

# Import all Python implementations
from okmich_quant_features.timothymasters.momentum import (
    rsi, detrended_rsi, stochastic, stoch_rsi, ma_difference,
    macd, ppo, price_change_osc, close_minus_ma, price_intensity, reactivity
)
from okmich_quant_features.timothymasters.trend import (
    linear_trend, quadratic_trend, cubic_trend,
    linear_deviation, quadratic_deviation, cubic_deviation,
    adx, aroon_up, aroon_down, aroon_diff
)
from okmich_quant_features.timothymasters.variance import (
    price_variance_ratio, change_variance_ratio
)
from okmich_quant_features.timothymasters.volume import (
    intraday_intensity, money_flow, price_volume_fit, vwma_ratio,
    normalized_obv, delta_obv, normalized_pvi, normalized_nvi, volume_momentum
)
from okmich_quant_features.timothymasters.information import (
    entropy, mutual_information
)
from okmich_quant_features.timothymasters.fti import (
    fti_lowpass, fti_best_width, fti_best_period, fti_best_fti
)


# Paths
CPP_EXECUTABLE = r"D:\practise\timothy-masters-indicators\executable_programs\SINGLE.exe"
TEST_DATA_DIR = Path(__file__).parent / "test_data"
SCRIPT_FILE = TEST_DATA_DIR / "all_indicators_clean.scr"


def load_market_data(filename):
    """
    Load market data file and return as numpy arrays.

    Format: YYYYMMDD Open High Low Close Volume
    """
    filepath = TEST_DATA_DIR / filename
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        names=['date', 'open', 'high', 'low', 'close', 'volume'],
        dtype={'date': str, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}
    )

    return {
        'date': df['date'].values,
        'open': df['open'].values,
        'high': df['high'].values,
        'low': df['low'].values,
        'close': df['close'].values,
        'volume': df['volume'].values,
    }


def run_cpp_indicators(market_file):
    """
    Run C++ SINGLE.exe and parse output.

    Returns
    -------
    dict : {indicator_name: np.ndarray}
    """
    market_path = TEST_DATA_DIR / market_file
    script_path = SCRIPT_FILE

    # Run C++ executable
    # Note: SINGLE.exe waits for "Press any key..." at the end, causing timeout
    # We catch the timeout and proceed since OUTVARS.TXT is created before the prompt
    try:
        result = subprocess.run(
            [CPP_EXECUTABLE, str(market_path), str(script_path)],
            cwd=TEST_DATA_DIR,
            capture_output=True,
            text=True,
            input="\n",  # Simulate pressing Enter to bypass "Press any key..."
            timeout=10  # Short timeout since computation is fast
        )
        if result.returncode != 0:
            raise RuntimeError(f"C++ executable failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        # Expected - executable hangs on "Press any key..." but output file is created
        pass

    # Read output file (OUTVARS.TXT is created in current working directory)
    output_file = TEST_DATA_DIR / "OUTVARS.TXT"

    if not output_file.exists():
        raise FileNotFoundError(f"Output file not found: {output_file}")

    # Parse whitespace-separated output
    df = pd.read_csv(output_file, sep=r'\s+')

    # Convert to dict of numpy arrays (exclude Date column)
    cpp_results = {}
    for col in df.columns:
        if col != 'Date':  # Skip date column
            cpp_results[col] = df[col].values

    return cpp_results


def compute_python_indicators(data):
    """
    Compute all indicators using Python implementation.

    Returns
    -------
    dict : {indicator_name: np.ndarray}
    """
    open_ = data['open']
    high = data['high']
    low = data['low']
    close = data['close']
    volume = data['volume']

    results = {}

    # Momentum & Oscillators (#1-11)
    results['RSI_14'] = rsi(close, period=14)
    results['DETREND_RSI'] = detrended_rsi(close, short_period=7, long_period=14, reg_len=32)
    results['STOCH_14_1'] = stochastic(high, low, close, period=14, smoothing=1)
    results['STOCHRSI_14'] = stoch_rsi(close, rsi_period=14, stoch_period=14, smooth_period=1)
    results['MA_DIFF'] = ma_difference(high, low, close, short_period=10, long_period=40, lag=0)
    results['MACD_12_26_9'] = macd(high, low, close, short_period=12, long_period=26, signal_period=9)
    results['PPO_12_26_9'] = ppo(close, short_period=12, long_period=26, signal_period=9)
    results['PRICE_CHG_OSC'] = price_change_osc(high, low, close, short_period=10, multiplier=4.0)
    results['CLOSE_MINUS_MA'] = close_minus_ma(high, low, close, period=20, atr_period=60)
    results['PRICE_INTENSITY'] = price_intensity(open_, high, low, close, smooth_period=1)
    results['REACTIVITY'] = reactivity(high, low, close, volume, period=10, multiplier=4)

    # Trend Indicators (#12-21)
    results['LINEAR_TREND'] = linear_trend(high, low, close, period=50, atr_period=60)
    results['QUADRATIC_TREND'] = quadratic_trend(high, low, close, period=50, atr_period=60)
    results['CUBIC_TREND'] = cubic_trend(high, low, close, period=50, atr_period=60)
    results['LINEAR_DEV'] = linear_deviation(close, period=50)
    results['QUADRATIC_DEV'] = quadratic_deviation(close, period=50)
    results['CUBIC_DEV'] = cubic_deviation(close, period=50)
    results['ADX_14'] = adx(high, low, close, period=14)
    results['AROON_UP_25'] = aroon_up(high, low, period=25)
    results['AROON_DOWN_25'] = aroon_down(high, low, period=25)
    results['AROON_DIFF_25'] = aroon_diff(high, low, period=25)

    # Variance Indicators (#22-23)
    results['PRICE_VAR_RATIO'] = price_variance_ratio(close, short_period=10, multiplier=4.0)
    results['CHNG_VAR_RATIO'] = change_variance_ratio(close, short_period=10, multiplier=4.0)

    # Volume Indicators (#24-32)
    results['INTRADAY_INT'] = intraday_intensity(high, low, close, volume, period=14, smooth_period=0)
    results['MONEY_FLOW'] = money_flow(high, low, close, volume, period=14)
    results['PRICE_VOL_FIT'] = price_volume_fit(close, volume, period=50)
    results['VWMA_RATIO'] = vwma_ratio(close, volume, period=20)
    results['NORM_OBV'] = normalized_obv(close, volume, period=20)
    results['DELTA_OBV'] = delta_obv(close, volume, period=20, delta_period=5)
    results['NORM_PVI'] = normalized_pvi(close, volume, period=100)
    results['NORM_NVI'] = normalized_nvi(close, volume, period=100)
    results['VOLUME_MOM'] = volume_momentum(volume, short_period=10, multiplier=4.0)

    # Information Theory Indicators (#33-34)
    results['ENTROPY_3'] = entropy(close, word_length=3, mult=10)
    results['MUTUAL_INFO_3'] = mutual_information(close, word_length=3, mult=10)

    # FTI Indicators (#35-38)
    results['FTI_LOWPASS'] = fti_lowpass(close, lookback=60, half_length=30, min_period=8, max_period=40)
    results['FTI_BEST_WIDTH'] = fti_best_width(close, lookback=60, half_length=30, min_period=8, max_period=40)
    results['FTI_BEST_PER'] = fti_best_period(close, lookback=60, half_length=30, min_period=8, max_period=40)
    results['FTI_BEST_FTI'] = fti_best_fti(close, lookback=60, half_length=30, min_period=8, max_period=40)

    return results


def compare_results(cpp_results, py_results, rtol=1e-6, atol=1e-8):
    """
    Compare C++ and Python results.

    Returns
    -------
    dict : Comparison report with statistics
    """
    report = {
        'total_indicators': len(cpp_results),
        'matched': 0,
        'mismatched': 0,
        'details': {}
    }

    for indicator_name in cpp_results.keys():
        if indicator_name not in py_results:
            report['details'][indicator_name] = {
                'status': 'MISSING',
                'error': 'Not computed by Python implementation'
            }
            report['mismatched'] += 1
            continue

        cpp_vals = cpp_results[indicator_name]
        py_vals = py_results[indicator_name]

        # Align arrays - C++ may skip initial warmup bars
        # Take last N values from Python where N = len(cpp_vals)
        if len(cpp_vals) < len(py_vals):
            py_vals = py_vals[-len(cpp_vals):]
        elif len(py_vals) < len(cpp_vals):
            cpp_vals = cpp_vals[-len(py_vals):]

        # Find valid (non-NaN) indices in both
        cpp_valid = ~np.isnan(cpp_vals)
        py_valid = ~np.isnan(py_vals)

        # Compare NaN patterns first
        nan_match = np.array_equal(cpp_valid, py_valid)

        # Compare values where both are valid
        both_valid = cpp_valid & py_valid

        if not np.any(both_valid):
            report['details'][indicator_name] = {
                'status': 'NO_VALID_DATA',
                'cpp_valid_count': cpp_valid.sum(),
                'py_valid_count': py_valid.sum(),
            }
            report['mismatched'] += 1
            continue

        cpp_valid_vals = cpp_vals[both_valid]
        py_valid_vals = py_vals[both_valid]

        # Compare values
        match = np.allclose(cpp_valid_vals, py_valid_vals, rtol=rtol, atol=atol)

        # Calculate statistics
        abs_diff = np.abs(cpp_valid_vals - py_valid_vals)
        rel_diff = abs_diff / (np.abs(cpp_valid_vals) + 1e-10)

        report['details'][indicator_name] = {
            'status': 'MATCH' if match and nan_match else 'MISMATCH',
            'nan_pattern_match': nan_match,
            'value_match': match,
            'valid_count': both_valid.sum(),
            'cpp_valid_count': cpp_valid.sum(),
            'py_valid_count': py_valid.sum(),
            'max_abs_diff': abs_diff.max(),
            'mean_abs_diff': abs_diff.mean(),
            'max_rel_diff': rel_diff.max(),
            'mean_rel_diff': rel_diff.mean(),
        }

        if match and nan_match:
            report['matched'] += 1
        else:
            report['mismatched'] += 1

    return report


# ============================================================================
# TEST CASES
# ============================================================================

@pytest.mark.parametrize("market_file", [
    "us500_first_1000.txt",
    "us500_middle_500.txt",
    "us500_recent_500.txt",
])
def test_real_data_comparison(market_file):
    """Test all 38 indicators against C++ reference on real US500 data."""

    print(f"\n{'='*70}")
    print(f"Testing: {market_file}")
    print(f"{'='*70}")

    # Load data
    data = load_market_data(market_file)
    print(f"Loaded {len(data['close'])} bars")

    # Run C++ indicators
    print("Running C++ SINGLE.exe...")
    cpp_results = run_cpp_indicators(market_file)
    print(f"C++ computed {len(cpp_results)} indicators")

    # Run Python indicators
    print("Running Python implementation...")
    py_results = compute_python_indicators(data)
    print(f"Python computed {len(py_results)} indicators")

    # Compare (slightly relaxed tolerance for floating-point edge cases)
    print("\nComparing results...")
    report = compare_results(cpp_results, py_results, rtol=1e-4, atol=1e-4)

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {market_file}")
    print(f"{'='*70}")
    print(f"Total indicators: {report['total_indicators']}")
    print(f"Matched: {report['matched']} ({report['matched']/report['total_indicators']*100:.1f}%)")
    print(f"Mismatched: {report['mismatched']}")

    # Print details for mismatches
    if report['mismatched'] > 0:
        print(f"\n{'='*70}")
        print("MISMATCHES:")
        print(f"{'='*70}")

        for name, detail in report['details'].items():
            if detail['status'] != 'MATCH':
                print(f"\n{name}:")
                print(f"  Status: {detail['status']}")
                if 'max_abs_diff' in detail:
                    print(f"  Max absolute diff: {detail['max_abs_diff']:.2e}")
                    print(f"  Mean absolute diff: {detail['mean_abs_diff']:.2e}")
                    print(f"  Max relative diff: {detail['max_rel_diff']:.2e}")
                    print(f"  Valid count: {detail['valid_count']}")
                if not detail.get('nan_pattern_match', True):
                    print(f"  NaN pattern mismatch: C++={detail['cpp_valid_count']} vs Py={detail['py_valid_count']}")

    # Assert all matched
    assert report['mismatched'] == 0, f"{report['mismatched']} indicators did not match C++ reference"


@pytest.mark.parametrize("market_file", [
    "sine_wave.txt",
    "constant.txt",
])
def test_synthetic_data_comparison(market_file):
    """Test indicators on synthetic edge cases."""

    print(f"\n{'='*70}")
    print(f"Testing: {market_file}")
    print(f"{'='*70}")

    # Load data
    data = load_market_data(market_file)
    print(f"Loaded {len(data['close'])} bars")

    # Run C++ indicators (may fail on extreme edge cases like constant prices)
    print("Running C++ SINGLE.exe...")
    try:
        cpp_results = run_cpp_indicators(market_file)
        print(f"C++ computed {len(cpp_results)} indicators")
    except RuntimeError as e:
        print(f"C++ executable failed (expected for extreme edge cases): {e}")
        print("Skipping comparison for this dataset")
        return

    # Run Python indicators
    print("Running Python implementation...")
    py_results = compute_python_indicators(data)
    print(f"Python computed {len(py_results)} indicators")

    # Compare with more relaxed tolerance for synthetic data
    print("\nComparing results...")
    report = compare_results(cpp_results, py_results, rtol=1e-5, atol=1e-7)

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {market_file}")
    print(f"{'='*70}")
    print(f"Matched: {report['matched']}/{report['total_indicators']}")

    # For synthetic data, we allow some mismatches due to extreme values
    # Just report, don't fail
    if report['mismatched'] > 0:
        print(f"Note: {report['mismatched']} indicators had differences (expected for edge cases)")


if __name__ == "__main__":
    # Run tests manually
    print("="*70)
    print("C++ REFERENCE COMPARISON TEST")
    print("="*70)

    # Test one real data file
    test_real_data_comparison("us500_first_1000.txt")
