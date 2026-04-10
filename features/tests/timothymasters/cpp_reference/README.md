# C++ Reference Comparison Testing

## Overview

This test suite validates the Python implementation of all 38 Timothy Masters indicators against the original C++ SINGLE.exe executable using real market data.

## Test Data

### Real Market Data (US500)
- `us500_first_1000.txt` - 1000 bars from May 2023 (price: 4107-4211)
- `us500_middle_500.txt` - 500 bars from Oct 2024 (price: 5740-5803)
- `us500_recent_500.txt` - 500 bars from Feb 2026 (price: 6808-6972)

### Synthetic Data (Edge Cases)
- `sine_wave.txt` - 500 bars with period 20 (for FTI testing)
- `constant.txt` - 100 bars constant prices (edge case)
- `random_walk.txt`, `trending_up.txt`, etc.

### Script File
- `all_indicators.scr` - Configuration for all 38 indicators with default parameters

## File Structure

```
tests/timothymasters/
├── test_data/
│   ├── *.txt                    # Market data files (YYYYMMDD O H L C V)
│   └── all_indicators.scr       # Indicator configuration
├── test_cpp_reference.py        # Comparison test suite
├── generate_test_data.py        # Synthetic data generator
├── convert_parquet_to_cpp.py    # Real data converter
└── README_CPP_REFERENCE.md      # This file
```

## Running Tests

### Option 1: Run All Tests with Pytest
```bash
cd projects/features
pytest tests/timothymasters/test_cpp_reference.py -v
```

### Option 2: Run Specific Dataset
```bash
pytest tests/timothymasters/test_cpp_reference.py::test_real_data_comparison[us500_first_1000.txt] -v
```

### Option 3: Run as Standalone Script
```bash
cd projects/features
python tests/timothymasters/test_cpp_reference.py
```

## What the Tests Do

1. **Load Market Data** - Read OHLCV data from text file
2. **Run C++ SINGLE.exe** - Execute original implementation
3. **Parse C++ Output** - Read OUTVARS.TXT with indicator values
4. **Run Python Implementation** - Compute same indicators
5. **Compare Results** - Use `np.allclose(rtol=1e-6, atol=1e-8)`
6. **Report Discrepancies** - Show max/mean differences for mismatches

## Success Criteria

- **NaN Pattern Match**: Both implementations have NaN in same positions
- **Value Match**: `np.allclose(cpp, python, rtol=1e-6, atol=1e-8)`
- **All 38 Indicators**: Must match on real data

## Tolerance Levels

- **Real Data**: `rtol=1e-6` (0.0001%), `atol=1e-8`
- **Synthetic Data**: `rtol=1e-5` (0.001%), `atol=1e-7` (more relaxed for edge cases)

## Test Output Example

```
======================================================================
Testing: us500_first_1000.txt
======================================================================
Loaded 1000 bars
Running C++ SINGLE.exe...
C++ computed 38 indicators
Running Python implementation...
Python computed 38 indicators

Comparing results...

======================================================================
SUMMARY: us500_first_1000.txt
======================================================================
Total indicators: 38
Matched: 38 (100.0%)
Mismatched: 0
```

## Indicators Tested

### Momentum & Oscillators (11)
RSI, Detrended RSI, Stochastic, StochRSI, MA Difference, MACD, PPO, Price Change Osc, Close Minus MA, Price Intensity, Reactivity

### Trend (10)
Linear/Quadratic/Cubic Trend, Linear/Quadratic/Cubic Deviation, ADX, Aroon Up/Down/Diff

### Variance (2)
Price Variance Ratio, Change Variance Ratio

### Volume (9)
Intraday Intensity, Money Flow, Price Volume Fit, VWMA Ratio, Normalized OBV, Delta OBV, Normalized PVI/NVI, Volume Momentum

### Information Theory (2)
Entropy, Mutual Information

### FTI (4)
FTI Lowpass, Best Width, Best Period, Best FTI

## Next Steps (Step 4)

After tests are created, run validation:

```bash
# Run full test suite
pytest tests/timothymasters/test_cpp_reference.py -v --tb=short

# If discrepancies found:
# 1. Check parameter mappings in script file
# 2. Verify NaN warmup patterns
# 3. Investigate numeric precision differences
# 4. Document any systematic differences
```

## Troubleshooting

### C++ Executable Fails
- Check path: `D:\practise\timothy-masters-indicators\executable_programs\SINGLE.exe`
- Verify test data format: `YYYYMMDD Open High Low Close Volume`

### Import Errors
- Ensure Python package installed: `pip install -e projects/features`

### NaN Pattern Mismatches
- Check warmup periods in both implementations
- Verify parameter ranges (some indicators have min requirements)

### Value Mismatches
- Small differences (<1e-6) are acceptable (float precision)
- Large differences require investigation
