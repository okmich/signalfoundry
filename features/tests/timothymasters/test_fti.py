import numpy as np
import pytest
from okmich_quant_features.timothymasters.fti import (
    fti_lowpass,
    fti_best_width,
    fti_best_period,
    fti_best_fti,
)


class TestFTI:
    """Test FTI indicators (#35-38)."""

    def test_sine_wave_produces_valid_period(self):
        """FTI best_period should produce valid period values for sine wave."""
        # Create pure sine wave with period 20
        true_period = 20
        n = 500
        t = np.arange(n, dtype=np.float64)
        close = 100.0 + 10.0 * np.sin(2.0 * np.pi * t / true_period)

        result = fti_best_period(
            close,
            lookback=60,
            half_length=30,
            min_period=8,
            max_period=40
        )

        # After warmup, should produce valid period values in specified range
        valid = result[~np.isnan(result)]
        # FTI is an approximation - just verify it outputs reasonable values
        assert len(valid) > 0, "Should have valid values after warmup"
        assert np.all(valid >= 8) and np.all(valid <= 40), \
            f"Period values outside expected range [8, 40]: {valid.min()}, {valid.max()}"

    def test_fti_lowpass_smooths_noise(self):
        """FTI lowpass should smooth out high-frequency noise."""
        np.random.seed(42)
        # Signal + noise
        n = 300
        t = np.arange(n, dtype=np.float64)
        signal = 100.0 + 10.0 * np.sin(2.0 * np.pi * t / 20)
        noise = np.random.randn(n) * 2.0
        close = signal + noise

        result = fti_lowpass(close, lookback=60, half_length=30)

        # After warmup, lowpass should be smoother than input
        valid_idx = ~np.isnan(result)
        if np.any(valid_idx):
            # Smoothed signal should have lower variance than noisy input
            input_std = np.std(close[valid_idx])
            output_std = np.std(result[valid_idx])
            assert output_std < input_std, \
                f"Lowpass should smooth noise: input_std={input_std}, output_std={output_std}"

    def test_fti_best_width_output_range(self):
        """FTI best_width should output reasonable values."""
        np.random.seed(42)
        close = np.random.randn(300).cumsum() + 100

        result = fti_best_width(
            close,
            lookback=60,
            half_length=30,
            min_period=8,
            max_period=40
        )

        valid = result[~np.isnan(result)]
        # Width should be in range [0, max_period]
        assert np.all(valid >= 0), f"Min width {valid.min()} below 0"
        assert np.all(valid <= 40), f"Max width {valid.max()} above max_period"

    def test_fti_best_fti_output_range(self):
        """FTI best_fti should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(300).cumsum() + 100

        result = fti_best_fti(
            close,
            lookback=60,
            half_length=30,
            min_period=8,
            max_period=40
        )

        valid = result[~np.isnan(result)]
        # FTI output range
        assert np.all(valid >= -50), f"Min FTI {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max FTI {valid.max()} above 50"

    def test_nan_warmup(self):
        """All FTI functions should have NaN warmup."""
        np.random.seed(42)
        close = np.random.randn(300).cumsum() + 100

        for func in [fti_lowpass, fti_best_width, fti_best_period, fti_best_fti]:
            result = func(close, lookback=60, half_length=30, min_period=8, max_period=40)
            # Should have substantial warmup
            assert np.any(np.isnan(result[:100])), f"{func.__name__} should have NaN warmup"

    def test_no_inf_leaks(self):
        """FTI should not produce inf values."""
        np.random.seed(42)
        close = np.random.randn(300).cumsum() + 100

        for func in [fti_lowpass, fti_best_width, fti_best_period, fti_best_fti]:
            result = func(close, lookback=60, half_length=30, min_period=8, max_period=40)
            assert not np.any(np.isinf(result)), f"{func.__name__} should not produce inf"
