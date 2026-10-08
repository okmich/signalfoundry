import numpy as np
import pandas as pd
import pywt
import talib
from numba import njit
from scipy.ndimage import convolve1d
from scipy.signal import savgol_filter
from scipy.signal.windows import gaussian
from statsmodels.nonparametric.smoothers_lowess import lowess


def smooth_ema(series: pd.Series, window=10):
    return pd.Series(talib.EMA(series.values, timeperiod=window), index=series.index)


def smooth_sma(series: pd.Series, window=10):
    return pd.Series(talib.SMA(series.values, timeperiod=window), index=series.index)


def smooth_wma(series: pd.Series, window=10):
    return pd.Series(talib.WMA(series.values, timeperiod=window), index=series.index)


def smooth_median(series: pd.Series, window=5, causal=True):
    """
    Median filter.

    Parameters
    ----------
    series : pd.Series
        Input time series
    window : int
        Window size (default: 5)
    causal : bool
        If True, only use past data (default: True for trading).
        If False, center the window (uses future data - NOT suitable for live trading).

    Returns
    -------
    pd.Series
        Smoothed series
    """
    if causal:
        return series.rolling(window=window, center=False).median()
    else:
        import warnings
        warnings.warn(
            "smooth_median with causal=False uses future data. "
            "Not suitable for live trading.",
            UserWarning
        )
        return series.rolling(window=window, center=True).median()


def smooth_gaussian(series: pd.Series, window=11, sigma=2.0, causal=True):
    """
    Gaussian smoothing filter.

    Parameters
    ----------
    series : pd.Series
        Input time series
    window : int
        Window size (default: 11)
    sigma : float
        Gaussian standard deviation (default: 2.0)
    causal : bool
        If True, use asymmetric kernel with only past data (default: True for trading).
        If False, use symmetric kernel centered at current bar (uses future data).

    Returns
    -------
    pd.Series
        Smoothed series
    """
    import warnings

    if window < 1:
        raise ValueError("Window size must be at least 1")

    if causal:
        # Create asymmetric (causal) Gaussian kernel
        # Only use past data: kernel spans from -window to 0
        x = np.linspace(-window, 0, window + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / np.sum(kernel)  # Normalize

        # Apply causal convolution
        result = np.zeros(len(series))
        values = series.values

        for i in range(len(series)):
            start = max(0, i - window)
            # Take appropriate slice of kernel for current position
            kernel_slice = kernel[-(i - start + 1):]
            # Re-normalize for edge cases
            kernel_slice = kernel_slice / np.sum(kernel_slice)
            result[i] = np.sum(values[start:i + 1] * kernel_slice)

        return pd.Series(result, index=series.index)
    else:
        # Original symmetric version (uses future data)
        warnings.warn(
            "smooth_gaussian with causal=False uses future data. "
            "Not suitable for live trading.",
            UserWarning
        )

        if window % 2 == 0:
            warnings.warn(
                "Odd window size recommended for Gaussian filter; incrementing by 1"
            )
            window += 1

        # Generate symmetric Gaussian kernel
        kernel = gaussian(window, sigma)
        kernel = kernel / np.sum(kernel)
        # Apply centered convolution
        smoothed = convolve1d(series.values, weights=kernel, mode="reflect")
        return pd.Series(smoothed, index=series.index)


def smooth_savitzky_golay(series: pd.Series, window=11, polyorder=2, causal=True):
    """
    Savitzky-Golay smoothing filter.

    Parameters
    ----------
    series : pd.Series
        Input time series
    window : int
        Window size, must be odd (default: 11)
    polyorder : int
        Polynomial order (default: 2)
    causal : bool
        If True, only use past data (default: True for trading).
        If False, center the window (uses future data).

    Returns
    -------
    pd.Series
        Smoothed series
    """
    if window % 2 == 0:
        raise ValueError("Window size must be odd for Savitzky-Golay filter")
    if polyorder >= window:
        raise ValueError("Polynomial order must be less than window size")

    # Track original NaN positions
    nan_mask = series.isna()

    # Fill NaN values temporarily to maintain index alignment
    filled_series = series.ffill().bfill()
    if filled_series.isna().any():
        return pd.Series(np.nan, index=series.index)

    if causal:
        # Causal implementation: fit polynomial using only past data
        result = np.zeros(len(filled_series))
        values = filled_series.values

        for i in range(len(values)):
            start = max(0, i - window + 1)
            end = i + 1

            # Fit polynomial to past window data
            if end - start >= polyorder + 1:
                x_window = np.arange(end - start)
                y_window = values[start:end]

                # Polynomial fit
                coeffs = np.polyfit(x_window, y_window, polyorder)
                # Evaluate at last point (current time)
                result[i] = np.polyval(coeffs, x_window[-1])
            else:
                # Not enough data yet, use raw value
                result[i] = values[i]

        smoothed = result
    else:
        # Original centered version (uses future data)
        import warnings
        warnings.warn(
            "smooth_savitzky_golay with causal=False uses future data. "
            "Not suitable for live trading.",
            UserWarning
        )
        smoothed = savgol_filter(filled_series.values, window_length=window, polyorder=polyorder)

    result = pd.Series(smoothed, index=series.index)

    # Restore NaN at original positions
    result[nan_mask] = np.nan
    return result


def _wavelet_denoise(values: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """Soft-threshold the detail coefficients (universal threshold) and reconstruct, same length as ``values``."""
    coeffs = pywt.wavedec(values, wavelet=wavelet, level=level)
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(values)))
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode="soft")
    smoothed = pywt.waverec(coeffs, wavelet=wavelet)
    if len(smoothed) > len(values):
        smoothed = smoothed[: len(values)]
    elif len(smoothed) < len(values):
        smoothed = np.pad(smoothed, (0, len(values) - len(smoothed)), mode="edge")
    return smoothed


def smooth_wavelet(series: pd.Series, wavelet="db4", level=2, causal=True, window=64):
    """
    Wavelet smoothing filter.

    Parameters
    ----------
    series : pd.Series
        Input time series
    wavelet : str
        Wavelet type (default: "db4")
    level : int
        Decomposition level (default: 2)
    causal : bool
        If True (default), the value at t is the last point of the denoised ``window`` bars ending at t — it uses no
        bar after t, so it is safe for backtests and live use. The first ``window - 1`` values (and any window that
        contains a NaN) are NaN. Cost is O(n · window).
        If False, the whole series is denoised in one pass: every value depends on future bars. Research/plotting
        only — NOT suitable for live trading or as a model input.
    window : int
        Trailing window, in bars, for the causal mode (default 64).

    Returns
    -------
    pd.Series
        Smoothed series
    """
    values = series.values.astype(np.float64)
    if not causal:
        import warnings
        warnings.warn(
            "smooth_wavelet with causal=False is NON-CAUSAL and uses future data. "
            "Not suitable for live trading.",
            UserWarning
        )
        return pd.Series(_wavelet_denoise(values, wavelet, level), index=series.index)

    if window < 2 ** (level + 1):
        raise ValueError(f"window={window} too short for a level-{level} decomposition")
    out = np.full(len(values), np.nan)
    for t in range(window - 1, len(values)):
        segment = values[t - window + 1: t + 1]
        if np.isfinite(segment).all():
            out[t] = _wavelet_denoise(segment, wavelet, level)[-1]
    return pd.Series(out, index=series.index)


@njit
def _kalman_1d(data, process_noise, measurement_noise, initial_error):
    n = len(data)
    smoothed = np.zeros(n)

    x = data[0] if not np.isnan(data[0]) else 0.0
    P = initial_error
    Q = process_noise
    R = measurement_noise

    for t in range(n):
        if np.isnan(data[t]):
            smoothed[t] = x
            continue

        # Predict
        x_pred = x
        P_pred = P + Q

        # Update
        K = P_pred / (P_pred + R)
        x = x_pred + K * (data[t] - x_pred)
        P = (1 - K) * P_pred

        smoothed[t] = x

    return smoothed


def smooth_kalman(series: pd.Series, process_noise=0.1, measurement_noise=1.0, initial_error=1.0):
    data = series.values.astype(np.float64)
    smoothed = _kalman_1d(data, process_noise, measurement_noise, initial_error)
    return pd.Series(smoothed, index=series.index)


@njit(cache=True)
def _one_sided_loess_kernel(y: np.ndarray, window: int) -> np.ndarray:
    """Tricube-weighted local-linear fit over the ``window`` bars ending at t, evaluated at t."""
    n = len(y)
    out = np.full(n, np.nan)
    span = float(window)
    for t in range(window - 1, n):
        sw = 0.0
        swx = 0.0
        swy = 0.0
        swxx = 0.0
        swxy = 0.0
        ok = True
        for k in range(window):
            v = y[t - k]
            if np.isnan(v):
                ok = False
                break
            d = k / span
            w = (1.0 - d * d * d) ** 3
            x = -float(k)
            sw += w
            swx += w * x
            swy += w * v
            swxx += w * x * x
            swxy += w * x * v
        if not ok:
            continue
        denom = sw * swxx - swx * swx
        if denom == 0.0:
            out[t] = swy / sw
        else:
            slope = (sw * swxy - swx * swy) / denom
            intercept = (swy - slope * swx) / sw
            out[t] = intercept  # the fit evaluated at x = 0, i.e. at bar t
    return out


def smooth_loess(series: pd.Series, frac=0.1, causal=True, window=50):
    """
    LOESS (locally weighted scatterplot smoothing) filter.

    Parameters
    ----------
    series : pd.Series
        Input time series
    frac : float
        Fraction of data used for smoothing in the non-causal mode (default: 0.1)
    causal : bool
        If True (default), one-sided LOESS: a tricube-weighted local-linear regression over the ``window`` bars ending
        at t, evaluated at t (no robustness iterations). It uses no bar after t, so it is safe for backtests and live
        use. The first ``window - 1`` values (and any window containing a NaN) are NaN.
        If False, statsmodels ``lowess`` over the whole series with ``frac``: every value depends on future bars.
        Research/plotting only — NOT suitable for live trading or as a model input.
    window : int
        Trailing window, in bars, for the causal mode (default 50).

    Returns
    -------
    pd.Series
        Smoothed series
    """
    if not causal:
        import warnings
        warnings.warn(
            "smooth_loess with causal=False is NON-CAUSAL and uses future data. "
            "Not suitable for live trading.",
            UserWarning
        )
        if not 0 < frac < 1:
            raise ValueError("Frac must be between 0 and 1")
        x = np.arange(len(series))
        y = series.values
        smoothed = lowess(y, x, frac=frac, return_sorted=False)
        return pd.Series(smoothed, index=series.index)

    if window < 3:
        raise ValueError("window must be >= 3 for a local-linear fit")
    smoothed = _one_sided_loess_kernel(series.values.astype(np.float64), int(window))
    return pd.Series(smoothed, index=series.index)
