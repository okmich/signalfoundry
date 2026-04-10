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


def smooth_wavelet(series: pd.Series, wavelet="db4", level=2):
    """
    Wavelet smoothing filter.

    WARNING: This filter is NON-CAUSAL and uses future data.
    NOT suitable for real-time trading strategies.
    Use for research, visualization, or offline analysis only.

    For causal alternatives, use smooth_ema(), smooth_kalman(), or smooth_gaussian(causal=True).

    Parameters
    ----------
    series : pd.Series
        Input time series
    wavelet : str
        Wavelet type (default: "db4")
    level : int
        Decomposition level (default: 2)

    Returns
    -------
    pd.Series
        Smoothed series
    """
    import warnings
    warnings.warn(
        "smooth_wavelet is NON-CAUSAL and uses future data. "
        "Not suitable for live trading. Use smooth_ema or smooth_kalman instead.",
        UserWarning
    )

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(series.values, wavelet=wavelet, level=level)

    # Zero out high-frequency coefficients for denoising
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(series)))
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode="soft")

    # Reconstruct smoothed signal
    smoothed = pywt.waverec(coeffs, wavelet=wavelet)

    # Ensure output length matches input
    if len(smoothed) > len(series):
        smoothed = smoothed[: len(series)]
    elif len(smoothed) < len(series):
        smoothed = np.pad(smoothed, (0, len(series) - len(smoothed)), mode="edge")

    return pd.Series(smoothed, index=series.index)


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


def smooth_loess(series: pd.Series, frac=0.1):
    """
    LOESS (locally weighted scatterplot smoothing) filter.

    WARNING: This filter is NON-CAUSAL and uses future data.
    NOT suitable for real-time trading strategies.
    Use for research, visualization, or offline analysis only.

    For causal alternatives, use smooth_ema(), smooth_kalman(), or smooth_gaussian(causal=True).

    Parameters
    ----------
    series : pd.Series
        Input time series
    frac : float
        Fraction of data used for smoothing (default: 0.1)

    Returns
    -------
    pd.Series
        Smoothed series
    """
    import warnings
    warnings.warn(
        "smooth_loess is NON-CAUSAL and uses future data. "
        "Not suitable for live trading. Use smooth_ema or smooth_kalman instead.",
        UserWarning
    )

    if not 0 < frac < 1:
        raise ValueError("Frac must be between 0 and 1")

    x = np.arange(len(series))
    y = series.values
    smoothed = lowess(y, x, frac=frac, return_sorted=False)
    return pd.Series(smoothed, index=series.index)
