"""
Tests for error handling and resilience features.

Tests cover:
- Retry logic with exponential backoff
- Error classification (transient vs permanent)
- Connection monitoring
- Exception types
"""
import time
import pytest
from unittest.mock import Mock, patch, MagicMock
import MetaTrader5 as mt5

from okmich_quant_mt5.resilience import (
    with_retry,
    MT5TransientError,
    MT5PermanentError,
    MT5ConnectionError,
    classify_mt5_error,
    is_mt5_connected,
    ConnectionMonitor,
    TRANSIENT_ERROR_CODES,
    PERMANENT_ERROR_CODES,
)


class TestErrorClassification:
    """Test MT5 error classification."""

    def test_classify_transient_errors(self):
        """Test that transient errors are classified correctly."""
        transient_codes = [
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_CONNECTION,
            mt5.TRADE_RETCODE_PRICE_CHANGED,
            mt5.TRADE_RETCODE_TIMEOUT,
            mt5.TRADE_RETCODE_REJECT,
        ]

        for code in transient_codes:
            assert classify_mt5_error(code) == 'transient', f"Code {code} should be transient"

    def test_classify_permanent_errors(self):
        """Test that permanent errors are classified correctly."""
        permanent_codes = [
            mt5.TRADE_RETCODE_INVALID,
            mt5.TRADE_RETCODE_INVALID_VOLUME,
            mt5.TRADE_RETCODE_INVALID_PRICE,
            mt5.TRADE_RETCODE_INVALID_STOPS,
            mt5.TRADE_RETCODE_TRADE_DISABLED,
            mt5.TRADE_RETCODE_MARKET_CLOSED,
            mt5.TRADE_RETCODE_NO_MONEY,
        ]

        for code in permanent_codes:
            assert classify_mt5_error(code) == 'permanent', f"Code {code} should be permanent"

    def test_classify_unknown_errors(self):
        """Test that unknown error codes return 'unknown'."""
        unknown_code = 99999
        assert classify_mt5_error(unknown_code) == 'unknown'

    def test_transient_error_codes_set_not_empty(self):
        """Test that transient error codes set is populated."""
        assert len(TRANSIENT_ERROR_CODES) > 0
        assert mt5.TRADE_RETCODE_REQUOTE in TRANSIENT_ERROR_CODES

    def test_permanent_error_codes_set_not_empty(self):
        """Test that permanent error codes set is populated."""
        assert len(PERMANENT_ERROR_CODES) > 0
        assert mt5.TRADE_RETCODE_NO_MONEY in PERMANENT_ERROR_CODES


class TestRetryDecorator:
    """Test retry decorator functionality."""

    def test_successful_operation_no_retry(self):
        """Test that successful operations don't trigger retries."""
        call_count = 0

        @with_retry(max_retries=3)
        def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=True):
            result = successful_operation()

        assert result == "success"
        assert call_count == 1, "Should only call once on success"

    def test_transient_error_retries(self):
        """Test that transient errors trigger retries."""
        call_count = 0

        @with_retry(max_retries=3, initial_delay=0.01)  # Fast retries for testing
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise MT5TransientError("Temporary failure", mt5.TRADE_RETCODE_REQUOTE)
            return "success"

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=True):
            result = failing_operation()

        assert result == "success"
        assert call_count == 3, "Should retry until success"

    def test_permanent_error_no_retry(self):
        """Test that permanent errors don't trigger retries."""
        call_count = 0

        @with_retry(max_retries=3)
        def permanent_error_operation():
            nonlocal call_count
            call_count += 1
            raise MT5PermanentError("Invalid volume", mt5.TRADE_RETCODE_INVALID_VOLUME)

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=True):
            with pytest.raises(MT5PermanentError):
                permanent_error_operation()

        assert call_count == 1, "Should not retry permanent errors"

    def test_max_retries_exceeded(self):
        """Test that retries stop after max_retries."""
        call_count = 0

        @with_retry(max_retries=3, initial_delay=0.01)
        def always_failing():
            nonlocal call_count
            call_count += 1
            raise MT5TransientError("Always fails", mt5.TRADE_RETCODE_TIMEOUT)

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=True):
            with pytest.raises(MT5TransientError):
                always_failing()

        assert call_count == 4, "Should try once + 3 retries = 4 total"

    def test_exponential_backoff(self):
        """Test that delays follow exponential backoff."""
        call_times = []

        @with_retry(max_retries=3, initial_delay=0.1, backoff_factor=2.0)
        def timed_operation():
            call_times.append(time.time())
            if len(call_times) < 4:
                raise MT5TransientError("Retry", mt5.TRADE_RETCODE_REQUOTE)
            return "success"

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=True):
            timed_operation()

        # Verify exponential backoff (with some tolerance for timing)
        assert len(call_times) == 4

        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        delay3 = call_times[3] - call_times[2]

        # Allow 50% tolerance for timing variance
        assert 0.08 <= delay1 <= 0.15, f"First delay should be ~0.1s, got {delay1:.3f}s"
        assert 0.16 <= delay2 <= 0.30, f"Second delay should be ~0.2s, got {delay2:.3f}s"
        assert 0.32 <= delay3 <= 0.60, f"Third delay should be ~0.4s, got {delay3:.3f}s"

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        call_times = []

        @with_retry(max_retries=5, initial_delay=10.0, backoff_factor=2.0, max_delay=0.2)
        def capped_delay_operation():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise MT5TransientError("Retry", mt5.TRADE_RETCODE_REQUOTE)
            return "success"

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=True):
            capped_delay_operation()

        # All delays should be capped at max_delay (0.2s)
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        assert delay1 <= 0.25, f"Delay should be capped at 0.2s, got {delay1:.3f}s"
        assert delay2 <= 0.25, f"Delay should be capped at 0.2s, got {delay2:.3f}s"

    def test_connection_error_retries(self):
        """Test that connection errors trigger retries when enabled."""
        call_count = 0

        @with_retry(max_retries=2, initial_delay=0.01, retry_on_connection_loss=True)
        def connection_error_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise MT5ConnectionError("Connection lost")
            return "success"

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=True):
            result = connection_error_operation()

        assert result == "success"
        assert call_count == 3

    def test_connection_error_no_retry_when_disabled(self):
        """Test that connection errors don't retry when retry_on_connection_loss=False."""
        call_count = 0

        @with_retry(max_retries=3, retry_on_connection_loss=False)
        def connection_error_operation():
            nonlocal call_count
            call_count += 1
            raise MT5ConnectionError("Connection lost")

        with pytest.raises(MT5ConnectionError):
            connection_error_operation()

        assert call_count == 1


class TestConnectionMonitor:
    """Test connection monitoring functionality."""

    def test_initialization(self):
        """Test ConnectionMonitor initialization."""
        monitor = ConnectionMonitor(check_interval=30.0)

        assert monitor.check_interval == 30.0
        assert monitor.is_healthy is True
        assert monitor.consecutive_failures == 0

    def test_healthy_connection(self):
        """Test that healthy connection returns True."""
        monitor = ConnectionMonitor(check_interval=1.0)

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=True):
            result = monitor.check_connection(force=True)

        assert result is True
        assert monitor.is_healthy is True
        assert monitor.consecutive_failures == 0

    def test_unhealthy_connection(self):
        """Test that unhealthy connection returns False."""
        monitor = ConnectionMonitor(check_interval=1.0)

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=False):
            result = monitor.check_connection(force=True)

        assert result is False
        assert monitor.is_healthy is False
        assert monitor.consecutive_failures == 1

    def test_consecutive_failures_tracking(self):
        """Test that consecutive failures are tracked."""
        monitor = ConnectionMonitor(check_interval=1.0)

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=False):
            monitor.check_connection(force=True)
            assert monitor.consecutive_failures == 1

            monitor.check_connection(force=True)
            assert monitor.consecutive_failures == 2

            monitor.check_connection(force=True)
            assert monitor.consecutive_failures == 3

    def test_recovery_resets_failures(self):
        """Test that successful connection resets consecutive failures."""
        monitor = ConnectionMonitor(check_interval=1.0)

        # Fail a few times
        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=False):
            monitor.check_connection(force=True)
            monitor.check_connection(force=True)
            assert monitor.consecutive_failures == 2

        # Recover
        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=True):
            monitor.check_connection(force=True)

        assert monitor.is_healthy is True
        assert monitor.consecutive_failures == 0

    def test_rate_limiting(self):
        """Test that checks are rate-limited by check_interval."""
        monitor = ConnectionMonitor(check_interval=10.0)

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=True) as mock_check:
            # First check should happen
            monitor.check_connection(force=False)
            assert mock_check.call_count == 1

            # Second check should be skipped (rate limited)
            monitor.check_connection(force=False)
            assert mock_check.call_count == 1, "Should skip second check due to rate limiting"

    def test_force_bypasses_rate_limiting(self):
        """Test that force=True bypasses rate limiting."""
        monitor = ConnectionMonitor(check_interval=10.0)

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=True) as mock_check:
            monitor.check_connection(force=True)
            assert mock_check.call_count == 1

            # Force should bypass rate limiting
            monitor.check_connection(force=True)
            assert mock_check.call_count == 2

    def test_reconnect_callback_on_failure(self):
        """Test that reconnect callback is called on connection failure."""
        reconnect_called = False

        def mock_reconnect():
            nonlocal reconnect_called
            reconnect_called = True
            return True

        monitor = ConnectionMonitor(
            check_interval=1.0,
            reconnect_callback=mock_reconnect
        )

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', side_effect=[False, True]):
            result = monitor.check_connection(force=True)

        assert reconnect_called is True
        assert result is True
        assert monitor.is_healthy is True

    def test_reconnect_callback_failure(self):
        """Test that failed reconnection is handled gracefully."""
        def mock_reconnect():
            return False

        monitor = ConnectionMonitor(
            check_interval=1.0,
            reconnect_callback=mock_reconnect
        )

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=False):
            result = monitor.check_connection(force=True)

        assert result is False
        assert monitor.is_healthy is False

    def test_get_health_status(self):
        """Test get_health_status returns correct metrics."""
        monitor = ConnectionMonitor(check_interval=1.0)

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=False):
            monitor.check_connection(force=True)
            monitor.check_connection(force=True)

        status = monitor.get_health_status()

        assert 'is_healthy' in status
        assert 'consecutive_failures' in status
        assert 'last_check_time' in status
        assert 'is_connected' in status

        assert status['is_healthy'] is False
        assert status['consecutive_failures'] == 2


class TestExceptionTypes:
    """Test exception type hierarchy."""

    def test_mt5_transient_error(self):
        """Test MT5TransientError creation."""
        error = MT5TransientError("Test error", mt5.TRADE_RETCODE_REQUOTE)

        assert str(error) == "Test error"
        assert error.retcode == mt5.TRADE_RETCODE_REQUOTE
        assert isinstance(error, Exception)

    def test_mt5_permanent_error(self):
        """Test MT5PermanentError creation."""
        error = MT5PermanentError("Invalid volume", mt5.TRADE_RETCODE_INVALID_VOLUME)

        assert str(error) == "Invalid volume"
        assert error.retcode == mt5.TRADE_RETCODE_INVALID_VOLUME
        assert isinstance(error, Exception)

    def test_mt5_connection_error(self):
        """Test MT5ConnectionError creation."""
        error = MT5ConnectionError("Connection lost")

        assert str(error) == "Connection lost"
        assert isinstance(error, Exception)

    def test_exceptions_are_catchable_separately(self):
        """Test that exception types can be caught independently."""
        # Test MT5TransientError
        try:
            raise MT5TransientError("Test", mt5.TRADE_RETCODE_REQUOTE)
        except MT5TransientError as e:
            assert e.retcode == mt5.TRADE_RETCODE_REQUOTE

        # Test MT5PermanentError
        try:
            raise MT5PermanentError("Test", mt5.TRADE_RETCODE_INVALID)
        except MT5PermanentError as e:
            assert e.retcode == mt5.TRADE_RETCODE_INVALID

        # Test MT5ConnectionError
        try:
            raise MT5ConnectionError("Test")
        except MT5ConnectionError:
            pass


class TestIsConnectionConnected:
    """Test is_mt5_connected function."""

    def test_connected_returns_true(self):
        """Test that connected terminal returns True."""
        mock_terminal_info = MagicMock()
        mock_terminal_info.connected = True

        with patch('MetaTrader5.terminal_info', return_value=mock_terminal_info):
            assert is_mt5_connected() is True

    def test_disconnected_returns_false(self):
        """Test that disconnected terminal returns False."""
        mock_terminal_info = MagicMock()
        mock_terminal_info.connected = False

        with patch('MetaTrader5.terminal_info', return_value=mock_terminal_info):
            assert is_mt5_connected() is False

    def test_no_terminal_info_returns_false(self):
        """Test that None terminal_info returns False."""
        with patch('MetaTrader5.terminal_info', return_value=None):
            assert is_mt5_connected() is False


class TestRetryIntegration:
    """Integration tests for retry behavior."""

    def test_retry_preserves_return_value(self):
        """Test that retry decorator preserves return values."""
        @with_retry(max_retries=2)
        def return_complex_value():
            return {"key": "value", "number": 42}

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=True):
            result = return_complex_value()
        assert result == {"key": "value", "number": 42}

    def test_retry_preserves_function_metadata(self):
        """Test that retry decorator preserves function metadata."""
        @with_retry(max_retries=2)
        def documented_function():
            """This is a documented function."""
            return "result"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function."

    def test_multiple_decorators(self):
        """Test that retry works correctly when stacked with other decorators."""
        inner_call_count = [0]
        outer_call_count = [0]

        def logging_decorator(func):
            def wrapper(*args, **kwargs):
                outer_call_count[0] += 1
                return func(*args, **kwargs)
            return wrapper

        @logging_decorator
        @with_retry(max_retries=2, initial_delay=0.01)
        def multi_decorated():
            inner_call_count[0] += 1
            if inner_call_count[0] < 3:
                raise MT5TransientError("Retry", mt5.TRADE_RETCODE_REQUOTE)
            return "success"

        with patch('okmich_quant_mt5.resilience.is_mt5_connected', return_value=True):
            result = multi_decorated()

        assert result == "success"
        # Outer decorator fires once (logging_decorator wraps the retrying function)
        assert outer_call_count[0] == 1
        # Inner function fires 3 times (1 initial + 2 retries)
        assert inner_call_count[0] == 3