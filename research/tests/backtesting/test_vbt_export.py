import json
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from okmich_quant_research.backtesting.vbt_export import FileFormat, QuantframeExportMixin, _ensure_datetime_series,\
    _guard_numeric, _sanitize_for_json, _validate_datetime_index

# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

N_BARS = 200
RNG = np.random.default_rng(42)


def _make_price_data(n: int = N_BARS) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = (RNG.standard_normal(n).cumsum() + 100).clip(min=10)
    high = close * (1 + RNG.uniform(0, 0.01, n))
    low = close * (1 - RNG.uniform(0, 0.01, n))
    return pd.DataFrame(
        {"open": close * (1 + RNG.standard_normal(n) * 0.002), "high": high, "low": low, "close": close},
        index=dates,
    )


class _ExporterStub(QuantframeExportMixin):
    """Concrete class to test the mixin."""
    pass


@pytest.fixture(scope="module")
def price_data() -> pd.DataFrame:
    return _make_price_data()


@pytest.fixture(scope="module")
def portfolio(price_data: pd.DataFrame) -> vbt.Portfolio:
    close = price_data["close"]
    fast_ma = close.rolling(5).mean()
    slow_ma = close.rolling(20).mean()
    entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    return vbt.Portfolio.from_signals(close=close, entries=entries, exits=exits, init_cash=10_000, fees=0.001, freq="D")


@pytest.fixture()
def exporter() -> _ExporterStub:
    return _ExporterStub()


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic(self):
        assert _ExporterStub._slugify("MACD Crossover") == "macd_crossover"

    def test_hyphens_and_spaces(self):
        assert _ExporterStub._slugify("RSI Mean-Reversion v2") == "rsi_mean_reversion_v2"

    def test_special_chars_stripped(self):
        assert _ExporterStub._slugify("Bollinger Band (20,2)") == "bollinger_band_202"


# ---------------------------------------------------------------------------
# _sanitize_path_component
# ---------------------------------------------------------------------------


class TestSanitizePathComponent:
    def test_basic_symbol(self):
        assert _ExporterStub._sanitize_path_component("AAPL") == "AAPL"

    def test_strips_path_separators(self):
        assert "/" not in _ExporterStub._sanitize_path_component("../evil")
        assert "\\" not in _ExporterStub._sanitize_path_component("..\\evil")

    def test_empty_fallback(self):
        assert _ExporterStub._sanitize_path_component("///") == "unknown"

    def test_hyphenated_symbol(self):
        result = _ExporterStub._sanitize_path_component("BTC-USD")
        assert "/" not in result
        assert "\\" not in result


# ---------------------------------------------------------------------------
# _guard_numeric
# ---------------------------------------------------------------------------


class TestGuardNumeric:
    def test_clean_array_passes(self):
        _guard_numeric(np.array([1.0, 2.0, 3.0]), "test")

    def test_inf_raises(self):
        with pytest.raises(RuntimeError, match="inf"):
            _guard_numeric(np.array([1.0, np.inf]), "test")

    def test_neg_inf_raises(self):
        with pytest.raises(RuntimeError, match="inf"):
            _guard_numeric(np.array([1.0, -np.inf]), "test")

    def test_nan_raises_by_default(self):
        with pytest.raises(RuntimeError, match="NaN"):
            _guard_numeric(np.array([1.0, np.nan]), "test")

    def test_nan_allowed(self):
        _guard_numeric(np.array([1.0, np.nan]), "test", allow_nan=True)


# ---------------------------------------------------------------------------
# _validate_datetime_index
# ---------------------------------------------------------------------------


class TestValidateDatetimeIndex:
    def test_naive_passthrough(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        result = _validate_datetime_index(idx)
        assert result.tz is None

    def test_tz_aware_converts(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="h", tz="US/Eastern")
        result = _validate_datetime_index(idx)
        assert str(result.tz) == str(idx.tz)

    def test_non_datetime_raises(self):
        idx = pd.RangeIndex(10)
        with pytest.raises(RuntimeError, match="DatetimeIndex"):
            _validate_datetime_index(idx)


# ---------------------------------------------------------------------------
# _ensure_datetime_series
# ---------------------------------------------------------------------------


class TestEnsureDatetimeSeries:
    def test_naive_passthrough(self):
        s = pd.Series(pd.date_range("2020-01-01", periods=3, freq="D"))
        result = _ensure_datetime_series(s)
        assert result.dt.tz is None

    def test_tz_aware_converts(self):
        s = pd.Series(pd.date_range("2020-01-01", periods=3, freq="h", tz="US/Eastern"))
        result = _ensure_datetime_series(s)
        assert str(result.dt.tz) == str(s.dt.tz)

    def test_parses_string_timestamps_without_normalizing(self):
        s = pd.Series(["2020-01-01T09:30:00-05:00", "2020-01-01T10:30:00-05:00"])
        result = _ensure_datetime_series(s)
        assert result.iloc[0].isoformat() == "2020-01-01T09:30:00-05:00"

    def test_rejects_mixed_naive_and_aware(self):
        s = pd.Series(["2020-01-01T09:30:00", "2020-01-01T10:30:00-05:00"])
        with pytest.raises(RuntimeError, match="mixes tz-aware and tz-naive"):
            _ensure_datetime_series(s)


# ---------------------------------------------------------------------------
# _sanitize_for_json
# ---------------------------------------------------------------------------


class TestSanitizeForJson:
    def test_numpy_int(self):
        result = _sanitize_for_json({"a": np.int64(42)})
        assert result == {"a": 42}
        assert isinstance(result["a"], int)

    def test_numpy_float(self):
        result = _sanitize_for_json({"a": np.float64(3.14)})
        assert result == {"a": 3.14}
        assert isinstance(result["a"], float)

    def test_nan_becomes_none(self):
        result = _sanitize_for_json({"a": float("nan")})
        assert result == {"a": None}

    def test_inf_becomes_none(self):
        result = _sanitize_for_json({"a": float("inf")})
        assert result == {"a": None}

    def test_numpy_nan_becomes_none(self):
        result = _sanitize_for_json({"a": np.float64("nan")})
        assert result == {"a": None}

    def test_ndarray(self):
        result = _sanitize_for_json({"a": np.array([1, 2, 3])})
        assert result == {"a": [1, 2, 3]}

    def test_timestamp(self):
        ts = pd.Timestamp("2024-01-15 09:30:00")
        result = _sanitize_for_json({"a": ts})
        assert result == {"a": "2024-01-15T09:30:00"}

    def test_nested_dict(self):
        obj = {"outer": {"inner": np.int64(5)}}
        result = _sanitize_for_json(obj)
        assert result == {"outer": {"inner": 5}}

    def test_result_is_json_serializable(self):
        obj = {"a": np.float64(1.5), "b": np.int64(2), "c": np.array([3, 4]), "d": float("nan")}
        sanitized = _sanitize_for_json(obj)
        text = json.dumps(sanitized, allow_nan=False)
        assert isinstance(text, str)

    def test_common_python_types(self):
        class Mode(Enum):
            FAST = "fast"

        model_path = Path("models") / "best.pkl"
        obj = {
            "dt": datetime(2024, 1, 15, 9, 30, 0),
            "d": date(2024, 1, 15),
            "path": model_path,
            "dec": Decimal("1.25"),
            "enum": Mode.FAST,
        }
        result = _sanitize_for_json(obj)
        assert result["dt"] == "2024-01-15T09:30:00"
        assert result["d"] == "2024-01-15"
        assert result["path"] == str(model_path)
        assert result["dec"] == 1.25
        assert result["enum"] == "fast"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_non_portfolio_raises_type_error(self, exporter: _ExporterStub, tmp_path: Path):
        with pytest.raises(TypeError, match="vbt.Portfolio"):
            exporter._validate_export_inputs("not_a_portfolio", tmp_path, "AAPL", "Strat", "parquet")

    def test_bad_format_raises(self, exporter: _ExporterStub, tmp_path: Path, portfolio: vbt.Portfolio):
        with pytest.raises(ValueError, match="parquet.*csv"):
            exporter._validate_export_inputs(portfolio, tmp_path, "AAPL", "Strat", "json")

    def test_empty_symbol_raises(self, exporter: _ExporterStub, tmp_path: Path, portfolio: vbt.Portfolio):
        with pytest.raises(ValueError, match="symbol"):
            exporter._validate_export_inputs(portfolio, tmp_path, "", "Strat", "parquet")

    def test_empty_strategy_raises(self, exporter: _ExporterStub, tmp_path: Path, portfolio: vbt.Portfolio):
        with pytest.raises(ValueError, match="strategy"):
            exporter._validate_export_inputs(portfolio, tmp_path, "AAPL", "", "parquet")

    def test_missing_output_dir_raises(self, exporter: _ExporterStub, portfolio: vbt.Portfolio):
        with pytest.raises(ValueError, match="output_dir does not exist"):
            exporter._validate_export_inputs(portfolio, "/nonexistent/path", "AAPL", "Strat", "parquet")

    def test_non_string_timeframe_raises(self, exporter: _ExporterStub, tmp_path: Path, portfolio: vbt.Portfolio):
        with pytest.raises(TypeError, match="timeframe must be a string"):
            exporter._validate_export_inputs(portfolio, tmp_path, "AAPL", "Strat", "parquet", timeframe=123)

    def test_empty_timeframe_raises(self, exporter: _ExporterStub, tmp_path: Path, portfolio: vbt.Portfolio):
        with pytest.raises(ValueError, match="timeframe must be a non-empty string"):
            exporter._validate_export_inputs(portfolio, tmp_path, "AAPL", "Strat", "parquet", timeframe="  ")


# ---------------------------------------------------------------------------
# Full export (integration)
# ---------------------------------------------------------------------------


class TestExportToQuantframe:
    def test_parquet_export_produces_all_files(self, exporter: _ExporterStub, portfolio: vbt.Portfolio, tmp_path: Path):
        result = exporter.export_to_quantframe(portfolio, tmp_path, "AAPL", "MA Crossover", "1D")
        assert result.is_dir()
        assert (result / "metadata.json").exists()
        assert (result / "equity_curve.parquet").exists()
        assert (result / "returns.parquet").exists()
        assert (result / "trades.parquet").exists()

    def test_csv_export_produces_all_files(self, exporter: _ExporterStub, portfolio: vbt.Portfolio, tmp_path: Path):
        result = exporter.export_to_quantframe(portfolio, tmp_path, "AAPL", "MA Crossover", "1D", file_format="csv")
        assert (result / "equity_curve.csv").exists()
        assert (result / "returns.csv").exists()
        assert (result / "trades.csv").exists()

    def test_metadata_content(self, exporter: _ExporterStub, portfolio: vbt.Portfolio, tmp_path: Path):
        params = {"fast": 5, "slow": 20}
        result = exporter.export_to_quantframe(portfolio, tmp_path, "AAPL", "MA Crossover", "1D", param_set=params)
        with open(result / "metadata.json") as f:
            meta = json.load(f)
        assert meta["symbol"] == "AAPL"
        assert meta["strategy"] == "MA Crossover"
        assert meta["timeframe"] == "1D"
        assert meta["param_set"] == {"fast": 5, "slow": 20}
        assert meta["source"] == "vectorbt"
        assert "exported_at" in meta
        assert "vbt_version" in meta
        assert meta["strategy_id"] == "AAPL_ma_crossover"

    def test_metadata_with_numpy_params(self, exporter: _ExporterStub, portfolio: vbt.Portfolio, tmp_path: Path):
        params = {"lr": np.float64(0.001), "epochs": np.int64(100), "bad": float("nan")}
        result = exporter.export_to_quantframe(portfolio, tmp_path, "SPY", "NN Strat", "1D", param_set=params)
        with open(result / "metadata.json") as f:
            meta = json.load(f)
        assert meta["param_set"]["lr"] == 0.001
        assert meta["param_set"]["epochs"] == 100
        assert meta["param_set"]["bad"] is None

    def test_equity_curve_schema(self, exporter: _ExporterStub, portfolio: vbt.Portfolio, tmp_path: Path):
        result = exporter.export_to_quantframe(portfolio, tmp_path, "AAPL", "Strat", "1D", file_format="csv")
        df = pd.read_csv(result / "equity_curve.csv")
        assert list(df.columns) == ["date", "equity"]
        assert len(df) > 2
        assert df["equity"].dtype == np.float64

    def test_returns_schema(self, exporter: _ExporterStub, portfolio: vbt.Portfolio, tmp_path: Path):
        result = exporter.export_to_quantframe(portfolio, tmp_path, "AAPL", "Strat", "1D", file_format="csv")
        df = pd.read_csv(result / "returns.csv")
        assert list(df.columns) == ["date", "return"]
        assert df["return"].notna().all()

    def test_trades_schema(self, exporter: _ExporterStub, portfolio: vbt.Portfolio, tmp_path: Path):
        result = exporter.export_to_quantframe(portfolio, tmp_path, "AAPL", "Strat", "1D", file_format="csv")
        df = pd.read_csv(result / "trades.csv")
        expected_cols = ["entry_date", "exit_date", "entry_price", "exit_price", "size", "pnl", "pnl_pct", "direction"]
        assert list(df.columns) == expected_cols
        assert set(df["direction"].unique()).issubset({"long", "short"})

    def test_intraday_timestamp_preserves_timezone(self, exporter: _ExporterStub, tmp_path: Path):
        dates = pd.date_range("2020-01-01", periods=N_BARS, freq="h", tz="US/Eastern")
        close = (RNG.standard_normal(N_BARS).cumsum() + 100).clip(min=10)
        fast_ma = pd.Series(close).rolling(5).mean()
        slow_ma = pd.Series(close).rolling(20).mean()
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        pf = vbt.Portfolio.from_signals(
            close=pd.Series(close, index=dates), entries=entries.values, exits=exits.values,
            init_cash=10_000, freq="h",
        )
        result = exporter.export_to_quantframe(pf, tmp_path, "AAPL", "Hourly Strat", "1H", file_format="csv")
        df = pd.read_csv(result / "equity_curve.csv")
        assert df["date"].iloc[0] == dates[0].isoformat()
        assert df["date"].iloc[-1] == dates[-1].isoformat()

    def test_daily_timestamp_preserved_without_truncation(self, exporter: _ExporterStub, portfolio: vbt.Portfolio,
                                                          tmp_path: Path):
        result = exporter.export_to_quantframe(portfolio, tmp_path, "AAPL", "Daily", "1D", file_format="csv")
        df = pd.read_csv(result / "equity_curve.csv")
        expected_start = portfolio.value().index[0].isoformat()
        expected_end = portfolio.value().index[-1].isoformat()
        assert df["date"].iloc[0] == expected_start
        assert df["date"].iloc[-1] == expected_end

    def test_stale_files_cleaned_on_format_switch(self, exporter: _ExporterStub, portfolio: vbt.Portfolio, tmp_path: Path):
        exporter.export_to_quantframe(portfolio, tmp_path, "AAPL", "Switch", "1D", file_format="parquet")
        target = tmp_path / "AAPL_switch"
        assert (target / "equity_curve.parquet").exists()

        exporter.export_to_quantframe(portfolio, tmp_path, "AAPL", "Switch", "1D", file_format="csv")
        assert (target / "equity_curve.csv").exists()
        assert not (target / "equity_curve.parquet").exists()

    def test_folder_name_is_sanitized(self, exporter: _ExporterStub, portfolio: vbt.Portfolio, tmp_path: Path):
        result = exporter.export_to_quantframe(portfolio, tmp_path, "../evil", "My Strat", "1D")
        assert result.resolve().is_relative_to(tmp_path.resolve())
        assert ".." not in result.name

    def test_overwrite_existing(self, exporter: _ExporterStub, portfolio: vbt.Portfolio, tmp_path: Path):
        exporter.export_to_quantframe(portfolio, tmp_path, "AAPL", "Overwrite", "1D")
        result = exporter.export_to_quantframe(portfolio, tmp_path, "AAPL", "Overwrite", "1D")
        assert (result / "metadata.json").exists()

    def test_returns_absolute_path(self, exporter: _ExporterStub, portfolio: vbt.Portfolio, tmp_path: Path):
        result = exporter.export_to_quantframe(portfolio, tmp_path, "AAPL", "AbsPath", "1D")
        assert result.is_absolute()

    def test_failed_export_does_not_clobber_existing(self, exporter: _ExporterStub, portfolio: vbt.Portfolio, tmp_path: Path,
                                                     monkeypatch):
        target = exporter.export_to_quantframe(portfolio, tmp_path, "AAPL", "Atomic", "1D", file_format="csv")
        original_metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))

        def _raise_write_error(df, path, file_format):
            raise RuntimeError("forced write failure")

        monkeypatch.setattr(_ExporterStub, "_write_dataframe", staticmethod(_raise_write_error))

        with pytest.raises(RuntimeError, match="forced write failure"):
            exporter.export_to_quantframe(portfolio, tmp_path, "AAPL", "Atomic", "1D", file_format="parquet")

        assert (target / "metadata.json").exists()
        assert (target / "equity_curve.csv").exists()
        assert (target / "returns.csv").exists()
        assert (target / "trades.csv").exists()
        assert not (target / "equity_curve.parquet").exists()
        assert json.loads((target / "metadata.json").read_text(encoding="utf-8")) == original_metadata
