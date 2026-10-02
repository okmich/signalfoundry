"""Tests for the harness that binds MQL5 to Python.

The real check needs MetaTrader to produce a dump, so what is verified here is
that the harness would actually catch a divergence: a synthetic dump built from
the Python reference passes, and perturbed dumps fail.
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_mt5.hmm_export import (LABEL_NONE, Mql5HmmKernel, Mql5InferenceMode, build_features, check_parity,
                                         load_dump)

BURN_IN = 60
MODE = Mql5InferenceMode.CAUSAL_VITERBI


@pytest.fixture
def timeline(bars):
    return pd.date_range("2024-01-01", periods=len(bars[2]), freq="h", tz="UTC")


@pytest.fixture
def reference(bars, params, timeline):
    """What the MQL5 indicator should produce, computed in Python."""
    high, low, close = bars
    features = build_features(high, low, close, params.spec)
    labels, confidence = Mql5HmmKernel(params, MODE).run(features, burn_in=BURN_IN)

    frame = pd.DataFrame(features, columns=list(params.spec.names))
    frame.insert(0, "bar", np.arange(len(labels)))
    frame.insert(0, "time", (timeline.astype("int64") // 10**9))
    frame["label"] = labels
    frame["confidence"] = confidence
    return frame[frame["label"] != LABEL_NONE].reset_index(drop=True)


def write_dump(frame, path):
    frame.to_csv(path, index=False)
    return path


class TestLoadDump:
    def test_parses_epoch_seconds_into_utc(self, reference, tmp_path):
        loaded = load_dump(write_dump(reference, tmp_path / "dump.csv"))

        assert loaded.index.tz is not None
        assert str(loaded.index[0]) == "2024-01-05 13:00:00+00:00"  # 49 warm-up + 60 burn-in -> bar 109
        assert loaded.index.is_monotonic_increasing

    def test_rejects_a_file_without_a_time_column(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"bar": [1, 2]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="no 'time' column"):
            load_dump(path)


class TestCheckParity:
    def test_matching_dump_passes(self, reference, bars, params, timeline, tmp_path):
        high, low, close = bars
        report = check_parity(write_dump(reference, tmp_path / "dump.csv"), timeline, high, low, close, params, MODE,
                              burn_in=BURN_IN)

        assert report.ok, str(report)
        assert report.label_mismatches == 0
        assert report.max_feature_abs_diff < 1e-12
        assert report.n_compared > 400

    def test_detects_a_shifted_feature(self, reference, bars, params, timeline, tmp_path):
        """The failure this whole harness exists for: MQL5 computing a feature differently."""
        high, low, close = bars
        tampered = reference.copy()
        tampered["macd_atr"] = tampered["macd_atr"] + 1e-6

        report = check_parity(write_dump(tampered, tmp_path / "dump.csv"), timeline, high, low, close, params, MODE,
                              burn_in=BURN_IN)

        assert not report.ok
        assert report.per_feature_abs_diff["macd_atr"] == pytest.approx(1e-6, rel=1e-3)
        assert report.per_feature_abs_diff["atr_close"] < 1e-12

    def test_detects_mismatched_labels(self, reference, bars, params, timeline, tmp_path):
        high, low, close = bars
        tampered = reference.copy()
        tampered.loc[tampered.index[:5], "label"] = (tampered["label"].iloc[:5] + 1) % params.n_states

        report = check_parity(write_dump(tampered, tmp_path / "dump.csv"), timeline, high, low, close, params, MODE,
                              burn_in=BURN_IN)

        assert not report.ok
        assert report.label_mismatches == 5

    def test_raises_when_asked_and_failing(self, reference, bars, params, timeline, tmp_path):
        high, low, close = bars
        tampered = reference.copy()
        tampered["confidence"] = tampered["confidence"] + 0.5

        with pytest.raises(AssertionError, match="parity check failed"):
            check_parity(write_dump(tampered, tmp_path / "dump.csv"), timeline, high, low, close, params, MODE,
                         burn_in=BURN_IN, raise_on_fail=True)

    def test_rejects_a_non_overlapping_dump(self, reference, bars, params, timeline, tmp_path):
        """A stale dump from a different symbol or session must not silently pass."""
        high, low, close = bars
        shifted = reference.copy()
        shifted["time"] = shifted["time"] + 10 * 365 * 24 * 3600

        with pytest.raises(ValueError, match="no overlapping bars"):
            check_parity(write_dump(shifted, tmp_path / "dump.csv"), timeline, high, low, close, params, MODE,
                         burn_in=BURN_IN)

    def test_compares_only_the_overlap(self, reference, bars, params, timeline, tmp_path):
        """A dump covering part of the pulled history is fine."""
        high, low, close = bars
        partial = reference.iloc[100:200]

        report = check_parity(write_dump(partial, tmp_path / "dump.csv"), timeline, high, low, close, params, MODE,
                              burn_in=BURN_IN)

        assert report.ok and report.n_compared == 100


class TestReport:
    def test_str_renders_pass_and_fail(self, reference, bars, params, timeline, tmp_path):
        high, low, close = bars
        report = check_parity(write_dump(reference, tmp_path / "dump.csv"), timeline, high, low, close, params, MODE,
                              burn_in=BURN_IN)

        rendered = str(report)
        assert "PASS" in rendered
        assert "macd_atr" in rendered and "atr_close" in rendered

    def test_empty_report_is_not_ok(self):
        from okmich_quant_mt5.hmm_export import ParityReport

        assert not ParityReport(n_compared=0).ok
