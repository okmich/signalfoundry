"""
Regression tests for label_eval_util.py.

Covers:
  - evaluate_all_labels_regime_returns_potentials does NOT mutate the caller's DataFrame
  - No log_return column is leaked onto the caller's frame
  - Original column set is preserved after the call
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.utils.label_eval_util import (
    evaluate_all_labels_regime_returns_potentials,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_df():
    """Minimal DataFrame with open/close + two label columns (open required by evaluate_regime_returns_potentials)."""
    n = 200
    rng = np.random.default_rng(42)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    opens = closes * rng.uniform(0.998, 1.002, n)
    labels_a = rng.choice([0, 1, 2], size=n)
    labels_b = rng.choice([0, 1, 2], size=n)
    df = pd.DataFrame(
        {
            "open": opens,
            "close": closes,
            "label_a": labels_a.astype(float),
            "label_b": labels_b.astype(float),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )
    return df


# ---------------------------------------------------------------------------
# No-mutation tests
# ---------------------------------------------------------------------------

class TestNoMutation:
    def test_caller_columns_unchanged(self, price_df):
        """Function must not add columns (e.g. log_return) to caller's DataFrame."""
        original_cols = set(price_df.columns)

        evaluate_all_labels_regime_returns_potentials(
            price_df, labels_cols=["label_a", "label_b"]
        )

        assert set(price_df.columns) == original_cols, (
            f"Columns added to caller frame: {set(price_df.columns) - original_cols}"
        )

    def test_no_log_return_column_on_caller(self, price_df):
        """Specifically, log_return must not appear on caller frame."""
        assert "log_return" not in price_df.columns

        evaluate_all_labels_regime_returns_potentials(
            price_df, labels_cols=["label_a"]
        )

        assert "log_return" not in price_df.columns

    def test_close_column_values_unchanged(self, price_df):
        """Values in existing columns must be identical before and after."""
        close_before = price_df["close"].copy()

        evaluate_all_labels_regime_returns_potentials(
            price_df, labels_cols=["label_a", "label_b"]
        )

        pd.testing.assert_series_equal(price_df["close"], close_before)

    def test_label_column_values_unchanged(self, price_df):
        """Label column values must be unchanged after the call."""
        label_before = price_df["label_a"].copy()

        evaluate_all_labels_regime_returns_potentials(
            price_df, labels_cols=["label_a"]
        )

        pd.testing.assert_series_equal(price_df["label_a"], label_before)

    def test_preexisting_log_return_not_overwritten(self):
        """If caller already has a log_return column with custom values, it is preserved."""
        n = 100
        rng = np.random.default_rng(1)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        opens = closes * rng.uniform(0.998, 1.002, n)
        sentinel = np.full(n, 999.0)  # sentinel value to detect overwrite
        df = pd.DataFrame(
            {
                "open": opens,
                "close": closes,
                "label_a": rng.choice([0, 1, 2], size=n).astype(float),
                "log_return": sentinel,
            },
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )
        log_ret_before = df["log_return"].copy()

        evaluate_all_labels_regime_returns_potentials(df, labels_cols=["label_a"])

        pd.testing.assert_series_equal(df["log_return"], log_ret_before)

    def test_empty_labels_cols_returns_none_without_mutation(self, price_df):
        """Passing an empty labels_cols list returns None and does not mutate frame."""
        original_cols = set(price_df.columns)

        result = evaluate_all_labels_regime_returns_potentials(price_df, labels_cols=[])

        assert result is None
        assert set(price_df.columns) == original_cols
