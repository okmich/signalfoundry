"""Tests for the posterior-native plotting utilities.

``Figure.show`` is patched to a no-op so the figure is built and inspected without
launching a renderer.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from okmich_quant_labelling.utils.posterior_plot_utils import (
    _confidence_alpha, _state_order, plot_price_multiple_posteriors_with_subplots,
    plot_price_with_posterior)


@pytest.fixture(autouse=True)
def _no_render(monkeypatch):
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)


def _build_df(T: int = 300, n_variants: int = 2, K: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, size=T)))
    df = pd.DataFrame({"close": close, "axis_feat": rng.normal(size=T)},
                      index=pd.date_range("2024-01-01", periods=T, freq="5min"))
    for v in range(1, n_variants + 1):
        logits = rng.normal(0, 1.0, size=(T, K))
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        for k in range(K):
            df[f"post_v{v}_s{k}"] = probs[:, k]
    return df


def test_confidence_alpha_bounds_and_extremes():
    onehot = np.array([[1.0, 0.0, 0.0]])
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert _confidence_alpha(onehot)[0] == pytest.approx(1.0)       # certain
    assert _confidence_alpha(uniform)[0] == pytest.approx(0.0, abs=1e-9)  # max entropy
    alpha = _confidence_alpha(np.array([[0.6, 0.3, 0.1]]))
    assert 0.0 < alpha[0] < 1.0


def test_state_order_follows_axis_rank():
    df = pd.DataFrame({"axis_feat": [2.0, 0.0]})
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])  # state0 sees axis 2.0, state1 sees 0.0
    # ascending: lowest-axis state (state1) should stack first
    assert _state_order(df, probs, axis_col="axis_feat", ascending_axis=True) == [1, 0]
    assert _state_order(df, probs, axis_col=None, ascending_axis=True) == [0, 1]


def test_plot_returns_figure_with_two_rows_per_variant():
    df = _build_df(n_variants=2)
    fig = plot_price_multiple_posteriors_with_subplots(df, axis_col="axis_feat",
                                                       rank_labels=["low", "mid", "high"])
    assert isinstance(fig, go.Figure)
    # per variant: 1 price line + 3 ribbon bands = 4 traces; 2 variants => 8
    assert len(fig.data) == 8


def test_plot_confidence_overlay_adds_marker_trace():
    df = _build_df(n_variants=1)
    base = plot_price_multiple_posteriors_with_subplots(df)
    overlay = plot_price_multiple_posteriors_with_subplots(df, color_price_by_confidence=True)
    assert len(overlay.data) == len(base.data) + 1


def test_plot_confidence_overlay_opacity_reflects_confidence():
    df = _build_df(n_variants=1)
    fig = plot_price_multiple_posteriors_with_subplots(df, color_price_by_confidence=True)
    marker_traces = [t for t in fig.data if getattr(t, "mode", None) == "markers"]
    assert len(marker_traces) == 1
    probs = df[["post_v1_s0", "post_v1_s1", "post_v1_s2"]].to_numpy()
    expected = _confidence_alpha(probs)
    # per-point opacity must equal confidence (ambiguous bars fade), not a constant
    np.testing.assert_allclose(np.asarray(marker_traces[0].marker.opacity, dtype=float), expected, atol=1e-9)
    assert np.ptp(expected) > 0  # there is genuine variation to encode


def test_plot_raises_when_no_variants():
    df = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="No posterior variants"):
        plot_price_multiple_posteriors_with_subplots(df)


def test_plot_raises_on_unknown_variant():
    df = _build_df(n_variants=1)
    with pytest.raises(ValueError, match="No post_"):
        plot_price_with_posterior(df, variant="ghost")


def test_plot_raises_on_missing_axis_col():
    df = _build_df(n_variants=1)
    with pytest.raises(ValueError, match="axis_col"):
        plot_price_multiple_posteriors_with_subplots(df, axis_col="nope")


def test_plot_raises_on_missing_price_col():
    df = _build_df(n_variants=1)
    with pytest.raises(ValueError, match="price_col"):
        plot_price_multiple_posteriors_with_subplots(df, price_col="not_a_col")


def test_single_variant_wrapper():
    df = _build_df(n_variants=2)
    fig = plot_price_with_posterior(df, variant="v1")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4  # 1 price + 3 bands for one variant