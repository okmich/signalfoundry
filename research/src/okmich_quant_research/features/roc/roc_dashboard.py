"""
Interactive Plotly / Dash dashboard for ROC analysis exploration.

Source / Attribution
--------------------
The dashboard concept — interactive threshold exploration, signal-vs-returns scatter, quantile bars, and cumulative
equity curves in a single view — is inspired by the visualisation guide distributed alongside:

    Timothy Masters, "Statistically Sound Indicators For Financial Market Prediction", Apress, 2013.

The ``InteractiveROCDashboard`` class is an original implementation for this project using Plotly and Dash.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .roc_analysis import ROCResults


class InteractiveROCDashboard:
    """
    Interactive four-panel ROC dashboard using Plotly and Dash.

    Provides two entry points:

    * :meth:`create_figure` — returns a static ``plotly.graph_objects.Figure``
      that can be displayed in a Jupyter notebook or saved as HTML.
    * :meth:`launch` — starts a local Dash web server (default port 8050) for
      fully interactive threshold exploration.

    Parameters
    ----------
    signals : np.ndarray
        Indicator values aligned with *returns*.
    returns : np.ndarray
        Forward log-returns aligned with *signals*.
    results : ROCResults
        Output of :meth:`~roc_analysis.ROCAnalyzer.analyze`.

    Examples
    --------
    Jupyter notebook::

        fig = InteractiveROCDashboard(signals, returns, results).create_figure()
        fig.show()

    Standalone server::

        InteractiveROCDashboard(signals, returns, results).launch(port=8050)
    """

    def __init__(self, signals: np.ndarray, returns: np.ndarray, results: ROCResults) -> None:
        self.signals = np.asarray(signals, dtype=np.float64)
        self.returns = np.asarray(returns, dtype=np.float64)
        self.results = results

    # ------------------------------------------------------------------ #
    # Static Plotly figure                                                 #
    # ------------------------------------------------------------------ #

    def create_figure(self):
        """
        Build and return a four-panel Plotly figure.

        Panels
        ------
        Top-left   : ROC curve (profit factor vs threshold)
        Top-right  : Signal vs returns scatter (coloured by return)
        Bottom-left: Mean return by decile (quantile bar chart)
        Bottom-right: Cumulative equity curves

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        results = self.results
        tbl = results.roc_table
        signals = self.signals
        returns = self.returns
        n = len(returns)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "ROC Curve",
                "Signal vs Returns",
                "Mean Return by Decile",
                "Cumulative Returns",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"},     {"type": "scatter"}],
            ],
            vertical_spacing=0.14,
            horizontal_spacing=0.10,
        )

        # ---- Top-left: ROC curve ---- #
        fig.add_trace(go.Scatter(
            x=tbl["threshold"], y=tbl["pf_long_above"],
            mode="lines+markers", name="Long PF",
            line=dict(color="green", width=3), marker=dict(size=8),
            hovertemplate="Threshold: %{x:.4f}<br>Long PF: %{y:.4f}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=tbl["threshold"], y=tbl["pf_short_below"],
            mode="lines+markers", name="Short PF",
            line=dict(color="red", width=3), marker=dict(size=8),
            hovertemplate="Threshold: %{x:.4f}<br>Short PF: %{y:.4f}<extra></extra>",
        ), row=1, col=1)

        fig.add_vline(x=results.long_threshold, line_dash="dash",
                      line_color="green",
                      annotation_text=f"Long: {results.long_threshold:.4f}",
                      row=1, col=1)
        fig.add_vline(x=results.short_threshold, line_dash="dash",
                      line_color="red",
                      annotation_text=f"Short: {results.short_threshold:.4f}",
                      row=1, col=1)
        fig.add_hline(y=1.0, line_dash="dot", line_color="black",
                      annotation_text="Break-even", row=1, col=1)

        # ---- Top-right: scatter ---- #
        fig.add_trace(go.Scattergl(
            x=signals, y=returns,
            mode="markers",
            marker=dict(size=3, color=returns, colorscale="RdYlGn",
                        showscale=True,
                        colorbar=dict(title="Return", x=1.12, thickness=12)),
            name="Observations",
            hovertemplate="Signal: %{x:.4f}<br>Return: %{y:.5f}<extra></extra>",
        ), row=1, col=2)

        fig.add_vline(x=results.long_threshold, line_dash="dash",
                      line_color="green", row=1, col=2)
        fig.add_vline(x=results.short_threshold, line_dash="dash",
                      line_color="red", row=1, col=2)
        fig.add_hline(y=0, line_dash="dot", line_color="black", row=1, col=2)

        # ---- Bottom-left: decile bar chart ---- #
        q_edges = np.percentile(signals, np.linspace(0, 100, 11))
        q_labels = np.clip(np.digitize(signals, q_edges[:-1]) - 1, 0, 9)
        q_means = [float(returns[q_labels == q].mean()) for q in range(10)]
        q_colors = ["green" if v >= 0 else "red" for v in q_means]

        fig.add_trace(go.Bar(
            x=list(range(1, 11)), y=q_means,
            marker_color=q_colors,
            name="Mean return by decile",
            hovertemplate="Decile %{x}<br>Mean return: %{y:.5f}<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="black", row=2, col=1)

        # ---- Bottom-right: cumulative equity ---- #
        idx = np.arange(n)
        long_mask = signals <= results.long_threshold
        short_mask = signals > results.short_threshold

        cum_long = np.exp(np.cumsum(np.where(long_mask, returns, 0))) - 1
        cum_short = np.exp(np.cumsum(np.where(short_mask, -returns, 0))) - 1
        cum_bh = np.exp(np.cumsum(returns)) - 1

        fig.add_trace(go.Scatter(
            x=idx, y=cum_long * 100, mode="lines",
            name=f"Long  PF={results.long_pf:.2f}",
            line=dict(color="green", width=2),
            hovertemplate="Bar %{x}<br>Cum. return: %{y:.2f}%<extra></extra>",
        ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=idx, y=cum_short * 100, mode="lines",
            name=f"Short PF={results.short_pf:.2f}",
            line=dict(color="red", width=2),
            hovertemplate="Bar %{x}<br>Cum. return: %{y:.2f}%<extra></extra>",
        ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=idx, y=cum_bh * 100, mode="lines",
            name="Buy & Hold",
            line=dict(color="grey", width=2, dash="dash"),
            hovertemplate="Bar %{x}<br>Cum. return: %{y:.2f}%<extra></extra>",
        ), row=2, col=2)

        fig.add_hline(y=0, line_dash="dot", line_color="black", row=2, col=2)

        # ---- Axis labels ---- #
        fig.update_xaxes(title_text="Threshold", row=1, col=1)
        fig.update_yaxes(title_text="Profit factor", row=1, col=1)
        fig.update_xaxes(title_text="Signal value", row=1, col=2)
        fig.update_yaxes(title_text="Forward return", row=1, col=2)
        fig.update_xaxes(title_text="Decile", row=2, col=1)
        fig.update_yaxes(title_text="Mean return", row=2, col=1)
        fig.update_xaxes(title_text="Bar index", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative return (%)", row=2, col=2)

        sig = "***" if results.best_pval < 0.01 else ("**" if results.best_pval < 0.05 else "")
        fig.update_layout(
            height=750,
            title_text=(
                f"ROC Analysis Dashboard — "
                f"best PF: {max(results.long_pf, results.short_pf):.3f}   "
                f"p={results.best_pval:.4f}{sig}"
            ),
            hovermode="closest",
            showlegend=True,
        )

        return fig

    # ------------------------------------------------------------------ #
    # Dash server                                                          #
    # ------------------------------------------------------------------ #

    def launch(self, port: int = 8050, debug: bool = False) -> None:
        """
        Start a local Dash server for interactive ROC exploration.

        Open a browser at ``http://localhost:{port}`` after calling this.

        Parameters
        ----------
        port : int, default 8050
        debug : bool, default False
        """
        try:
            from dash import Dash, dcc, html
        except ImportError as exc:
            raise ImportError(
                "Dash is required to launch the interactive dashboard. "
                "Install it with:  pip install dash"
            ) from exc

        app = Dash(__name__)
        fig = self.create_figure()

        sig_color = "green" if self.results.best_pval < 0.05 else "red"
        app.layout = html.Div([
            html.H2(
                "Interactive ROC Analysis Dashboard",
                style={"textAlign": "center", "fontFamily": "sans-serif"},
            ),
            html.P(
                f"Best p-value: {self.results.best_pval:.4f}  "
                f"({'significant' if self.results.best_pval < 0.05 else 'not significant'})",
                style={"textAlign": "center", "color": sig_color,
                       "fontFamily": "sans-serif", "fontSize": "16px"},
            ),
            dcc.Graph(id="roc-graph", figure=fig, style={"height": "780px"}),
        ])

        print(f"Dashboard running at http://localhost:{port}/")
        app.run_server(debug=debug, port=port)
