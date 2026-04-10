import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_price_with_regression_prediction(df, prediction_col, price_col="close", quantile_thresholds=None, title=None, colorscale=None):
    """
    Plot actual prices with regression prediction line.

    If quantile_thresholds is None: plots prediction as a simple line.
    If quantile_thresholds is provided: colors prediction line segments by qcut classes.

    Parameters:
    - df: DataFrame with prices and predictions
    - prediction_col: column with continuous regression predictions
    - price_col: column with actual prices
    - quantile_thresholds: optional list of quantiles for qcut (e.g., [0.2, 0.4, 0.6, 0.8])
                          Creates len(thresholds)+1 classes. If None, plots simple line.
    - title: optional plot title
    - colorscale: optional plotly colorscale name (e.g., 'RdYlGn', 'Viridis', 'Plasma')
                  Default uses Set1 discrete colors
    """
    if prediction_col not in df.columns:
        raise ValueError(f"{prediction_col} not found in DataFrame.")

    title = title or f"Price with Regression Prediction: {prediction_col}"

    fig = go.Figure()

    # Add actual closing price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[price_col],
            mode="lines",
            line=dict(color="black", width=1),
            name="Actual Price",
        )
    )

    # Plot prediction line
    if quantile_thresholds is None:
        # Simple prediction line without classification
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[price_col],
                mode="lines",
                line=dict(color="cyan", width=2, dash="dash"),
                name=f"{prediction_col}",
                hovertemplate=(
                    f"<b>{prediction_col}</b><br>"
                    f"Time: %{x}<br>"
                    f"Price: %{y:.2f}<br>"
                    f"Prediction: {df[prediction_col]:.4f}<br>"
                    "<extra></extra>"
                ),
            )
        )
    else:
        # Bin predictions using qcut and color by class
        df_copy = df.copy()
        df_copy["pred_class"] = pd.qcut(
            df_copy[prediction_col], q=quantile_thresholds, labels=False, duplicates="drop"
        )

        # Get unique classes for coloring
        unique_classes = sorted(df_copy["pred_class"].dropna().unique())
        n_classes = len(unique_classes)

        # Select colorscale
        if colorscale:
            colors = px.colors.sample_colorscale(colorscale, n_classes)
        else:
            discrete_colors = px.colors.qualitative.Set1
            colors = [discrete_colors[i % len(discrete_colors)] for i in range(n_classes)]

        # Add line segments for each class
        for i, class_id in enumerate(unique_classes):
            class_data = df_copy[df_copy["pred_class"] == class_id]

            fig.add_trace(
                go.Scatter(
                    x=class_data.index,
                    y=class_data[price_col],
                    mode="lines+markers",
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=4, color=colors[i]),
                    name=f"Class {int(class_id)}",
                    legendgroup=f"class_{i}",
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{prediction_col}</b><br>"
                        f"Class: {int(class_id)}<br>"
                        f"Time: %{x}<br>"
                        f"Price: %{y:.2f}<br>"
                        f"Prediction: {class_data[prediction_col]:.4f}<br>"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        hovermode="x unified",
        autosize=True,
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50),
    )
    fig.show()


def plot_price_multiple_regression_predictions_with_subplots(
        df, prediction_cols, price_col="close", quantile_thresholds=None, title=None, colorscale=None
):
    """
    Plot actual prices with multiple regression predictions in subplots.

    Similar to plot_price_multiple_labels_with_subplots but for continuous predictions.

    Parameters:
    - df: DataFrame with prices and predictions
    - prediction_cols: list of columns with continuous regression predictions
    - price_col: column with actual prices
    - quantile_thresholds: optional list of quantiles for qcut (e.g., [0.2, 0.4, 0.6, 0.8])
                          If None, plots simple lines. Applied to all predictions.
    - title: optional plot title
    - colorscale: optional plotly colorscale name for all subplots
    """
    for pred_col in prediction_cols:
        if pred_col not in df.columns:
            raise ValueError(f"{pred_col} not found in DataFrame.")

    n_plots = len(prediction_cols)
    n_rows = n_plots
    n_cols = 1

    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{col}" for col in prediction_cols],
        vertical_spacing=0.03,
        horizontal_spacing=0.05,
        shared_xaxes=True,
    )

    # Plot each prediction
    for i, pred_col in enumerate(prediction_cols):
        row = i + 1
        col = 1

        # Add actual price line (gray background)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[price_col],
                mode="lines",
                line=dict(color="#666", width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Plot prediction line
        if quantile_thresholds is None:
            # Simple prediction line
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[price_col],
                    mode="lines",
                    line=dict(color="cyan", width=2, dash="dash"),
                    name=f"{pred_col}",
                    legendgroup=f"pred_{i}",
                    showlegend=True,
                    legend=f"legend{row}",
                    hovertemplate=(
                        f"<b>{pred_col}</b><br>"
                        f"Time: %{x}<br>"
                        f"Price: %{y:.2f}<br>"
                        f"Prediction: {df[pred_col]:.4f}<br>"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        else:
            # Bin predictions and color by class
            df_copy = df.copy()
            df_copy["pred_class"] = pd.qcut(
                df_copy[pred_col], q=quantile_thresholds, labels=False, duplicates="drop"
            )

            unique_classes = sorted(df_copy["pred_class"].dropna().unique())
            n_classes = len(unique_classes)

            # Select colorscale
            if colorscale:
                colors = px.colors.sample_colorscale(colorscale, n_classes)
            else:
                discrete_colors = px.colors.qualitative.Set1
                colors = [discrete_colors[j % len(discrete_colors)] for j in range(n_classes)]

            # Add line segments for each class
            for j, class_id in enumerate(unique_classes):
                class_data = df_copy[df_copy["pred_class"] == class_id]

                fig.add_trace(
                    go.Scatter(
                        x=class_data.index,
                        y=class_data[price_col],
                        mode="lines+markers",
                        line=dict(color=colors[j], width=2),
                        marker=dict(size=3, color=colors[j]),
                        name=f"Class {int(class_id)}",
                        legendgroup=f"{pred_col}_class_{j}",
                        showlegend=True,
                        legend=f"legend{row}",
                        hovertemplate=(
                            f"<b>{pred_col}</b><br>"
                            f"Class: {int(class_id)}<br>"
                            f"Time: %{x}<br>"
                            f"Price: %{y:.2f}<br>"
                            f"Prediction: {class_data[pred_col]:.4f}<br>"
                            "<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

        fig.update_yaxes(title_text="Price", row=row, col=col)

    # Update layout
    layout_update = dict(
        title=title or f"Regression Predictions: {', '.join(prediction_cols)}",
        template="plotly_dark",
        height=250 * n_rows,
        autosize=True,
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    # Configure legend positioning for each subplot
    for i in range(n_rows):
        legend_name = f"legend{i + 1}" if i > 0 else "legend1"
        y_position = 1 - (i / n_rows) - 0.02
        y_anchor = "top"

        layout_update[legend_name] = dict(
            x=1.02,
            y=y_position,
            xanchor="left",
            yanchor=y_anchor,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1,
        )

    fig.update_layout(**layout_update)
    fig.show()
