import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_price_with_labels(df, label_col, price_col="close", title=None):
    """
    Interactive Plotly subplot scatter plot of price over time colored by cluster IDs.

    Parameters:
    - df: DataFrame with clustering results
    - label_col: column name of label
    - price_col: column with close prices
    - title: optional plot title
    """

    title = title or "Price over time"

    if label_col not in df.columns:
        raise ValueError(f"{label_col} not found in DataFrame.")

    fig = go.Figure()

    # closing price
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[price_col],
            mode="lines",
            line=dict(color="black", width=1),
            name="Closing Price",
        )
    )

    # Get unique clusters for consistent coloring
    unique_clusters = sorted(df[label_col].unique())
    discrete_colors = px.colors.qualitative.Set1

    # Add scatter traces for each cluster
    for j, cluster_id in enumerate(unique_clusters):
        cluster_data = df[df[label_col] == cluster_id]
        color_index = j % len(discrete_colors)
        color = discrete_colors[color_index]

        fig.add_trace(
            go.Scatter(
                x=cluster_data.index,
                y=cluster_data[price_col],
                mode="markers",
                marker=dict(size=5, opacity=0.8, color=color),
                name=f"{label_col}_C{cluster_id}",
                legendgroup=f"{label_col}_{j}",
                showlegend=True,
                hovertemplate=(
                    f"<b>{label_col}</b><br>"
                    f"Cluster: {cluster_id}<br>"
                    f"Time: %{{x}}<br>"
                    f"Price: %{{y:.2f}}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Closing Price",
        template="plotly_dark",
        hovermode="x unified",
        autosize=True,
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50),
    )
    fig.show()


def plot_price_multiple_labels_with_subplots(df, labels, price_col="close", title=None):
    """
    Interactive Plotly subplot scatter plots of close price over time,
    colored by cluster IDs for multiple algorithms.

    Parameters:
    - df: DataFrame with clustering results
    - algos: list of columns with cluster trend
    - price_col: column with close prices
    - title: optional plot title
    """

    for label in labels:
        if label not in df.columns:
            raise ValueError(f"{label} not found in DataFrame.")

    n_plots = len(labels)
    n_cols = 1
    n_rows = n_plots

    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{algo}" for algo in labels],
        vertical_spacing=0.03,
        horizontal_spacing=0.05,
        shared_xaxes=True,  # Share x-axis across all subplots
    )

    # Plot each algorithm starting from row=1
    for i, label in enumerate(labels):
        cluster_col = label
        row = i + 1
        col = 1

        # Add continuous line trace for the entire closing price series
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[price_col],
                mode="lines",
                line=dict(color="#ccc", width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Get unique clusters for consistent coloring
        unique_clusters = sorted(df[cluster_col].unique())
        discrete_colors = px.colors.qualitative.Set1

        # Add scatter traces for each cluster
        for j, cluster_id in enumerate(unique_clusters):
            cluster_data = df[df[cluster_col] == cluster_id]
            color_index = j % len(discrete_colors)
            color = discrete_colors[color_index]

            # Add marker trace for cluster points
            fig.add_trace(
                go.Scatter(
                    x=cluster_data.index,
                    y=cluster_data[price_col],
                    mode="markers",
                    marker=dict(size=5, opacity=0.8, color=color),
                    name=f"{cluster_col}_C{cluster_id}",
                    legendgroup=f"{cluster_col}_{j}",
                    showlegend=True,
                    legend=f"legend{row}",
                    hovertemplate=(
                        f"<b>{cluster_col}</b><br>"
                        f"Cluster: {cluster_id}<br>"
                        f"Time: %{{x}}<br>"
                        f"Price: %{{y:.2f}}<br>"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        fig.update_yaxes(title_text="Close Price", row=row, col=col)

    # Update layout for full width and better sizing
    layout_update = dict(
        title=title or f"Cluster Visualizations: {', '.join(labels)}",
        template="plotly_dark",
        height=250 * n_rows,  # Increased height per subplot
        autosize=True,  # Full width of page
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    # Configure individual legend positioning for each subplot
    for i in range(n_rows):
        legend_name = f"legend{i + 1}" if i > 0 else "legend1"
        y_position = 1 - (i / n_rows) - 0.02  # Position legend at top of each subplot
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
