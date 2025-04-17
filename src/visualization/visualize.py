"""
Visualization module for the customer segmentation project.
This module contains functions for creating visualizations for analysis and the dashboard.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any


def plot_rfm_distribution(
    df: pd.DataFrame, save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create distribution plots for RFM metrics.

    Args:
        df (pd.DataFrame): The dataframe with RFM metrics
        save_path (str, optional): Path to save the figure

    Returns:
        Dict[str, Any]: Dictionary containing the figure objects
    """
    # Set up the figure
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Recency Distribution",
            "Frequency Distribution",
            "Monetary Value Distribution",
        ),
    )

    # Add histogram traces
    fig.add_trace(
        go.Histogram(x=df["Recency"], name="Recency", marker_color="#636EFA"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(x=df["Frequency"], name="Frequency", marker_color="#EF553B"),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Histogram(
            x=df["MonetaryValue"], name="Monetary Value", marker_color="#00CC96"
        ),
        row=1,
        col=3,
    )

    # Update layout
    fig.update_layout(
        title_text="RFM Metrics Distribution", showlegend=False, height=500, width=1000
    )

    # Save the figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
        print(f"RFM distribution plot saved to {save_path}")

    return {"fig": fig}


def plot_segment_distribution(
    df: pd.DataFrame, segment_col: str = "SegmentName", save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a pie chart showing the distribution of customer segments.

    Args:
        df (pd.DataFrame): The dataframe with customer segments
        segment_col (str): Name of the column containing segment labels
        save_path (str, optional): Path to save the figure

    Returns:
        Dict[str, Any]: Dictionary containing the figure objects
    """
    # Count the number of customers in each segment
    segment_counts = df[segment_col].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]
    segment_counts["Percentage"] = (
        segment_counts["Count"] / segment_counts["Count"].sum() * 100
    )

    # Create the pie chart
    fig = px.pie(
        segment_counts,
        names="Segment",
        values="Count",
        title="Customer Segment Distribution",
        hover_data=["Percentage"],
        labels={"Count": "Number of Customers"},
        color_discrete_sequence=px.colors.qualitative.G10,
    )

    # Update layout
    fig.update_layout(legend_title="Segments", height=600, width=800)

    # Save the figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
        print(f"Segment distribution plot saved to {save_path}")

    return {"fig": fig, "data": segment_counts}


def plot_cluster_3d(
    df: pd.DataFrame,
    features: List[str] = ["Recency", "Frequency", "MonetaryValue"],
    cluster_col: str = "Cluster",
    segment_col: Optional[str] = "SegmentName",
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create an interactive 3D scatter plot of the clusters.

    Args:
        df (pd.DataFrame): The dataframe with cluster labels
        features (List[str]): The three features to use for 3D visualization
        cluster_col (str): Name of the column containing cluster labels
        segment_col (str, optional): Name of the column containing segment names
        save_path (str, optional): Path to save the figure

    Returns:
        Dict[str, Any]: Dictionary containing the figure objects
    """
    # Ensure we have exactly 3 features for 3D visualization
    if len(features) != 3:
        raise ValueError("Exactly 3 features must be provided for 3D visualization")

    # Prepare data for plotting
    if segment_col and segment_col in df.columns:
        hover_data = [segment_col, cluster_col] + features
        color_col = segment_col
    else:
        hover_data = [cluster_col] + features
        color_col = cluster_col

    # Create the 3D scatter plot
    fig = px.scatter_3d(
        df,
        x=features[0],
        y=features[1],
        z=features[2],
        color=color_col,
        hover_name=df.index if df.index.name is not None else None,
        hover_data=hover_data,
        title="3D Cluster Visualization",
        labels={
            features[0]: features[0],
            features[1]: features[1],
            features[2]: features[2],
        },
    )

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title=features[0], yaxis_title=features[1], zaxis_title=features[2]
        ),
        height=800,
        width=1000,
    )

    # Save the figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
        print(f"3D cluster visualization saved to {save_path}")

    return {"fig": fig}


def plot_segment_radar(
    df: pd.DataFrame,
    segment_col: str = "SegmentName",
    features: List[str] = ["Recency", "Frequency", "MonetaryValue"],
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a radar chart comparing the average values for each segment.

    Args:
        df (pd.DataFrame): The dataframe with customer segments
        segment_col (str): Name of the column containing segment labels
        features (List[str]): Features to include in the radar chart
        save_path (str, optional): Path to save the figure

    Returns:
        Dict[str, Any]: Dictionary containing the figure objects
    """
    # Calculate the mean values for each segment and feature
    segment_means = df.groupby(segment_col)[features].mean().reset_index()

    # Normalize the values to a 0-1 scale for better visualization
    for feature in features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        segment_means[f"{feature}_norm"] = (segment_means[feature] - min_val) / (
            max_val - min_val
        )

    # For recency, invert the normalized values (lower recency is better)
    if "Recency_norm" in segment_means.columns:
        segment_means["Recency_norm"] = 1 - segment_means["Recency_norm"]

    # Create the radar chart
    fig = go.Figure()

    # Add a trace for each segment
    for segment in segment_means[segment_col]:
        segment_data = segment_means[segment_means[segment_col] == segment]

        # Prepare the data for the radar chart
        r_values = [segment_data[f"{feature}_norm"].values[0] for feature in features]
        theta_values = features

        # Add the trace
        fig.add_trace(
            go.Scatterpolar(r=r_values, theta=theta_values, fill="toself", name=segment)
        )

    # Update layout
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Segment Comparison (Normalized Values)",
        showlegend=True,
        height=600,
        width=800,
    )

    # Save the figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
        print(f"Segment radar chart saved to {save_path}")

    return {"fig": fig, "data": segment_means}


def plot_segment_boxplots(
    df: pd.DataFrame,
    segment_col: str = "SegmentName",
    features: List[str] = ["Recency", "Frequency", "MonetaryValue"],
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create box plots comparing the distribution of features across segments.

    Args:
        df (pd.DataFrame): The dataframe with customer segments
        segment_col (str): Name of the column containing segment labels
        features (List[str]): Features to include in the box plots
        save_path (str, optional): Path to save the figure

    Returns:
        Dict[str, Any]: Dictionary containing the figure objects
    """
    # Create subplots
    fig = make_subplots(
        rows=len(features),
        cols=1,
        subplot_titles=[f"{feature} by Segment" for feature in features],
        vertical_spacing=0.05,
    )

    # Add a box plot for each feature
    for i, feature in enumerate(features):
        box_fig = px.box(df, x=segment_col, y=feature, color=segment_col)

        # Add the trace to the subplot
        for trace in box_fig.data:
            fig.add_trace(trace, row=i + 1, col=1)

    # Update layout
    fig.update_layout(
        title="Feature Distribution by Segment",
        showlegend=False,
        height=300 * len(features),
        width=1000,
        boxmode="group",
    )

    # Save the figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
        print(f"Segment box plots saved to {save_path}")

    return {"fig": fig}


def plot_segment_heatmap(
    df: pd.DataFrame,
    segment_col: str = "SegmentName",
    features: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a heatmap showing the average values of features for each segment.

    Args:
        df (pd.DataFrame): The dataframe with customer segments
        segment_col (str): Name of the column containing segment labels
        features (List[str], optional): Features to include in the heatmap
        save_path (str, optional): Path to save the figure

    Returns:
        Dict[str, Any]: Dictionary containing the figure objects
    """
    # If features is not provided, use all numeric columns except the segment column
    if features is None:
        features = df.select_dtypes(include=["number"]).columns.tolist()
        if segment_col in features:
            features.remove(segment_col)
        # Remove any cluster columns
        features = [f for f in features if "Cluster" not in f]

    # Calculate the mean values for each segment and feature
    segment_means = df.groupby(segment_col)[features].mean()

    # Create a copy of the data for normalization
    normalized_means = segment_means.copy()

    # Normalize the values for better visualization
    for feature in features:
        min_val = normalized_means[feature].min()
        max_val = normalized_means[feature].max()
        # Avoid division by zero
        if max_val > min_val:
            normalized_means[feature] = (normalized_means[feature] - min_val) / (
                max_val - min_val
            )
        else:
            normalized_means[feature] = 0

    # Create the heatmap
    fig = px.imshow(
        normalized_means.T,
        labels=dict(x=segment_col, y="Feature", color="Normalized Value"),
        x=normalized_means.index,
        y=features,
        title="Segment Profiles Heatmap (Normalized Values)",
        color_continuous_scale="viridis",
    )

    # Update layout
    fig.update_layout(height=600, width=800)

    # Add annotations with the actual values
    for i, segment in enumerate(segment_means.index):
        for j, feature in enumerate(features):
            fig.add_annotation(
                x=i,
                y=j,
                text=f"{segment_means.loc[segment, feature]:.2f}",
                showarrow=False,
                font=dict(color="white"),
            )

    # Save the figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
        print(f"Segment heatmap saved to {save_path}")

    return {"fig": fig, "data": segment_means}


def plot_customer_lifetime_value(
    df: pd.DataFrame,
    segment_col: str = "SegmentName",
    clv_col: str = "MonetaryValue",
    top_n: int = 10,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a bar chart showing the average CLV by segment and a table of the top N customers by CLV.

    Args:
        df (pd.DataFrame): The dataframe with customer segments and CLV
        segment_col (str): Name of the column containing segment labels
        clv_col (str): Name of the column containing CLV or monetary value
        top_n (int): Number of top customers to include in the table
        save_path (str, optional): Path to save the figure

    Returns:
        Dict[str, Any]: Dictionary containing the figure objects
    """
    # Calculate the average CLV by segment
    avg_clv_by_segment = df.groupby(segment_col)[clv_col].mean().reset_index()
    avg_clv_by_segment = avg_clv_by_segment.sort_values(by=clv_col, ascending=False)

    # Create the bar chart
    fig1 = px.bar(
        avg_clv_by_segment,
        x=segment_col,
        y=clv_col,
        color=segment_col,
        title=f"Average {clv_col} by Segment",
        labels={clv_col: f"Average {clv_col}"},
    )

    # Get the top N customers by CLV
    top_customers = df.sort_values(by=clv_col, ascending=False).head(top_n)

    # Create the table
    fig2 = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "Customer ID",
                        segment_col,
                        clv_col,
                        "Recency",
                        "Frequency",
                    ],
                    fill_color="paleturquoise",
                    align="left",
                ),
                cells=dict(
                    values=[
                        top_customers["Customer ID"],
                        top_customers[segment_col],
                        top_customers[clv_col],
                        top_customers["Recency"],
                        top_customers["Frequency"],
                    ],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )

    fig2.update_layout(title=f"Top {top_n} Customers by {clv_col}")

    # Save the figures if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig1.write_image(f"{os.path.splitext(save_path)[0]}_bar.png")
        fig2.write_image(f"{os.path.splitext(save_path)[0]}_table.png")
        print(f"CLV visualizations saved to {os.path.dirname(save_path)}")

    return {
        "bar_fig": fig1,
        "table_fig": fig2,
        "avg_clv": avg_clv_by_segment,
        "top_customers": top_customers,
    }


if __name__ == "__main__":
    # Example usage
    try:
        # Load the customer segments data
        input_path = os.path.join("data", "processed", "customer_segments.csv")
        if not os.path.exists(input_path):
            print(f"Error: File not found - {input_path}")
            exit(1)

        df = pd.read_csv(input_path)
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns.")

        # Create output directory for figures
        figures_dir = os.path.join("reports", "figures")
        os.makedirs(figures_dir, exist_ok=True)

        # Generate visualizations
        print("Generating visualizations...")

        # RFM distribution
        plot_rfm_distribution(
            df, save_path=os.path.join(figures_dir, "rfm_distribution.png")
        )

        # Segment distribution
        plot_segment_distribution(
            df, save_path=os.path.join(figures_dir, "segment_distribution.png")
        )

        # 3D cluster visualization
        plot_cluster_3d(df, save_path=os.path.join(figures_dir, "cluster_3d.png"))

        # Segment radar chart
        plot_segment_radar(df, save_path=os.path.join(figures_dir, "segment_radar.png"))

        # Segment box plots
        plot_segment_boxplots(
            df, save_path=os.path.join(figures_dir, "segment_boxplots.png")
        )

        # Segment heatmap
        plot_segment_heatmap(
            df, save_path=os.path.join(figures_dir, "segment_heatmap.png")
        )

        # Customer lifetime value
        plot_customer_lifetime_value(
            df, save_path=os.path.join(figures_dir, "customer_lifetime_value.png")
        )

        print("All visualizations generated successfully!")

    except Exception as e:
        import traceback

        print(f"Error generating visualizations: {e}")
        print(traceback.format_exc())
