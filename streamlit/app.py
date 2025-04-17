"""
Main Streamlit application for the customer segmentation project.
This is the entry point for the Streamlit dashboard.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add the project root to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.visualization.visualize import (
    plot_rfm_distribution,
    plot_segment_distribution,
    plot_cluster_3d,
    plot_segment_radar,
    plot_segment_boxplots,
    plot_segment_heatmap,
    plot_customer_lifetime_value,
)


# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Sidebar for navigation
def create_sidebar():
    with st.sidebar:
        st.title("📊 Customer Segmentation")
        st.markdown("---")

        # Upload options
        st.subheader("Load Data")
        data_option = st.radio(
            "Choose data source:", ["Use Sample Data", "Upload Custom Data"], index=0
        )

        uploaded_file = None
        if data_option == "Upload Custom Data":
            uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

        st.markdown("---")

        # Filtering options
        st.subheader("Filter Data")

        if "df" in st.session_state and st.session_state.df is not None:
            df = st.session_state.df

            # Segment filter
            if "SegmentName" in df.columns:
                segments = df["SegmentName"].unique().tolist()
                selected_segments = st.multiselect(
                    "Select segments to display:", segments, default=segments
                )

                # Store the filter in session state
                st.session_state.selected_segments = selected_segments

            # Recency filter
            if "Recency" in df.columns:
                min_recency, max_recency = (
                    int(df["Recency"].min()),
                    int(df["Recency"].max()),
                )
                recency_range = st.slider(
                    "Recency range (days):",
                    min_value=min_recency,
                    max_value=max_recency,
                    value=(min_recency, max_recency),
                )

                st.session_state.recency_range = recency_range

        st.markdown("---")

        # About section
        st.subheader("About")
        st.markdown("""
        This dashboard visualizes customer segments based on RFM (Recency, Frequency, Monetary value) analysis.
        
        Navigate through the pages to explore different aspects of the segmentation.
        """)

        st.markdown("---")
        st.markdown("Developed by: Abdiwahid Hussein Ali")

    return uploaded_file


# Load the data
@st.cache_data
def load_data(file_path=None):
    if file_path is None:
        # Use the sample data
        file_path = os.path.join("data", "processed", "customer_segments.csv")

    try:
        df = pd.read_csv(file_path)
        # Debug print to verify data is loaded correctly
        print(
            f"Data loaded successfully: {len(df)} rows, columns: {df.columns.tolist()}"
        )
        print(
            f"MonetaryValue stats: min={df['MonetaryValue'].min()}, max={df['MonetaryValue'].max()}, mean={df['MonetaryValue'].mean()}"
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Apply filters to the data
def filter_data(df):
    if df is None:
        return None

    filtered_df = df.copy()

    # Apply segment filter
    if "selected_segments" in st.session_state and "SegmentName" in filtered_df.columns:
        if st.session_state.selected_segments:
            filtered_df = filtered_df[
                filtered_df["SegmentName"].isin(st.session_state.selected_segments)
            ]

    # Apply recency filter
    if "recency_range" in st.session_state and "Recency" in filtered_df.columns:
        min_recency, max_recency = st.session_state.recency_range
        filtered_df = filtered_df[
            (filtered_df["Recency"] >= min_recency)
            & (filtered_df["Recency"] <= max_recency)
        ]

    return filtered_df


# Main dashboard layout
def dashboard():
    # Display header
    st.title("📊 Customer Segmentation Dashboard")
    st.markdown("---")

    # Check if data is loaded
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("No data loaded. Please upload a CSV file or use the sample data.")
        return

    # Get the filtered data
    df = filter_data(st.session_state.df)

    if df is None or df.empty:
        st.warning("No data available after applying filters.")
        return

    # Display key metrics in the header
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{len(df):,}")

    with col2:
        # Ensure the MonetaryValue column exists and has valid data
        if "MonetaryValue" in df.columns:
            avg_monetary = df["MonetaryValue"].mean()
            st.metric("Avg. Monetary Value", f"£{avg_monetary:.2f}")
        else:
            st.metric("Avg. Monetary Value", "N/A")
            st.caption("MonetaryValue column not found")

    with col3:
        # Ensure the Frequency column exists and has valid data
        if "Frequency" in df.columns:
            avg_frequency = df["Frequency"].mean()
            st.metric("Avg. Purchase Frequency", f"{avg_frequency:.2f}")
        else:
            st.metric("Avg. Purchase Frequency", "N/A")
            st.caption("Frequency column not found")

    with col4:
        # Ensure the Recency column exists and has valid data
        if "Recency" in df.columns:
            avg_recency = df["Recency"].mean()
            st.metric("Avg. Recency (days)", f"{avg_recency:.0f}")
        else:
            st.metric("Avg. Recency (days)", "N/A")
            st.caption("Recency column not found")

    st.markdown("---")

    # Main content area - split into two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Customer Segment Distribution")
        if "SegmentName" in df.columns:
            fig_data = plot_segment_distribution(df)
            st.plotly_chart(fig_data["fig"], use_container_width=True)
        else:
            st.warning(
                "SegmentName column not found. Cannot display segment distribution."
            )

    with col2:
        st.subheader("Segment Profiles")

        # Table with segment statistics
        if "SegmentName" in df.columns:
            segment_stats = (
                df.groupby("SegmentName")
                .agg(
                    {
                        "Recency": "mean",
                        "Frequency": "mean",
                        "MonetaryValue": "mean",
                        "Customer ID": "count",
                    }
                )
                .reset_index()
            )

            segment_stats.rename(columns={"Customer ID": "Count"}, inplace=True)

            # Add percentage column
            segment_stats["Percentage"] = (
                segment_stats["Count"] / segment_stats["Count"].sum() * 100
            )

            # Format the table
            formatted_stats = segment_stats.copy()
            formatted_stats["Recency"] = formatted_stats["Recency"].round(1)
            formatted_stats["Frequency"] = formatted_stats["Frequency"].round(1)
            formatted_stats["MonetaryValue"] = formatted_stats["MonetaryValue"].round(2)
            formatted_stats["Percentage"] = (
                formatted_stats["Percentage"].round(1).astype(str) + "%"
            )

            st.dataframe(
                formatted_stats,
                column_config={
                    "SegmentName": "Segment",
                    "Recency": "Avg. Recency (days)",
                    "Frequency": "Avg. Frequency",
                    "MonetaryValue": st.column_config.NumberColumn(
                        "Avg. Monetary Value", format="£%.2f"
                    ),
                    "Count": "Customers",
                    "Percentage": "% of Total",
                },
                use_container_width=True,
            )
        else:
            st.warning("SegmentName column not found. Cannot display segment profiles.")

    st.markdown("---")

    # Check if we have the required RFM columns before attempting to display visualizations
    required_cols = ["Recency", "Frequency", "MonetaryValue"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.warning(
            f"Missing required columns: {', '.join(missing_cols)}. Some visualizations will not be available."
        )
    else:
        # RFM Distribution
        st.subheader("RFM Metrics Distribution")

        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(
            ["Distribution", "3D Visualization", "Segment Comparison"]
        )

        with tab1:
            # RFM distribution
            rfm_data = plot_rfm_distribution(df)
            st.plotly_chart(rfm_data["fig"], use_container_width=True)

        with tab2:
            # 3D cluster visualization
            cluster_data = plot_cluster_3d(df)
            st.plotly_chart(cluster_data["fig"], use_container_width=True)

        with tab3:
            # Segment radar chart
            if "SegmentName" in df.columns:
                radar_data = plot_segment_radar(df)
                st.plotly_chart(radar_data["fig"], use_container_width=True)
            else:
                st.warning(
                    "SegmentName column not found. Cannot display segment comparison."
                )

        st.markdown("---")

        # Customer Value Analysis
        st.subheader("Customer Lifetime Value Analysis")

        clv_data = plot_customer_lifetime_value(df)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.plotly_chart(clv_data["bar_fig"], use_container_width=True)

        with col2:
            st.plotly_chart(clv_data["table_fig"], use_container_width=True)

        st.markdown("---")

        # Segment Details
        st.subheader("Segment Details")

        # Feature box plots by segment
        if "SegmentName" in df.columns:
            boxplot_data = plot_segment_boxplots(df)
            st.plotly_chart(boxplot_data["fig"], use_container_width=True)

            # Segment heatmap
            heatmap_data = plot_segment_heatmap(df)
            st.plotly_chart(heatmap_data["fig"], use_container_width=True)
        else:
            st.warning("SegmentName column not found. Cannot display segment details.")


# Main function
def main():
    # Create the sidebar and get the uploaded file
    uploaded_file = create_sidebar()

    # Load the data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
    elif "df" not in st.session_state:
        st.session_state.df = load_data()

    # Display the dashboard
    dashboard()


if __name__ == "__main__":
    main()
