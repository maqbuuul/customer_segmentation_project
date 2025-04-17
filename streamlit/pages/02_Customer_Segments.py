"""
Streamlit page for detailed customer segment analysis.
This page provides in-depth information about each customer segment.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

# Add the project root to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.visualization.visualize import (
    plot_segment_radar,
    plot_segment_boxplots,
    plot_segment_heatmap,
)


# Page configuration
st.set_page_config(
    page_title="Customer Segments Analysis",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load the data
@st.cache_data
def load_data():
    file_path = os.path.join("data", "processed", "customer_segments.csv")

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Apply filters to the data
def filter_data(df):
    filtered_df = df.copy()

    # Apply segment filter
    if "selected_segment" in st.session_state and "SegmentName" in filtered_df.columns:
        if st.session_state.selected_segment != "All Segments":
            filtered_df = filtered_df[
                filtered_df["SegmentName"] == st.session_state.selected_segment
            ]

    return filtered_df


# Create segment profile card
def segment_profile_card(segment_name, segment_df):
    # Calculate segment metrics
    avg_recency = segment_df["Recency"].mean()
    avg_frequency = segment_df["Frequency"].mean()
    avg_monetary = segment_df["MonetaryValue"].mean()
    segment_size = len(segment_df)
    segment_pct = segment_size / len(st.session_state.df) * 100

    # Create the card
    st.subheader(f"Profile: {segment_name}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Customers", f"{segment_size:,}")

    with col2:
        st.metric("% of Total", f"{segment_pct:.1f}%")

    with col3:
        st.metric("Avg. Recency (days)", f"{avg_recency:.1f}")

    with col4:
        st.metric("Avg. Frequency", f"{avg_frequency:.1f}")

    # Create additional metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg. Monetary Value", f"£{avg_monetary:.2f}")

    with col2:
        if "CLV" in segment_df.columns:
            avg_clv = segment_df["CLV"].mean()
            st.metric("Avg. CLV", f"£{avg_clv:.2f}")
        else:
            total_monetary = segment_df["MonetaryValue"].sum()
            st.metric("Total Revenue", f"£{total_monetary:.2f}")

    with col3:
        avg_monetary_per_transaction = (
            avg_monetary / avg_frequency if avg_frequency > 0 else 0
        )
        st.metric("Avg. Transaction Value", f"£{avg_monetary_per_transaction:.2f}")

    with col4:
        if "EngagementScore" in segment_df.columns:
            avg_engagement = segment_df["EngagementScore"].mean()
            st.metric("Engagement Score", f"{avg_engagement:.2f}")
        else:
            # Calculate a simple engagement score
            recency_score = (
                1 - (avg_recency / segment_df["Recency"].max())
                if segment_df["Recency"].max() > 0
                else 0
            )
            frequency_score = (
                avg_frequency / segment_df["Frequency"].max()
                if segment_df["Frequency"].max() > 0
                else 0
            )
            monetary_score = (
                avg_monetary / segment_df["MonetaryValue"].max()
                if segment_df["MonetaryValue"].max() > 0
                else 0
            )

            engagement_score = (recency_score + frequency_score + monetary_score) / 3
            st.metric("Engagement Score", f"{engagement_score:.2f}")

    # Business description and recommendations
    business_descriptions = {
        "Champions": {
            "description": "Your best customers who buy often and recently, spending the most.",
            "recommendations": [
                "Reward these customers through loyalty programs",
                "Seek their feedback for new products/services",
                "Use them as brand ambassadors",
                "Ensure VIP treatment to maintain their loyalty",
            ],
        },
        "Loyal Customers": {
            "description": "Regular customers who purchase frequently but spend less than Champions.",
            "recommendations": [
                "Encourage them to spend more through upselling",
                "Offer special discounts on premium products",
                "Provide excellent customer service",
                "Seek feedback on improving products/services",
            ],
        },
        "Big Spenders": {
            "description": "Customers who make large purchases but shop less frequently.",
            "recommendations": [
                "Increase purchase frequency with limited-time offers",
                "Create a special loyalty program for big spenders",
                "Personalized communication about new premium products",
                "Exclusive previews and early access to new collections",
            ],
        },
        "Dormant": {
            "description": "Previously active customers who haven't purchased in a long time.",
            "recommendations": [
                "Re-engagement campaigns with special offers",
                "Surveys to understand why they stopped purchasing",
                "New product announcements to spark interest",
                "Win-back promotions with time-limited discounts",
            ],
        },
        "At Risk": {
            "description": "Customers who purchased regularly but haven't returned recently.",
            "recommendations": [
                "Targeted retention campaigns before they become dormant",
                "Personalized offers based on their previous purchases",
                "Request feedback to address potential issues",
                "Remind them of your value proposition",
            ],
        },
        "New Customers": {
            "description": "Recently acquired customers with few transactions.",
            "recommendations": [
                "Welcome series to introduce your brand and products",
                "Educational content about product usage",
                "Incentives for second purchase to establish habit",
                "Collect feedback on their first experience",
            ],
        },
        "Rare Shoppers": {
            "description": "Customers who purchase infrequently and spend little.",
            "recommendations": [
                "Targeted campaigns to increase purchase frequency",
                "Bundle offers to increase average order value",
                "Loyalty program to encourage repeat business",
                "Cross-sell related products based on past purchases",
            ],
        },
        "Average Customers": {
            "description": "Moderately active and moderately spending customers.",
            "recommendations": [
                "Regular engagement to prevent them from becoming inactive",
                "Personalized recommendations to increase spending",
                "Reward milestones to encourage loyalty",
                "Test different marketing approaches with this segment",
            ],
        },
    }

    # Get the business description and recommendations for the segment
    segment_info = business_descriptions.get(
        segment_name,
        {
            "description": "Customer segment with unique purchasing patterns.",
            "recommendations": [
                "Analyze this segment further to understand their behavior",
                "Test different marketing approaches",
                "Monitor changes in their purchasing patterns",
                "Develop targeted offerings based on their preferences",
            ],
        },
    )

    st.markdown("---")
    st.markdown(f"### Segment Description")
    st.markdown(segment_info["description"])

    st.markdown("### Strategic Recommendations")
    for i, rec in enumerate(segment_info["recommendations"], 1):
        st.markdown(f"{i}. {rec}")


# Create segment comparison function
def segment_comparison():
    # Get the full dataset
    df = st.session_state.df

    # Create radar chart for segment comparison
    st.subheader("Segment Comparison")

    radar_data = plot_segment_radar(df)
    st.plotly_chart(radar_data["fig"], use_container_width=True)

    # Create segment feature distribution
    st.subheader("Feature Distribution by Segment")

    boxplot_data = plot_segment_boxplots(df)
    st.plotly_chart(boxplot_data["fig"], use_container_width=True)

    # Create heatmap of segment profiles
    st.subheader("Segment Profiles Heatmap")

    heatmap_data = plot_segment_heatmap(df)
    st.plotly_chart(heatmap_data["fig"], use_container_width=True)

    # Create segment size comparison
    st.subheader("Segment Size Comparison")

    segment_counts = df["SegmentName"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]
    segment_counts["Percentage"] = (
        segment_counts["Count"] / segment_counts["Count"].sum() * 100
    )

    fig = px.bar(
        segment_counts,
        x="Segment",
        y="Count",
        color="Segment",
        text="Percentage",
        labels={"Count": "Number of Customers", "Segment": "Segment Name"},
        title="Customer Distribution by Segment",
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

    st.plotly_chart(fig, use_container_width=True)


# Create customer explorer function
def customer_explorer(segment_df):
    st.subheader("Customer Explorer")

    # Create a searchable table of customers
    search_col, filter_col = st.columns([1, 1])

    with search_col:
        search_id = st.text_input("Search by Customer ID")

    with filter_col:
        sort_by = st.selectbox(
            "Sort by", options=["MonetaryValue", "Frequency", "Recency"], index=0
        )
        ascending = st.checkbox("Ascending order", value=False)

    # Filter and sort the data
    explorer_df = segment_df.copy()

    if search_id:
        explorer_df = explorer_df[
            explorer_df["Customer ID"].astype(str).str.contains(search_id)
        ]

    explorer_df = explorer_df.sort_values(by=sort_by, ascending=ascending)

    # Display the top customers
    if len(explorer_df) > 0:
        # Select columns to display
        display_cols = [
            "Customer ID",
            "SegmentName",
            "Recency",
            "Frequency",
            "MonetaryValue",
        ]

        # Add additional columns if available
        for col in ["CLV", "EngagementScore", "RF_Ratio", "MonetaryDensity"]:
            if col in explorer_df.columns:
                display_cols.append(col)

        # Create the customer table
        st.dataframe(
            explorer_df[display_cols],
            column_config={
                "Customer ID": st.column_config.NumberColumn("Customer ID"),
                "SegmentName": "Segment",
                "Recency": "Recency (days)",
                "Frequency": "Purchase Frequency",
                "MonetaryValue": st.column_config.NumberColumn(
                    "Monetary Value", format="£%.2f"
                ),
                "CLV": st.column_config.NumberColumn(
                    "Customer Lifetime Value", format="£%.2f"
                ),
                "EngagementScore": st.column_config.NumberColumn(
                    "Engagement Score", format="%.2f"
                ),
                "RF_Ratio": st.column_config.NumberColumn(
                    "Recency/Frequency Ratio", format="%.2f"
                ),
                "MonetaryDensity": st.column_config.NumberColumn(
                    "Value per Transaction", format="£%.2f"
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

        st.caption(f"Showing {len(explorer_df)} customers out of {len(segment_df)}")
    else:
        st.warning("No customers found matching the criteria.")


# Main function
def main():
    # Display header
    st.title("👥 Customer Segments Analysis")
    st.markdown("Detailed analysis of customer segments based on RFM clustering")
    st.markdown("---")

    # Load the data if not already in session state
    if "df" not in st.session_state:
        st.session_state.df = load_data()

    if st.session_state.df is None:
        st.error("Failed to load data. Please check the data source.")
        return

    # Get the list of segments
    segments = ["All Segments"] + sorted(
        st.session_state.df["SegmentName"].unique().tolist()
    )

    # Sidebar for segment selection
    with st.sidebar:
        st.title("Segment Selection")

        selected_segment = st.selectbox(
            "Choose a segment to analyze:", segments, index=0
        )

        st.session_state.selected_segment = selected_segment

        st.markdown("---")

        # Display segment statistics in the sidebar
        if selected_segment != "All Segments":
            segment_df = st.session_state.df[
                st.session_state.df["SegmentName"] == selected_segment
            ]

            st.subheader(f"{selected_segment} Segment")
            st.metric("Customers", f"{len(segment_df):,}")
            st.metric(
                "% of Total", f"{len(segment_df) / len(st.session_state.df) * 100:.1f}%"
            )

            # Show quick RFM metrics
            st.metric("Avg. Recency (days)", f"{segment_df['Recency'].mean():.1f}")
            st.metric("Avg. Frequency", f"{segment_df['Frequency'].mean():.1f}")
            st.metric(
                "Avg. Monetary Value", f"£{segment_df['MonetaryValue'].mean():.2f}"
            )

    # Get the filtered data based on the selected segment
    filtered_df = filter_data(st.session_state.df)

    # Create tabs for different analysis views
    tab1, tab2, tab3 = st.tabs(
        ["Segment Profile", "Segment Comparison", "Customer Explorer"]
    )

    with tab1:
        if st.session_state.selected_segment == "All Segments":
            st.info(
                "Please select a specific segment from the sidebar to view its profile."
            )
        else:
            segment_profile_card(st.session_state.selected_segment, filtered_df)

    with tab2:
        segment_comparison()

    with tab3:
        customer_explorer(filtered_df)


if __name__ == "__main__":
    main()
