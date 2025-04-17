"""
Streamlit page for data overview and exploration.
This page provides an overview of the data and allows basic exploration.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from datetime import datetime

# Add the project root to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# Page configuration
st.set_page_config(
    page_title="Data Overview",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load the data
@st.cache_data
def load_data():
    # Load both the raw and processed data
    try:
        # Load retail data
        retail_path = os.path.join("data", "processed", "cleaned_retail_data.csv")
        retail_df = pd.read_csv(retail_path)

        # Convert date columns
        retail_df["InvoiceDate"] = pd.to_datetime(retail_df["InvoiceDate"])

        # Load customer segment data
        segment_path = os.path.join("data", "processed", "customer_segments.csv")
        segment_df = pd.read_csv(segment_path)

        return retail_df, segment_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


# Calculate data quality metrics
def calculate_quality_metrics(df):
    metrics = {}

    # Basic statistics
    metrics["rows"] = len(df)
    metrics["columns"] = len(df.columns)
    metrics["duplicates"] = df.duplicated().sum()
    metrics["duplicates_pct"] = (
        (metrics["duplicates"] / metrics["rows"]) * 100 if metrics["rows"] > 0 else 0
    )

    # Missing values
    missing_values = df.isnull().sum()
    metrics["missing_values"] = missing_values.sum()
    metrics["missing_values_pct"] = (
        metrics["missing_values"] / (metrics["rows"] * metrics["columns"])
    ) * 100
    metrics["missing_by_column"] = {
        col: {"count": count, "pct": (count / metrics["rows"]) * 100}
        for col, count in missing_values.items()
        if count > 0
    }

    # Data types
    metrics["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return metrics


# Create data summary
def data_summary(retail_df, segment_df):
    st.subheader("Data Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Retail Transactions")

        if retail_df is not None:
            # Calculate basic metrics
            total_transactions = retail_df["Invoice"].nunique()
            total_customers = retail_df["Customer ID"].nunique()
            total_products = retail_df["StockCode"].nunique()
            total_countries = retail_df["Country"].nunique()
            total_sales = (retail_df["Quantity"] * retail_df["Price"]).sum()
            date_range = (
                retail_df["InvoiceDate"].min(),
                retail_df["InvoiceDate"].max(),
            )
            date_range_str = f"{date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}"

            # Display metrics
            st.metric("Total Transactions", f"{total_transactions:,}")
            st.metric("Total Customers", f"{total_customers:,}")
            st.metric("Total Products", f"{total_products:,}")
            st.metric("Total Countries", f"{total_countries:,}")
            st.metric("Total Sales", f"£{total_sales:,.2f}")
            st.metric("Date Range", date_range_str)
        else:
            st.warning("Retail transaction data not available.")

    with col2:
        st.markdown("#### Customer Segments")

        if segment_df is not None:
            # Calculate segment metrics
            total_segments = (
                segment_df["SegmentName"].nunique()
                if "SegmentName" in segment_df.columns
                else 0
            )
            recency_range = (
                (segment_df["Recency"].min(), segment_df["Recency"].max())
                if "Recency" in segment_df.columns
                else (0, 0)
            )
            frequency_range = (
                (segment_df["Frequency"].min(), segment_df["Frequency"].max())
                if "Frequency" in segment_df.columns
                else (0, 0)
            )
            monetary_range = (
                (segment_df["MonetaryValue"].min(), segment_df["MonetaryValue"].max())
                if "MonetaryValue" in segment_df.columns
                else (0, 0)
            )

            # Display metrics
            st.metric("Total Customers", f"{len(segment_df):,}")
            st.metric("Number of Segments", f"{total_segments:,}")
            st.metric(
                "Recency Range (days)",
                f"{recency_range[0]:.0f} - {recency_range[1]:.0f}",
            )
            st.metric(
                "Frequency Range",
                f"{frequency_range[0]:.0f} - {frequency_range[1]:.0f}",
            )
            st.metric(
                "Monetary Value Range",
                f"£{monetary_range[0]:.2f} - £{monetary_range[1]:.2f}",
            )
        else:
            st.warning("Customer segment data not available.")

    st.markdown("---")


# Create data quality section
def data_quality(retail_df, segment_df):
    st.subheader("Data Quality Assessment")

    # Create tabs for different datasets
    tab1, tab2 = st.tabs(["Retail Transactions", "Customer Segments"])

    with tab1:
        if retail_df is not None:
            # Calculate quality metrics
            metrics = calculate_quality_metrics(retail_df)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Rows", f"{metrics['rows']:,}")
                st.metric("Columns", f"{metrics['columns']:,}")

            with col2:
                st.metric(
                    "Duplicate Rows",
                    f"{metrics['duplicates']:,} ({metrics['duplicates_pct']:.2f}%)",
                )
                st.metric(
                    "Missing Values",
                    f"{metrics['missing_values']:,} ({metrics['missing_values_pct']:.2f}%)",
                )

            with col3:
                # Create a mini data profiling summary
                numeric_cols = retail_df.select_dtypes(include=["number"]).columns
                st.metric("Numeric Columns", f"{len(numeric_cols)}")
                categorical_cols = retail_df.select_dtypes(include=["object"]).columns
                st.metric("Categorical Columns", f"{len(categorical_cols)}")

            # Display missing values by column if any
            if metrics["missing_values"] > 0:
                st.markdown("#### Missing Values by Column")

                missing_data = []
                for col, stats in metrics["missing_by_column"].items():
                    if stats["count"] > 0:
                        missing_data.append(
                            {
                                "Column": col,
                                "Missing Count": stats["count"],
                                "Missing Percentage": stats["pct"],
                            }
                        )

                missing_df = pd.DataFrame(missing_data)
                missing_df = missing_df.sort_values(by="Missing Count", ascending=False)

                # Display as a bar chart
                fig = px.bar(
                    missing_df,
                    x="Column",
                    y="Missing Count",
                    color="Missing Percentage",
                    labels={"Missing Count": "Number of Missing Values"},
                    title="Missing Values by Column",
                    color_continuous_scale="Reds",
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values detected in the dataset.")

            # Display data types
            st.markdown("#### Column Data Types")

            dtype_data = []
            for col, dtype in metrics["dtypes"].items():
                dtype_data.append({"Column": col, "Data Type": dtype})

            dtype_df = pd.DataFrame(dtype_data)

            st.dataframe(
                dtype_df,
                column_config={"Column": "Column Name", "Data Type": "Data Type"},
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.warning("Retail transaction data not available.")

    with tab2:
        if segment_df is not None:
            # Calculate quality metrics
            metrics = calculate_quality_metrics(segment_df)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Rows", f"{metrics['rows']:,}")
                st.metric("Columns", f"{metrics['columns']:,}")

            with col2:
                st.metric(
                    "Duplicate Rows",
                    f"{metrics['duplicates']:,} ({metrics['duplicates_pct']:.2f}%)",
                )
                st.metric(
                    "Missing Values",
                    f"{metrics['missing_values']:,} ({metrics['missing_values_pct']:.2f}%)",
                )

            with col3:
                # Create a mini data profiling summary
                numeric_cols = segment_df.select_dtypes(include=["number"]).columns
                st.metric("Numeric Columns", f"{len(numeric_cols)}")
                categorical_cols = segment_df.select_dtypes(include=["object"]).columns
                st.metric("Categorical Columns", f"{len(categorical_cols)}")

            # Display missing values by column if any
            if metrics["missing_values"] > 0:
                st.markdown("#### Missing Values by Column")

                missing_data = []
                for col, stats in metrics["missing_by_column"].items():
                    if stats["count"] > 0:
                        missing_data.append(
                            {
                                "Column": col,
                                "Missing Count": stats["count"],
                                "Missing Percentage": stats["pct"],
                            }
                        )

                missing_df = pd.DataFrame(missing_data)
                missing_df = missing_df.sort_values(by="Missing Count", ascending=False)

                # Display as a bar chart
                fig = px.bar(
                    missing_df,
                    x="Column",
                    y="Missing Count",
                    color="Missing Percentage",
                    labels={"Missing Count": "Number of Missing Values"},
                    title="Missing Values by Column",
                    color_continuous_scale="Reds",
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values detected in the dataset.")

            # Display data types
            st.markdown("#### Column Data Types")

            dtype_data = []
            for col, dtype in metrics["dtypes"].items():
                dtype_data.append({"Column": col, "Data Type": dtype})

            dtype_df = pd.DataFrame(dtype_data)

            st.dataframe(
                dtype_df,
                column_config={"Column": "Column Name", "Data Type": "Data Type"},
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.warning("Customer segment data not available.")

    st.markdown("---")


# Create time series analysis section
def time_series_analysis(retail_df):
    st.subheader("Time Series Analysis")

    if retail_df is None or len(retail_df) == 0:
        st.warning("Retail transaction data not available.")
        return

    # Ensure the InvoiceDate column is in datetime format
    retail_df["InvoiceDate"] = pd.to_datetime(retail_df["InvoiceDate"])

    # Add sales column if not already present
    if "SalesLineTotal" not in retail_df.columns:
        retail_df["SalesLineTotal"] = retail_df["Quantity"] * retail_df["Price"]

    # Create date components
    retail_df["Year"] = retail_df["InvoiceDate"].dt.year
    retail_df["Month"] = retail_df["InvoiceDate"].dt.month
    retail_df["Day"] = retail_df["InvoiceDate"].dt.day
    retail_df["DayOfWeek"] = retail_df["InvoiceDate"].dt.dayofweek
    retail_df["Quarter"] = retail_df["InvoiceDate"].dt.quarter

    # Aggregate by day
    daily_sales = (
        retail_df.groupby(retail_df["InvoiceDate"].dt.date)
        .agg({"Invoice": "nunique", "Customer ID": "nunique", "SalesLineTotal": "sum"})
        .reset_index()
    )

    daily_sales.columns = ["Date", "Orders", "Customers", "Sales"]

    # Aggregate by month
    monthly_sales = (
        retail_df.groupby([retail_df["Year"], retail_df["Month"]])
        .agg({"Invoice": "nunique", "Customer ID": "nunique", "SalesLineTotal": "sum"})
        .reset_index()
    )

    # Create Month-Year string for better display - FIXED by converting to int first
    monthly_sales["Month-Year"] = monthly_sales.apply(
        lambda x: f"{int(x['Year'])}-{int(x['Month']):02d}", axis=1
    )

    monthly_sales.columns = [
        "Year",
        "Month",
        "Orders",
        "Customers",
        "Sales",
        "Month-Year",
    ]

    # Aggregate by day of week
    day_of_week_mapping = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    retail_df["DayName"] = retail_df["DayOfWeek"].map(day_of_week_mapping)

    day_of_week_sales = (
        retail_df.groupby("DayName")
        .agg({"Invoice": "nunique", "Customer ID": "nunique", "SalesLineTotal": "sum"})
        .reset_index()
    )

    day_of_week_sales.columns = ["Day", "Orders", "Customers", "Sales"]

    # Sort by day of week
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    day_of_week_sales["DaySort"] = day_of_week_sales["Day"].apply(
        lambda x: day_order.index(x)
    )
    day_of_week_sales = day_of_week_sales.sort_values("DaySort")
    day_of_week_sales = day_of_week_sales.drop(columns=["DaySort"])

    # Create tabs for different time aggregations
    tab1, tab2, tab3 = st.tabs(["Daily", "Monthly", "Day of Week"])

    with tab1:
        # Daily sales analysis
        st.markdown("#### Daily Sales Trend")

        # Create a time series plot for daily sales
        fig = px.line(
            daily_sales,
            x="Date",
            y="Sales",
            title="Daily Sales Trend",
            labels={"Sales": "Sales (£)", "Date": "Date"},
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales (£)",
            yaxis_tickprefix="£",
            yaxis_tickformat=",.0f",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Create a dual-axis plot for orders and customers
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=daily_sales["Date"], y=daily_sales["Orders"], name="Orders"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=daily_sales["Date"],
                y=daily_sales["Customers"],
                name="Unique Customers",
            ),
            secondary_y=True,
        )

        fig.update_layout(title_text="Daily Orders and Customers", xaxis_title="Date")

        fig.update_yaxes(title_text="Number of Orders", secondary_y=False)
        fig.update_yaxes(title_text="Number of Customers", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Monthly sales analysis
        st.markdown("#### Monthly Sales Trend")

        # Create a bar chart for monthly sales
        fig = px.bar(
            monthly_sales,
            x="Month-Year",
            y="Sales",
            title="Monthly Sales Trend",
            labels={"Sales": "Sales (£)", "Month-Year": "Month"},
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Sales (£)",
            yaxis_tickprefix="£",
            yaxis_tickformat=",.0f",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Create a line chart for monthly customers and orders
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=monthly_sales["Month-Year"], y=monthly_sales["Orders"], name="Orders"
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=monthly_sales["Month-Year"],
                y=monthly_sales["Customers"],
                name="Unique Customers",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title_text="Monthly Orders and Customers", xaxis_title="Month"
        )

        fig.update_yaxes(title_text="Number of Orders", secondary_y=False)
        fig.update_yaxes(title_text="Number of Customers", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Day of week analysis
        st.markdown("#### Sales by Day of Week")

        # Create a bar chart for sales by day of week
        fig = px.bar(
            day_of_week_sales,
            x="Day",
            y="Sales",
            title="Sales by Day of Week",
            labels={"Sales": "Sales (£)", "Day": "Day of Week"},
        )

        fig.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Sales (£)",
            yaxis_tickprefix="£",
            yaxis_tickformat=",.0f",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Create a bar chart for orders and customers by day of week
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=day_of_week_sales["Day"], y=day_of_week_sales["Orders"], name="Orders"
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=day_of_week_sales["Day"],
                y=day_of_week_sales["Customers"],
                name="Unique Customers",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title_text="Orders and Customers by Day of Week", xaxis_title="Day of Week"
        )

        fig.update_yaxes(title_text="Number of Orders", secondary_y=False)
        fig.update_yaxes(title_text="Number of Customers", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")


# Create geographic analysis section
def geographic_analysis(retail_df):
    st.subheader("Geographic Analysis")

    if retail_df is None or len(retail_df) == 0:
        st.warning("Retail transaction data not available.")
        return

    # Add sales column if not already present
    if "SalesLineTotal" not in retail_df.columns:
        retail_df["SalesLineTotal"] = retail_df["Quantity"] * retail_df["Price"]

    # Aggregate by country
    country_sales = (
        retail_df.groupby("Country")
        .agg({"Invoice": "nunique", "Customer ID": "nunique", "SalesLineTotal": "sum"})
        .reset_index()
    )

    country_sales.columns = ["Country", "Orders", "Customers", "Sales"]

    # Sort by sales
    country_sales = country_sales.sort_values("Sales", ascending=False)

    # Calculate percentage of total
    total_sales = country_sales["Sales"].sum()
    country_sales["SalesPercentage"] = country_sales["Sales"] / total_sales * 100

    # Display top countries
    st.markdown("#### Sales by Country")

    col1, col2 = st.columns(2)

    with col1:
        # Create a bar chart for top countries by sales
        top_countries = country_sales.head(10)

        fig = px.bar(
            top_countries,
            x="Country",
            y="Sales",
            title="Top 10 Countries by Sales",
            labels={"Sales": "Sales (£)", "Country": "Country"},
            color="Country",
        )

        fig.update_layout(
            xaxis_title="Country",
            yaxis_title="Sales (£)",
            yaxis_tickprefix="£",
            yaxis_tickformat=",.0f",
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Create a pie chart for sales percentage by country
        fig = px.pie(
            top_countries,
            values="SalesPercentage",
            names="Country",
            title="Sales Percentage by Country (Top 10)",
            hover_data=["Sales"],
            labels={"SalesPercentage": "% of Total Sales"},
        )

        fig.update_traces(textposition="inside", textinfo="percent+label")

        st.plotly_chart(fig, use_container_width=True)

    # Display table with all countries
    st.markdown("#### Complete Country Breakdown")

    st.dataframe(
        country_sales,
        column_config={
            "Country": "Country",
            "Orders": st.column_config.NumberColumn("Orders"),
            "Customers": st.column_config.NumberColumn("Customers"),
            "Sales": st.column_config.NumberColumn("Sales (£)", format="£%.2f"),
            "SalesPercentage": st.column_config.NumberColumn(
                "% of Total", format="%.2f%%"
            ),
        },
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("---")


# Create product analysis section
def product_analysis(retail_df):
    st.subheader("Product Analysis")

    if retail_df is None or len(retail_df) == 0:
        st.warning("Retail transaction data not available.")
        return

    # Add sales column if not already present
    if "SalesLineTotal" not in retail_df.columns:
        retail_df["SalesLineTotal"] = retail_df["Quantity"] * retail_df["Price"]

    # Aggregate by product
    product_sales = (
        retail_df.groupby(["StockCode", "Description"])
        .agg(
            {
                "Quantity": "sum",
                "SalesLineTotal": "sum",
                "Invoice": "nunique",
                "Customer ID": "nunique",
            }
        )
        .reset_index()
    )

    product_sales.columns = [
        "StockCode",
        "Description",
        "Quantity",
        "Sales",
        "Orders",
        "Customers",
    ]

    # Calculate average price
    product_sales["AvgPrice"] = product_sales["Sales"] / product_sales["Quantity"]

    # Sort by sales
    product_sales = product_sales.sort_values("Sales", ascending=False)

    # Get top products
    top_products = product_sales.head(20)

    # Create tabs for different product analyses
    tab1, tab2 = st.tabs(["Top Products", "Product Explorer"])

    with tab1:
        st.markdown("#### Top 20 Products by Sales")

        col1, col2 = st.columns(2)

        with col1:
            # Create a bar chart for top products by sales
            fig = px.bar(
                top_products,
                x="StockCode",
                y="Sales",
                title="Top 20 Products by Sales",
                labels={"Sales": "Sales (£)", "StockCode": "Stock Code"},
                color="Sales",
                hover_data=["Description", "Quantity", "Orders", "Customers"],
            )

            fig.update_layout(
                xaxis_title="Stock Code",
                yaxis_title="Sales (£)",
                yaxis_tickprefix="£",
                yaxis_tickformat=",.0f",
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Create a bar chart for top products by quantity
            top_by_quantity = product_sales.sort_values(
                "Quantity", ascending=False
            ).head(20)

            fig = px.bar(
                top_by_quantity,
                x="StockCode",
                y="Quantity",
                title="Top 20 Products by Quantity",
                labels={"Quantity": "Quantity Sold", "StockCode": "Stock Code"},
                color="Quantity",
                hover_data=["Description", "Sales", "Orders", "Customers"],
            )

            fig.update_layout(xaxis_title="Stock Code", yaxis_title="Quantity Sold")

            st.plotly_chart(fig, use_container_width=True)

        # Display table with top products
        st.markdown("#### Top Products Details")

        st.dataframe(
            top_products,
            column_config={
                "StockCode": "Stock Code",
                "Description": "Product Description",
                "Quantity": st.column_config.NumberColumn("Quantity Sold"),
                "Sales": st.column_config.NumberColumn("Sales (£)", format="£%.2f"),
                "Orders": st.column_config.NumberColumn("Number of Orders"),
                "Customers": st.column_config.NumberColumn("Unique Customers"),
                "AvgPrice": st.column_config.NumberColumn(
                    "Avg. Price (£)", format="£%.2f"
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

    with tab2:
        st.markdown("#### Product Explorer")

        # Create a search box for products
        search_query = st.text_input(
            "Search for products by description or stock code:"
        )

        filtered_products = product_sales

        if search_query:
            # Filter products by search query
            filtered_products = product_sales[
                (
                    product_sales["Description"].str.contains(
                        search_query, case=False, na=False
                    )
                )
                | (
                    product_sales["StockCode"].str.contains(
                        search_query, case=False, na=False
                    )
                )
            ]

        # Display filtered products
        if len(filtered_products) > 0:
            st.dataframe(
                filtered_products,
                column_config={
                    "StockCode": "Stock Code",
                    "Description": "Product Description",
                    "Quantity": st.column_config.NumberColumn("Quantity Sold"),
                    "Sales": st.column_config.NumberColumn("Sales (£)", format="£%.2f"),
                    "Orders": st.column_config.NumberColumn("Number of Orders"),
                    "Customers": st.column_config.NumberColumn("Unique Customers"),
                    "AvgPrice": st.column_config.NumberColumn(
                        "Avg. Price (£)", format="£%.2f"
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                f"Showing {len(filtered_products)} products out of {len(product_sales)}"
            )
        else:
            st.warning("No products match your search criteria.")

    st.markdown("---")


# Main function
def main():
    # Display header
    st.title("📊 Data Overview")
    st.markdown("Explore and understand the retail data used for customer segmentation")
    st.markdown("---")

    # Load the data
    retail_df, segment_df = load_data()

    if retail_df is None or segment_df is None:
        st.error("Failed to load data. Please check the data source.")
        return

    # Store data in session state for other pages
    st.session_state.retail_df = retail_df
    st.session_state.df = segment_df  # For compatibility with other pages

    # Display data summary
    data_summary(retail_df, segment_df)

    # Display data quality assessment
    data_quality(retail_df, segment_df)

    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(
        ["Time Series Analysis", "Geographic Analysis", "Product Analysis"]
    )

    with tab1:
        time_series_analysis(retail_df)

    with tab2:
        geographic_analysis(retail_df)

    with tab3:
        product_analysis(retail_df)


if __name__ == "__main__":
    main()
