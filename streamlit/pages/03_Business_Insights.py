"""
Streamlit page for business insights and recommendations.
This page provides actionable business recommendations based on customer segments.
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


# Page configuration
st.set_page_config(
    page_title="Business Insights & Recommendations",
    page_icon="💡",
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


# Calculate business metrics
def calculate_business_metrics(df):
    metrics = {}

    # Basic customer metrics
    metrics["total_customers"] = len(df)
    metrics["total_revenue"] = df["MonetaryValue"].sum()
    metrics["avg_revenue_per_customer"] = (
        metrics["total_revenue"] / metrics["total_customers"]
    )

    # Segment distribution
    segment_distribution = df["SegmentName"].value_counts(normalize=True).to_dict()
    metrics["segment_distribution"] = segment_distribution

    # Segment revenue contribution
    segment_revenue = df.groupby("SegmentName")["MonetaryValue"].sum()
    segment_revenue_pct = segment_revenue / segment_revenue.sum()
    metrics["segment_revenue"] = segment_revenue.to_dict()
    metrics["segment_revenue_pct"] = segment_revenue_pct.to_dict()

    # Customer retention risk
    if "Recency" in df.columns:
        high_risk_threshold = df["Recency"].quantile(0.75)
        high_risk_customers = df[df["Recency"] > high_risk_threshold]
        metrics["high_risk_customers"] = len(high_risk_customers)
        metrics["high_risk_pct"] = len(high_risk_customers) / len(df) * 100
        metrics["high_risk_revenue"] = high_risk_customers["MonetaryValue"].sum()
        metrics["high_risk_revenue_pct"] = (
            metrics["high_risk_revenue"] / metrics["total_revenue"] * 100
        )

    # Customer acquisition and retention metrics
    if "Recency" in df.columns and "Frequency" in df.columns:
        new_customers = df[(df["Recency"] <= 30) & (df["Frequency"] <= 2)]
        retained_customers = df[(df["Recency"] <= 90) & (df["Frequency"] > 2)]

        metrics["new_customers"] = len(new_customers)
        metrics["new_customers_pct"] = len(new_customers) / len(df) * 100
        metrics["retained_customers"] = len(retained_customers)
        metrics["retained_customers_pct"] = len(retained_customers) / len(df) * 100

    return metrics


# Create revenue opportunity card
def revenue_opportunity_card(df, metrics):
    st.subheader("Revenue Opportunities")

    # Calculate potential revenue increase
    potential_scenarios = {
        "Reduce churn by 10%": {
            "description": "Reducing customer churn in high-risk segments by 10%",
            "calculation": metrics["high_risk_revenue"] * 0.1,
            "strategy": "Implement targeted retention campaigns for customers with high recency values",
        },
        "Increase frequency by 5%": {
            "description": "Increasing purchase frequency across all segments by 5%",
            "calculation": metrics["total_revenue"] * 0.05,
            "strategy": "Implement loyalty programs, regular promotions, and personalized product recommendations",
        },
        "Upsell to mid-tier customers": {
            "description": "Increasing monetary value of mid-tier segments by 15%",
            "calculation": df[
                df["SegmentName"].isin(["Average Customers", "Rare Shoppers"])
            ]["MonetaryValue"].sum()
            * 0.15,
            "strategy": "Targeted upselling and cross-selling campaigns for mid-tier segments",
        },
        "Reactivate dormant customers": {
            "description": "Reactivating 20% of dormant customers",
            "calculation": df[df["SegmentName"] == "Dormant"]["MonetaryValue"].mean()
            * len(df[df["SegmentName"] == "Dormant"])
            * 0.2,
            "strategy": "Win-back campaigns with special offers for dormant customers",
        },
        "Convert new customers to loyal": {
            "description": "Converting 25% of new customers to loyal customers",
            "calculation": (
                df[df["SegmentName"] == "Loyal Customers"]["MonetaryValue"].mean()
                - df[df["SegmentName"] == "New Customers"]["MonetaryValue"].mean()
            )
            * len(df[df["SegmentName"] == "New Customers"])
            * 0.25,
            "strategy": "Onboarding programs and second-purchase incentives for new customers",
        },
    }

    # Display the opportunities
    col1, col2 = st.columns(2)

    with col1:
        # Create table of revenue opportunities
        opportunities_data = []

        for scenario, data in potential_scenarios.items():
            opportunities_data.append(
                {
                    "Opportunity": scenario,
                    "Description": data["description"],
                    "Potential Revenue": data["calculation"],
                    "Strategy": data["strategy"],
                }
            )

        # Convert to dataframe and sort by potential revenue
        opportunities_df = pd.DataFrame(opportunities_data)
        opportunities_df = opportunities_df.sort_values(
            by="Potential Revenue", ascending=False
        )

        # Format the dataframe for display
        display_df = opportunities_df.copy()
        display_df["Potential Revenue"] = display_df["Potential Revenue"].apply(
            lambda x: f"£{x:,.2f}"
        )

        # Display the table
        st.dataframe(
            display_df,
            column_config={
                "Opportunity": "Revenue Opportunity",
                "Description": "Description",
                "Potential Revenue": "Potential Revenue Increase",
                "Strategy": "Recommended Strategy",
            },
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        # Create chart of revenue opportunities
        fig = px.bar(
            opportunities_df,
            x="Opportunity",
            y="Potential Revenue",
            color="Opportunity",
            title="Potential Revenue Increase by Opportunity",
            labels={"Potential Revenue": "Potential Revenue Increase (£)"},
        )

        fig.update_layout(
            xaxis={"categoryorder": "total descending"},
            yaxis_tickprefix="£",
            yaxis_tickformat=",.0f",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Calculate total potential revenue increase
    total_potential = opportunities_df["Potential Revenue"].sum()
    current_revenue = metrics["total_revenue"]
    potential_increase_pct = (total_potential / current_revenue) * 100

    st.info(
        f"Total potential revenue increase: £{total_potential:,.2f} ({potential_increase_pct:.1f}% of current revenue)"
    )


# Create segment strategy recommendations
def segment_strategy_recommendations():
    st.subheader("Segment Strategy Recommendations")

    # Define strategies for each segment
    segment_strategies = {
        "Champions": {
            "Retention": {
                "title": "VIP Loyalty Program",
                "description": "Create an exclusive VIP program with premium benefits, personalized service, and special events.",
                "impact": "High",
                "effort": "Medium",
                "timeframe": "3-6 months",
            },
            "Growth": {
                "title": "Referral Incentives",
                "description": "Incentivize champions to refer friends and family with dual-sided rewards.",
                "impact": "Medium",
                "effort": "Low",
                "timeframe": "1-3 months",
            },
            "Engagement": {
                "title": "Product Feedback Loop",
                "description": "Establish a formal feedback program to gather insights from your best customers.",
                "impact": "Medium",
                "effort": "Low",
                "timeframe": "1-2 months",
            },
        },
        "Loyal Customers": {
            "Retention": {
                "title": "Tiered Loyalty Program",
                "description": "Implement a tiered loyalty program with clear path to VIP status.",
                "impact": "High",
                "effort": "Medium",
                "timeframe": "2-4 months",
            },
            "Growth": {
                "title": "Premium Upsell Campaign",
                "description": "Create targeted campaigns to introduce premium products with loyalty discounts.",
                "impact": "High",
                "effort": "Medium",
                "timeframe": "1-2 months",
            },
            "Engagement": {
                "title": "Early Access Program",
                "description": "Provide early access to new products and collections for loyal customers.",
                "impact": "Medium",
                "effort": "Low",
                "timeframe": "Ongoing",
            },
        },
        "Big Spenders": {
            "Retention": {
                "title": "Personal Shopper Service",
                "description": "Offer personalized shopping assistance or consultations for big spenders.",
                "impact": "High",
                "effort": "High",
                "timeframe": "3-6 months",
            },
            "Growth": {
                "title": "Frequency Boosting Program",
                "description": "Create incentives for more frequent purchases with time-limited exclusive offers.",
                "impact": "High",
                "effort": "Medium",
                "timeframe": "2-3 months",
            },
            "Engagement": {
                "title": "Premium Product Previews",
                "description": "Exclusive previews of high-end products before general release.",
                "impact": "Medium",
                "effort": "Low",
                "timeframe": "Ongoing",
            },
        },
        "Dormant": {
            "Retention": {
                "title": "Win-Back Campaign Series",
                "description": "Multi-touch campaign to re-engage dormant customers with escalating offers.",
                "impact": "Medium",
                "effort": "Medium",
                "timeframe": "2-3 months",
            },
            "Growth": {
                "title": "Dormant Customer Survey",
                "description": "Survey to understand why they stopped purchasing, with incentive for completion.",
                "impact": "Medium",
                "effort": "Low",
                "timeframe": "1 month",
            },
            "Engagement": {
                "title": "Product Update Communication",
                "description": "Share what's new since they last purchased to spark renewed interest.",
                "impact": "Low",
                "effort": "Low",
                "timeframe": "Ongoing",
            },
        },
        "At Risk": {
            "Retention": {
                "title": "Preemptive Retention Program",
                "description": "Identify and engage at-risk customers before they become dormant.",
                "impact": "High",
                "effort": "Medium",
                "timeframe": "Ongoing",
            },
            "Growth": {
                "title": "Personalized Re-engagement",
                "description": "Highly personalized offers based on previous purchase history.",
                "impact": "Medium",
                "effort": "Medium",
                "timeframe": "1-2 months",
            },
            "Engagement": {
                "title": "Satisfaction Survey",
                "description": "Survey to identify potential issues with a special offer for completion.",
                "impact": "Medium",
                "effort": "Low",
                "timeframe": "1 month",
            },
        },
        "New Customers": {
            "Retention": {
                "title": "Onboarding Journey",
                "description": "Structured communication flow to educate and engage new customers.",
                "impact": "High",
                "effort": "Medium",
                "timeframe": "2-3 months",
            },
            "Growth": {
                "title": "Second Purchase Incentive",
                "description": "Special offer to encourage the critical second purchase.",
                "impact": "High",
                "effort": "Low",
                "timeframe": "Ongoing",
            },
            "Engagement": {
                "title": "Welcome Survey",
                "description": "Brief survey to learn about preferences and first experience.",
                "impact": "Medium",
                "effort": "Low",
                "timeframe": "Ongoing",
            },
        },
        "Rare Shoppers": {
            "Retention": {
                "title": "Frequency Reward Program",
                "description": "Program that rewards multiple purchases within a defined timeframe.",
                "impact": "Medium",
                "effort": "Medium",
                "timeframe": "2-3 months",
            },
            "Growth": {
                "title": "Bundle Promotions",
                "description": "Create attractive bundles to increase average order value.",
                "impact": "Medium",
                "effort": "Low",
                "timeframe": "1-2 months",
            },
            "Engagement": {
                "title": "Regular Re-engagement",
                "description": "Consistent communication and offers to keep your brand top-of-mind.",
                "impact": "Low",
                "effort": "Low",
                "timeframe": "Ongoing",
            },
        },
        "Average Customers": {
            "Retention": {
                "title": "Mid-tier Loyalty Program",
                "description": "Loyalty program designed for average customers with achievable benefits.",
                "impact": "Medium",
                "effort": "Medium",
                "timeframe": "2-4 months",
            },
            "Growth": {
                "title": "Cross-sell Campaign",
                "description": "Targeted cross-selling to increase product categories per customer.",
                "impact": "Medium",
                "effort": "Medium",
                "timeframe": "1-2 months",
            },
            "Engagement": {
                "title": "Milestone Rewards",
                "description": "Celebrate and reward purchase milestones to encourage continued engagement.",
                "impact": "Medium",
                "effort": "Low",
                "timeframe": "Ongoing",
            },
        },
    }

    # Create tabs for strategy categories
    tabs = st.tabs(["All Strategies", "Retention", "Growth", "Engagement"])

    with tabs[0]:
        # Display all strategies
        for segment, strategies in segment_strategies.items():
            st.markdown(f"### {segment}")

            for strategy_type, strategy in strategies.items():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                with col1:
                    st.markdown(f"**{strategy['title']}** - {strategy['description']}")

                with col2:
                    impact_color = {
                        "High": "green",
                        "Medium": "orange",
                        "Low": "blue",
                    }.get(strategy["impact"], "gray")

                    st.markdown(
                        f"Impact: <span style='color:{impact_color};font-weight:bold'>{strategy['impact']}</span>",
                        unsafe_allow_html=True,
                    )

                with col3:
                    effort_color = {
                        "Low": "green",
                        "Medium": "orange",
                        "High": "red",
                    }.get(strategy["effort"], "gray")

                    st.markdown(
                        f"Effort: <span style='color:{effort_color};font-weight:bold'>{strategy['effort']}</span>",
                        unsafe_allow_html=True,
                    )

                with col4:
                    st.markdown(f"Timeframe: {strategy['timeframe']}")

            st.markdown("---")

    # Create filtered views for each strategy type
    for i, strategy_type in enumerate(["Retention", "Growth", "Engagement"], 1):
        with tabs[i]:
            # Create a table of strategies for this type
            strategies_data = []

            for segment, strategies in segment_strategies.items():
                if strategy_type in strategies:
                    strategy = strategies[strategy_type]
                    strategies_data.append(
                        {
                            "Segment": segment,
                            "Strategy": strategy["title"],
                            "Description": strategy["description"],
                            "Impact": strategy["impact"],
                            "Effort": strategy["effort"],
                            "Timeframe": strategy["timeframe"],
                        }
                    )

            # Convert to dataframe and sort by impact
            impact_order = {"High": 0, "Medium": 1, "Low": 2}
            strategies_df = pd.DataFrame(strategies_data)
            strategies_df["Impact_Order"] = strategies_df["Impact"].map(impact_order)
            strategies_df = strategies_df.sort_values(by=["Impact_Order", "Effort"])
            strategies_df = strategies_df.drop(columns=["Impact_Order"])

            # Display the table
            st.dataframe(
                strategies_df,
                column_config={
                    "Segment": "Customer Segment",
                    "Strategy": f"{strategy_type} Strategy",
                    "Description": "Description",
                    "Impact": st.column_config.Column(
                        "Business Impact", help="Estimated impact on business metrics"
                    ),
                    "Effort": st.column_config.Column(
                        "Implementation Effort",
                        help="Estimated effort required to implement",
                    ),
                    "Timeframe": st.column_config.Column(
                        "Timeframe", help="Estimated time to implement"
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )


# Create campaign planning function
def campaign_planning():
    st.subheader("Campaign Planning")

    # Define campaign ideas
    campaign_ideas = [
        {
            "name": "VIP Appreciation Event",
            "description": "Exclusive virtual or in-person event for Champions and Big Spenders",
            "segments": ["Champions", "Big Spenders"],
            "channel": "Email + Direct Mail",
            "goal": "Retention",
            "kpis": ["Attendance rate", "Post-event purchases", "Feedback score"],
            "timeframe": "Quarterly",
        },
        {
            "name": "Win-Back Campaign",
            "description": "Multi-touch campaign with escalating offers to reactivate dormant customers",
            "segments": ["Dormant", "At Risk"],
            "channel": "Email + SMS",
            "goal": "Reactivation",
            "kpis": ["Response rate", "Conversion rate", "ROI"],
            "timeframe": "Monthly",
        },
        {
            "name": "Cross-Sell Recommendations",
            "description": "Personalized product recommendations based on purchase history",
            "segments": ["Loyal Customers", "Average Customers"],
            "channel": "Email + Website",
            "goal": "Growth",
            "kpis": ["Click-through rate", "Conversion rate", "AOV increase"],
            "timeframe": "Bi-weekly",
        },
        {
            "name": "Second Purchase Incentive",
            "description": "Limited-time offer to encourage second purchase from new customers",
            "segments": ["New Customers"],
            "channel": "Email + App Notification",
            "goal": "Conversion",
            "kpis": ["Conversion rate", "Time to second purchase", "AOV"],
            "timeframe": "Ongoing",
        },
        {
            "name": "Frequency Booster",
            "description": "Campaign to increase purchase frequency with time-limited offers",
            "segments": ["Big Spenders", "Rare Shoppers"],
            "channel": "Email + Social Media",
            "goal": "Frequency",
            "kpis": ["Response rate", "Purchase frequency", "Revenue lift"],
            "timeframe": "Monthly",
        },
        {
            "name": "Feedback Survey",
            "description": "Survey to collect feedback with incentive for completion",
            "segments": ["Champions", "At Risk"],
            "channel": "Email",
            "goal": "Insight",
            "kpis": ["Response rate", "Completion rate", "NPS score"],
            "timeframe": "Quarterly",
        },
        {
            "name": "Loyalty Program Launch",
            "description": "Introduction of tiered loyalty program with clear benefits",
            "segments": ["All Segments"],
            "channel": "All Channels",
            "goal": "Retention",
            "kpis": ["Enrollment rate", "Program engagement", "Revenue impact"],
            "timeframe": "One-time with ongoing promotion",
        },
        {
            "name": "Bundle Promotions",
            "description": "Curated product bundles to increase average order value",
            "segments": ["Average Customers", "Rare Shoppers", "New Customers"],
            "channel": "Email + Website",
            "goal": "AOV Increase",
            "kpis": ["Bundle adoption rate", "AOV increase", "Revenue per customer"],
            "timeframe": "Monthly rotation",
        },
        {
            "name": "Early Access Program",
            "description": "Exclusive early access to new products for loyal customers",
            "segments": ["Champions", "Loyal Customers"],
            "channel": "Email + App Notification",
            "goal": "Engagement",
            "kpis": ["Early access purchases", "Social sharing", "Feedback quality"],
            "timeframe": "With new product launches",
        },
        {
            "name": "Milestone Rewards",
            "description": "Celebrate and reward customer purchase milestones",
            "segments": ["All Segments"],
            "channel": "Email + App Notification",
            "goal": "Retention",
            "kpis": [
                "Milestone achievement rate",
                "Post-milestone purchases",
                "Customer satisfaction",
            ],
            "timeframe": "Automated/Ongoing",
        },
    ]

    # Convert to dataframe
    campaigns_df = pd.DataFrame(campaign_ideas)

    # Create campaign calendar visualization
    st.markdown("### Campaign Calendar")

    # Define campaign schedule (example)
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    # Create a heatmap of campaign schedule
    # For this example, we'll create a simple schedule based on timeframe
    schedule_data = []

    for i, campaign in enumerate(campaign_ideas):
        schedule_row = {"Campaign": campaign["name"]}

        # Determine which months to run the campaign based on timeframe
        if campaign["timeframe"] == "Quarterly":
            active_months = ["Jan", "Apr", "Jul", "Oct"]
        elif campaign["timeframe"] == "Monthly":
            active_months = months
        elif campaign["timeframe"] == "Bi-weekly":
            active_months = months
        elif campaign["timeframe"] == "Ongoing":
            active_months = months
        elif "One-time" in campaign["timeframe"]:
            active_months = ["Jan"]  # Assume January launch for this example
        elif "Monthly rotation" in campaign["timeframe"]:
            active_months = months
        elif "With new product launches" in campaign["timeframe"]:
            # Assume quarterly product launches
            active_months = ["Mar", "Jun", "Sep", "Dec"]
        else:
            active_months = []

        # Set values for each month (1 for active, 0 for inactive)
        for month in months:
            schedule_row[month] = 1 if month in active_months else 0

        schedule_data.append(schedule_row)

    # Convert to dataframe
    schedule_df = pd.DataFrame(schedule_data)

    # Create the heatmap
    fig = px.imshow(
        schedule_df.iloc[:, 1:],
        x=months,
        y=schedule_df["Campaign"],
        color_continuous_scale=[[0, "white"], [1, "green"]],
        labels=dict(x="Month", y="Campaign", color="Active"),
        title="Campaign Calendar",
    )

    # Update layout
    fig.update_layout(
        xaxis={"side": "top"}, coloraxis_showscale=False, height=400, width=800
    )

    # Add annotations (checkmarks for active months)
    for i, campaign in enumerate(schedule_df["Campaign"]):
        for j, month in enumerate(months):
            if schedule_df.iloc[i, j + 1] == 1:
                fig.add_annotation(
                    x=j,
                    y=i,
                    text="✓",
                    showarrow=False,
                    font=dict(size=14, color="darkgreen"),
                )

    st.plotly_chart(fig, use_container_width=True)

    # Display campaign details
    st.markdown("### Campaign Details")

    # Allow filtering by segment and goal
    col1, col2 = st.columns(2)

    with col1:
        # Get all unique segments
        all_segments = []
        for campaign in campaign_ideas:
            all_segments.extend(campaign["segments"])
        unique_segments = sorted(list(set(all_segments)))

        selected_segments = st.multiselect(
            "Filter by segment:",
            options=["All Segments"] + unique_segments,
            default=["All Segments"],
        )

    with col2:
        # Get all unique goals
        all_goals = [campaign["goal"] for campaign in campaign_ideas]
        unique_goals = sorted(list(set(all_goals)))

        selected_goals = st.multiselect(
            "Filter by goal:", options=unique_goals, default=unique_goals
        )

    # Filter campaigns based on selection
    filtered_campaigns = campaigns_df.copy()

    if not (len(selected_segments) == 1 and "All Segments" in selected_segments):
        filtered_campaigns = filtered_campaigns[
            filtered_campaigns["segments"].apply(
                lambda x: any(segment in x for segment in selected_segments)
            )
        ]

    if selected_goals:
        filtered_campaigns = filtered_campaigns[
            filtered_campaigns["goal"].isin(selected_goals)
        ]

    # Display campaign cards
    if len(filtered_campaigns) > 0:
        for i, campaign in filtered_campaigns.iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"**{campaign['name']}**")
                st.markdown(campaign["description"])

            with col2:
                st.markdown(f"**Segments:** {', '.join(campaign['segments'])}")
                st.markdown(f"**Goal:** {campaign['goal']}")

            with col3:
                st.markdown(f"**Channel:** {campaign['channel']}")
                st.markdown(f"**Timeframe:** {campaign['timeframe']}")

            st.markdown(f"**KPIs:** {', '.join(campaign['kpis'])}")
            st.markdown("---")
    else:
        st.warning("No campaigns match the selected filters.")


# Create ROI calculator
def roi_calculator():
    st.subheader("Campaign ROI Calculator")

    st.markdown("""
    Use this calculator to estimate the ROI of marketing campaigns targeted at specific customer segments.
    Adjust the parameters to see how different scenarios affect the potential return on investment.
    """)

    # Create input form
    with st.form("roi_calculator_form"):
        col1, col2 = st.columns(2)

        with col1:
            # Campaign details
            campaign_name = st.text_input("Campaign Name", "Win-Back Campaign")

            segment_options = [
                "Champions",
                "Loyal Customers",
                "Big Spenders",
                "Dormant",
                "At Risk",
                "New Customers",
                "Rare Shoppers",
                "Average Customers",
            ]

            target_segment = st.selectbox(
                "Target Segment",
                options=segment_options,
                index=3,  # Default to Dormant
            )

            campaign_cost = st.number_input(
                "Campaign Cost (£)",
                min_value=100,
                max_value=100000,
                value=5000,
                step=100,
            )

        with col2:
            # Response rates and conversion
            reach = st.slider(
                "Segment Reach (%)",
                min_value=10,
                max_value=100,
                value=80,
                step=5,
                help="Percentage of the segment that will receive the campaign",
            )

            response_rate = st.slider(
                "Response Rate (%)",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                help="Percentage of recipients who will respond to the campaign",
            )

            conversion_rate = st.slider(
                "Conversion Rate (%)",
                min_value=1,
                max_value=100,
                value=25,
                step=1,
                help="Percentage of responders who will make a purchase",
            )

            avg_order_value = st.number_input(
                "Average Order Value (£)",
                min_value=10,
                max_value=10000,
                value=100,
                step=10,
            )

        calculate_button = st.form_submit_button("Calculate ROI")

    # If calculate button is pressed or form is submitted
    if calculate_button or "roi_results" in st.session_state:
        # Get segment size from the data
        if "df" in st.session_state:
            df = st.session_state.df
            segment_size = len(df[df["SegmentName"] == target_segment])
        else:
            # Default values if data is not loaded
            segment_size = 1000

        # Calculate metrics
        reached_customers = segment_size * (reach / 100)
        responders = reached_customers * (response_rate / 100)
        conversions = responders * (conversion_rate / 100)
        revenue = conversions * avg_order_value
        profit = revenue - campaign_cost
        roi_percent = (profit / campaign_cost) * 100 if campaign_cost > 0 else 0

        # Store results in session state
        st.session_state.roi_results = {
            "segment_size": segment_size,
            "reached_customers": reached_customers,
            "responders": responders,
            "conversions": conversions,
            "revenue": revenue,
            "profit": profit,
            "roi_percent": roi_percent,
        }

        # Display results
        st.markdown("### Campaign ROI Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Segment Size", f"{segment_size:,.0f} customers")
            st.metric("Reached Customers", f"{reached_customers:,.0f} customers")
            st.metric("Responders", f"{responders:,.0f} customers")
            st.metric("Conversions", f"{conversions:,.0f} customers")

        with col2:
            st.metric("Campaign Cost", f"£{campaign_cost:,.2f}")
            st.metric("Generated Revenue", f"£{revenue:,.2f}")
            st.metric("Net Profit", f"£{profit:,.2f}")

            # Color the ROI metric based on value
            roi_color = "normal"
            if roi_percent < 0:
                roi_color = "off"
            elif roi_percent > 100:
                roi_color = "inverse"

            st.metric("ROI", f"{roi_percent:,.1f}%", delta_color=roi_color)

        # Create ROI visualization
        st.markdown("### ROI Visualization")

        # Create a funnel chart
        funnel_data = {
            "Stage": ["Segment Size", "Reached", "Responded", "Converted"],
            "Customers": [segment_size, reached_customers, responders, conversions],
        }

        funnel_df = pd.DataFrame(funnel_data)

        fig = go.Figure(
            go.Funnel(
                y=funnel_df["Stage"],
                x=funnel_df["Customers"],
                textinfo="value+percent initial",
                marker={"color": ["#2c3e50", "#3498db", "#1abc9c", "#2ecc71"]},
            )
        )

        fig.update_layout(title="Campaign Conversion Funnel", width=800, height=500)

        st.plotly_chart(fig, use_container_width=True)

        # Add sensitivity analysis
        st.markdown("### Sensitivity Analysis")
        st.markdown("See how changes in response rate and conversion rate affect ROI:")

        # Create a sensitivity matrix
        response_rates = [
            max(1, response_rate - 5),
            response_rate,
            min(50, response_rate + 5),
        ]
        conversion_rates = [
            max(1, conversion_rate - 10),
            conversion_rate,
            min(90, conversion_rate + 10),
        ]

        sensitivity_data = []

        for resp_rate in response_rates:
            for conv_rate in conversion_rates:
                resp = reached_customers * (resp_rate / 100)
                conv = resp * (conv_rate / 100)
                rev = conv * avg_order_value
                prof = rev - campaign_cost
                roi = (prof / campaign_cost) * 100 if campaign_cost > 0 else 0

                sensitivity_data.append(
                    {
                        "Response Rate": f"{resp_rate}%",
                        "Conversion Rate": f"{conv_rate}%",
                        "Revenue": rev,
                        "Profit": prof,
                        "ROI": roi,
                    }
                )

        sensitivity_df = pd.DataFrame(sensitivity_data)

        # Create a heatmap of ROI values
        sensitivity_pivot = sensitivity_df.pivot_table(
            values="ROI", index="Response Rate", columns="Conversion Rate"
        )

        fig = px.imshow(
            sensitivity_pivot,
            labels=dict(x="Conversion Rate", y="Response Rate", color="ROI (%)"),
            x=sensitivity_pivot.columns,
            y=sensitivity_pivot.index,
            color_continuous_scale="RdYlGn",
            title="ROI Sensitivity Analysis",
        )

        # Add annotations with the ROI values
        for i, resp_rate in enumerate(sensitivity_pivot.index):
            for j, conv_rate in enumerate(sensitivity_pivot.columns):
                roi_value = sensitivity_pivot.iloc[i, j]
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{roi_value:.1f}%",
                    showarrow=False,
                    font=dict(
                        color="black"
                        if (roi_value > 50 and roi_value < 200)
                        else "white"
                    ),
                )

        st.plotly_chart(fig, use_container_width=True)


# Main function
def main():
    # Display header
    st.title("💡 Business Insights & Recommendations")
    st.markdown(
        "Actionable insights and recommendations based on customer segmentation"
    )
    st.markdown("---")

    # Load the data if not already in session state
    if "df" not in st.session_state:
        st.session_state.df = load_data()

    if st.session_state.df is None:
        st.error("Failed to load data. Please check the data source.")
        return

    # Calculate business metrics
    metrics = calculate_business_metrics(st.session_state.df)

    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Revenue Opportunities",
            "Segment Strategies",
            "Campaign Planning",
            "ROI Calculator",
        ]
    )

    with tab1:
        revenue_opportunity_card(st.session_state.df, metrics)

    with tab2:
        segment_strategy_recommendations()

    with tab3:
        campaign_planning()

    with tab4:
        roi_calculator()


if __name__ == "__main__":
    main()
