"""
Feature engineering module for the customer segmentation project.
This module handles creating and transforming features for analysis.
"""

import os
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_rfm_features(
    df: pd.DataFrame, reference_date: pd.Timestamp = None
) -> pd.DataFrame:
    """
    Create RFM (Recency, Frequency, Monetary Value) features from a dataframe.

    Args:
        df (pd.DataFrame): Input dataframe with transaction data
        reference_date (pd.Timestamp, optional): Reference date for recency calculation

    Returns:
        pd.DataFrame: Dataframe with RFM features
    """
    # Ensure the dataframe has the required columns
    required_cols = ["Customer ID", "Invoice", "InvoiceDate", "Quantity", "Price"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create SalesLineTotal if it doesn't exist
    if "SalesLineTotal" not in df.columns:
        df["SalesLineTotal"] = df["Quantity"] * df["Price"]

    # If reference_date is not provided, use the max date in the dataset
    if reference_date is None:
        reference_date = df["InvoiceDate"].max()

    # Aggregate at customer level to create RFM metrics
    rfm_df = (
        df.groupby("Customer ID")
        .agg(
            {
                "InvoiceDate": lambda x: (reference_date - x.max()).days,  # Recency
                "Invoice": "nunique",  # Frequency
                "SalesLineTotal": "sum",  # Monetary
            }
        )
        .reset_index()
    )

    # Rename columns
    rfm_df.rename(
        columns={
            "InvoiceDate": "Recency",
            "Invoice": "Frequency",
            "SalesLineTotal": "MonetaryValue",
        },
        inplace=True,
    )

    return rfm_df


def normalize_features(
    df: pd.DataFrame,
    columns_to_scale: List[str],
    scaler_type: str = "standard",
    save_scaler: bool = True,
    scaler_path: str = None,
) -> Tuple[pd.DataFrame, Any]:
    """
    Normalize the features in the given dataframe.

    Args:
        df (pd.DataFrame): The input dataframe
        columns_to_scale (List[str]): List of column names to scale
        scaler_type (str): Type of scaler to use ('standard' or 'minmax')
        save_scaler (bool): Whether to save the scaler to disk
        scaler_path (str): Path to save the scaler

    Returns:
        Tuple[pd.DataFrame, Any]: Scaled dataframe and the scaler object
    """
    # Create a copy of the dataframe
    scaled_df = df.copy()

    # Select the scaler
    if scaler_type.lower() == "standard":
        scaler = StandardScaler()
    elif scaler_type.lower() == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(
            f"Unknown scaler type: {scaler_type}. Use 'standard' or 'minmax'."
        )

    # Fit and transform the selected columns
    scaled_values = scaler.fit_transform(df[columns_to_scale])

    # Replace the values in the dataframe
    for i, col in enumerate(columns_to_scale):
        scaled_df[col] = scaled_values[:, i]

    # Save the scaler if requested
    if save_scaler:
        if scaler_path is None:
            scaler_path = os.path.join("models", f"{scaler_type}_scaler.pkl")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

        # Save the scaler
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    return scaled_df, scaler


def create_customer_segments(
    rfm_df: pd.DataFrame, segment_by: List[str] = None, num_segments: int = 3
) -> pd.DataFrame:
    """
    Create customer segments based on RFM features using quantiles.

    Args:
        rfm_df (pd.DataFrame): RFM dataframe
        segment_by (List[str]): Features to use for segmentation (default: ['Recency', 'Frequency', 'MonetaryValue'])
        num_segments (int): Number of segments to create

    Returns:
        pd.DataFrame: Dataframe with segment labels
    """
    if segment_by is None:
        segment_by = ["Recency", "Frequency", "MonetaryValue"]

    segmented_df = rfm_df.copy()

    # Create segment labels for each feature
    for feature in segment_by:
        # For Recency, lower values are better (more recent)
        if feature == "Recency":
            segmented_df[f"{feature}_Segment"] = pd.qcut(
                segmented_df[feature], q=num_segments, labels=range(num_segments, 0, -1)
            )
        # For other features, higher values are better
        else:
            segmented_df[f"{feature}_Segment"] = pd.qcut(
                segmented_df[feature],
                q=num_segments,
                labels=range(1, num_segments + 1),
                duplicates="drop",
            )

    # Calculate overall RFM score
    segmented_df["RFM_Score"] = segmented_df[
        [f"{feature}_Segment" for feature in segment_by]
    ].sum(axis=1)

    # Create RFM segments
    score_max = num_segments * len(segment_by)
    bins = [0] + [score_max * (i + 1) / 5 for i in range(5)]
    segment_labels = ["Low-Value", "Bronze", "Silver", "Gold", "Platinum"]

    segmented_df["Customer_Segment"] = pd.cut(
        segmented_df["RFM_Score"], bins=bins, labels=segment_labels, include_lowest=True
    )

    return segmented_df


def create_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features for analysis.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with additional features
    """
    # Make a copy of the dataframe
    enhanced_df = df.copy()

    # Calculate Customer Lifetime Value (simple version)
    if "MonetaryValue" in enhanced_df.columns and "Frequency" in enhanced_df.columns:
        # CLV = Average Order Value * Purchase Frequency
        enhanced_df["CLV"] = (
            enhanced_df["MonetaryValue"]
            / enhanced_df["Frequency"]
            * enhanced_df["Frequency"]
        )

    # Calculate Monetary Density (avg spending per transaction)
    if "MonetaryValue" in enhanced_df.columns and "Frequency" in enhanced_df.columns:
        enhanced_df["MonetaryDensity"] = (
            enhanced_df["MonetaryValue"] / enhanced_df["Frequency"]
        )

    # Calculate Recency-Frequency Ratio (lower means more frequent relative to recency)
    if "Recency" in enhanced_df.columns and "Frequency" in enhanced_df.columns:
        # Add 1 to avoid division by zero
        enhanced_df["RF_Ratio"] = enhanced_df["Recency"] / (
            enhanced_df["Frequency"] + 1
        )

    # Calculate Engagement Score (higher is better)
    if all(
        col in enhanced_df.columns for col in ["Recency", "Frequency", "MonetaryValue"]
    ):
        # Normalized values for a simple score
        max_recency = enhanced_df["Recency"].max()
        max_frequency = enhanced_df["Frequency"].max()
        max_monetary = enhanced_df["MonetaryValue"].max()

        recency_score = 1 - (enhanced_df["Recency"] / max_recency)
        frequency_score = enhanced_df["Frequency"] / max_frequency
        monetary_score = enhanced_df["MonetaryValue"] / max_monetary

        enhanced_df["EngagementScore"] = (
            recency_score + frequency_score + monetary_score
        ) / 3

    return enhanced_df


def save_features(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the features dataframe to a CSV file.

    Args:
        df (pd.DataFrame): The dataframe to save
        output_path (str): Path to save the CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the dataframe
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")


def feature_engineering_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.

    Args:
        input_path (str): Path to the input CSV file
        output_path (str): Path to save the output CSV file

    Returns:
        pd.DataFrame: The processed dataframe
    """
    # Load the data
    df = pd.read_csv(input_path)

    # Convert date columns to datetime
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Create RFM features if not already created
    if not all(col in df.columns for col in ["Recency", "Frequency", "MonetaryValue"]):
        print("Creating RFM features...")
        df = create_rfm_features(df)

    # Normalize the features
    rfm_columns = ["Recency", "Frequency", "MonetaryValue"]
    print("Normalizing features...")
    normalized_df, scaler = normalize_features(
        df,
        columns_to_scale=rfm_columns,
        scaler_type="standard",
        save_scaler=True,
        scaler_path=os.path.join("models", "rfm_scaler.pkl"),
    )

    # Create customer segments
    print("Creating customer segments...")
    segmented_df = create_customer_segments(normalized_df, segment_by=rfm_columns)

    # Create additional features
    print("Creating additional features...")
    enhanced_df = create_additional_features(segmented_df)

    # Save the processed data
    save_features(enhanced_df, output_path)

    return enhanced_df


if __name__ == "__main__":
    # Run the feature engineering pipeline
    try:
        input_path = os.path.join("data", "processed", "cleaned_retail_data.csv")
        output_path = os.path.join("data", "processed", "customer_features.csv")

        print(f"Starting feature engineering pipeline...")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        # Process the data
        df = feature_engineering_pipeline(input_path, output_path)

        print("\nFeature engineering pipeline completed successfully!")
        print(f"Output shape: {df.shape}")
        print(f"Features: {', '.join(df.columns)}")

        # Display segment distribution
        if "Customer_Segment" in df.columns:
            segment_counts = df["Customer_Segment"].value_counts()
            print("\nCustomer Segment Distribution:")
            for segment, count in segment_counts.items():
                print(
                    f"  - {segment}: {count} customers ({count / len(df) * 100:.1f}%)"
                )

    except Exception as e:
        print(f"Error in feature engineering pipeline: {e}")
