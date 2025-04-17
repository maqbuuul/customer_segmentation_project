"""
Data processing module for the customer segmentation project.
This module handles cleaning, preprocessing, and transforming the retail dataset.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from .data_loader import load_retail_data, validate_data


def clean_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the retail dataset by removing invalid entries and handling missing values.

    Args:
        df (pd.DataFrame): The raw dataframe

    Returns:
        pd.DataFrame: The cleaned dataframe
    """
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()

    # Convert Invoice to string
    cleaned_df["Invoice"] = cleaned_df["Invoice"].astype("str")

    # Filter for valid invoices (regular sales, not returns/cancelled orders)
    # Regular invoices have 6 digits
    mask = cleaned_df["Invoice"].str.match("^\\d{6}$") == True
    cleaned_df = cleaned_df[mask]

    # Filter for valid stock codes
    cleaned_df["StockCode"] = cleaned_df["StockCode"].astype("str")
    mask = (
        (cleaned_df["StockCode"].str.match("^\\d{5}$") == True)
        | (cleaned_df["StockCode"].str.match("^\\d{5}[a-zA-Z]+$") == True)
        | (cleaned_df["StockCode"].str.match("^PADS$") == True)
    )
    cleaned_df = cleaned_df[mask]

    # Remove rows with missing Customer IDs
    cleaned_df.dropna(subset=["Customer ID"], inplace=True)

    # Remove rows with zero or negative prices
    cleaned_df = cleaned_df[cleaned_df["Price"] > 0.0]

    # Remove rows with missing descriptions (optional)
    # cleaned_df.dropna(subset=['Description'], inplace=True)

    # Calculate sales line total
    cleaned_df["SalesLineTotal"] = cleaned_df["Quantity"] * cleaned_df["Price"]

    return cleaned_df


def preprocess_for_rfm(
    df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Preprocess the data for RFM analysis by aggregating at the customer level.

    Args:
        df (pd.DataFrame): The cleaned dataframe
        reference_date (pd.Timestamp, optional): The reference date for recency calculation.
                                                If None, use the max date in the dataset.

    Returns:
        pd.DataFrame: The aggregated dataframe with RFM metrics
    """
    # If no reference date is provided, use the max date in the dataset
    if reference_date is None:
        reference_date = df["InvoiceDate"].max()

    # Aggregate the data by customer
    rfm_df = df.groupby(by="Customer ID", as_index=False).agg(
        MonetaryValue=("SalesLineTotal", "sum"),
        Frequency=("Invoice", "nunique"),
        LastInvoiceDate=("InvoiceDate", "max"),
    )

    # Calculate Recency in days
    rfm_df["Recency"] = (reference_date - rfm_df["LastInvoiceDate"]).dt.days

    return rfm_df


def identify_outliers(
    df: pd.DataFrame, columns: list, method: str = "iqr", threshold: float = 1.5
) -> pd.DataFrame:
    """
    Identify outliers in the specified columns.

    Args:
        df (pd.DataFrame): The dataframe
        columns (list): List of column names to check for outliers
        method (str): Method to use ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection (default 1.5 for IQR, 3 for z-score)

    Returns:
        pd.DataFrame: Dataframe with an additional 'is_outlier' column
    """
    result_df = df.copy()
    result_df["is_outlier"] = False

    for col in columns:
        if method == "iqr":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

        elif method == "zscore":
            mean = df[col].mean()
            std = df[col].std()

            z_scores = np.abs((df[col] - mean) / std)
            outliers = z_scores > threshold

        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")

        result_df.loc[outliers, "is_outlier"] = True

    return result_df


def split_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into outliers and non-outliers.

    Args:
        df (pd.DataFrame): The dataframe with an 'is_outlier' column

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (non_outliers_df, outliers_df)
    """
    non_outliers_df = df[~df["is_outlier"]].copy()
    outliers_df = df[df["is_outlier"]].copy()

    return non_outliers_df, outliers_df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the processed dataframe to a CSV file.

    Args:
        df (pd.DataFrame): The dataframe to save
        output_path (str): Path to save the CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the dataframe
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


def process_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the complete data processing pipeline.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (cleaned_df, rfm_df, rfm_no_outliers_df)
    """
    # Load the data
    raw_data_path = os.path.join("data", "raw", "online_retail_combined_dataset.csv")
    df = load_retail_data(raw_data_path)

    # Validate the data
    is_valid, _ = validate_data(df)
    if not is_valid:
        print("Warning: Data validation failed, but processing will continue.")

    # Clean the data
    cleaned_df = clean_retail_data(df)

    # Save the cleaned data
    cleaned_data_path = os.path.join("data", "processed", "cleaned_retail_data.csv")
    save_processed_data(cleaned_df, cleaned_data_path)

    # Preprocess for RFM analysis
    rfm_df = preprocess_for_rfm(cleaned_df)

    # Identify outliers
    rfm_with_outliers = identify_outliers(
        rfm_df, ["MonetaryValue", "Frequency", "Recency"]
    )

    # Split into outliers and non-outliers
    rfm_no_outliers_df, rfm_outliers_df = split_outliers(rfm_with_outliers)

    # Save the RFM data
    rfm_data_path = os.path.join("data", "processed", "rfm_data.csv")
    save_processed_data(rfm_df, rfm_data_path)

    # Save the RFM data without outliers
    rfm_no_outliers_path = os.path.join("data", "processed", "rfm_no_outliers_data.csv")
    save_processed_data(rfm_no_outliers_df, rfm_no_outliers_path)

    # Save the RFM outliers
    rfm_outliers_path = os.path.join("data", "processed", "rfm_outliers_data.csv")
    save_processed_data(rfm_outliers_df, rfm_outliers_path)

    return cleaned_df, rfm_df, rfm_no_outliers_df


if __name__ == "__main__":
    # Run the pipeline
    try:
        cleaned_df, rfm_df, rfm_no_outliers_df = process_pipeline()

        print("\nProcessing complete!")
        print(f"Cleaned data shape: {cleaned_df.shape}")
        print(f"RFM data shape: {rfm_df.shape}")
        print(f"RFM data without outliers shape: {rfm_no_outliers_df.shape}")

    except Exception as e:
        print(f"Error in processing pipeline: {e}")
