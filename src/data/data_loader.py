"""
Data loading module for the customer segmentation project.
This module handles loading the data from various sources and performs initial validation.
"""

import os
import pandas as pd
from typing import Optional, Dict, Any, Tuple


def load_retail_data(filepath: str) -> pd.DataFrame:
    """
    Load the retail dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: The loaded dataframe

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is empty or has invalid format
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist")

    # Load the dataset
    df = pd.read_csv(filepath)

    # Verify the data is not empty
    if df.empty:
        raise ValueError(f"The file {filepath} is empty")

    # Check for required columns
    required_columns = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"The file is missing the following required columns: {missing_columns}"
        )

    # Convert datatypes
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    return df


def validate_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate the dataframe and return statistics about data quality.

    Args:
        df (pd.DataFrame): The dataframe to validate

    Returns:
        Tuple[bool, Dict[str, Any]]: (is_valid, validation_stats)
    """
    validation_stats = {
        "row_count": len(df),
        "null_counts": df.isnull().sum().to_dict(),
        "negative_quantities": (df["Quantity"] < 0).sum(),
        "zero_prices": (df["Price"] == 0).sum(),
        "missing_customer_ids": df["Customer ID"].isnull().sum(),
        "unique_countries": df["Country"].nunique(),
        "date_range": (df["InvoiceDate"].min(), df["InvoiceDate"].max()),
    }

    # Define validation criteria
    is_valid = (
        validation_stats["row_count"] > 0
        and validation_stats["negative_quantities"] / validation_stats["row_count"]
        < 0.1
        and validation_stats["zero_prices"] / validation_stats["row_count"] < 0.01
        and validation_stats["missing_customer_ids"] / validation_stats["row_count"]
        < 0.3
    )

    return is_valid, validation_stats


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of the dataset including statistics and quality metrics.

    Args:
        df (pd.DataFrame): The dataframe to summarize

    Returns:
        Dict[str, Any]: A dictionary of summary statistics
    """
    # Basic statistics
    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        "time_period": f"{df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}",
        "customer_count": df["Customer ID"].nunique(),
        "country_count": df["Country"].nunique(),
        "top_countries": df["Country"].value_counts().head(5).to_dict(),
        "product_count": df["StockCode"].nunique(),
        "transaction_count": df["Invoice"].nunique(),
        "avg_products_per_transaction": df.groupby("Invoice")["StockCode"]
        .nunique()
        .mean(),
        "avg_quantity_per_transaction": df.groupby("Invoice")["Quantity"].sum().mean(),
        "total_sales": (df["Quantity"] * df["Price"]).sum(),
    }

    return summary


if __name__ == "__main__":
    # Example usage
    try:
        data_path = os.path.join("data", "raw", "online_retail_combined_dataset.csv")
        df = load_retail_data(data_path)
        is_valid, validation_stats = validate_data(df)

        print(f"Data loaded successfully: {len(df)} rows")
        print(f"Data validation: {'PASSED' if is_valid else 'FAILED'}")

        if not is_valid:
            print("Validation issues:")
            for key, value in validation_stats.items():
                print(f"  - {key}: {value}")

        summary = get_data_summary(df)
        print("\nData Summary:")
        for key, value in summary.items():
            print(f"  - {key}: {value}")

    except Exception as e:
        print(f"Error: {e}")
