"""
Clustering module for the customer segmentation project.
This module handles the application of clustering algorithms to segment customers.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Tuple, List, Dict, Any, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def find_optimal_clusters(
    df: pd.DataFrame,
    features: List[str],
    max_clusters: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Find the optimal number of clusters using the elbow method and silhouette score.

    Args:
        df (pd.DataFrame): The input dataframe
        features (List[str]): List of features to use for clustering
        max_clusters (int): Maximum number of clusters to try
        random_state (int): Random state for reproducibility

    Returns:
        Dict[str, Any]: Dictionary containing the results
    """
    if max_clusters < 2:
        raise ValueError("max_clusters must be at least 2")

    # Initialize results
    results = {
        "k_values": list(range(2, max_clusters + 1)),
        "inertia": [],
        "silhouette_scores": [],
    }

    # Extract feature data
    X = df[features].values

    # Calculate inertia and silhouette score for each k
    for k in results["k_values"]:
        print(f"Testing k={k}...")
        # Initialize and fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Store results
        results["inertia"].append(kmeans.inertia_)

        # Calculate silhouette score if k >= 2
        if k >= 2:
            sil_score = silhouette_score(X, cluster_labels)
            results["silhouette_scores"].append(sil_score)
        else:
            results["silhouette_scores"].append(None)

    # Determine optimal k based on silhouette score
    valid_scores = [s for s in results["silhouette_scores"] if s is not None]
    optimal_k_silhouette = results["k_values"][
        results["silhouette_scores"].index(max(valid_scores))
    ]
    results["optimal_k_silhouette"] = optimal_k_silhouette

    # Determine optimal k based on elbow method (simple heuristic)
    # Look for the point where the decrease in inertia starts to slow down
    inertia_diff = np.diff(results["inertia"])
    elbow_index = 0
    for i in range(len(inertia_diff) - 1):
        if inertia_diff[i] / inertia_diff[i + 1] > 2:  # Rate of change slows down
            elbow_index = i
            break

    if elbow_index == 0 and len(inertia_diff) > 1:
        # If no clear elbow is found, use the point with the largest second derivative
        second_derivative = np.diff(inertia_diff)
        if len(second_derivative) > 0:
            elbow_index = np.argmax(second_derivative) + 1
        else:
            elbow_index = 0

    optimal_k_elbow = results["k_values"][elbow_index]
    results["optimal_k_elbow"] = optimal_k_elbow

    print(f"Optimal k based on silhouette score: {optimal_k_silhouette}")
    print(f"Optimal k based on elbow method: {optimal_k_elbow}")

    return results


def apply_kmeans_clustering(
    df: pd.DataFrame,
    features: List[str],
    n_clusters: int,
    random_state: int = 42,
    save_model: bool = True,
    model_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """
    Apply KMeans clustering to the data.

    Args:
        df (pd.DataFrame): The input dataframe
        features (List[str]): List of features to use for clustering
        n_clusters (int): Number of clusters
        random_state (int): Random state for reproducibility
        save_model (bool): Whether to save the model
        model_path (str, optional): Path to save the model

    Returns:
        Tuple[pd.DataFrame, Any]: Dataframe with cluster labels and the KMeans model
    """
    # Make a copy of the dataframe
    clustered_df = df.copy()

    # Extract feature data
    X = df[features].values

    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # Add cluster labels to the dataframe
    clustered_df["Cluster"] = cluster_labels

    # Save the model if requested
    if save_model:
        if model_path is None:
            model_path = os.path.join("models", "kmeans_model.pkl")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save the model
        joblib.dump(kmeans, model_path)
        print(f"KMeans model saved to {model_path}")

    return clustered_df, kmeans


def assign_segment_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign business-friendly segment names based on cluster characteristics.

    Args:
        df (pd.DataFrame): Dataframe with cluster labels

    Returns:
        pd.DataFrame: Dataframe with segment names
    """
    # Make a copy of the dataframe
    df_with_segments = df.copy()

    # Calculate cluster profiles
    profiles = df.groupby("Cluster")[["Recency", "Frequency", "MonetaryValue"]].mean()

    # Normalize profiles for easier comparison
    for col in ["Recency", "Frequency", "MonetaryValue"]:
        profiles[f"{col}_Norm"] = (profiles[col] - profiles[col].min()) / (
            profiles[col].max() - profiles[col].min()
        )

    # For recency, lower is better, so invert the normalized value
    profiles["Recency_Norm"] = 1 - profiles["Recency_Norm"]

    # Assign segment names based on characteristics
    segment_names = {}

    for cluster in profiles.index:
        recency = profiles.loc[cluster, "Recency_Norm"]
        frequency = profiles.loc[cluster, "Frequency_Norm"]
        monetary = profiles.loc[cluster, "MonetaryValue_Norm"]

        # Set thresholds for high/medium/low classification
        high_threshold = 0.66
        low_threshold = 0.33

        # Determine segment name based on RFM values
        if (
            recency > high_threshold
            and frequency > high_threshold
            and monetary > high_threshold
        ):
            name = "Champions"
        elif recency > high_threshold and frequency > high_threshold:
            name = "Loyal Customers"
        elif recency > high_threshold and monetary > high_threshold:
            name = "Big Spenders"
        elif (
            recency < low_threshold
            and frequency < low_threshold
            and monetary < low_threshold
        ):
            name = "Dormant"
        elif recency < low_threshold:
            name = "At Risk"
        elif recency > high_threshold and frequency < low_threshold:
            name = "New Customers"
        elif frequency < low_threshold and monetary < low_threshold:
            name = "Rare Shoppers"
        else:
            name = "Average Customers"

        segment_names[cluster] = name

    # Map segment names to the dataframe
    df_with_segments["SegmentName"] = df_with_segments["Cluster"].map(segment_names)

    return df_with_segments


def save_clustering_results(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the clustering results to a CSV file.

    Args:
        df (pd.DataFrame): The dataframe with clustering results
        output_path (str): Path to save the CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the dataframe
    df.to_csv(output_path, index=False)
    print(f"Clustering results saved to {output_path}")


def clustering_pipeline(
    input_path: str, output_path: str, optimal_k: Optional[int] = None
) -> pd.DataFrame:
    """
    Run the complete clustering pipeline.

    Args:
        input_path (str): Path to the input CSV file
        output_path (str): Path to save the output CSV file
        optimal_k (int, optional): Number of clusters (if None, it will be determined automatically)

    Returns:
        pd.DataFrame: The dataframe with clustering results
    """
    # Load the data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    # Features for clustering
    features = ["Recency", "Frequency", "MonetaryValue"]
    print(f"Using features: {features}")

    # Find optimal number of clusters if not provided
    if optimal_k is None:
        print("Finding optimal number of clusters...")
        results = find_optimal_clusters(df, features, max_clusters=10)
        # Use silhouette score's recommendation as it's more reliable
        optimal_k = results["optimal_k_silhouette"]

    print(f"Applying KMeans clustering with {optimal_k} clusters...")
    clustered_df, kmeans_model = apply_kmeans_clustering(
        df,
        features=features,
        n_clusters=optimal_k,
        save_model=True,
        model_path=os.path.join("models", "kmeans_model.pkl"),
    )

    # Assign segment names
    print("Assigning segment names...")
    labeled_df = assign_segment_names(clustered_df)

    # Save the results
    save_clustering_results(labeled_df, output_path)

    return labeled_df


if __name__ == "__main__":
    # Run the clustering pipeline
    try:
        input_path = os.path.join("data", "processed", "customer_features.csv")
        output_path = os.path.join("data", "processed", "customer_segments.csv")

        print(f"Starting clustering pipeline...")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        # Process the data
        df = clustering_pipeline(input_path, output_path)

        print("\nClustering pipeline completed successfully!")
        print(f"Output shape: {df.shape}")

        # Display segment distribution
        if "SegmentName" in df.columns:
            segment_counts = df["SegmentName"].value_counts()
            print("\nCustomer Segment Distribution:")
            for segment, count in segment_counts.items():
                print(
                    f"  - {segment}: {count} customers ({count / len(df) * 100:.1f}%)"
                )

    except Exception as e:
        import traceback

        print(f"Error in clustering pipeline: {e}")
        print(traceback.format_exc())
