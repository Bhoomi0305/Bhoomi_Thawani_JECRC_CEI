# ==========================================================
# CUSTOMER SEGMENTATION AND PREDICTIVE ANALYTICS SYSTEM
# K-Means + DBSCAN + Random Forest + XGBoost
# ==========================================================

# Import Libraries

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# DATA LOADING
# ==========================================================

def load_country_dataset(file_path):
    """
    Load dataset from CSV file.
    """
    return pd.read_csv(file_path)


# ==========================================================
# DATA PREPROCESSING
# ==========================================================

def preprocess_dataset(dataframe):

    feature_data = dataframe.drop("country", axis=1)

    standardizer = StandardScaler()

    normalized_features = standardizer.fit_transform(
        feature_data
    )

    return feature_data, normalized_features


# ==========================================================
# K-MEANS CLUSTERING
# ==========================================================

def perform_kmeans_clustering(
    normalized_features,
    total_clusters=3
):

    segment_model = KMeans(
        n_clusters=total_clusters,
        random_state=42,
        n_init=10
    )

    segment_ids = segment_model.fit_predict(
        normalized_features
    )

    return segment_model, segment_ids


# ==========================================================
# DBSCAN CLUSTERING
# ==========================================================

def perform_dbscan_clustering(
    normalized_features
):

    density_model = DBSCAN(
        eps=1.5,
        min_samples=5
    )

    density_clusters = density_model.fit_predict(
        normalized_features
    )

    return density_clusters


# ==========================================================
# ELBOW METHOD
# ==========================================================

def visualize_elbow_curve(
    normalized_features
):

    inertia_values = []

    for cluster_count in range(2, 11):

        model = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=10
        )

        model.fit(normalized_features)

        inertia_values.append(
            model.inertia_
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(2, 11),
        inertia_values,
        marker='o'
    )

    plt.title(
        "Elbow Method for Optimal Clusters"
    )

    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")

    plt.grid(True)

    plt.show()


# ==========================================================
# PCA VISUALIZATION
# ==========================================================

def visualize_customer_segments(
    normalized_features,
    segment_ids
):

    pca_model = PCA(
        n_components=2
    )

    pca_features = pca_model.fit_transform(
        normalized_features
    )

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        x=pca_features[:, 0],
        y=pca_features[:, 1],
        hue=segment_ids,
        palette="Set2",
        s=100
    )

    plt.title(
        "Customer Segments using PCA"
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.show()


# ==========================================================
# RANDOM FOREST MODEL
# ==========================================================

def train_random_forest_model(
    train_features,
    train_labels
):

    rf_classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )

    rf_classifier.fit(
        train_features,
        train_labels
    )

    return rf_classifier


# ==========================================================
# XGBOOST MODEL
# ==========================================================

def train_xgboost_model(
    train_features,
    train_labels,
    total_classes
):

    boosting_classifier = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        objective="multi:softmax",
        num_class=total_classes,
        random_state=42
    )

    boosting_classifier.fit(
        train_features,
        train_labels
    )

    return boosting_classifier


# ==========================================================
# MODEL EVALUATION
# ==========================================================

def evaluate_model(
    actual_labels,
    predicted_labels,
    model_name
):

    print("\n")
    print("=" * 50)
    print(model_name)
    print("=" * 50)

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nClassification Report\n")

    print(
        classification_report(
            actual_labels,
            predicted_labels
        )
    )

    print("\nConfusion Matrix\n")

    print(
        confusion_matrix(
            actual_labels,
            predicted_labels
        )
    )

    return accuracy


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def display_feature_importance(
    trained_model,
    feature_names
):

    important_features = pd.DataFrame({

        "Feature": feature_names,

        "Importance":
            trained_model.feature_importances_

    })

    important_features = (
        important_features
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=important_features,
        x="Importance",
        y="Feature"
    )

    plt.title(
        "Feature Importance Analysis"
    )

    plt.show()

    return important_features


# ==========================================================
# MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":

    print("\nLoading Dataset...\n")

    country_df = load_country_dataset(
        "Country-data.csv"
    )

    print(country_df.head())

    print(
        "\nDataset Shape:",
        country_df.shape
    )

    # ----------------------------------------
    # Preprocessing
    # ----------------------------------------

    feature_data, normalized_features = (
        preprocess_dataset(country_df)
    )

    # ----------------------------------------
    # Elbow Method
    # ----------------------------------------

    visualize_elbow_curve(
        normalized_features
    )

    # ----------------------------------------
    # K-Means Clustering
    # ----------------------------------------

    segment_model, segment_ids = (
        perform_kmeans_clustering(
            normalized_features,
            total_clusters=3
        )
    )

    country_df["Segment"] = segment_ids

    print(
        "\nSegment Distribution:\n"
    )

    print(
        country_df["Segment"]
        .value_counts()
    )

    silhouette_value = silhouette_score(
        normalized_features,
        segment_ids
    )

    print(
        "\nSilhouette Score:",
        round(
            silhouette_value,
            4
        )
    )

    # ----------------------------------------
    # DBSCAN
    # ----------------------------------------

    density_clusters = (
        perform_dbscan_clustering(
            normalized_features
        )
    )

    country_df[
        "Density_Cluster"
    ] = density_clusters

    print(
        "\nDBSCAN Cluster Counts:\n"
    )

    print(
        country_df[
            "Density_Cluster"
        ].value_counts()
    )

    # ----------------------------------------
    # PCA Visualization
    # ----------------------------------------

    visualize_customer_segments(
        normalized_features,
        segment_ids
    )

    # ----------------------------------------
    # Train-Test Split
    # ----------------------------------------

    target_labels = segment_ids

    (
        train_features,
        test_features,
        train_labels,
        test_labels

    ) = train_test_split(

        normalized_features,
        target_labels,

        test_size=0.20,

        random_state=42,

        stratify=target_labels
    )

    # ----------------------------------------
    # Random Forest
    # ----------------------------------------

    rf_classifier = (
        train_random_forest_model(
            train_features,
            train_labels
        )
    )

    rf_predictions = (
        rf_classifier.predict(
            test_features
        )
    )

    rf_accuracy = evaluate_model(
        test_labels,
        rf_predictions,
        "Random Forest Classifier"
    )

    # ----------------------------------------
    # XGBoost
    # ----------------------------------------

    boosting_classifier = (
        train_xgboost_model(
            train_features,
            train_labels,
            total_classes=3
        )
    )

    boost_predictions = (
        boosting_classifier.predict(
            test_features
        )
    )

    xgb_accuracy = evaluate_model(
        test_labels,
        boost_predictions,
        "XGBoost Classifier"
    )

    # ----------------------------------------
    # Accuracy Comparison
    # ----------------------------------------

    comparison_df = pd.DataFrame({

        "Model": [
            "Random Forest",
            "XGBoost"
        ],

        "Accuracy": [
            rf_accuracy,
            xgb_accuracy
        ]

    })

    print(
        "\nModel Comparison\n"
    )

    print(comparison_df)

    plt.figure(figsize=(6, 5))

    sns.barplot(
        data=comparison_df,
        x="Model",
        y="Accuracy"
    )

    plt.title(
        "Model Performance Comparison"
    )

    plt.show()

    # ----------------------------------------
    # Feature Importance
    # ----------------------------------------

    print(
        "\nTop Important Features\n"
    )

    display_feature_importance(
        rf_classifier,
        feature_data.columns
    )

    # ----------------------------------------
    # Segment Analysis
    # ----------------------------------------

    cluster_analysis = (
        country_df.groupby("Segment")
        .mean(numeric_only=True)
    )

    print(
        "\nCustomer Segment Analysis\n"
    )

    print(cluster_analysis)

    # ----------------------------------------
    # Export Results
    # ----------------------------------------

    country_df.to_csv(
        "Customer_Segmentation_Output.csv",
        index=False
    )

    print(
        "\nProject Execution Completed Successfully!"
    )
