import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

SEGMENT_FEATURES = ["estimated_income", "selling_price"]

df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")

X = df[SEGMENT_FEATURES]

kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
df["cluster_id"] = kmeans.fit_predict(X)

centers = kmeans.cluster_centers_

# Sort clusters by income
sorted_clusters = centers[:, 0].argsort()
cluster_mapping = {
    sorted_clusters[0]: "Economy",
    sorted_clusters[1]: "Standard",
    sorted_clusters[2]: "Premium",
}

df["client_class"] = df["cluster_id"].map(cluster_mapping)

joblib.dump(kmeans, "model_generators/clustering/clustering_model.pkl")

silhouette_avg = round(silhouette_score(X, df["cluster_id"]), 2)

# Calculate Coefficient of Variation for each cluster
cluster_cv = {}
for cluster_name in ["Economy", "Standard", "Premium"]:
    cluster_data = df[df["client_class"] == cluster_name][SEGMENT_FEATURES]
    # CV = (standard deviation / mean) * 100
    cv_income = (cluster_data["estimated_income"].std() / cluster_data["estimated_income"].mean()) * 100
    cv_price = (cluster_data["selling_price"].std() / cluster_data["selling_price"].mean()) * 100
    cluster_cv[cluster_name] = {
        "income_cv": round(cv_income, 2),
        "price_cv": round(cv_price, 2),
        "avg_cv": round((cv_income + cv_price) / 2, 2)
    }

# Overall coefficient of variation
overall_cv_income = (df["estimated_income"].std() / df["estimated_income"].mean()) * 100
overall_cv_price = (df["selling_price"].std() / df["selling_price"].mean()) * 100
overall_cv = round((overall_cv_income + overall_cv_price) / 2, 2)

cluster_summary = df.groupby("client_class")[SEGMENT_FEATURES].mean()
cluster_counts = df["client_class"].value_counts().reset_index()
cluster_counts.columns = ["client_class", "count"]
cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")

comparison_df = df[["client_name", "estimated_income", "selling_price", "client_class"]]

def evaluate_clustering_model():
    return {
        "silhouette": silhouette_avg,
        "coefficient_variation": overall_cv,
        "cluster_cv": cluster_cv,
        "summary": cluster_summary.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
    }
