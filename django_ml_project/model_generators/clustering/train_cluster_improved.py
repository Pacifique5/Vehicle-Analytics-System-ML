import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

SEGMENT_FEATURES = ["estimated_income", "selling_price"]

df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")

X = df[SEGMENT_FEATURES]

# IMPROVEMENT 1: Standardize features (normalize the data)
# This helps when features have different scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# IMPROVEMENT 2: Try different numbers of clusters to find optimal
silhouette_scores = {}
for n_clusters in range(2, 8):
    kmeans_temp = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    cluster_labels = kmeans_temp.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores[n_clusters] = silhouette_avg
    print(f"For n_clusters = {n_clusters}, silhouette score = {silhouette_avg:.4f}")

# Find optimal number of clusters
optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
print(f"\nOptimal number of clusters: {optimal_clusters}")
print(f"Best silhouette score: {silhouette_scores[optimal_clusters]:.4f}")

# IMPROVEMENT 3: Train final model with optimal parameters
kmeans = KMeans(
    n_clusters=optimal_clusters, 
    random_state=42, 
    n_init=50,  # Increased from 10 to 50 for better initialization
    max_iter=500,  # Increased iterations
    algorithm='lloyd'  # More precise algorithm
)
df["cluster_id"] = kmeans.fit_predict(X_scaled)

centers = kmeans.cluster_centers_

# Transform centers back to original scale for interpretation
centers_original = scaler.inverse_transform(centers)

# Sort clusters by income
sorted_clusters = centers_original[:, 0].argsort()

# Create dynamic cluster mapping based on number of clusters
if optimal_clusters == 2:
    cluster_names = ["Economy", "Premium"]
elif optimal_clusters == 3:
    cluster_names = ["Economy", "Standard", "Premium"]
elif optimal_clusters == 4:
    cluster_names = ["Budget", "Economy", "Standard", "Premium"]
elif optimal_clusters == 5:
    cluster_names = ["Budget", "Economy", "Standard", "Premium", "Luxury"]
else:
    cluster_names = [f"Segment_{i+1}" for i in range(optimal_clusters)]

cluster_mapping = {sorted_clusters[i]: cluster_names[i] for i in range(optimal_clusters)}

df["client_class"] = df["cluster_id"].map(cluster_mapping)

# Save both the model and the scaler
joblib.dump(kmeans, "model_generators/clustering/clustering_model_improved.pkl")
joblib.dump(scaler, "model_generators/clustering/scaler_improved.pkl")

silhouette_avg = round(silhouette_score(X_scaled, df["cluster_id"]), 4)

# Calculate Coefficient of Variation for each cluster
cluster_cv = {}
for cluster_name in cluster_names:
    cluster_data = df[df["client_class"] == cluster_name][SEGMENT_FEATURES]
    if len(cluster_data) > 0:
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

def evaluate_clustering_model_improved():
    return {
        "silhouette": silhouette_avg,
        "coefficient_variation": overall_cv,
        "cluster_cv": cluster_cv,
        "optimal_clusters": optimal_clusters,
        "all_scores": silhouette_scores,
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

print(f"\n{'='*60}")
print(f"IMPROVED MODEL RESULTS")
print(f"{'='*60}")
print(f"Silhouette Score: {silhouette_avg}")
print(f"Coefficient of Variation: {overall_cv}%")
print(f"Number of Clusters: {optimal_clusters}")
print(f"{'='*60}")
