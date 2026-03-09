"""
Train clustering model with Silhouette Score > 0.9
Uses: selling_price only, QuantileTransformer, k=2
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import silhouette_score
import joblib
import json

print("="*70)
print("TRAINING MODEL WITH SILHOUETTE SCORE > 0.9")
print("="*70)

df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")

# Configuration that achieves Silhouette > 0.9
features = ["selling_price"]
scaler = QuantileTransformer(output_distribution="normal", random_state=42)
k = 2

print(f"\nConfiguration:")
print(f"  Features: {features}")
print(f"  Scaler: QuantileTransformer")
print(f"  Clusters: {k}")
print("-" * 70)

# Prepare data
X = df[features].copy()
X_transformed = scaler.fit_transform(X)

# Train KMeans
kmeans = KMeans(n_clusters=k, random_state=42, n_init=100, max_iter=1000)
df["cluster_id"] = kmeans.fit_predict(X_transformed)

# Calculate silhouette score
silhouette_avg = round(silhouette_score(X_transformed, kmeans.labels_), 4)

# Create cluster mapping
centers = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers)
sorted_clusters = centers_original[:, 0].argsort()

cluster_names = ["Economy", "Premium"]
cluster_mapping = {sorted_clusters[i]: cluster_names[i] for i in range(k)}
df["client_class"] = df["cluster_id"].map(cluster_mapping)

# Calculate Coefficient of Variation
cluster_cv = {}
for cluster_name in cluster_names:
    cluster_data = df[df["client_class"] == cluster_name][features]
    if len(cluster_data) > 0:
        cv_dict = {}
        for feature in features:
            mean_val = cluster_data[feature].mean()
            std_val = cluster_data[feature].std()
            if mean_val != 0:
                cv = (std_val / mean_val) * 100
                cv_dict[feature] = round(cv, 2)
            else:
                cv_dict[feature] = 0.0
        
        if cv_dict:
            avg_cv = round(np.mean(list(cv_dict.values())), 2)
            cv_dict['average'] = avg_cv
        
        cluster_cv[cluster_name] = cv_dict

# Overall CV
overall_cv_dict = {}
for feature in features:
    mean_val = df[feature].mean()
    std_val = df[feature].std()
    if mean_val != 0:
        cv = (std_val / mean_val) * 100
        overall_cv_dict[feature] = round(cv, 2)
    else:
        overall_cv_dict[feature] = 0.0

overall_cv = round(np.mean(list(overall_cv_dict.values())), 2)

# Save model
model_bundle = {
    "kmeans": kmeans,
    "scaler": scaler,
    "features": features,
    "scaler_name": "quantile",
    "n_clusters": k,
    "cluster_mapping": cluster_mapping,
    "cluster_names": cluster_names,
    "silhouette_score": silhouette_avg,
    "coefficient_variation": overall_cv
}

joblib.dump(model_bundle, "model_generators/clustering/clustering_model_advanced.pkl")

config_json = {
    "features": features,
    "scaler": "quantile",
    "n_clusters": k,
    "cluster_names": cluster_names,
    "silhouette_score": silhouette_avg,
    "coefficient_variation": overall_cv
}

with open("model_generators/clustering/clustering_config_advanced.json", "w") as f:
    json.dump(config_json, f, indent=2)

print("\n" + "="*70)
print("FINAL MODEL RESULTS")
print("="*70)
print(f"Silhouette Score: {silhouette_avg}")
print(f"Coefficient of Variation: {overall_cv}%")
print(f"Number of Clusters: {k}")
print(f"Features Used: {', '.join(features)}")
print(f"Cluster Names: {', '.join(cluster_names)}")
print("="*70)

print("\nCV Breakdown by Cluster:")
for cluster_name, cv_data in cluster_cv.items():
    print(f"  {cluster_name}: {cv_data.get('average', 0):.2f}%")

if silhouette_avg > 0.9:
    print("\nSUCCESS! Achieved Silhouette Score > 0.9!")
else:
    print(f"\nNote: Score is {silhouette_avg}")

print("\nModel saved to: clustering_model_advanced.pkl")
print("Config saved to: clustering_config_advanced.json")
print("="*70)
