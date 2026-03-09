"""
Test more clusters to reduce CV further
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import silhouette_score
import joblib
import json

print("="*70)
print("TESTING MORE CLUSTERS TO REDUCE CV")
print("="*70)

df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")

# Test with selling_price + year (best CV so far)
features = ["selling_price", "year"]
scaler = QuantileTransformer(output_distribution="normal", random_state=42)

print(f"\nFeatures: {features}")
print(f"Scaler: QuantileTransformer")
print("-" * 70)
print(f"{'k':<5} {'Silhouette':<12} {'Overall CV':<12} {'Avg Cluster CV':<15} {'Max Cluster CV':<15} {'Status'}")
print("-" * 70)

best_config = None
best_cv = 999

for k in range(2, 15):  # Test up to 14 clusters
    # Prepare data
    X = df[features].copy()
    X_transformed = scaler.fit_transform(X)
    
    # Train KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=100, max_iter=1000)
    labels = kmeans.fit_predict(X_transformed)
    
    # Calculate silhouette score
    silhouette = silhouette_score(X_transformed, labels)
    
    # Calculate overall CV
    temp_df = df[features].copy()
    temp_df['cluster'] = labels
    
    cv_values = []
    for feature in features:
        mean_val = temp_df[feature].mean()
        std_val = temp_df[feature].std()
        if mean_val != 0:
            cv = (std_val / mean_val) * 100
            cv_values.append(cv)
    
    avg_cv = round(np.mean(cv_values), 2) if cv_values else 0
    
    # Calculate per-cluster CV
    cluster_cvs = []
    for cluster_id in range(k):
        cluster_data = temp_df[temp_df['cluster'] == cluster_id][features]
        if len(cluster_data) > 1:
            cluster_cv_vals = []
            for feature in features:
                mean_val = cluster_data[feature].mean()
                std_val = cluster_data[feature].std()
                if mean_val != 0:
                    cv = (std_val / mean_val) * 100
                    cluster_cv_vals.append(cv)
            if cluster_cv_vals:
                cluster_cvs.append(np.mean(cluster_cv_vals))
    
    max_cluster_cv = round(max(cluster_cvs), 2) if cluster_cvs else 0
    avg_cluster_cv = round(np.mean(cluster_cvs), 2) if cluster_cvs else 0
    
    status = ""
    if silhouette > 0.9 and avg_cluster_cv < 30:
        status = "✅ BOTH!"
    elif silhouette > 0.9:
        status = "✅ Sil"
    elif avg_cluster_cv < 30:
        status = "✅ CV"
    
    print(f"{k:<5} {silhouette:<12.4f} {avg_cv:<12.2f} {avg_cluster_cv:<15.2f} {max_cluster_cv:<15.2f} {status}")
    
    # Track best configuration (prioritize low avg cluster CV if silhouette > 0.7)
    if silhouette >= 0.7 and avg_cluster_cv < best_cv:
        best_cv = avg_cluster_cv
        best_config = {
            'k': k,
            'silhouette': silhouette,
            'cv': avg_cv,
            'avg_cluster_cv': avg_cluster_cv,
            'max_cluster_cv': max_cluster_cv,
            'kmeans': kmeans,
            'labels': labels,
            'features': features
        }

print("-" * 70)

if best_config:
    print(f"\n✅ BEST CONFIGURATION FOUND:")
    print(f"   Clusters: {best_config['k']}")
    print(f"   Silhouette: {best_config['silhouette']:.4f}")
    print(f"   Overall CV: {best_config['cv']:.2f}%")
    print(f"   Avg Cluster CV: {best_cv:.2f}%")
    print(f"   Max Cluster CV: {best_config['max_cluster_cv']:.2f}%")
    
    # Use best config
    k = best_config['k']
    silhouette_avg = round(best_config['silhouette'], 4)
    overall_cv = round(best_config['cv'], 2)
    kmeans = best_config['kmeans']
    features = best_config['features']
    
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL")
    print("="*70)
    
    # Retrain and save
    X = df[features].copy()
    X_transformed = scaler.fit_transform(X)
    df["cluster_id"] = kmeans.predict(X_transformed)
    
    # Create cluster mapping
    centers = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers)
    sorted_clusters = centers_original[:, 0].argsort()
    
    # Dynamic cluster names
    if k == 2:
        cluster_names = ["Economy", "Premium"]
    elif k == 3:
        cluster_names = ["Economy", "Standard", "Premium"]
    elif k == 4:
        cluster_names = ["Budget", "Economy", "Standard", "Premium"]
    elif k == 5:
        cluster_names = ["Budget", "Economy", "Mid-Range", "Standard", "Premium"]
    elif k == 6:
        cluster_names = ["Budget", "Economy", "Mid-Range", "Standard", "Premium", "Luxury"]
    elif k == 7:
        cluster_names = ["Budget", "Economy", "Mid-Economy", "Mid-Range", "Standard", "Premium", "Luxury"]
    elif k == 8:
        cluster_names = ["Ultra-Budget", "Budget", "Economy", "Mid-Economy", "Mid-Range", "Standard", "Premium", "Luxury"]
    elif k == 9:
        cluster_names = ["Ultra-Budget", "Budget", "Economy", "Mid-Economy", "Mid-Range", "Standard", "Premium", "Luxury", "Ultra-Luxury"]
    elif k == 10:
        cluster_names = ["Ultra-Budget", "Budget", "Low-Economy", "Economy", "Mid-Economy", "Mid-Range", "Standard", "Premium", "Luxury", "Ultra-Luxury"]
    else:
        cluster_names = [f"Segment_{i+1}" for i in range(k)]
    
    cluster_mapping = {sorted_clusters[i]: cluster_names[i] for i in range(k)}
    df["client_class"] = df["cluster_id"].map(cluster_mapping)
    
    # Calculate CV per cluster
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
                avg_cv_cluster = round(np.mean(list(cv_dict.values())), 2)
                cv_dict['average'] = avg_cv_cluster
            
            cluster_cv[cluster_name] = cv_dict
    
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
    
    print("\n📊 CV Breakdown by Cluster:")
    for cluster_name, cv_data in cluster_cv.items():
        print(f"  {cluster_name}: {cv_data.get('average', 0):.2f}%")
    
    print("\n✅ Model saved to: clustering_model_advanced.pkl")
    print("✅ Config saved to: clustering_config_advanced.json")
    print("="*70)
else:
    print("\n❌ No suitable configuration found")
