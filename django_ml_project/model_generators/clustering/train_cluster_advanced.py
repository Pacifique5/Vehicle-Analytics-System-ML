import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.metrics import silhouette_score
import joblib
import json

def evaluate_clustering_model_advanced():
    """Evaluate the advanced clustering model without retraining"""
    # Load the saved model bundle
    model_bundle = joblib.load("model_generators/clustering/clustering_model_advanced.pkl")
    
    # Load the dataset
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    
    # Extract model components
    kmeans = model_bundle["kmeans"]
    scaler = model_bundle["scaler"]
    features = model_bundle["features"]
    cluster_mapping = model_bundle["cluster_mapping"]
    cluster_names = model_bundle["cluster_names"]
    silhouette_avg = model_bundle["silhouette_score"]
    overall_cv = model_bundle["coefficient_variation"]
    
    # Prepare data
    X = df[features].copy()
    
    # Apply scaler if exists
    if scaler is not None:
        X_transformed = scaler.transform(X)
    else:
        X_transformed = X.values
    
    # Predict clusters
    df["cluster_id"] = kmeans.predict(X_transformed)
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
            
            # Calculate average CV for the cluster
            if cv_dict:
                avg_cv = round(np.mean(list(cv_dict.values())), 2)
                cv_dict['average'] = avg_cv
            
            cluster_cv[cluster_name] = cv_dict
    
    # Create summary tables
    cluster_summary = df.groupby("client_class")[features].mean()
    cluster_counts = df["client_class"].value_counts().reset_index()
    cluster_counts.columns = ["client_class", "count"]
    cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")
    
    comparison_df = df[["client_name"] + features + ["client_class"]]
    
    return {
        "silhouette": silhouette_avg,
        "coefficient_variation": overall_cv,
        "cluster_cv": cluster_cv,
        "optimal_clusters": len(cluster_names),
        "features_used": features,
        "scaler_used": model_bundle["scaler_name"],
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

if __name__ == "__main__":
    print("="*70)
    print("ADVANCED CLUSTERING MODEL - Systematic Feature & Scaler Search")
    print("="*70)

    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")

    # Define feature combinations to test
    # Focus on combinations that might reduce CV by creating tighter clusters
    feature_configs = [
        ["selling_price"],  # Current best for silhouette
        ["estimated_income", "selling_price"],
        ["selling_price", "year"],
        ["selling_price", "kilometers_driven"],
        ["selling_price", "seating_capacity"],
        ["estimated_income"],
        ["estimated_income", "selling_price", "year"],
        ["estimated_income", "selling_price", "kilometers_driven"],
    ]

    # Define scaler options
    scaler_configs = {
        "none": None,
        "standard": StandardScaler(),
        "quantile": QuantileTransformer(output_distribution="normal", random_state=42),
        "robust": RobustScaler(),
    }

    # Define cluster counts to test
    # More clusters typically reduce CV by creating more homogeneous groups
    cluster_counts = [2, 3, 4, 5, 6, 7]

    print("\nSearching for best configuration (optimizing for Silhouette > 0.9 AND low CV)...")
    print("-" * 70)

    best_score = -1
    best_config = None
    all_results = []

    for features in feature_configs:
        for scaler_name, scaler in scaler_configs.items():
            for k in cluster_counts:
                try:
                    # Prepare data
                    X = df[features].copy().dropna()
                    
                    # Apply scaling if specified
                    if scaler is not None:
                        X_transformed = scaler.fit_transform(X)
                    else:
                        X_transformed = X.values
                    
                    # Train KMeans
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=500)
                    labels = kmeans.fit_predict(X_transformed)
                    
                    # Calculate silhouette score
                    score = silhouette_score(X_transformed, labels)
                    
                    # Calculate CV for this configuration
                    temp_df = df[features].copy()
                    temp_df['cluster'] = labels
                    
                    # Calculate overall CV
                    cv_values = []
                    for feature in features:
                        mean_val = temp_df[feature].mean()
                        std_val = temp_df[feature].std()
                        if mean_val != 0:
                            cv = (std_val / mean_val) * 100
                            cv_values.append(cv)
                    
                    avg_cv = np.mean(cv_values) if cv_values else 0
                    
                    result = {
                        "features": features,
                        "scaler": scaler_name,
                        "k": k,
                        "score": score,
                        "cv": round(avg_cv, 2)
                    }
                    all_results.append(result)
                    
                    # Track best - NEW STRATEGY: Balance silhouette and CV
                    # Prioritize configurations with CV < 40% and silhouette >= 0.80
                    is_better = False
                    
                    if score > 0.9:
                        # If we achieve > 0.9, prefer lower CV
                        if best_score < 0.9 or (best_score > 0.9 and avg_cv < best_config.get('cv', 999)):
                            is_better = True
                    elif score >= 0.80 and avg_cv < 40:
                        # PRIORITIZE: Good silhouette (>= 0.80) with very low CV (< 40%)
                        if best_score < 0.9:
                            if avg_cv < best_config.get('cv', 999) or score > best_score:
                                is_better = True
                    elif score > best_score and best_score < 0.80:
                        # If we haven't found good solution yet, take best silhouette
                        is_better = True
                    
                    if is_better:
                        best_score = score
                        best_config = result.copy()
                        best_config["scaler_obj"] = scaler
                        best_config["kmeans"] = kmeans
                        best_config["X_transformed"] = X_transformed
                        best_config["labels"] = labels
                    
                    print(f"Features: {str(features):50s} Scaler: {scaler_name:10s} k={k} Score: {score:.4f} CV: {avg_cv:.2f}%")
                    
                except Exception as e:
                    print(f"Features: {str(features):50s} Scaler: {scaler_name:10s} k={k} ERROR: {str(e)}")

    print("\n" + "="*70)
    print("BEST CONFIGURATION FOUND")
    print("="*70)
    print(f"Features: {best_config['features']}")
    print(f"Scaler: {best_config['scaler']}")
    print(f"Clusters: {best_config['k']}")
    print(f"Silhouette Score: {best_config['score']:.4f}")
    print(f"Coefficient of Variation: {best_config['cv']:.2f}%")
    print("="*70)

    # Train final model with best configuration
    print("\nTraining final model with best configuration...")

    X_final = df[best_config['features']].copy()

    # Apply best scaler
    if best_config['scaler_obj'] is not None:
        scaler_final = best_config['scaler_obj']
        X_final_transformed = scaler_final.fit_transform(X_final)
    else:
        scaler_final = None
        X_final_transformed = X_final.values

    # Train final KMeans
    kmeans_final = KMeans(n_clusters=best_config['k'], random_state=42, n_init=50, max_iter=500)
    df["cluster_id"] = kmeans_final.fit_predict(X_final_transformed)

    # Create cluster mapping
    centers = kmeans_final.cluster_centers_

    # Transform centers back to original scale if scaler was used
    if scaler_final is not None:
        centers_original = scaler_final.inverse_transform(centers)
    else:
        centers_original = centers

    # Sort clusters by first feature (usually income or price)
    sorted_clusters = centers_original[:, 0].argsort()

    # Create dynamic cluster names
    if best_config['k'] == 2:
        cluster_names = ["Economy", "Premium"]
    elif best_config['k'] == 3:
        cluster_names = ["Economy", "Standard", "Premium"]
    elif best_config['k'] == 4:
        cluster_names = ["Budget", "Economy", "Standard", "Premium"]
    elif best_config['k'] == 5:
        cluster_names = ["Budget", "Economy", "Mid-Range", "Standard", "Premium"]
    elif best_config['k'] == 6:
        cluster_names = ["Budget", "Economy", "Mid-Range", "Standard", "Premium", "Luxury"]
    elif best_config['k'] == 7:
        cluster_names = ["Budget", "Economy", "Mid-Economy", "Mid-Range", "Standard", "Premium", "Luxury"]
    else:
        cluster_names = [f"Segment_{i+1}" for i in range(best_config['k'])]

    cluster_mapping = {sorted_clusters[i]: cluster_names[i] for i in range(best_config['k'])}
    df["client_class"] = df["cluster_id"].map(cluster_mapping)

    # Calculate metrics
    silhouette_avg = round(best_config['score'], 4)

    # Calculate Coefficient of Variation
    cluster_cv = {}
    for cluster_name in cluster_names:
        cluster_data = df[df["client_class"] == cluster_name][best_config['features']]
        if len(cluster_data) > 0:
            cv_dict = {}
            for feature in best_config['features']:
                mean_val = cluster_data[feature].mean()
                std_val = cluster_data[feature].std()
                if mean_val != 0:
                    cv = (std_val / mean_val) * 100
                    cv_dict[feature] = round(cv, 2)
                else:
                    cv_dict[feature] = 0.0
            
            # Calculate average CV for the cluster
            if cv_dict:
                avg_cv = round(np.mean(list(cv_dict.values())), 2)
                cv_dict['average'] = avg_cv
            
            cluster_cv[cluster_name] = cv_dict

    # Overall CV
    overall_cv_dict = {}
    for feature in best_config['features']:
        mean_val = df[feature].mean()
        std_val = df[feature].std()
        if mean_val != 0:
            cv = (std_val / mean_val) * 100
            overall_cv_dict[feature] = round(cv, 2)
        else:
            overall_cv_dict[feature] = 0.0

    # Calculate average overall CV
    if overall_cv_dict:
        overall_cv = round(np.mean(list(overall_cv_dict.values())), 2)
    else:
        overall_cv = 0.0

    # Save model and metadata
    model_bundle = {
        "kmeans": kmeans_final,
        "scaler": scaler_final,
        "features": best_config['features'],
        "scaler_name": best_config['scaler'],
        "n_clusters": best_config['k'],
        "cluster_mapping": cluster_mapping,
        "cluster_names": cluster_names,
        "silhouette_score": silhouette_avg,
        "coefficient_variation": overall_cv
    }

    joblib.dump(model_bundle, "model_generators/clustering/clustering_model_advanced.pkl")

    # Save config as JSON for reference
    config_json = {
        "features": best_config['features'],
        "scaler": best_config['scaler'],
        "n_clusters": best_config['k'],
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
    print(f"Number of Clusters: {best_config['k']}")
    print(f"Features Used: {', '.join(best_config['features'])}")
    print(f"Scaler Used: {best_config['scaler']}")
    print(f"Cluster Names: {', '.join(cluster_names)}")
    print("="*70)

    if silhouette_avg > 0.9:
        print("\n🎉 SUCCESS! Achieved Silhouette Score > 0.9!")
    else:
        print(f"\n⚠️  Note: Score is {silhouette_avg} (< 0.9)")
        print("   This represents the best achievable clustering for this dataset.")
        print("   Real-world customer data rarely achieves perfect separation (0.9+).")

    print("\nModel saved to: clustering_model_advanced.pkl")
    print("Config saved to: clustering_config_advanced.json")
    print("="*70)
