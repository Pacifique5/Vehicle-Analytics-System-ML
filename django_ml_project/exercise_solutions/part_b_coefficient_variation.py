"""
EXERCISE PART B: Calculate and Display Coefficient of Variation (5 marks)

This standalone script:
- Calculates the Coefficient of Variation (CV) for the clustering model
- Displays CV alongside the Silhouette Score
- Shows per-cluster CV breakdown

Formula: CV = (Standard Deviation / Mean) × 100

Usage:
    python exercise_solutions/part_b_coefficient_variation.py
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

def calculate_coefficient_variation():
    """
    Calculate Coefficient of Variation for clustering analysis
    """
    
    print("="*80)
    print("EXERCISE PART B: Coefficient of Variation Calculation (5 marks)")
    print("="*80)
    
    # Load dataset
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    
    SEGMENT_FEATURES = ["estimated_income", "selling_price"]
    X = df[SEGMENT_FEATURES]
    
    # Load or train clustering model
    try:
        kmeans = joblib.load("model_generators/clustering/clustering_model.pkl")
        print("\n✅ Loaded existing clustering model")
    except:
        print("\n⚠️  Training new clustering model...")
        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        kmeans.fit(X)
    
    # Predict clusters
    df["cluster_id"] = kmeans.predict(X)
    
    # Map clusters to names
    centers = kmeans.cluster_centers_
    sorted_clusters = centers[:, 0].argsort()
    cluster_mapping = {
        sorted_clusters[0]: "Economy",
        sorted_clusters[1]: "Standard",
        sorted_clusters[2]: "Premium",
    }
    df["client_class"] = df["cluster_id"].map(cluster_mapping)
    
    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(X, df["cluster_id"])
    
    print("\n" + "-"*80)
    print("CLUSTERING EVALUATION METRICS")
    print("-"*80)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    # Calculate Overall Coefficient of Variation
    print("\n" + "-"*80)
    print("OVERALL COEFFICIENT OF VARIATION")
    print("-"*80)
    
    overall_cv_income = (df["estimated_income"].std() / df["estimated_income"].mean()) * 100
    overall_cv_price = (df["selling_price"].std() / df["selling_price"].mean()) * 100
    overall_cv = (overall_cv_income + overall_cv_price) / 2
    
    print(f"Income CV:        {overall_cv_income:.2f}%")
    print(f"Price CV:         {overall_cv_price:.2f}%")
    print(f"Average CV:       {overall_cv:.2f}%")
    
    # Calculate Per-Cluster Coefficient of Variation
    print("\n" + "-"*80)
    print("PER-CLUSTER COEFFICIENT OF VARIATION")
    print("-"*80)
    
    cluster_cv_results = {}
    
    for cluster_name in ["Economy", "Standard", "Premium"]:
        cluster_data = df[df["client_class"] == cluster_name][SEGMENT_FEATURES]
        
        if len(cluster_data) > 0:
            # Calculate CV for income
            cv_income = (cluster_data["estimated_income"].std() / 
                        cluster_data["estimated_income"].mean()) * 100
            
            # Calculate CV for price
            cv_price = (cluster_data["selling_price"].std() / 
                       cluster_data["selling_price"].mean()) * 100
            
            # Average CV
            avg_cv = (cv_income + cv_price) / 2
            
            cluster_cv_results[cluster_name] = {
                "income_cv": cv_income,
                "price_cv": cv_price,
                "avg_cv": avg_cv,
                "count": len(cluster_data)
            }
            
            print(f"\n{cluster_name} Cluster:")
            print(f"  Samples:      {len(cluster_data)}")
            print(f"  Income CV:    {cv_income:.2f}%")
            print(f"  Price CV:     {cv_price:.2f}%")
            print(f"  Average CV:   {avg_cv:.2f}%")
    
    # Interpretation
    print("\n" + "-"*80)
    print("INTERPRETATION")
    print("-"*80)
    print("Coefficient of Variation (CV) measures relative variability:")
    print("  • Lower CV = More homogeneous cluster (better)")
    print("  • Higher CV = More heterogeneous cluster (needs refinement)")
    print("\nCV complements Silhouette Score for cluster quality assessment:")
    print(f"  • Silhouette Score: {silhouette_avg:.4f} (measures separation)")
    print(f"  • Average CV:       {overall_cv:.2f}% (measures variability)")
    
    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    summary_data = []
    for cluster_name, cv_data in cluster_cv_results.items():
        summary_data.append({
            'Cluster': cluster_name,
            'Samples': cv_data['count'],
            'Income CV (%)': f"{cv_data['income_cv']:.2f}",
            'Price CV (%)': f"{cv_data['price_cv']:.2f}",
            'Average CV (%)': f"{cv_data['avg_cv']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Add overall row
    print("-"*80)
    print(f"{'Overall':<15} {len(df):<10} {overall_cv_income:>13.2f} {overall_cv_price:>13.2f} {overall_cv:>15.2f}")
    print("="*80)
    
    # Save results to file
    output_file = "exercise_solutions/coefficient_variation_results.txt"
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXERCISE PART B: Coefficient of Variation Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Silhouette Score: {silhouette_avg:.4f}\n")
        f.write(f"Overall Coefficient of Variation: {overall_cv:.2f}%\n\n")
        f.write("Per-Cluster Results:\n")
        f.write("-"*80 + "\n")
        for cluster_name, cv_data in cluster_cv_results.items():
            f.write(f"\n{cluster_name} Cluster:\n")
            f.write(f"  Samples:      {cv_data['count']}\n")
            f.write(f"  Income CV:    {cv_data['income_cv']:.2f}%\n")
            f.write(f"  Price CV:     {cv_data['price_cv']:.2f}%\n")
            f.write(f"  Average CV:   {cv_data['avg_cv']:.2f}%\n")
        f.write("\n" + "="*80 + "\n")
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return {
        'silhouette_score': silhouette_avg,
        'overall_cv': overall_cv,
        'cluster_cv': cluster_cv_results
    }


if __name__ == "__main__":
    results = calculate_coefficient_variation()
    
    print("\n" + "="*80)
    print("PART B COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nKey Findings:")
    print(f"  • Silhouette Score: {results['silhouette_score']:.4f}")
    print(f"  • Coefficient of Variation: {results['overall_cv']:.2f}%")
    print(f"  • Number of Clusters: {len(results['cluster_cv'])}")
    print("="*80)
