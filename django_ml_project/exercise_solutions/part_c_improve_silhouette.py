"""
EXERCISE PART C: Improve Silhouette Score to > 0.9 (5 marks)

This standalone script improves the clustering model using:
1. Feature Standardization (StandardScaler)
2. Optimal Cluster Selection (testing 2-7 clusters)
3. Enhanced KMeans Parameters (more iterations and initializations)
4. Outlier Removal (optional, for maximum improvement)

Target: Silhouette Score > 0.9

Usage:
    python exercise_solutions/part_c_improve_silhouette.py
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score
import joblib
import matplotlib.pyplot as plt

def improve_clustering_model(remove_outliers=True):
    """
    Improve clustering model to achieve Silhouette Score > 0.9
    
    Args:
        remove_outliers: If True, removes outliers using IQR method
    """
    
    print("="*80)
    print("EXERCISE PART C: Improve Silhouette Score to > 0.9 (5 marks)")
    print("="*80)
    
    # Load dataset
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    
    SEGMENT_FEATURES = ["estimated_income", "selling_price"]
    X = df[SEGMENT_FEATURES].copy()
    
    print(f"\nOriginal dataset size: {len(X)} samples")
    
    # STEP 1: Baseline Model (Original)
    print("\n" + "-"*80)
    print("STEP 1: Baseline Model (Original)")
    print("-"*80)
    
    kmeans_baseline = KMeans(n_clusters=3, random_state=42, n_init=10)
    baseline_labels = kmeans_baseline.fit_predict(X)
    baseline_score = silhouette_score(X, baseline_labels)
    
    print(f"Baseline Silhouette Score: {baseline_score:.4f}")
    print("Status: ❌ Below target (0.9)")
    
    # STEP 2: Remove Outliers (if enabled)
    if remove_outliers:
        print("\n" + "-"*80)
        print("STEP 2: Outlier Removal (IQR Method)")
        print("-"*80)
        
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)
        X_clean = X[mask]
        df_clean = df[mask].copy()
        
        outliers_removed = len(X) - len(X_clean)
        print(f"Outliers removed: {outliers_removed}")
        print(f"Remaining samples: {len(X_clean)}")
        
        X_working = X_clean
        df_working = df_clean
    else:
        print("\n" + "-"*80)
        print("STEP 2: Outlier Removal - SKIPPED")
        print("-"*80)
        X_working = X
        df_working = df
    
    # STEP 3: Feature Standardization
    print("\n" + "-"*80)
    print("STEP 3: Feature Standardization")
    print("-"*80)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_working)
    
    print("✅ Applied StandardScaler")
    print(f"   Mean: {X_scaled.mean(axis=0)}")
    print(f"   Std:  {X_scaled.std(axis=0)}")
    
    # STEP 4: Optimal Cluster Selection
    print("\n" + "-"*80)
    print("STEP 4: Optimal Cluster Selection (Testing 2-7 clusters)")
    print("-"*80)
    
    silhouette_scores = {}
    best_score = 0
    best_n = 2
    
    for n_clusters in range(2, 8):
        kmeans_temp = KMeans(
            n_clusters=n_clusters,
            n_init=50,  # Increased from 10
            max_iter=500,  # Increased from 300
            random_state=42,
            algorithm='lloyd',
            tol=1e-6
        )
        cluster_labels = kmeans_temp.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores[n_clusters] = score
        
        status = "✅" if score > 0.9 else "⚠️" if score > 0.7 else "❌"
        print(f"  n_clusters = {n_clusters}: {score:.4f} {status}")
        
        if score > best_score:
            best_score = score
            best_n = n_clusters
    
    print(f"\n🎯 Optimal: {best_n} clusters with score {best_score:.4f}")
    
    # STEP 5: Train Final Model
    print("\n" + "-"*80)
    print("STEP 5: Training Final Optimized Model")
    print("-"*80)
    
    kmeans_final = KMeans(
        n_clusters=best_n,
        n_init=100,  # Maximum initializations
        max_iter=1000,  # Maximum iterations
        random_state=42,
        algorithm='lloyd',
        tol=1e-6
    )
    
    df_working["cluster_id"] = kmeans_final.fit_predict(X_scaled)
    final_score = silhouette_score(X_scaled, df_working["cluster_id"])
    
    print(f"Final Silhouette Score: {final_score:.4f}")
    
    if final_score > 0.9:
        print("Status: ✅ TARGET ACHIEVED! (> 0.9)")
    elif final_score > 0.8:
        print("Status: ⚠️  Close to target (0.8-0.9)")
    else:
        print("Status: ❌ Below target (< 0.8)")
    
    # Create cluster names
    centers = kmeans_final.cluster_centers_
    centers_original = scaler.inverse_transform(centers)
    sorted_clusters = centers_original[:, 0].argsort()
    
    if best_n == 2:
        cluster_names = ["Economy", "Premium"]
    elif best_n == 3:
        cluster_names = ["Economy", "Standard", "Premium"]
    elif best_n == 4:
        cluster_names = ["Budget", "Economy", "Standard", "Premium"]
    else:
        cluster_names = [f"Segment_{i+1}" for i in range(best_n)]
    
    cluster_mapping = {sorted_clusters[i]: cluster_names[i] for i in range(best_n)}
    df_working["client_class"] = df_working["cluster_id"].map(cluster_mapping)
    
    # STEP 6: Comparison Summary
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY")
    print("="*80)
    
    improvement_data = [
        ["Metric", "Baseline", "Improved", "Change"],
        ["-"*20, "-"*15, "-"*15, "-"*15],
        ["Silhouette Score", f"{baseline_score:.4f}", f"{final_score:.4f}", 
         f"+{(final_score - baseline_score):.4f}"],
        ["Number of Clusters", "3 (fixed)", f"{best_n} (optimal)", "Dynamic"],
        ["Feature Scaling", "No", "Yes (StandardScaler)", "✅"],
        ["Outlier Removal", "No", "Yes" if remove_outliers else "No", 
         "✅" if remove_outliers else "❌"],
        ["KMeans n_init", "10", "100", "+90"],
        ["KMeans max_iter", "300", "1000", "+700"],
    ]
    
    for row in improvement_data:
        print(f"{row[0]:<20} {row[1]:<15} {row[2]:<15} {row[3]:<15}")
    
    print("="*80)
    
    # STEP 7: Techniques Applied
    print("\n" + "-"*80)
    print("OPTIMIZATION TECHNIQUES APPLIED")
    print("-"*80)
    print("1. ✅ Feature Standardization (StandardScaler)")
    print("   - Ensures equal weight for all features")
    print("   - Improves distance-based clustering")
    
    if remove_outliers:
        print("\n2. ✅ Outlier Removal (IQR Method)")
        print(f"   - Removed {outliers_removed} outliers")
        print("   - Creates more cohesive clusters")
    
    print("\n3. ✅ Optimal Cluster Selection")
    print(f"   - Tested 2-7 clusters")
    print(f"   - Selected {best_n} clusters (best score)")
    
    print("\n4. ✅ Enhanced KMeans Parameters")
    print("   - n_init: 100 (more initialization attempts)")
    print("   - max_iter: 1000 (more iterations)")
    print("   - tol: 1e-6 (stricter convergence)")
    
    # STEP 8: Save Results
    print("\n" + "-"*80)
    print("SAVING RESULTS")
    print("-"*80)
    
    # Save model and scaler
    joblib.dump(kmeans_final, "exercise_solutions/improved_clustering_model.pkl")
    joblib.dump(scaler, "exercise_solutions/improved_scaler.pkl")
    print("✅ Model saved: exercise_solutions/improved_clustering_model.pkl")
    print("✅ Scaler saved: exercise_solutions/improved_scaler.pkl")
    
    # Save detailed report
    report_file = "exercise_solutions/silhouette_improvement_report.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXERCISE PART C: Silhouette Score Improvement Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"Baseline Silhouette Score: {baseline_score:.4f}\n")
        f.write(f"Improved Silhouette Score: {final_score:.4f}\n")
        f.write(f"Improvement: +{(final_score - baseline_score):.4f}\n\n")
        f.write(f"Target Achievement: {'✅ YES' if final_score > 0.9 else '❌ NO'} (Target: > 0.9)\n\n")
        f.write("Techniques Applied:\n")
        f.write("1. Feature Standardization (StandardScaler)\n")
        if remove_outliers:
            f.write(f"2. Outlier Removal ({outliers_removed} outliers)\n")
        f.write(f"3. Optimal Cluster Selection ({best_n} clusters)\n")
        f.write("4. Enhanced KMeans Parameters (n_init=100, max_iter=1000)\n\n")
        f.write("Silhouette Scores by Cluster Count:\n")
        for n, score in silhouette_scores.items():
            f.write(f"  {n} clusters: {score:.4f}\n")
        f.write("\n" + "="*80 + "\n")
    
    print(f"✅ Report saved: {report_file}")
    
    # Create visualization
    create_improvement_visualization(silhouette_scores, baseline_score, final_score)
    
    return {
        'baseline_score': baseline_score,
        'final_score': final_score,
        'improvement': final_score - baseline_score,
        'optimal_clusters': best_n,
        'all_scores': silhouette_scores,
        'target_achieved': final_score > 0.9
    }


def create_improvement_visualization(scores_dict, baseline, final):
    """
    Create visualization comparing silhouette scores
    """
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Silhouette scores by cluster count
    plt.subplot(1, 2, 1)
    clusters = list(scores_dict.keys())
    scores = list(scores_dict.values())
    
    plt.plot(clusters, scores, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=0.9, color='g', linestyle='--', label='Target (0.9)')
    plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.3f})')
    plt.xlabel('Number of Clusters', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score by Cluster Count', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Baseline vs Improved
    plt.subplot(1, 2, 2)
    models = ['Baseline\n(Original)', 'Improved\n(Optimized)']
    scores_comparison = [baseline, final]
    colors = ['#e74c3c', '#2ecc71' if final > 0.9 else '#f39c12']
    
    bars = plt.bar(models, scores_comparison, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(y=0.9, color='g', linestyle='--', linewidth=2, label='Target (0.9)')
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Model Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.legend()
    
    # Add value labels on bars
    for bar, score in zip(bars, scores_comparison):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = "exercise_solutions/silhouette_improvement_chart.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    # Run with outlier removal for maximum improvement
    results = improve_clustering_model(remove_outliers=True)
    
    print("\n" + "="*80)
    print("PART C COMPLETED!")
    print("="*80)
    print(f"\nResults:")
    print(f"  • Baseline Score:  {results['baseline_score']:.4f}")
    print(f"  • Improved Score:  {results['final_score']:.4f}")
    print(f"  • Improvement:     +{results['improvement']:.4f}")
    print(f"  • Optimal Clusters: {results['optimal_clusters']}")
    print(f"  • Target Achieved:  {'✅ YES' if results['target_achieved'] else '❌ NO'}")
    
    if not results['target_achieved']:
        print("\n💡 Note: If target not achieved, this indicates natural data overlap.")
        print("   Consider: More features, different algorithms, or feature engineering.")
    
    print("="*80)
