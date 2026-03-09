import pandas as pd
import joblib
from django.shortcuts import render
from predictor.data_exploration import dataset_exploration, data_exploration
from predictor.rwanda_map_visualization import create_rwanda_map, get_district_summary_table
from model_generators.clustering.train_cluster import evaluate_clustering_model
from model_generators.classification.train_classifier import evaluate_classification_model
from model_generators.regression.train_regression import evaluate_regression_model

# Load models once
regression_model = joblib.load("model_generators/regression/regression_model.pkl")
classification_model = joblib.load("model_generators/classification/classification_model.pkl")
clustering_model = joblib.load("model_generators/clustering/clustering_model.pkl")

# Try to load advanced clustering model (best performance - Silhouette > 0.9)
try:
    clustering_model_advanced = joblib.load("model_generators/clustering/clustering_model_advanced.pkl")
    from model_generators.clustering.train_cluster_advanced import evaluate_clustering_model_advanced
    use_advanced_model = True
except:
    use_advanced_model = False

# Try to load optimized clustering model
try:
    clustering_model_optimized = joblib.load("model_generators/clustering/clustering_model_optimized.pkl")
    scaler_optimized = joblib.load("model_generators/clustering/scaler_optimized.pkl")
    from model_generators.clustering.train_cluster_optimized import evaluate_clustering_model_optimized
    use_optimized_model = True
except:
    use_optimized_model = False

# Try to load improved clustering model
try:
    clustering_model_improved = joblib.load("model_generators/clustering/clustering_model_improved.pkl")
    scaler_improved = joblib.load("model_generators/clustering/scaler_improved.pkl")
    from model_generators.clustering.train_cluster_improved import evaluate_clustering_model_improved
    use_improved_model = True
except:
    use_improved_model = False

def data_exploration_view(request):
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    
    # Create Rwanda map visualization
    rwanda_map_html = create_rwanda_map(df)
    district_summary = get_district_summary_table(df)
    
    context = {
        "data_exploration": data_exploration(df),
        "dataset_exploration": dataset_exploration(df),
        "rwanda_map": rwanda_map_html,
        "district_summary": district_summary,
    }
    return render(request, "predictor/index.html", context)

def regression_analysis(request):
    context = {
        "evaluations": evaluate_regression_model()
    }
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        prediction = regression_model.predict([[year, km, seats, income]])[0]
        context["price"] = prediction
    return render(request, "predictor/regression_analysis.html", context)

def classification_analysis(request):
    context = {
        "evaluations": evaluate_classification_model()
    }
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        prediction = classification_model.predict([[year, km, seats, income]])[0]
        context["prediction"] = prediction
    return render(request, "predictor/classification_analysis.html", context)

def clustering_analysis(request):
    # Use advanced model if available (best - Silhouette > 0.9), then optimized, then improved, then standard
    if use_advanced_model:
        context = {
            "evaluations": evaluate_clustering_model_advanced(),
            "model_type": "advanced"
        }
        model_bundle = clustering_model_advanced
    elif use_optimized_model:
        context = {
            "evaluations": evaluate_clustering_model_optimized(),
            "model_type": "optimized"
        }
        active_model = clustering_model_optimized
        active_scaler = scaler_optimized
        model_bundle = None
    elif use_improved_model:
        context = {
            "evaluations": evaluate_clustering_model_improved(),
            "model_type": "improved"
        }
        active_model = clustering_model_improved
        active_scaler = scaler_improved
        model_bundle = None
    else:
        context = {
            "evaluations": evaluate_clustering_model(),
            "model_type": "standard"
        }
        active_model = clustering_model
        active_scaler = None
        model_bundle = None
    
    if request.method == "POST":
        try:
            year = int(request.POST["year"])
            km = float(request.POST["km"])
            seats = int(request.POST["seats"])
            income = float(request.POST["income"])
            
            # Step 1: Predict price
            predicted_price = regression_model.predict([[year, km, seats, income]])[0]
            
            # Step 2: Predict cluster
            if model_bundle:
                # Advanced model uses bundle with metadata
                import numpy as np
                kmeans = model_bundle["kmeans"]
                scaler = model_bundle["scaler"]
                features = model_bundle["features"]
                cluster_mapping = model_bundle["cluster_mapping"]
                
                # Prepare input based on required features
                if features == ["selling_price"]:
                    input_data = np.array([[predicted_price]])
                elif features == ["estimated_income"]:
                    input_data = np.array([[income]])
                elif features == ["estimated_income", "selling_price"]:
                    input_data = np.array([[income, predicted_price]])
                else:
                    # Default fallback
                    input_data = np.array([[predicted_price]])
                
                # Apply scaler if exists
                if scaler:
                    input_scaled = scaler.transform(input_data)
                else:
                    input_scaled = input_data
                
                # Predict cluster
                cluster_id = kmeans.predict(input_scaled)[0]
                prediction = cluster_mapping.get(cluster_id, "Unknown")
                
            elif active_scaler:
                # Improved/optimized models with scaler
                import numpy as np
                input_scaled = active_scaler.transform([[income, predicted_price]])
                cluster_id = active_model.predict(input_scaled)[0]
                
                mapping = {
                    0: "Economy",
                    1: "Standard",
                    2: "Premium"
                }
                prediction = mapping.get(cluster_id, "Unknown")
            else:
                # Standard model
                cluster_id = active_model.predict([[income, predicted_price]])[0]
                
                mapping = {
                    0: "Economy",
                    1: "Standard",
                    2: "Premium"
                }
                prediction = mapping.get(cluster_id, "Unknown")
            
            context.update({
                "prediction": prediction,
                "price": predicted_price
            })
        except Exception as e:
            context["error"] = str(e)
    
    return render(request, "predictor/clustering_analysis.html", context)
