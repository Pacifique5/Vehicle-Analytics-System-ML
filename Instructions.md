Django Machine Learning Lab 
Manual/30marks
Building a Vehicle Analytics System with Regression, Classification, and Client Segmentation
Designed as a step‑by‑step tutorial for students learning Python/Django and Machine 
Learning integration.
Contents
PART I: Learning Objectives.................................................................................................................................2
PART II: Project Overview ....................................................................................................................................2
1. Technology Stack: ......................................................................................................................................2
2. Required Software.....................................................................................................................................3
PART III: Creating project.....................................................................................................................................3
3. Create Project Folder................................................................................................................................3
4. Create Virtual Environment...................................................................................................................3
5. Install Required Libraries.......................................................................................................................4
6. Create Django Project and App.............................................................................................................4
7. Register the Application..........................................................................................................................4
8. Add the Dataset...........................................................................................................................................4
PART IV: Data Exploration Using Pandas .......................................................................................................5
9. Running the Server....................................................................................................................................7
Part V: Model training.............................................................................................................................................8
10. Feature Selection...................................................................................................................................8
11. Regression Model – Predict Selling Price ....................................................................................8
12. Classification Model – Income Level .............................................................................................9
13. Clustering Model.................................................................................................................................10
14. Update the views.py..........................................................................................................................11
15. Create templates for the project ..................................................................................................12
16. Update the app urls.py .....................................................................................................................19
17. Running the Server............................................................................................................................19
18. Testing the System.............................................................................................................................19
19. Exercise ..................................................................................................................................................19
PART I: Learning Objectives
By the end of this lab students should be able to:
• Create and configure a Django project
• Prepare a dataset for machine learning
• Train machine learning models using scikit‑learn
• Implement three ML techniques:
 Regression (predict selling price)
 Classification (predict income level)
 Clustering (client segmentation)
• Integrate ML models into a Django web application
• Build a web interface to interact with ML predictions
• Deploy and test the ML system
PART II: Project Overview
In this project we build a Vehicle Analytics System. The system will analyze a dataset of 
vehicle sales and provide:
1. Price Prediction: Predict the selling price of a vehicle using regression.
2. Income Classification: Predict the customer's income level using classification.
3. Client Segmentation: Group customers into clusters using K‑Means clustering.
1. Technology Stack:
 Django
 Python
 Pandas
 Scikit‑learn
 HTML templates
2. Required Software
Install the following tools:
Python 3.9 or newer
Visual Studio Code (recommended)
Git (optional)
Google Chrome or Firefox
Python packages required:
django
pandas
scikit-learn
matplotlib
seaborn
joblib
PART III: Creating project
3. Create Project Folder
Open a terminal and run:
mkdir django_ml_project
cd django_ml_project
4. Create Virtual Environment
Create a Python virtual environment.
python -m venv venv
Activate environment
i. Windows:
venv\Scripts\activate
ii. Mac/Linux:
source venv/bin/activate
5. Install Required Libraries
Install project dependencies using requirements.txt
Create a file named requirements.txt and add:
django 
pandas 
scikit-learn 
matplotlib 
seaborn 
joblib
plotly
Or Generate automatically from your environment:
pip freeze > requirements.txt
Install all packages at once:
pip install -r requirements.txt
6. Create Django Project and App
django-admin startproject config .
python manage.py startapp predictor 
Project structure becomes:
django_ml_project/
 manage.py
 config/
 predictor/
 requirements.txt
7. Register the Application
Open settings.py and add the new app 'predictor' on existing apps:
INSTALLED_APPS = [
 #NB: leave existing apps (DO NOT DELETE)
 'predictor',
]
8. Add the Dataset
Within your project folder, create another folder named dummy-data for dataset of your 
project.
django_ml_project/
 dummy-data/
 vehicles_ml_dataset.csv
PART IV: Data Exploration Using Pandas
Create a file: predictor/data_eploration.py
import pandas as pd
# Data Exploration
def dataset_exploration(df):
 table_html = df.head().to_html(
 classes="table table-bordered table-striped table-sm",
 float_format="%.2f",
 justify="center",
 index=False,
 )
 return table_html
# Data description
def data_exploration(df):
 table_html = df.head().to_html(
 classes="table table-bordered table-striped table-sm",
 float_format="%.2f",
 justify="center",
 )
 return table_html
Add bellow in predictor/views.py
import pandas as pd
from django.shortcuts import render
from predictor.data_exploration import dataset_exploration, data_exploration
def data_exploration_view(request):
 df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
 context = {
 "data_exploration": data_exploration(df),
 "dataset_exploration": dataset_exploration(df),
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
Create file: predictor/templates/predictor/index.html
<!DOCTYPE html>
<html lang="en">
<head>
 <meta charset="UTF-8">
 <meta name="viewport" content="width=device-width, initial-scale=1.0">
 <title>Vehicle ML Dashboard</title>
 <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" 
rel="stylesheet">
</head>
<body class="bg-light">
 <div class="container-fluid">
 <div class="row">
 <nav id="sidebarMenu" class="col-md-3 col-lg-2 d-md-block bg-white border-end sticky-top 
vh-100 overflow-y-auto p-3">
 <div class="mb-4 ps-2">
 <h5 class="text-primary fw-bold mb-0">Vehicle ML</h5>
 <small class="text-muted">EDA</small>
 </div>
 <div id="data_exploration-nav" class="nav nav-pills flex-column">
 <a class="nav-link mb-1 active" href="#dataset_exploration">Data Exploration</a>
 <a class="nav-link mb-1" href="#data_exploration">Data Description</a>
 <a class="nav-link mb-1" href="{% url 'regression_analysis' %}">Regression 
Analysis</a>
 <a class="nav-link mb-1" href="{% url 'classification_analysis' 
%}">Classification Analysis</a>
 <a class="nav-link mb-1" href="{% url 'clustering_analysis' %}">Clustering 
Analysis</a>
 </div>
 </nav>
 <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4 py-4" 
 data-bs-spy="scroll" 
 data-bs-target="#data_exploration-nav" 
 data-bs-smooth-scroll="true" 
 tabindex="0">
 
 <header class="mb-4 border-bottom pb-3">
 <h1 class="fw-bold text-dark">Vehicle Insights Dashboard</h1>
 <p class="text-secondary mb-0">Exploratory Data Analysis</p>
 </header>
 <section id="dataset_exploration" class="mb-5" style="padding-top: 20px;">
 <div class="card shadow-sm border-0">
 <div class="card-header bg-white border-bottom py-3">
 <span class="fw-bold">Data Exploration</span>
 </div>
 <div class="card-body overflow-x-auto">
 {{ dataset_exploration|safe }}
 </div>
 </div>
 </section>
 <section id="data_exploration" class="mb-5" style="padding-top: 20px;">
 <div class="card shadow-sm border-0">
 <div class="card-header bg-white border-bottom py-3">
 <span class="fw-bold">Statistical Analysis</span>
 </div>
 <div class="card-body overflow-x-auto">
 {{ data_exploration|safe }}
 </div>
 </div>
 </section>
 </main>
 </div>
 </div>
 <script 
src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
Create file and add below code: predictor/urls.py
from django.urls import path
from predictor import views
urlpatterns = [
 path("data_exploration", views.data_exploration_view, name="data_exploration"),]
Within config/urls.py add below code:
from django.urls import path, include
urlpatterns = [
 path("", include("predictor.urls")),
]
9. Running the Server
Run the Django development server.
python manage.py runserver
Part V: Model training
10.Feature Selection
Select features useful for machine learning. Example features: year, kilometers_driven, 
seating_capacity, estimated_income, 
Target variables: selling_price for regression analysis and prediction, income_level for 
classification. For clustering, it is an unsupervised learning no target variable is needed
11.Regression Model – Predict Selling Price
Create file within the project folder: model_generators/regression/train_regression.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
features = ["year", "kilometers_driven", "seating_capacity", "estimated_income"]
target = "selling_price"
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)
# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Save model
joblib.dump(model, "regression_model.pkl")
# Predict
predictions = model.predict(X_test)
# Calculate R2 Score
r2 = round(r2_score(y_test, predictions) * 100, 2)
# Create a Comparison DataFrame for the data_exploration
comparison_df = pd.DataFrame(
 {
 "Actual": y_test.values,
 "Predicted": predictions.round(2),
 "Difference": (y_test.values - predictions).round(2),
 }
)
def evaluate_regression_model():
 return {
 "r2": r2,
 "comparison": comparison_df.head(10).to_html(
 classes="table table-bordered table-striped table-sm",
 float_format="%.2f",
 justify="center",
 ),
 }
12.Classification Model – Income Level
Create file within the project folder: model_generators/classification/train_classifier.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
# Define features and target (target moved up for logical flow)
features = ["year", "kilometers_driven", "seating_capacity", "estimated_income"]
target = "income_level" 
X = df[features]
y = df[target]
# Split data
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)
# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Save model
joblib.dump(model, "classification_model.pkl")
# Predict
predictions = model.predict(X_test)
# Calculate Accuracy Score (equivalent to R2 in your regression example)
accuracy = round(accuracy_score(y_test, predictions) * 100, 2)
# Create a Comparison DataFrame for the data_exploration
comparison_df = pd.DataFrame(
 {
 "Actual": y_test.values,
 "Predicted": predictions,
 "Match": y_test.values == predictions,
 }
)
def evaluate_classification_model():
 return {
 "accuracy": accuracy,
 "comparison": comparison_df.head(10).to_html(
 classes="table table-bordered table-striped table-sm",
 justify="center",
 ),
 }
13.Clustering Model
Create file within the project folder: model_generators/clustering/train_cluster.py
import pandas as pd
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
cluster_summary = df.groupby("client_class")[SEGMENT_FEATURES].mean()
cluster_counts = df["client_class"].value_counts().reset_index()
cluster_counts.columns = ["client_class", "count"]
cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")
comparison_df = df[["client_name", "estimated_income", "selling_price", "client_class"]]
def evaluate_clustering_model():
 return {
 "silhouette": silhouette_avg,
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
14.Update the views.py
import joblib
from model_generators.clustering.train_cluster import evaluate_clustering_model
from model_generators.classification.train_classifier import evaluate_classification_model
from model_generators.regression.train_regression import evaluate_regression_model
# Load models once
regression_model = joblib.load("model_generators/regression/regression_model.pkl")
classification_model = joblib.load("model_generators/classification/classification_model.pkl")
clustering_model = joblib.load("model_generators/clustering/clustering_model.pkl")
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
 context = {
 "evaluations": evaluate_clustering_model()
 }
 if request.method == "POST":
 try:
 year = int(request.POST["year"])
 km = float(request.POST["km"])
 seats = int(request.POST["seats"])
 income = float(request.POST["income"])
 # Step 1: Predict price
 predicted_price = regression_model.predict([[year, km, seats, income]])[0]
 # Step 2: Predict cluster
 cluster_id = clustering_model.predict([[income, predicted_price]])[0]
 mapping = {
 0: "Economy",
 1: "Standard",
 2: "Premium"
 }
 context.update({
 "prediction": mapping.get(cluster_id, "Unknown"),
 "price": predicted_price
 })
 except Exception as e:
 context["error"] = str(e)
 return render(request, "predictor/clustering_analysis.html", context)
15.Create templates for the project
Create file within the app folder: templates/predictor/index.html
<!DOCTYPE html>
<html lang="en">
<head>
 <meta charset="UTF-8">
 <meta name="viewport" content="width=device-width, initial-scale=1.0">
 <title>Vehicle ML Dashboard</title>
 <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
 <div class="container-fluid">
 <div class="row">
 <nav id="sidebarMenu" class="col-md-3 col-lg-2 d-md-block bg-white border-end sticky-top vh-100 overflow-y-auto p-3">
 <div class="mb-4 ps-2">
 <h5 class="text-primary fw-bold mb-0">Vehicle ML</h5>
 <small class="text-muted">EDA</small>
 </div>
 <div id="data_exploration-nav" class="nav nav-pills flex-column">
 <a class="nav-link mb-1 active" href="#dataset_exploration">Data Exploration</a>
 <a class="nav-link mb-1" href="#data_exploration">Data Description</a>
 <a class="nav-link mb-1" href="{% url 'regression_analysis' %}">Regression Analysis</a>
 <a class="nav-link mb-1" href="{% url 'classification_analysis' %}">Classification Analysis</a>
 <a class="nav-link mb-1" href="{% url 'clustering_analysis' %}">Clustering Analysis</a>
 </div>
 </nav>
 <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4 py-4" 
 data-bs-spy="scroll" 
 data-bs-target="#data_exploration-nav" 
 data-bs-smooth-scroll="true" 
 tabindex="0">
 
 <header class="mb-4 border-bottom pb-3">
 <h1 class="fw-bold text-dark">Vehicle Insights Dashboard</h1>
 <p class="text-secondary mb-0">Exploratory Data Analysis</p>
 </header>
 <section id="dataset_exploration" class="mb-5" style="padding-top: 20px;">
 <div class="card shadow-sm border-0">
 <div class="card-header bg-white border-bottom py-3">
 <span class="fw-bold">Data Exploration</span>
 </div>
 <div class="card-body overflow-x-auto">
 {{ dataset_exploration|safe }}
 </div>
 </div>
 </section>
 <section id="data_exploration" class="mb-5" style="padding-top: 20px;">
 <div class="card shadow-sm border-0">
 <div class="card-header bg-white border-bottom py-3">
 <span class="fw-bold">Statistical Analysis</span>
 </div>
 <div class="card-body overflow-x-auto">
 {{ data_exploration|safe }}
 </div>
 </div>
 </section>
 </main>
 </div>
 </div>
 <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
Create file within the app folder: templates/predictor/regretion_analysis.html
<!DOCTYPE html>
<html lang="en">
<head>
 <meta charset="UTF-8">
 <meta name="viewport" content="width=device-width, initial-scale=1.0">
 <title>Vehicle ML Dashboard</title>
 <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
 <div class="container-fluid">
 <div class="row">
 <nav id="sidebarMenu"
 class="col-md-3 col-lg-2 d-md-block bg-white border-end sticky-top vh-100 overflow-y-auto p-3">
 <div class="mb-4 ps-2">
 <h5 class="text-primary fw-bold mb-0">Vehicle ML</h5>
 <small class="text-muted">Regression Analysis</small>
 </div>
 <div id="data_exploration-nav" class="nav nav-pills flex-column">
 <a class="nav-link mb-1 active" href="#regression-analysis">Regression Analysis</a>
 <a class="nav-link mb-1" href="#evaluation_metrics">Evaluation Metrics</a>
 <a class="nav-link mb-1" href="{% url 'data_exploration' %}">Exploratory Data Analysis</a>
 <a class="nav-link mb-1" href="{% url 'classification_analysis' %}">Classification Analysis</a>
 <a class="nav-link mb-1" href="{% url 'clustering_analysis' %}">Clustering Analysis</a>
 </div>
 </nav>
 <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4 py-4" data-bs-spy="scroll"
 data-bs-target="#data_exploration-nav" data-bs-smooth-scroll="true" tabindex="0">
 <header class="mb-4 border-bottom pb-3">
 <h1 class="fw-bold text-dark">Vehicle Insights Dashboard</h1>
 <p class="text-secondary mb-0">Real-time Machine Learning Inference</p>
 </header>
 <section id="regression-analysis" class="mb-5" style="padding-top: 20px;">
 <div class="row g-4">
 <div class="col-lg-7">
 <div class="card shadow-sm border-0 h-100">
 <div class="card-header bg-primary text-white py-3">
 <h6 class="mb-0 fw-bold">Input Vehicle Specifications</h6>
 </div>
 <div class="card-body p-4">
 <form method="POST" action="">
 {% csrf_token %}
 <div class="row g-3">
 <div class="col-md-6">
 <label class="form-label fw-semibold">Model Year</label>
 <input type="number" name="year" class="form-control"
 placeholder="e.g. 2022" required>
 </div>
 <div class="col-md-6">
 <label class="form-label fw-semibold">Kilometers Driven</label>
 <input type="number" step="any" name="km" class="form-control"
 placeholder="e.g. 15000" required>
 </div>
 <div class="col-md-6">
 <label class="form-label fw-semibold">Number of Seats</label>
 <input type="number" name="seats" class="form-control"
 placeholder="e.g. 5" required>
 </div>
 <div class="col-md-6">
 <label class="form-label fw-semibold">Owner Income</label>
 <input type="number" step="any" name="income" class="form-control"
 placeholder="e.g. 50000" required>
 </div>
 <div class="col-12 mt-4">
 <button type="submit" class="btn btn-primary w-100 fw-bold py-2">Predict
 Market Price</button>
 </div>
 </div>
 </form>
 </div>
 </div>
 </div>
 <div class="col-lg-5">
 <div class="card shadow-sm border-0 h-100 bg-white">
 <div class="card-header bg-dark text-white py-3">
 <h6 class="mb-0 fw-bold">Prediction Result</h6>
 </div>
 <div
 class="card-body d-flex flex-column align-items-center justify-content-center text-center p-4">
 {% if price %}
 <div class="mb-2 text-muted text-uppercase small fw-bold">Estimated Value</div>
 <h2 class="display-4 fw-bold text-success mb-2">${{ price|floatformat:2 }}</h2>
 {% else %}
 <div class="py-5">
 <div class="text-muted mb-3 opacity-25">
 Waiting...
 </div>
 <p class="text-secondary">Fill out the form to generate a price prediction.</p>
 </div>
 {% endif %}
 </div>
 </div>
 </div>
 </div>
 </section>
 <section id="evaluation_metrics" class="mb-5" style="padding-top: 20px;">
 <div class="card shadow-sm border-0">
 <div class="card-header bg-white border-bottom py-3">
 <span class="fw-bold">Evaluation Metrics</span>
 </div>
 <div class="card-body overflow-x-auto">
 Then, {{ evaluations.r2|safe }}% measures how well these selected features explain the
 variation in selling prices.
 {{ evaluations.comparison|safe }}
 </div>
 </div>
 </section>
 </main>
 </div>
 </div>
 <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
Create file within the app folder: templates/predictor/classification_analysis.html
<!DOCTYPE html>
<html lang="en">
<head>
 <meta charset="UTF-8">
 <meta name="viewport" content="width=device-width, initial-scale=1.0">
 <title>Vehicle ML Dashboard - Classification</title>
 <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
 <div class="container-fluid">
 <div class="row">
 <nav id="sidebarMenu" class="col-md-3 col-lg-2 d-md-block bg-white border-end sticky-top vh-100 overflow-y-auto p-3">
 <div class="mb-4 ps-2">
 <h5 class="text-primary fw-bold mb-0">Vehicle ML</h5>
 <small class="text-muted">Classification Analysis</small>
 </div>
 <div id="data_exploration-nav" class="nav nav-pills flex-column">
 <a class="nav-link mb-1 active" href="#classification-analysis">Classification Analysis</a>
 <a class="nav-link mb-1" href="#evaluation_metrics">Evaluation Metrics</a>
 <a class="nav-link active mb-1" href="{% url 'data_exploration' %}">Exploratory Data Analysis</a>
 <a class="nav-link mb-1" href="{% url 'regression_analysis' %}">Regression Analysis</a>
 <a class="nav-link mb-1" href="{% url 'clustering_analysis' %}">Clustering Analysis</a>
 </div>
 </nav>
 <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4 py-4" 
 data-bs-spy="scroll" 
 data-bs-target="#data_exploration-nav" 
 data-bs-smooth-scroll="true" 
 tabindex="0">
 
 <header class="mb-4 border-bottom pb-3">
 <h1 class="fw-bold text-dark">Vehicle Insights Dashboard</h1>
 <p class="text-secondary mb-0">Classification: Predicting Owner Income Category</p>
 </header>
 <section id="classification-analysis" class="mb-5" style="padding-top: 20px;">
 <div class="row g-4">
 <div class="col-lg-7">
 <div class="card shadow-sm border-0 h-100">
 <div class="card-header bg-primary text-white py-3">
 <h6 class="mb-0 fw-bold">Input Vehicle Specifications</h6>
 </div>
 <div class="card-body p-4">
 <form method="POST" action="">
 {% csrf_token %}
 <div class="row g-3">
 <div class="col-md-6">
 <label class="form-label fw-semibold">Model Year</label>
 <input type="number" name="year" class="form-control" placeholder="e.g. 2022" required>
 </div>
 <div class="col-md-6">
 <label class="form-label fw-semibold">Kilometers Driven</label>
 <input type="number" step="any" name="km" class="form-control" placeholder="e.g. 15000" required>
 </div>
 <div class="col-md-6">
 <label class="form-label fw-semibold">Number of Seats</label>
 <input type="number" name="seats" class="form-control" placeholder="e.g. 5" required>
 </div>
 <div class="col-md-6">
 <label class="form-label fw-semibold">Reference Income</label>
 <input type="number" step="any" name="income" class="form-control" placeholder="e.g. 50000" required>
 </div>
 <div class="col-12 mt-4">
 <button type="submit" class="btn btn-primary w-100 fw-bold py-2">Predict Income Category</button>
 </div>
 </div>
 </form>
 </div>
 </div>
 </div>
 <div class="col-lg-5">
 <div class="card shadow-sm border-0 h-100 bg-white">
 <div class="card-header bg-dark text-white py-3">
 <h6 class="mb-0 fw-bold">Classification Result</h6>
 </div>
 <div class="card-body d-flex flex-column align-items-center justify-content-center text-center p-4">
 {% if prediction %}
 <div class="mb-2 text-muted text-uppercase small fw-bold">Predicted Category</div>
 <h2 class="display-4 fw-bold text-success mb-2">{{ prediction }}</h2>
 {% else %}
 <div class="py-5">
 <div class="text-muted mb-3 opacity-25">
 Waiting...
 </div>
 <p class="text-secondary">Fill out the form to classify the income level.</p>
 </div>
 {% endif %}
 </div>
 </div>
 </div>
 </div>
 </section>
 <section id="evaluation_metrics" class="mb-5" style="padding-top: 20px;">
 <div class="card shadow-sm border-0">
 <div class="card-header bg-white border-bottom py-3">
 <span class="fw-bold">Evaluation Metrics</span>
 </div>
 <div class="card-body overflow-x-auto">
 <p class="mb-3">The model has an accuracy of <strong>{{ evaluations.accuracy }}%</strong> in correctly categorizing the 
income level based on historical vehicle data.</p>
 <div class="table-responsive">
 {{ evaluations.comparison|safe }}
 </div>
 </div>
 </div>
 </section>
 </main>
 </div>
 </div>
 <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
Create file within the app folder: templates/predictor/clustering_analysis.html
<!DOCTYPE html>
<html lang="en">
<head>
 <meta charset="UTF-8">
 <meta name="viewport" content="width=device-width, initial-scale=1.0">
 <title>Vehicle ML Dashboard</title>
 <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
 <div class="container-fluid">
 <div class="row">
 <nav id="sidebarMenu"
 class="col-md-3 col-lg-2 d-md-block bg-white border-end sticky-top vh-100 overflow-y-auto p-3">
 <div class="mb-4 ps-2 border-bottom pb-3">
 <h5 class="text-primary fw-bold mb-0">Vehicle ML</h5>
 <small class="text-muted">Clustering Analysis</small>
 </div>
 <div id="data_exploration-nav" class="nav nav-pills flex-column">
 <a class="nav-link mb-1 active" href="#clustering-analysis">Clustering Analysis</a>
 <a class="nav-link mb-1" href="#evaluation-metrics">Evaluation Metrics</a>
 <a class="nav-link mb-1" href="#comparison-analysis">Comparison Analysis</a>
 <a class="nav-link mb-1" href="{% url 'data_exploration' %}">Exploratory Data Analysis</a>
 <a class="nav-link mb-1" href="{% url 'regression_analysis' %}">Regression Analysis</a>
 <a class="nav-link mb-1" href="{% url 'classification_analysis' %}">Classification Analysis</a>
 </div>
 </nav>
 <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4 py-4" data-bs-spy="scroll"
 data-bs-target="#data_exploration-nav" data-bs-smooth-scroll="true" tabindex="0">
 <header class="mb-4 border-bottom pb-3 d-flex justify-content-between align-items-center">
 <div>
 <h1 class="fw-bold text-dark">Vehicle Insights Dashboard</h1>
 <p class="text-secondary mb-0">Regression & Clustering Analysis</p>
 </div>
 </header>
 <section id="clustering-analysis" class="mb-5 pt-3">
 <div class="row g-4">
 <div class="col-lg-7">
 <div class="card shadow-sm border-0 h-100">
 <div class="card-header bg-primary text-white py-3">
 <h6 class="mb-0 fw-bold">Input Specifications</h6>
 </div>
 <div class="card-body p-4">
 <form method="POST" action="">
 {% csrf_token %}
 <div class="row g-3">
 <div class="col-md-6">
 <label class="form-label fw-semibold small">Model Year</label>
 <input type="number" name="year" class="form-control" placeholder="2022"
 required>
 </div>
 <div class="col-md-6">
 <label class="form-label fw-semibold small">Kilometers Driven</label>
 <input type="number" step="any" name="km" class="form-control"
 placeholder="15000" required>
 </div>
 <div class="col-md-6">
 <label class="form-label fw-semibold small">Seats</label>
 <input type="number" name="seats" class="form-control" placeholder="5"
 required>
 </div>
 <div class="col-md-6">
 <label class="form-label fw-semibold small">Owner Annual Income</label>
 <input type="number" step="any" name="income" class="form-control"
 placeholder="50000" required>
 </div>
 <div class="col-12 mt-4">
 <button type="submit"
 class="btn btn-primary w-100 fw-bold py-2 shadow-sm">Run Combined
 Inference</button>
 </div>
 </div>
 </form>
 </div>
 </div>
 </div>
 <div class="col-lg-5">
 <div class="card shadow-sm border-0 h-100 bg-white overflow-hidden">
 <div class="card-header bg-dark text-white py-3">
 <h6 class="mb-0 fw-bold">Dual-Model Result</h6>
 </div>
 <div
 class="card-body d-flex flex-column align-items-center justify-content-center text-center p-4">
 {% if price %}
 <div class="mb-4">
 <small class="text-muted text-uppercase fw-bold ls-wide">Estimated Value</small>
 <h2 class="display-4 fw-bold text-primary mb-0">${{ price|floatformat:2 }}</h2>
 </div>
 <hr class="w-75 my-3 opacity-10">
 <div class="mt-2">
 <small class="text-muted text-uppercase fw-bold ls-wide">Client Cluster</small>
 <div class="mt-2">
 <span class="badge rounded-pill bg-success px-4 py-2 fs-6">
 {{prediction|safe }}</span>
 </div>
 </div>
 {% else %}
 <div class="py-5">
 <div class="text-muted mb-3 opacity-25">Waiting...</div>
 <p class="text-secondary">
 Awaiting input to generate prediction and customer profile.
 </p>
 </div>
 {% endif %}
 </div>
 </div>
 </div>
 </div>
 </section>
 <section id="evaluation-metrics" class="mb-5 pt-3">
 <div class="card shadow-sm border-0">
 <div
 class="card-header bg-white border-bottom py-3 d-flex justify-content-between align-items-center">
 <span class="fw-bold text-dark">Evaluation Metrics</span>
 <span class="badge bg-light text-primary border border-primary">
 Silhouette Score: {{evaluations.silhouette|safe }}
 </span>
 </div>
 <div class="card-body"> 
 <div class="table-responsive">
 {{ evaluations.summary|safe }}
 </div>
 </div>
 </div>
 </section>
 <section id="comparison-analysis" class="mb-5 pt-3">
 <div class="card shadow-sm border-0">
 <div class="card-header bg-white border-bottom py-3">
 <span class="fw-bold">Comparison Analysis</span>
 </div>
 <div class="card-body p-0">
 <div class="table-responsive mt-3">
 {{ evaluations.comparison|safe }}
 </div>
 </div>
 </div>
 </section>
 </main>
 </div>
 </div>
 <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
16.Update the app urls.py
path("regression_analysis",views.regression_analysis, name="regression_analysis"),
path("classification_analysis",views.classification_analysis,name="classification_analysis"),
path("clustering_analysis",views.clustering_analysis, name="clustering_analysis"),
17.Running the Server
Run the Django development server.
python manage.py runserver
18.Testing the System
Open browser: http://127.0.0.1:8000
Enter vehicle data and check the predicted price.
19.Exercise
Students can add dashboards using Plotly.
a) On exploratory data analysis, display the Rwanda map with names and districts 
boundaries, number of vehicle clients in each district. (20 marks)
b) The Silhouette Score is 0.68, 
• Calculate the coefficient of variation and display it along with Silhouette 
Score. (5 marks)
• Refine the model Silhouette Score to have the Silhouette Score above 
0.9(5marks)