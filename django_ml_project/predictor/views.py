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
