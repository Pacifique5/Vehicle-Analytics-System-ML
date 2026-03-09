# Django Machine Learning Project

Vehicle Analytics System with Regression, Classification, and Clustering models.

## Quick Start

### 1. Start Django Server
```bash
cd django_ml_project
./venv/Scripts/activate
python manage.py runserver
```

Visit: http://127.0.0.1:8000/data_exploration

### 2. Run Exercise Solutions
```bash
python RUN_ALL_EXERCISES.py
```

## Django Application URLs

- `/data_exploration` - Data exploration with Rwanda map visualization
- `/regression_analysis` - Vehicle price prediction
- `/classification_analysis` - Income level classification
- `/clustering_analysis` - Customer segmentation

## Exercise Solutions (30 marks)

Standalone scripts in `exercise_solutions/` folder:
- `part_a_rwanda_map.py` - Rwanda Map (20 marks)
- `part_b_coefficient_variation.py` - Coefficient of Variation (5 marks)
- `part_c_improve_silhouette.py` - Improve Silhouette Score (5 marks)

## Project Structure

```
django_ml_project/
├── config/                 # Django settings
├── predictor/              # Main app
│   ├── views.py
│   ├── urls.py
│   ├── data_exploration.py
│   ├── rwanda_map_visualization.py
│   └── templates/
├── model_generators/       # ML models
│   ├── regression/
│   ├── classification/
│   └── clustering/
├── dummy-data/             # Dataset
├── exercise_solutions/     # Exercise scripts
└── manage.py
```

## Technologies

- Django 6.0.3
- Pandas
- Scikit-learn
- Plotly
- Matplotlib
- Bootstrap 5
