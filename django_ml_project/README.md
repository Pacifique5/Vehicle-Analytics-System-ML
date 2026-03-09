# Django Machine Learning Project

Vehicle Analytics System with ML models and exercise solutions integrated.

## Quick Start

```bash
cd django_ml_project
./venv/Scripts/activate
python manage.py runserver
```

Visit: http://127.0.0.1:8000/data_exploration

## Application URLs

- `/data_exploration` - Data exploration with Rwanda map (Exercise Part A)
- `/regression_analysis` - Vehicle price prediction
- `/classification_analysis` - Income level classification  
- `/clustering_analysis` - Customer segmentation (Exercise Parts B & C)

## Exercise Solutions (30 marks)

All exercises are integrated into the Django application:

### Part A: Rwanda Map (20 marks)
**Location:** `/data_exploration`
- Rwanda geographical map
- District names and boundaries
- Vehicle client distribution

### Part B: Coefficient of Variation (5 marks)
**Location:** `/clustering_analysis`
- CV calculation displayed
- Per-cluster breakdown
- Shown with Silhouette Score

### Part C: Improve Silhouette Score (5 marks)
**Location:** `/clustering_analysis`
- Optimized clustering model
- Feature standardization
- Enhanced parameters

## Project Structure

```
django_ml_project/
├── config/                 # Django settings
├── predictor/              # Main app
│   ├── views.py
│   ├── urls.py
│   ├── data_exploration.py
│   ├── rwanda_map_visualization.py  # Exercise Part A
│   └── templates/
├── model_generators/       # ML models
│   ├── regression/
│   ├── classification/
│   └── clustering/
├── dummy-data/             # Dataset
└── manage.py
```

## Technologies

- Django 6.0.3
- Pandas
- Scikit-learn
- Plotly
- Bootstrap 5
