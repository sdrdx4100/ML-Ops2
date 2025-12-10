# ML-Ops Mini Platform

A unified machine learning platform for training, inference, and model management built with Django.

![ML-Ops Home](https://github.com/user-attachments/assets/c5559459-0bfc-49cf-b6bd-111cd43f41f2)

## Features

- **Training App**: New training and fine-tuning with support for scikit-learn, PyTorch, and XGBoost models
- **Inference App**: Run predictions with SHAP/LIME feature analysis
- **Model Registry**: Version control and lineage tracking for trained models
- **Datasets**: CSV/Parquet upload with DuckDB integration for efficient querying
- **Simple Grey-Tone UI**: Clean design without external CSS frameworks

## Project Structure

```
mlops_project/
├── config/              # Django project settings
├── training_app/        # New training and fine-tuning
├── inference_app/       # Predictions and analysis
├── model_registry/      # Model version management
├── datasets/            # Data upload and preprocessing
├── shared/              # Common utilities
│   ├── utils/           # Logging, timer, exceptions
│   ├── preprocess/      # Scalers, encoders, feature engineering
│   ├── metrics/         # MAE, MSE, F1, ROC-AUC, etc.
│   └── visualize/       # Matplotlib/Plotly visualizations
└── templates/           # Django templates
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run migrations
cd mlops_project
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser

# Run development server
python manage.py runserver
```

## Supported Models

- **Linear Regression** (scikit-learn)
- **Random Forest** (scikit-learn)
- **SGD Regressor** (scikit-learn, supports partial_fit for fine-tuning)
- **Random Forest Classifier** (scikit-learn)
- **PyTorch MLP** (neural network with configurable layers)

## Usage

1. **Upload Dataset**: Go to Datasets → Upload Dataset (CSV or Parquet)
2. **Train Model**: Go to Training → Select dataset, model type, and hyperparameters
3. **View Models**: Check Model Registry for trained models and their lineage
4. **Run Inference**: Go to Inference → Select model version and input data

## API Endpoints

The platform includes REST API endpoints for programmatic access:

- `POST /training/api/train/` - Start training job
- `POST /inference/api/predict/` - Run prediction
- `GET /models/api/list/` - List all models
- `GET /models/api/get/<version>/` - Get model details
- `GET /datasets/api/list/` - List all datasets
- `GET /datasets/api/get/<name>/` - Get dataset details

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_training.py -v
pytest tests/test_inference.py -v
pytest tests/test_datasets.py -v
pytest tests/test_shared.py -v
```

## Extensibility

- **Celery/RQ**: Training jobs can be offloaded to background workers
- **REST API**: Inference endpoints are ready for separate deployment
- **Model Types**: Easy to add new model types in `training_app/services/trainer.py`

## Technologies

- Django 4.2+
- scikit-learn
- PyTorch
- XGBoost
- DuckDB + Parquet
- SHAP/LIME for explainability
- Matplotlib/Plotly for visualization

## License

MIT License