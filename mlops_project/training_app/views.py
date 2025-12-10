from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib import messages
import json

from .models import TrainingJob
from .services import Trainer
from .services.trainer import DuplicateDatasetWarning
from model_registry.models import ModelArtifact, TrainingDataHistory
from datasets.models import Dataset


def index(request):
    """Training app main page."""
    jobs = TrainingJob.objects.all()[:20]
    datasets = Dataset.objects.all()
    models = ModelArtifact.objects.all()
    
    # Prepare dataset info with column data for JavaScript
    datasets_json = json.dumps([
        {
            'name': d.name,
            'columns': d.column_names,
            'row_count': d.row_count,
        }
        for d in datasets
    ])
    
    context = {
        'jobs': jobs,
        'datasets': datasets,
        'datasets_json': datasets_json,
        'models': models,
        'model_types': [
            ('linear_regression', 'Linear Regression'),
            ('random_forest', 'Random Forest'),
            ('sgd_regressor', 'SGD Regressor'),
            ('random_forest_classifier', 'Random Forest Classifier'),
            ('torch_mlp', 'PyTorch MLP'),
        ],
    }
    return render(request, 'training_app/index.html', context)


@require_http_methods(["POST"])
def train(request):
    """Start a training job with support for multiple datasets."""
    try:
        # Get multiple dataset names (checkbox/multi-select)
        dataset_names = request.POST.getlist('dataset_names')
        
        # Backward compatibility: also check for single dataset_name
        if not dataset_names:
            single_name = request.POST.get('dataset_name')
            if single_name:
                dataset_names = [single_name]
        
        if not dataset_names:
            raise ValueError("At least one dataset must be selected")
        
        model_type = request.POST.get('model_type')
        training_mode = request.POST.get('training_mode', 'new')
        target_column = request.POST.get('target_column')
        feature_columns = request.POST.getlist('feature_columns')
        base_model_version = request.POST.get('base_model_version')
        allow_duplicates = request.POST.get('allow_duplicates') == 'on'
        
        # Parse hyperparameters from form
        hyperparameters = {}
        for key in request.POST:
            if key.startswith('hp_'):
                param_name = key[3:]  # Remove 'hp_' prefix
                value = request.POST[key]
                # Try to convert to appropriate type
                try:
                    if '.' in value:
                        hyperparameters[param_name] = float(value)
                    else:
                        hyperparameters[param_name] = int(value)
                except ValueError:
                    hyperparameters[param_name] = value
        
        config = {
            'dataset_names': dataset_names,
            'model_type': model_type,
            'training_mode': training_mode,
            'target_column': target_column,
            'feature_columns': feature_columns if feature_columns else None,
            'hyperparameters': hyperparameters,
            'base_model_version': base_model_version if training_mode == 'fine_tune' else None,
            'allow_duplicates': allow_duplicates,
        }
        
        trainer = Trainer()
        version, metrics = trainer.train(config)
        
        messages.success(
            request, 
            f'Training completed! Model version: {version} '
            f'(trained on {len(dataset_names)} dataset(s))'
        )
        
    except DuplicateDatasetWarning as e:
        messages.warning(request, f'Training blocked: {str(e)}')
        
    except Exception as e:
        messages.error(request, f'Training failed: {str(e)}')
    
    return redirect('training_app:index')


def job_detail(request, job_id):
    """View training job details."""
    try:
        job = TrainingJob.objects.get(job_id=job_id)
    except TrainingJob.DoesNotExist:
        messages.error(request, 'Job not found')
        return redirect('training_app:index')
    
    # Get training history for this job's result model
    training_history = []
    if job.result_version:
        try:
            model = ModelArtifact.objects.get(version=job.result_version)
            training_history = TrainingDataHistory.objects.filter(model_version=model)
        except ModelArtifact.DoesNotExist:
            pass
    
    context = {
        'job': job,
        'training_history': training_history,
    }
    return render(request, 'training_app/job_detail.html', context)


def api_train(request):
    """API endpoint for training with multiple dataset support."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        config = json.loads(request.body)
        
        # Normalize dataset_names
        if 'dataset_names' not in config and 'dataset_name' in config:
            config['dataset_names'] = [config['dataset_name']]
        
        trainer = Trainer()
        version, metrics = trainer.train(config)
        
        return JsonResponse({
            'status': 'success',
            'version': version,
            'metrics': metrics,
            'datasets_used': config.get('dataset_names', []),
        })
    except DuplicateDatasetWarning as e:
        return JsonResponse({
            'status': 'warning',
            'error': str(e),
            'error_type': 'duplicate_dataset',
        }, status=409)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e),
        }, status=400)
