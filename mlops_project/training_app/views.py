from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib import messages
import json

from .models import TrainingJob
from .services import Trainer
from model_registry.models import ModelArtifact
from datasets.models import Dataset


def index(request):
    """Training app main page."""
    jobs = TrainingJob.objects.all()[:20]
    datasets = Dataset.objects.all()
    models = ModelArtifact.objects.all()
    
    context = {
        'jobs': jobs,
        'datasets': datasets,
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
    """Start a training job."""
    try:
        dataset_name = request.POST.get('dataset_name')
        model_type = request.POST.get('model_type')
        training_mode = request.POST.get('training_mode', 'new')
        target_column = request.POST.get('target_column')
        feature_columns = request.POST.getlist('feature_columns')
        base_model_version = request.POST.get('base_model_version')
        
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
            'dataset_name': dataset_name,
            'model_type': model_type,
            'training_mode': training_mode,
            'target_column': target_column,
            'feature_columns': feature_columns if feature_columns else None,
            'hyperparameters': hyperparameters,
            'base_model_version': base_model_version if training_mode == 'fine_tune' else None,
        }
        
        trainer = Trainer()
        version, metrics = trainer.train(config)
        
        messages.success(request, f'Training completed! Model version: {version}')
        
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
    
    context = {'job': job}
    return render(request, 'training_app/job_detail.html', context)


def api_train(request):
    """API endpoint for training."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        config = json.loads(request.body)
        trainer = Trainer()
        version, metrics = trainer.train(config)
        
        return JsonResponse({
            'status': 'success',
            'version': version,
            'metrics': metrics,
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e),
        }, status=400)
