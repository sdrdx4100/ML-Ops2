from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages

from .models import ModelArtifact
from .services import ModelRegistry


def index(request):
    """Model registry main page."""
    models = ModelArtifact.objects.all()
    
    context = {
        'models': models,
    }
    return render(request, 'model_registry/index.html', context)


def model_detail(request, version):
    """View model details."""
    try:
        model = ModelArtifact.objects.get(version=version)
    except ModelArtifact.DoesNotExist:
        messages.error(request, 'Model not found')
        return redirect('model_registry:index')
    
    lineage = model.get_lineage()
    children = model.children.all()
    
    context = {
        'model': model,
        'lineage': lineage,
        'children': children,
    }
    return render(request, 'model_registry/model_detail.html', context)


def delete_model(request, version):
    """Delete a model."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        registry = ModelRegistry()
        registry.delete(version)
        messages.success(request, f'Model version {version} deleted')
    except Exception as e:
        messages.error(request, f'Delete failed: {str(e)}')
    
    return redirect('model_registry:index')


def api_list(request):
    """API endpoint to list models."""
    models = ModelArtifact.objects.all()
    data = [
        {
            'version': m.version,
            'model_type': m.model_type,
            'training_mode': m.training_mode,
            'metrics': m.metrics,
            'created_at': m.created_at.isoformat(),
        }
        for m in models
    ]
    return JsonResponse({'models': data})


def api_get(request, version):
    """API endpoint to get model details."""
    try:
        model = ModelArtifact.objects.get(version=version)
        data = {
            'version': model.version,
            'model_type': model.model_type,
            'training_mode': model.training_mode,
            'parent': model.parent.version if model.parent else None,
            'metadata': model.metadata,
            'metrics': model.metrics,
            'created_at': model.created_at.isoformat(),
        }
        return JsonResponse(data)
    except ModelArtifact.DoesNotExist:
        return JsonResponse({'error': 'Model not found'}, status=404)
