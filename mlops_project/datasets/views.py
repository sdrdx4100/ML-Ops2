from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages

from .models import Dataset
from .services import DataLoader


def index(request):
    """Datasets main page."""
    datasets = Dataset.objects.all()
    
    context = {
        'datasets': datasets,
    }
    return render(request, 'datasets/index.html', context)


def upload(request):
    """Upload a new dataset."""
    if request.method != 'POST':
        return render(request, 'datasets/upload.html')
    
    try:
        name = request.POST.get('name')
        file = request.FILES.get('file')
        
        if not name or not file:
            messages.error(request, 'Name and file are required')
            return render(request, 'datasets/upload.html')
        
        loader = DataLoader()
        dataset = loader.upload(file, name)
        
        messages.success(request, f'Dataset "{name}" uploaded successfully')
        return redirect('datasets:detail', name=name)
        
    except Exception as e:
        messages.error(request, f'Upload failed: {str(e)}')
        return render(request, 'datasets/upload.html')


def detail(request, name):
    """View dataset details."""
    try:
        dataset = Dataset.objects.get(name=name)
    except Dataset.DoesNotExist:
        messages.error(request, 'Dataset not found')
        return redirect('datasets:index')
    
    # Get preview
    loader = DataLoader()
    try:
        preview = loader.get_preview(name, max_rows=100)
        preview_html = preview.to_html(classes='data-table', index=False)
    except Exception:
        preview_html = None
    
    context = {
        'dataset': dataset,
        'preview_html': preview_html,
    }
    return render(request, 'datasets/detail.html', context)


def delete(request, name):
    """Delete a dataset."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        loader = DataLoader()
        loader.delete(name)
        messages.success(request, f'Dataset "{name}" deleted')
    except Exception as e:
        messages.error(request, f'Delete failed: {str(e)}')
    
    return redirect('datasets:index')


def api_list(request):
    """API endpoint to list datasets."""
    datasets = Dataset.objects.all()
    data = [
        {
            'name': d.name,
            'file_type': d.file_type,
            'row_count': d.row_count,
            'column_names': d.column_names,
            'created_at': d.created_at.isoformat(),
        }
        for d in datasets
    ]
    return JsonResponse({'datasets': data})


def api_get(request, name):
    """API endpoint to get dataset details."""
    try:
        dataset = Dataset.objects.get(name=name)
        data = {
            'name': dataset.name,
            'file_type': dataset.file_type,
            'row_count': dataset.row_count,
            'column_names': dataset.column_names,
            'statistics': dataset.statistics,
            'created_at': dataset.created_at.isoformat(),
        }
        return JsonResponse(data)
    except Dataset.DoesNotExist:
        return JsonResponse({'error': 'Dataset not found'}, status=404)
