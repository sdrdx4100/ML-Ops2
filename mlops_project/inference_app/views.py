from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib import messages
import json
import pandas as pd
import io

from .models import InferenceLog
from .services import Predictor
from model_registry.models import ModelArtifact


def index(request):
    """Inference app main page."""
    models = ModelArtifact.objects.all()
    logs = InferenceLog.objects.all()[:20]
    
    context = {
        'models': models,
        'logs': logs,
    }
    return render(request, 'inference_app/index.html', context)


@require_http_methods(["POST"])
def predict(request):
    """Run prediction."""
    try:
        model_version = request.POST.get('model_version')
        input_type = request.POST.get('input_type', 'single')
        analyze = request.POST.get('analyze', 'false').lower() == 'true'
        analysis_method = request.POST.get('analysis_method', 'shap')
        
        predictor = Predictor()
        
        if input_type == 'file' and request.FILES.get('input_file'):
            # Handle file upload
            uploaded_file = request.FILES['input_file']
            content = uploaded_file.read()
            
            if uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(content))
            else:
                df = pd.read_csv(io.BytesIO(content))
            
            result = predictor.predict(model_version, df, analyze, analysis_method)
        else:
            # Handle single input
            input_data = {}
            for key in request.POST:
                if key.startswith('input_'):
                    feature_name = key[6:]  # Remove 'input_' prefix
                    value = request.POST[key]
                    try:
                        if '.' in value:
                            input_data[feature_name] = float(value)
                        else:
                            input_data[feature_name] = int(value)
                    except ValueError:
                        input_data[feature_name] = value
            
            result = predictor.predict(model_version, input_data, analyze, analysis_method)
        
        context = {
            'result': result,
            'model_version': model_version,
        }
        return render(request, 'inference_app/result.html', context)
        
    except Exception as e:
        messages.error(request, f'Prediction failed: {str(e)}')
        return redirect('inference_app:index')


def log_detail(request, log_id):
    """View inference log details."""
    try:
        log = InferenceLog.objects.get(id=log_id)
    except InferenceLog.DoesNotExist:
        messages.error(request, 'Log not found')
        return redirect('inference_app:index')
    
    context = {'log': log}
    return render(request, 'inference_app/log_detail.html', context)


def api_predict(request):
    """API endpoint for predictions."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        model_version = data.get('model_version')
        input_data = data.get('input_data')
        analyze = data.get('analyze', False)
        analysis_method = data.get('analysis_method', 'shap')
        
        predictor = Predictor()
        result = predictor.predict(model_version, input_data, analyze, analysis_method)
        
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e),
        }, status=400)
