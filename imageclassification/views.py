from django.shortcuts import render
import base64
from django.shortcuts import render
from django.http import JsonResponse
from .predict import predict_class

def index(request):
    return render(request, 'index.html')

def classify(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        predicted_class = predict_class()
        predicted = predicted_class.pred(image)

        image_data = base64.b64encode(image.read()).decode('utf-8')
        image.seek(0)  # Reset file pointer

        return render(request, 'child.html', {'predicted_class': predicted, 'image_data': image_data})
        # return JsonResponse({'class': predicted})
    else:
        return JsonResponse({'error': 'No image file found.'}, status=400)

def about(request):
    return render(request, 'child.html')

def home(request):
    return render(request, 'home.html')