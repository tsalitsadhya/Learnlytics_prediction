from django.shortcuts import render
from .forms import GradePredictionForm
import pandas as pd
import joblib
import os

def prediksi_nilai_view(request):
    result = None
    if request.method == 'POST':
        form = GradePredictionForm(request.POST)
        if form.is_valid():
            fitur = form.cleaned_data
            result = predict_grade(fitur)
    else:
        form = GradePredictionForm()

    return render(request, 'Learnlytics/keizia/predict.html', {'form': form, 'result': result})


MODEL_PATH = os.path.join('ml_models', 'grade_predictor.pkl')

def load_model():
    return joblib.load(MODEL_PATH)

def predict_grade(features_dict):
    model = load_model()
    ordered_keys = model.feature_names_in_

    # Create a single-row DataFrame with correct column names
    input_df = pd.DataFrame([{
        key: features_dict.get(key, 0) for key in ordered_keys
    }])

    prediction = model.predict(input_df)
    return round(prediction[0], 2)
