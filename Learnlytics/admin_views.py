from django.shortcuts import get_object_or_404, redirect
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib import messages
from django.utils.timezone import now
from .models import ModelInfo
import os
import joblib
from django.core.files.base import ContentFile
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report

@staff_member_required
def retrain_model_view(request, model_id):
    model_instance = get_object_or_404(ModelInfo, id=model_id)

    try:
        new_model_filename = f"{model_instance.model_name}_retrained.pkl"
        new_model_path = os.path.join("save_model/riska", new_model_filename)

        # Load the existing model for retraining
        model_data = joblib.load(os.path.join("save_model/riska", "random_forest_model.pkl"))
        model = model_data["model"]

        # Simulate new data for retraining
        X_train = np.random.rand(100, 8)  # 100 samples, 8 features
        y_train = np.random.randint(0, 5, 100)  # 5 possible activities (classes)

        # Retrain the model
        new_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        new_model.fit(X_train, y_train)  # Fit the model with updated data
        joblib.dump(new_model, new_model_path)  # Save the new model

        # Generate classification report
        y_pred = new_model.predict(X_train)
        report = classification_report(y_train, y_pred)

        # Save model file into ModelInfo
        with open(new_model_path, "rb") as f:
            file_content = f.read()

        # Update the model file and summary
        model_instance.model_file.save(new_model_filename, ContentFile(file_content), save=False)
        model_instance.model_summary = f"Model retrained with new data. Classification Report: {report}"
        model_instance.training_date = now()
        model_instance.save()

        # Send success message
        messages.success(request, f"Model '{model_instance.model_name}' retrained successfully.")
    except Exception as e:
        messages.error(request, f"Failed to retrain model: {e}")

    return redirect("admin:Learnlytics_modelinfo_changelist")  # Redirect to model info changelist

