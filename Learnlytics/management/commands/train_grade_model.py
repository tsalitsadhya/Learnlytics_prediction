#lisa
from django.core.management.base import BaseCommand
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from django.conf import settings
from Learnlytics.models import ModelInfo
from datetime import datetime

class Command(BaseCommand):
    help = 'Train classification model to predict graduation and save model file + metadata'

    def handle(self, *args, **kwargs):
        dataset_path = 'student_dataset.csv'

        if not os.path.exists(dataset_path):
            self.stdout.write(self.style.ERROR(f"Dataset file '{dataset_path}' not found. Run 'etl_dataset.py' first."))
            return

        df = pd.read_csv(dataset_path)

        # Define features and target variable
        features = ['course_id', 'grade']
        X = df[features]
        y = df['label']

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        accuracy = model.score(X_test, y_test)

        # Output results
        self.stdout.write(self.style.SUCCESS(f'Classification Report:\n{report}'))
        self.stdout.write(self.style.SUCCESS(f'Accuracy: {accuracy*100:.2f}%'))

        # Save the model to a file
        model_file = os.path.join(settings.BASE_DIR, 'course_model.pkl')
        joblib.dump(model, model_file)
        self.stdout.write(self.style.SUCCESS(f'Model saved to {model_file}'))

        # Save model metadata to database
        model_info, created = ModelInfo.objects.update_or_create(
            model_name='Graduation Course Prediction Model',
            defaults={
                'model_file': model_file,
                'training_data': dataset_path,
                'training_date': datetime.now(),
                'model_summary': 'RandomForest model trained to predict student graduation based on course and grade'
            }
        )
        self.stdout.write(self.style.SUCCESS('ModelInfo metadata saved/updated'))
