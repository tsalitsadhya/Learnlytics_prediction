#kejo
from django.core.management.base import BaseCommand
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

class Command(BaseCommand):
    help = 'Train model regresi prediksi nilai mahasiswa'

    def handle(self, *args, **options):
        df = pd.read_csv('etl_output/student_activity_features.csv')

        X = df.drop(columns=['stu_id', 'course_id', 'grade'])
        y = df['grade']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.stdout.write(f'MAE: {mae:.2f}')
        self.stdout.write(f'R2 Score: {r2:.2f}')

        os.makedirs('ml_models', exist_ok=True)
        joblib.dump(model, 'ml_models/grade_predictor.pkl')

        self.stdout.write(self.style.SUCCESS('Model berhasil dilatih dan disimpan.'))