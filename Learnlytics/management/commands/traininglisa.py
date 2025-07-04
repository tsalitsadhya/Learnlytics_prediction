import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from django.core.management.base import BaseCommand
import psycopg2
from django.conf import settings

# Sesuaikan konfigurasi DB kamu
DB_CONFIG = {
    'host': 'localhost',
    'database': 'examdb',
    'user': 'postgres',
    'password': 'riska06',  # ganti dengan passwordmu
    'port': 5432
}

class Command(BaseCommand):
    help = 'Train RandomForest model to predict student grade and save as grade_predict.pkl'

    def handle(self, *args, **kwargs):
        MODEL_DIR = os.path.join(settings.BASE_DIR, 'Learnlytics', 'ml_models')
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, 'grade_predict.pkl')

        self.stdout.write("Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)

        query = """
            SELECT e.stu_id, e.course_id, e.grade,
                   COALESCE(SUM(EXTRACT(EPOCH FROM (sal.activity_end - sal.activity_start)) / 60), 0) AS total_activity_minutes,
                   COUNT(sal.activity_id) AS activity_count
            FROM enrollment e
            LEFT JOIN student_activity_log sal ON e.stu_id = sal.stu_id
            LEFT JOIN course_activity ca ON sal.activity_id = ca.activity_id AND e.course_id = ca.course_id
            GROUP BY e.stu_id, e.course_id, e.grade
        """
        df = pd.read_sql(query, conn)
        conn.close()
        self.stdout.write(f"Data retrieved: {len(df)} rows")

        # Preprocessing
        df['pass_fail'] = (df['grade'] >= 60).astype(int)
        X = df[['stu_id', 'course_id', 'total_activity_minutes', 'activity_count']]
        y = df['pass_fail']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.stdout.write("Training RandomForest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save model
        joblib.dump(model, model_path)
        self.stdout.write(self.style.SUCCESS(f'Model saved to {model_path}'))
