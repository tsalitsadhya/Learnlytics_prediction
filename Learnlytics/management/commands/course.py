#riska
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, matthews_corrcoef
import joblib
import os
from django.core.management.base import BaseCommand
from django.db import connection
from django.conf import settings
from django.utils.timezone import now
from Learnlytics.models import ModelInfo

class Command(BaseCommand):
    help = "Train model to predict the activity name using interests, past courses, grades, and gender"

    def handle(self, *args, **options):
        self.stdout.write("Starting training...")

        # SQL query to fetch the data
        query = """
        SELECT
            s.stu_id,
            s.name AS student_name,
            s.gender,
            e.grade,
            it.interes_name,
            ca.activity_name,   -- Target column (activity_name)
            ca.activity_start_date,
            ca.activity_end_date,
            c.course_name,
            c.course_id,
            ca.type_id,
            at.type_name
        FROM 
            student s
        JOIN 
            enrollment e ON s.stu_id = e.stu_id
        JOIN 
            course c ON e.course_id = c.course_id
        JOIN 
            course_activity ca ON c.course_id = ca.course_id
        JOIN 
            activity_type at ON ca.type_id = at.type_id
        JOIN 
            interes_type it ON ca.activity_id = it.activity_id
        WHERE
            e.grade IS NOT NULL
        ORDER BY 
            s.stu_id, ca.activity_start_date;
        """

        # Fetch data and create DataFrame
        df = pd.read_sql(query, connection)

        # Data preprocessing
        df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
        df['interes_name'] = df['interes_name'].astype('category').cat.codes  # Label encoding interests
        df['type_name'] = df['type_name'].astype('category').cat.codes  # Label encoding activity type
        df['days_since_activity'] = (pd.to_datetime('now') - pd.to_datetime(df['activity_start_date'])).dt.days

        # Creating multiple course and interest columns
        df["interest1"] = df.groupby("stu_id")["interes_name"].shift(0)
        df["interest2"] = df.groupby("stu_id")["interes_name"].shift(1)
        df["course1"] = df.groupby("stu_id")["course_id"].shift(0)
        df["course2"] = df.groupby("stu_id")["course_id"].shift(1)
        df["grade1"] = df.groupby("stu_id")["grade"].shift(0)
        df["grade2"] = df.groupby("stu_id")["grade"].shift(1)

        df = df.dropna(subset=["interest1", "interest2", "course1", "course2", "grade1", "grade2", "days_since_activity"])

        # Encoding
        le_course = LabelEncoder()
        le_course.fit(pd.concat([df["course1"], df["course2"]]).unique())

        le_interest = LabelEncoder()
        le_interest.fit(pd.concat([df["interest1"], df["interest2"]]).unique())

        le_gender = LabelEncoder()
        le_gender.fit(df["gender"].unique())

        le_activity = LabelEncoder()
        le_activity.fit(df["activity_name"].unique())

        df["course1_enc"] = le_course.transform(df["course1"])
        df["course2_enc"] = le_course.transform(df["course2"])
        df["interest1_enc"] = le_interest.transform(df["interest1"])
        df["interest2_enc"] = le_interest.transform(df["interest2"])
        df["gender_enc"] = le_gender.transform(df["gender"])
        df["activity_name_enc"] = le_activity.transform(df["activity_name"])

        # Feature set
        X = df[["interest1_enc", "interest2_enc", "course1_enc", "course2_enc", "grade1", "grade2", "gender_enc"]]
        y = df["activity_name_enc"]  # Using activity_name as the target

        self.stdout.write(f"Dataset ready. X shape: {X.shape}, y shape: {y.shape}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train the model
        model_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_rf.fit(X_train, y_train)

        # Predict the classes for the test set
        y_pred = model_rf.predict(X_test)

        # Get the class labels (activity names)
        class_labels = le_activity.classes_

        # Get classification report
        report = classification_report(y_test, y_pred, target_names=class_labels)
        self.stdout.write("Classification Report:")
        self.stdout.write(report)

        # Calculating MCC for Random Forest
        mcc_rf = matthews_corrcoef(y_test, y_pred)
        self.stdout.write(f"MCC for Random Forest: {mcc_rf}")

        # Save everything into one file
        model_dir = os.path.join(settings.BASE_DIR, "save_model", "riska")
        os.makedirs(model_dir, exist_ok=True)

        model_data = {
            "model": model_rf,
            "le_course": le_course,
            "le_interest": le_interest,
            "le_gender": le_gender,
            "le_activity": le_activity,
        }

        
        model_filename = "random_forest_model.pkl"
        model_path = os.path.join(model_dir, model_filename)

        # Save the model
        joblib.dump(model_data, model_path)

        self.stdout.write(self.style.SUCCESS(f"Model and encoders saved in {model_dir}"))

        # Create ModelInfo entry
        model_info = ModelInfo(
            model_name="random_forest_model",  # Name of the model
            model_file=model_path,  # Path where the model is saved
            training_data="SQL Query Data",  # Description of data source
            training_date=now(),  # Current date and time
            model_summary=report  # Using classification report as model summary
        )
        model_info.save()  # Save model info into the database
        self.stdout.write(self.style.SUCCESS("ModelInfo has been saved to the database"))