import os
import joblib
import pandas as pd
from django.shortcuts import render
from django.conf import settings
from django.db import connection

MODEL_DIR = os.path.join(settings.BASE_DIR, 'Learnlytics', 'ml_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'grade_predict.pkl')

def get_students():
    with connection.cursor() as cursor:
        cursor.execute("SELECT stu_id, name FROM student ORDER BY name")
        return cursor.fetchall()

def get_courses():
    with connection.cursor() as cursor:
        cursor.execute("SELECT course_id, course_name FROM course ORDER BY course_name")
        return cursor.fetchall()

def get_student_activity_features(stu_id, course_id):
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT
                COALESCE(SUM(EXTRACT(EPOCH FROM (sal.activity_end - sal.activity_start)) / 60), 0) AS total_activity_minutes,
                COUNT(sal.activity_id) AS activity_count
            FROM student_activity_log sal
            JOIN course_activity ca ON sal.activity_id = ca.activity_id
            WHERE sal.stu_id = %s AND ca.course_id = %s
        """, [stu_id, course_id])
        result = cursor.fetchone()
    return result if result else (0, 0)

def predict_graduation(request):
    students = get_students()
    courses = get_courses()

    context = {
        'students': students,
        'courses': courses,
        'prediction': None,
        'error': None,
    }

    if request.method == 'POST':
        stu_id_raw = request.POST.get('stu_id')
        course_id_raw = request.POST.get('course_id')

        if not stu_id_raw or not course_id_raw:
            context['error'] = "Student dan Course harus dipilih."
            return render(request, 'Learnlytics/lisa/predict_graduations.html', context)

        try:
            stu_id = int(stu_id_raw)
            course_id = int(course_id_raw)
        except ValueError:
            context['error'] = "Input tidak valid."
            return render(request, 'Learnlytics/lisa/predict_graduations.html', context)

        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            context['error'] = f"Error loading model: {e}"
            return render(request, 'Learnlytics/lisa/predict_graduations.html', context)

        total_minutes, activity_count = get_student_activity_features(stu_id, course_id)

        input_df = pd.DataFrame([{
            'stu_id': stu_id,
            'course_id': course_id,
            'total_activity_minutes': total_minutes,
            'activity_count': activity_count
        }])

        try:
            prediction = model.predict(input_df)[0]
            try:
                proba = model.predict_proba(input_df)[0][prediction]
                confidence = f"{round(proba * 100, 2)}%"
            except AttributeError:
                confidence = "N/A"

            status = "Passed" if prediction == 1 else "Failed"
            context['prediction'] = {
                'status': status,
                'confidence': confidence
            }
        except Exception as e:
            context['error'] = f"Prediction error: {e}"

    return render(request, 'Learnlytics/lisa/predict_graduations.html', context)
