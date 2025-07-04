import os
import random
import joblib
import pandas as pd
from datetime import datetime
from django.conf import settings
from django.shortcuts import render
from Learnlytics.forms import UserInputForm
from django.db import connection
from Learnlytics.models import UserInput, Course

#riska 
def predict_course(request):
    if request.method == 'POST':
        form = UserInputForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            
            # save user to model userinput
            user_input = UserInput(
                interes_name_1=data['interes_name_1'],
                interes_name_2=data['interes_name_2'],
                last_course_1=data['last_course_1'],
                score_last_course_1=data['score_last_course_1'],
                last_course_2=data['last_course_2'],
                score_last_course_2=data['score_last_course_2'],
                gender=data['gender']
            )
            user_input.save()  # Save to database

            model_path = os.path.join(settings.BASE_DIR, 'save_model', 'riska', 'random_forest_model.pkl')
            try:
                model_data = joblib.load(model_path)
                model_rf = model_data["model"]
                le_course = model_data["le_course"]
                le_interest = model_data["le_interest"]
                le_gender = model_data["le_gender"]
                le_activity = model_data["le_activity"]
            except FileNotFoundError:
                return render(request, 'error.html', {'error_message': 'Model not found.'})

            # Mapping Gender and Interests
            gender_map = {'Male': 0, 'Female': 1}
            interes_map = {
                'Leadership': 0, 'Strategy': 1, 'Empathy & Social': 2, 'Time Management': 3,
                'Problem Solving': 4, 'Teamwork': 5, 'Purpose': 6, 'Critical Thinking': 7
            }

            gender = gender_map.get(data['gender'], 0)
            interes_name1 = interes_map.get(data['interes_name_1'], 0)
            interes_name2 = interes_map.get(data['interes_name_2'], 0)

            # Fetch the course_id from the Course model if needed
            course1 = Course.objects.get(course_name=data['last_course_1']).course_id
            course2 = Course.objects.get(course_name=data['last_course_2']).course_id

            today = pd.to_datetime('now')
            days_since_activity = (today - pd.to_datetime('2025-05-26')).days

            input_data = pd.DataFrame({
                'gender': [gender],
                'score_last_course_1': [data['score_last_course_1']],
                'score_last_course_2': [data['score_last_course_2']],
                'interes_name_1': [interes_name1],
                'interes_name_2': [interes_name2],
                'type_name1': [course1],
                'type_name2': [course2],
                'days_since_activity': [days_since_activity],
                'grade1': [data['score_last_course_1']],
                'grade2': [data['score_last_course_2']]
            })

            # Encoding
            input_data['gender_enc'] = le_gender.transform(input_data['gender'])
            input_data['interest1_enc'] = le_interest.transform(input_data['interes_name_1'])
            input_data['interest2_enc'] = le_interest.transform(input_data['interes_name_2'])
            input_data['course1_enc'] = le_course.transform(input_data['type_name1'])
            input_data['course2_enc'] = le_course.transform(input_data['type_name2'])

            columns_order = ['interest1_enc', 'interest2_enc', 'course1_enc', 'course2_enc', 'grade1', 'grade2', 'gender_enc']
            input_data = input_data[columns_order]

            y_pred_proba = model_rf.predict_proba(input_data)
            top_3_courses_indices = y_pred_proba[0].argsort()[-3:][::-1]
            all_courses = le_activity.classes_
            top_3_courses = [all_courses[i] for i in top_3_courses_indices]
            progress_bar_values = [y_pred_proba[0][i] * 100 for i in top_3_courses_indices]
            top_3_courses_with_progress = list(zip(top_3_courses, progress_bar_values))

            course_descriptions = {
                "Course 1 - Job": {"description": "Introduces the world of work, career preparation, and basic skills needed to succeed in the modern era."},
                "Course 1 - Century": {"description": "Focuses on 21st-century skills such as critical thinking, creativity, collaboration, and communication."},
                "Course 1 - Summer": {"description": "A self-development program during the summer break that includes interest exploration and soft skills development."},
                "Course 2 - Ahead": {"description": "Guides students in creating future plans with long-term learning strategies."},
                "Course 2 - Group": {"description": "Focuses on effective teamwork and group dynamics."},
                "Course 2 - Charge": {"description": "Fosters initiative and personal responsibility to lead and take action."},
                "Course 3 - Week": {"description": "Teaches weekly time management, productivity, and effective planning techniques."},
                "Course 3 - Do": {"description": "Encourages turning ideas into real actions and building an execution mindset."},
                "Course 3 - Key": {"description": "Helps identify personal and professional success factors and build positive habits."},
                "Course 4 - Third": {"description": "Explores third-party or alternative perspectives for problem-solving."},
                "Course 4 - Look": {"description": "Enhances observation, analysis, and reflection skills to better understand situations."},
                "Course 4 - Purpose": {"description": "Helps students discover their life purpose and core personal values."},
                "Course 5 - Their": {"description": "Develops empathy, understanding of others' perspectives, and inclusive communication."},
                "Course 5 - Lead": {"description": "Covers leadership fundamentals, influence-building, and decision-making."},
                "Course 5 - Need": {"description": "Identifies human needs (individual/group) and how to fulfill them ethically and effectively."}
            }

            course_details = []
            for course, progress in top_3_courses_with_progress:
                detail = course_descriptions.get(course, {"description": "N/A"})
                course_details.append({
                    "name": course,
                    "progress": progress,
                    "description": detail["description"],
                })

            gender_distribution = {
                course_name: [random.randint(10, 100), random.randint(10, 100)] for course_name in top_3_courses
            }

            return render(request, 'Learnlytics/riska/course.html', {
                'form': form,
                'top_3_courses_with_progress': top_3_courses_with_progress,
                'score_last_course_1': data['score_last_course_1'],
                'score_last_course_2': data['score_last_course_2'],
                'top_3_courses': top_3_courses,
                'top_3_course_details': course_details,
                'gender_values': gender_distribution,
                'gender_labels': ['Male', 'Female']
            })
    else:
        form = UserInputForm()

    return render(request, 'Learnlytics/riska/course.html', {'form': form})

