#lisa
from django.core.management.base import BaseCommand
from Learnlytics.models import Student, Enrollment
import pandas as pd

class Command(BaseCommand):
    help = 'Create dataset to predict student grade based on enrollment information only'

    def handle(self, *args, **kwargs):
        data = []

        students = Student.objects.all()

        for student in students:
            enrollments = Enrollment.objects.filter(stu=student)

            for enroll in enrollments:
                if enroll.grade is not None:  
                    data.append({
                        'stu_id': student.stu_id,
                        'course_id': enroll.course_id,
                        'grade': enroll.grade
                    })

        df = pd.DataFrame(data)
        df.to_csv('student_grade_dataset.csv', index=False)
        
        # Output message after dataset creation
        self.stdout.write(self.style.SUCCESS(f'Dataset untuk prediksi grade dibuat: {len(data)} baris'))
