#shalwa
import random
from django.core.management.base import BaseCommand
from Learnlytics.models import Student, CourseActivity
import pandas as pd

class Command(BaseCommand):
    help = 'Dummy ETL: Buat data simulasi aktivitas mahasiswa untuk apriori dengan grade random'

    def handle(self, *args, **kwargs):
        students = Student.objects.all()
        activities = list(CourseActivity.objects.all())

        data = []
        for student in students:
            
            aktivitas_pilihan = random.sample(activities, k=random.randint(2,5))

            aktivitas_nama = []
            for a in aktivitas_pilihan:
                grade = random.randint(50, 100)
                aktivitas_nama.append(f"{a.activity_name}:{grade}")

            data.append({
                'student_id': student.stu_id,
                'student_name': student.name,
                'activities': aktivitas_nama
            })

        df = pd.DataFrame(data)
        csv_path = 'dummy_student_activities.csv'
        df.to_csv(csv_path, index=False)

        self.stdout.write(self.style.SUCCESS(f'Dummy data berhasil disimpan di {csv_path}'))
