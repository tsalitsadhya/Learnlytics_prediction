#kejp
from django.core.management.base import BaseCommand
from Learnlytics.models import Student, StudentActivityLog, Enrollment, ActivityType
import pandas as pd
import os

class Command(BaseCommand):
    help = 'ETL fitur untuk prediksi nilai menggunakan regresi'

    def handle(self, *args, **kwargs):
        self.stdout.write("Mengambil data log aktivitas...")

        logs = StudentActivityLog.objects.select_related('stu_id', 'activity_id__type_id', 'activity_id__course_id')
        enrollments = Enrollment.objects.select_related('stu_id', 'course_id')
        activity_types = {a.type_id: a.type_name for a in ActivityType.objects.all()}

        data = []

        for log in logs:
            student = log.stu_id
            activity = log.activity_id
            course_id = activity.course_id_id
            type_name = activity.type_id.type_name if activity.type_id else 'Unknown'

            if not log.activity_end:
                continue  # skip if activity not finished

            duration = (log.activity_end - log.activity_start).total_seconds() / 60.0

            enrollment = enrollments.filter(stu_id=student, course_id=course_id).first()
            if not enrollment:
                continue

            data.append({
                'stu_id': student.stu_id,
                'course_id': course_id,
                'activity_type': type_name,
                'duration': duration,
                'grade': enrollment.grade
            })

        if not data:
            self.stdout.write(self.style.WARNING('Tidak ada data yang dapat diproses.'))
            return

        df = pd.DataFrame(data)

        self.stdout.write("Melakukan agregasi fitur...")

        # Agregasi fitur numerik
        grouped = df.groupby(['stu_id', 'course_id', 'grade']).agg(
            total_duration=('duration', 'sum'),
            avg_duration=('duration', 'mean'),
            activity_count=('duration', 'count')
        ).reset_index()

        # Pivot jenis aktivitas
        pivot_types = df.pivot_table(index=['stu_id', 'course_id'], 
                                     columns='activity_type',
                                     values='duration',
                                     aggfunc='sum').reset_index().fillna(0)

        # Gabungkan semua
        final_df = pd.merge(grouped, pivot_types, on=['stu_id', 'course_id'])

        # Output ke CSV
        output_folder = 'etl_output'
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, 'student_activity_features.csv')

        final_df.to_csv(output_path, index=False)

        self.stdout.write(self.style.SUCCESS(f'Data ETL berhasil disimpan di: {output_path}'))