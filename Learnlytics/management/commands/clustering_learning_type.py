#embun
import pandas as pd
from django.core.management.base import BaseCommand
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from Learnlytics.models import ActivityLog, CourseAct, LearningProfile

class Command(BaseCommand):
    help = 'Cluster students into learning types with NaN handling'

    def handle(self, *args, **options):
        try:
            # 1. Ambil data dan hitung fitur
            features = self.calculate_features()
            
            # 2. Handle NaN values
            features_clean = self.handle_nan(features)
            
            # 3. Normalisasi dan clustering
            self.run_clustering(features_clean)
            
            self.stdout.write(self.style.SUCCESS("Clustering selesai!"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))

    def calculate_features(self):
        """Hitung fitur dengan penanganan error untuk data kosong."""
        logs = ActivityLog.objects.all().values(
            'student_id', 'start_time', 'end_time', 'session_id'
        )
        df = pd.DataFrame.from_records(logs)
        
        if df.empty:
            raise ValueError("Data aktivitas kosong! Pastikan tabel StudentActivityLog terisi.")
        
        # Convert start_time and end_time to datetime format
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        # Hitung durasi (dalam menit)
        df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
        
        # Group by student
        features = df.groupby('student_id').apply(
            lambda x: pd.Series({
                'avg_duration': x['duration'].mean(),
                'sessions_per_week': x['session_id'].nunique() / 4,
                'night_activity_freq': self.get_night_freq(x),
                'forum_vs_task': self.get_forum_ratio(x['student_id'].iloc[0])
            })
        ).reset_index()
        
        return features

    def get_night_freq(self, df_group):
        """Hitung frekuensi aktivitas malam (20:00-06:00) dengan penanganan divisi zero."""
        night_mask = (df_group['start_time'].dt.hour >= 20) | (df_group['start_time'].dt.hour <= 6)
        night_count = night_mask.sum()
        total = len(df_group)
        return night_count / total if total > 0 else 0.0

    def get_forum_ratio(self, student_id):
        """Hitung rasio forum/tugas dengan penanganan data kosong."""
        activities = CourseAct.objects.filter(
            student_id=student_id,
            activity_type__in=['forum', 'task']
        ).values('activity_type')
        
        if not activities:
            return 0.5  # Nilai default jika tidak ada data
        
        df = pd.DataFrame.from_records(activities)
        counts = df['activity_type'].value_counts()
        forum = counts.get('forum', 0)
        task = counts.get('task', 0)
        return forum / (forum + task) if (forum + task) > 0 else 0.5

    def handle_nan(self, df):
        """Ganti NaN dengan nilai median atau default."""
        # Gunakan SimpleImputer untuk mengisi NaN
        imputer = SimpleImputer(strategy='median')
        df[['avg_duration', 'sessions_per_week', 'night_activity_freq', 'forum_vs_task']] = imputer.fit_transform(df[['avg_duration', 'sessions_per_week', 'night_activity_freq', 'forum_vs_task']])
        return df

    def run_clustering(self, df):
        """Normalisasi data dan jalankan KMeans."""
        scaler = StandardScaler()
        X = scaler.fit_transform(df[['avg_duration', 'sessions_per_week', 'night_activity_freq', 'forum_vs_task']])
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)

        # Ambil centroid
        centroids = kmeans.cluster_centers_

        # Hitung skor (semakin besar, semakin intensif)
        centroid_scores = []
        for idx, c in enumerate(centroids):
            # Skor berdasarkan banyaknya durasi dan sesi
            # (pakai indeks ke-0 dan ke-1, karena itu adalah avg_duration dan sessions_per_week)
            score = c[0] + c[1]  # Bisa tambahkan bobot jika mau
            centroid_scores.append((idx, score))

        # Urutkan skor untuk mapping
        sorted_clusters = sorted(centroid_scores, key=lambda x: x[1], reverse=True)
        cluster_label_map = {
            sorted_clusters[0][0]: 'Intensive',
            sorted_clusters[1][0]: 'Relaxed',
            sorted_clusters[2][0]: 'Passive'
        }

        # Map ke learning type
        df['learning_type'] = df['cluster'].map(cluster_label_map)

        # Simpan ke database
        for _, row in df.iterrows():
            LearningProfile.objects.update_or_create(
                student_id=row['student_id'],
                defaults={
                    'avg_duration': row['avg_duration'],
                    'sessions_per_week': row['sessions_per_week'],
                    'night_activity_freq': row['night_activity_freq'],
                    'forum_vs_task': row['forum_vs_task'],
                    'learning_type': row['learning_type'],
                    'cluster': row['cluster']
                }
            )
