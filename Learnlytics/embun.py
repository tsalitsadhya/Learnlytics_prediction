from django.core.paginator import Paginator
from django.shortcuts import render
from django.db.models import Count
from .models import LearningProfile, ActivityLog, CourseAct
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from .forms import LearningTypeForm

def learning_type_dashboard(request):
    # Ambil parameter filter dari URL
    learning_type_filter = request.GET.get('type', None)
    student_query = request.GET.get('student', None)

    # Query data dengan filter jika ada
    if learning_type_filter:
        profiles = LearningProfile.objects.filter(learning_type=learning_type_filter).order_by('student_id')
        student_activity = ActivityLog.objects.filter(learning_type=learning_type_filter).order_by('student_id')
        course = CourseAct.objects.filter(learning_type=learning_type_filter).order_by('student_id')
    else:
        profiles = LearningProfile.objects.all().order_by('student_id')
        student_activity = ActivityLog.objects.all().order_by('student_id')
        course = CourseAct.objects.all().order_by('student_id')
        
    # Data untuk dropdown
    student_list = LearningProfile.objects.all().values_list('student_id', flat=True)
    student_choices = [(s, s) for s in student_list]

    # Visualisasi detail mahasiswa terpilih
    selected_student = request.GET.get('student_id', student_list[0] if student_list else None)
    student_data = None
    
    if selected_student:
        student_data = LearningProfile.objects.filter(student_id=selected_student).first()
        
        if student_data:
            # Siapkan data untuk barchart
            metrics = {
                'Average Duration (minutes)': student_data.avg_duration,
                'Sessions per Week': student_data.sessions_per_week,
                'Night Activity Frequencies': student_data.night_activity_freq * 100,  # Konversi ke persen
                'Forum/Task Ratio': student_data.forum_vs_task * 100
            }
            
            student_chart = px.bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                title=f"Profil Belajar - {selected_student}",
                labels={'x': 'Metrik', 'y': 'Nilai'},
                color=list(metrics.keys()),
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            student_chart.update_layout(showlegend=False)
            student_viz = plot(student_chart, output_type='div')
    
    # Setup pagination
    paginator = Paginator(profiles, 10)  # 10 item per halaman
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Hitung total untuk setiap tipe belajar
    type_counts = LearningProfile.objects.values('learning_type').annotate(total=Count('learning_type'))
    
    # Buat DataFrame untuk visualisasi
    data = {
        'student_id': [p.student_id for p in profiles],
        'learning_type': [p.learning_type for p in profiles],
        'avg_duration': [p.avg_duration for p in profiles],
        'sessions_per_week': [p.sessions_per_week for p in profiles],
        'night_activity': [p.night_activity_freq for p in profiles]
    }
    df = pd.DataFrame(data)

    # 1. Chart distribusi tipe belajar
    type_chart = px.bar(
        type_counts,
        x='learning_type',
        y='total',
        title='Student Learning Type Dist.',
        color='learning_type',
        color_discrete_map={
            'Intensive': '#1f77b4',
            'Relaxed': '#ff7f0e',
            'Passive': '#2ca02c'
        }
    )
    type_chart.update_layout(showlegend=False)
    type_chart = plot(type_chart, output_type='div')
    
    # Samakan data berdasarkan student_id
    activity_map = {s.student_id: s for s in student_activity}
    course_map = {c.student_id: c for c in course}
    profile_map = {p.student_id: p for p in profiles}

    common_ids = set(activity_map) & set(course_map) & set(profile_map)

    heatmap_data = {
        'start_time': [activity_map[sid].start_time for sid in common_ids],
        'duration': [course_map[sid].duration for sid in common_ids],
        'learning_type': [profile_map[sid].learning_type for sid in common_ids],
    }
    heatmap_df = pd.DataFrame(heatmap_data)

    # Buat heatmap jam belajar
    heatmap_fig = px.density_heatmap(
        heatmap_df,
        x=heatmap_df['start_time'].dt.hour,  # Jam dalam sehari
        y=heatmap_df['start_time'].dt.dayofweek,  # Hari dalam seminggu (0=Senin)
        z=heatmap_df['duration'],
        histfunc="avg",
        title="Learning Activity Pattern by Days & Hours",
        labels={'x': 'Hours', 'y': 'Days', 'z': 'Average Duration (minutes)'},
        facet_col="learning_type"
    )
    new_titles = ["Relaxed", "Intensive", "Passive"]

    for i, annotation in enumerate(heatmap_fig.layout.annotations):
        annotation.text = new_titles[i]
    heatmap_fig.update_yaxes(categoryorder='array', 
                           categoryarray=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    heatmap_fig.update_layout(
        updatemenus=[dict(
            buttons=[
                dict(label="Semua Tipe", method="update", args=[{"visible": [True, True, True]}]),
                dict(label="Intensive", method="update", args=[{"visible": [True, False, False]}]),
            ],
            direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.1, y=1.1
        )],
    )
    
    heatmap_fig.update_xaxes(
        showticklabels=False, showline=False, ticks="",
    )
    heatmap = plot(heatmap_fig, output_type='div')
    
    context = {
        'type_chart': type_chart,
        'heatmap': heatmap,
        'page_obj': page_obj,
        'profiles': profiles,
        'total_students': len(profiles),
        'current_filter': learning_type_filter,
        'type_counts': {t['learning_type']: t['total'] for t in type_counts},
        'student_list': student_list,
        'selected_student': selected_student,
        'student_viz': student_viz if selected_student else None,
        'student_data': student_data
    }
    
    return render(request, 'Learnlytics/embun/dashboard.html', context)


def predict_learning_type(request):
    predicted_type = None
    silhouette_score_value = None
    form = LearningTypeForm()

    if request.method == 'POST':
        form = LearningTypeForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            input_df = pd.DataFrame([data])

            # Ambil data historis dari database
            qs = LearningProfile.objects.all().values(
                'avg_duration', 'sessions_per_week', 'night_activity_freq', 'forum_vs_task', 'cluster'
            )
            df = pd.DataFrame.from_records(qs)

            if len(df) >= 3:
                # Normalisasi data historis
                feature_cols = ['avg_duration', 'sessions_per_week', 'night_activity_freq', 'forum_vs_task']
                X = df[feature_cols]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Latih ulang model KMeans
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(X_scaled)

                # Hitung skor silhouette
                silhouette_score_value = silhouette_score(X_scaled, clusters)

                # Normalisasi data input user dan prediksi klasternya
                input_X = input_df[feature_cols]
                X_input_scaled = scaler.transform(input_X)
                cluster = kmeans.predict(X_input_scaled)[0]

                # Hitung skor centroid untuk mapping label
                centroids = kmeans.cluster_centers_
                centroid_scores = []
                for idx, c in enumerate(centroids):
                    score = c[0] + c[1]  # avg_duration + sessions_per_week
                    centroid_scores.append((idx, score))

                sorted_clusters = sorted(centroid_scores, key=lambda x: x[1], reverse=True)
                cluster_label_map = {
                    sorted_clusters[0][0]: 'Intensive',
                    sorted_clusters[1][0]: 'Relaxed',
                    sorted_clusters[2][0]: 'Passive'
                }

                predicted_type = cluster_label_map[cluster]

                # Simpan ke database
                LearningProfile.objects.create(
                    student_id=data['student_id'],
                    avg_duration=data['avg_duration'],
                    sessions_per_week=data['sessions_per_week'],
                    night_activity_freq=data['night_activity_freq'],
                    forum_vs_task=data['forum_vs_task'],
                    learning_type=predicted_type,
                    cluster=cluster
                )

    return render(request, 'Learnlytics/embun/predict.html', {
        'form': form,
        'predicted_type': predicted_type,
        'silhouette_score': silhouette_score_value
    })
