from django.urls import path
from . import embun, keizia, lisa, riska, shalwa
from . import admin_views, views


urlpatterns = [
    path("", views.home, name="home"),
    path("about/", views.about, name="about"),
    path('course/', riska.predict_course, name='course'),
    path(
        "admin/retrain-model/<int:model_id>/",
        admin_views.retrain_model_view,
        name="retrain_model",
    ),

    path('dashboard/', embun.learning_type_dashboard, name='dashboard'),
    path('predict/', embun.predict_learning_type, name='predict_learning_type'),
    #lisa
    path('predict_graduation/', lisa.predict_graduation, name='predict_graduation'),
    path('find_partner/', shalwa.find_partner, name='find_partner'),
    path('admin/retrain-model/<int:model_id>/', admin_views.retrain_model_view, name='retrain_model'),
   
    path('predict_grade/', keizia.prediksi_nilai_view, name='predict_grade'),
]