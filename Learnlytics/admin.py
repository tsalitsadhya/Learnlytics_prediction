from django.contrib import admin
from django.utils.html import format_html
from django.urls import path
from .models import ModelInfo, UserInput  # Import model
from .admin_views import retrain_model_view  # For retraining button

#riska
admin.site.register(UserInput)
@admin.register(ModelInfo)
class ModelInfoAdmin(admin.ModelAdmin):
    list_display = (
        "model_name",
        "training_date",
        "training_data",
        "short_summary",
        "retrain_button",
    )
    search_fields = ("model_name", "training_data")

    def short_summary(self, obj):
        return (obj.model_summary[:75] + "...") if obj.model_summary else "-"

    short_summary.short_description = "Summary"
    
    def model_file(self, obj):
        if obj.model_file:
            return obj.model_file.name  # Display the file name in the admin
        return "-"

    def retrain_button(self, obj):
        return format_html('<a class="button" href="retrain/{}/">Retrain</a>', obj.id)

    retrain_button.short_description = "Retrain"

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "retrain/<int:model_id>/",
                self.admin_site.admin_view(self.retrain_model),
                name="modelinfo_retrain",
            ),
        ]
        return custom_urls + urls


    def retrain_model(self, request, model_id):
        return retrain_model_view(request, model_id)
    

#embun
from .models import LearningProfile
admin.site.register(LearningProfile)

