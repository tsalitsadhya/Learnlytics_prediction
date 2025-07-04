from django import forms
from .models import InterestType, Course

#riska
class UserInputForm(forms.Form):
    interest_choices = [(item, item) for item in InterestType.objects.values_list('interes_name', flat=True).distinct()]
    course_queryset = Course.objects.all()

    interes_name_1 = forms.ChoiceField(
        choices=interest_choices,
        label="Interest Subject",
        required=True,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    interes_name_2 = forms.ChoiceField(
        choices=interest_choices,
        label="Interest Subject",
        required=True,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    last_course_1 = forms.ModelChoiceField(
        queryset=course_queryset,
        label="Last Course",
        empty_label="Choose a course",
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    score_last_course_1 = forms.FloatField(
        label="Score for Last Course",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )

    last_course_2 = forms.ModelChoiceField(
        queryset=course_queryset,
        label="Last Course",
        empty_label="Choose a course",
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    score_last_course_2 = forms.FloatField(
        label="Score for Last Course",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )

    gender = forms.ChoiceField(
        choices=[('Male', 'Male'), ('Female', 'Female')],
        label="Gender",
        widget=forms.Select(attrs={'class': 'form-control'})
    )


#embun
from .models import LearningProfile

class LearningTypeForm(forms.ModelForm):
    class Meta:
        model = LearningProfile
        fields = [
            'student_id',
            'avg_duration',
            'sessions_per_week',
            'night_activity_freq',
            'forum_vs_task',
        ]
        labels = {
            'student_id': 'Student ID',
            'avg_duration': 'Average Duration (minutes)',
            'sessions_per_week': 'Sessions per Week',
            'night_activity_freq': 'Night Activity Frequency (0-1)',
            'forum_vs_task': 'Forum/Task Ratio (0-1)',
        }
 
#shalwa
class PartnerForm(forms.Form):
    name = forms.CharField(
        label='Your Name',
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter your name'})
    )
    activity = forms.CharField(
        label='Activity Names (separate by comma)',
        max_length=200,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., Course 1 - Century, Course 2 - Ahead'})
    )
    grade = forms.CharField(
        label='Grades (separate by comma, match activity order)',
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 85, 90'})
    )

#keizia
class GradePredictionForm(forms.Form):
    total_duration = forms.FloatField(label="Total Activity Duration (minutes)")
    avg_duration = forms.FloatField(label="Average Activity Duration (minutes)")
    activity_count = forms.IntegerField(label="Total Activities")
    Lecture = forms.FloatField(label="Lecture Duration (minutes)", required=False)
    Quiz = forms.FloatField(label="Quiz Duration (minutes)", required=False)
    Assignment = forms.FloatField(label="Assignment Duration (minutes)", required=False)



from Learnlytics.models import Student

class StudentCourseForm(forms.Form):
    stu_id = forms.ModelChoiceField(queryset=Student.objects.all(), label='Select Student')
    course_id = forms.ModelChoiceField(queryset=Course.objects.all(), label='Select Course')


# lisa form
from django import forms
from Learnlytics.models import Student, Course

class StudentCourseForm(forms.Form):
    stu_id = forms.ChoiceField(label='Select Student')
    course_id = forms.ChoiceField(label='Select Course')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['stu_id'].choices = [(s.stu_id, s.name) for s in Student.objects.all()]
        self.fields['course_id'].choices = [(c.course_id, c.course_name) for c in Course.objects.all()]
