from django.db import models

# Model untuk Student (Mahasiswa)
class Student(models.Model):
    stu_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    gender = models.CharField(max_length=10, blank=True, null=True)

    class Meta:
        db_table = "student"
        managed = False

    def __str__(self):
        return self.name

class ActivityType(models.Model):
    type_id = models.AutoField(primary_key=True)
    type_name = models.CharField(max_length=50)

    class Meta:
        db_table = 'activity_type'
        managed = False


# Model untuk Course (Kursus)
class Course(models.Model):
    course_id = models.AutoField(primary_key=True)
    course_name = models.CharField(max_length=255)

    class Meta:
        db_table = "course"
        managed = False

    def __str__(self):
        return self.course_name


# Model untuk Enrollment (Pendaftaran Kursus oleh Mahasiswa)
class Enrollment(models.Model):
    enroll_id = models.AutoField(primary_key=True)
    stu = models.ForeignKey(Student, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    grade = models.FloatField(blank=True, null=True)

    class Meta:
        unique_together = (
            "stu",
            "course",
        )  # Correct field name here
        db_table = "enrollment"
        managed = False

    def __str__(self):
        return f"{self.stu.name} - {self.course.course_name}"


# Model untuk InterestType (Tipe Minat)
class InterestType(models.Model):
    interes_id = models.AutoField(primary_key=True)
    interes_name = models.CharField(max_length=255, unique=True)

    class Meta:
        db_table = "interes_type"
        managed = False

    def __str__(self):
        return self.interes_name


# Model untuk CourseActivity (Aktivitas Kursus)
class CourseActivity(models.Model):
    activity_id = models.AutoField(primary_key=True)
    type_id = models.ForeignKey(ActivityType, on_delete=models.DO_NOTHING, db_column='type_id')
    course_id = models.ForeignKey(Course, on_delete=models.DO_NOTHING, db_column='course_id')
    activity_name = models.CharField(max_length=100, null=True, blank=True)
    activity_start_date = models.DateTimeField(null=True, blank=True)
    activity_end_date = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'course_activity'
        managed = False

    def __str__(self):
        return self.activity_name
    
class StudentActivityLog(models.Model):
    log_id = models.AutoField(primary_key=True)
    stu_id = models.ForeignKey(Student, on_delete=models.CASCADE, db_column='stu_id')  # Ensure this is correct
    activity_id = models.ForeignKey(CourseActivity, on_delete=models.CASCADE, db_column='activity_id')  # Ensure correct ForeignKey reference
    activity_start = models.DateTimeField(blank=True, null=True)
    activity_end = models.DateTimeField(blank=True, null=True)

    class Meta:
        db_table = "student_activity_log"
        managed = False  # Make sure this is correct if you're not managing the schema via migrations

    def __str__(self):
        return f"Log by {self.stu_id.name} for activity."

# Model untuk ModelInfo (Informasi Model Machine Learning)
class ModelInfo(models.Model):
    model_name = models.CharField(max_length=100)
    model_file = models.FileField(upload_to="models/", blank=True, null=True)
    training_data = models.CharField(max_length=255)
    training_date = models.DateTimeField()
    model_summary = models.TextField(blank=True)

    def __str__(self):
        return f"{self.model_name} - {self.training_date.strftime('%Y-%m-%d %H:%M:%S')}"


# Model untuk UserInput (Input Pengguna untuk Prediksi)
class UserInput(models.Model):
    id = models.BigAutoField(primary_key=True)
    interes_name_1 = models.CharField(max_length=255, blank=True, null=True)
    interes_name_2 = models.CharField(max_length=255, blank=True, null=True)
    last_course_1 = models.CharField(max_length=255, blank=True, null=True)
    score_last_course_1 = models.FloatField(blank=True, null=True)
    last_course_2 = models.CharField(max_length=255, blank=True, null=True)
    score_last_course_2 = models.FloatField(blank=True, null=True)
    gender = models.CharField(max_length=10, blank=True, null=True)

    class Meta:
        db_table = "Learnlytics_userinput"
        managed =  True

    def __str__(self):
        return f"Input ID: {self.id}"


class UserActivity(models.Model):
    id = models.BigAutoField(primary_key=True)  # Auto-incrementing id
    name = models.CharField(max_length=100)
    activity = models.CharField(max_length=100)
    grade = models.IntegerField()

    class Meta:
        db_table = "learnlytics_useractivity"  # Name of the table in the database
        managed = True  # Allow Django to manage this table

    def __str__(self):
        return self.name

class ActivityLog(models.Model):
    # Primary Key (auto-generated)
    session_id = models.BigAutoField(primary_key=True)  # Use this as primary key
    student_id = models.CharField(max_length=50)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField(null=True, default=None)

    class Meta:
        db_table = 'student_learning_studentactivitylog'
        indexes = [
            models.Index(fields=['student_id']),
            models.Index(fields=['start_time']),
        ]


class CourseAct(models.Model):
    ACTIVITY_TYPES = [
        ('forum', 'Discussion Forum'),
        ('task', 'Assignment'),
        ('video', 'Learning Video'),
        ('quiz', 'Quiz'),
    ]
    
    student_id = models.CharField(max_length=50)  # Could also be a ForeignKey to Student
    activity_type = models.CharField(max_length=20, choices=ACTIVITY_TYPES)
    timestamp = models.DateTimeField()  # Timestamp for when the activity occurred
    duration = models.PositiveIntegerField(help_text="Duration in minutes")  # Duration of the activity in minutes
    
    class Meta:
        db_table = 'student_learning_courseactivity'  
        managed =  True
        
        # Optional: Custom table name
    
    def __str__(self):
        return f"{self.activity_type} activity for Student {self.student_id}"

# Model for Learning Profile (Student Learning Profile)
class LearningProfile(models.Model):
    student_id = models.CharField(max_length=50, primary_key=True)  # Unique identifier for each student
    avg_duration = models.FloatField(verbose_name="Average Learning Duration (minutes)")  # Average time spent on learning
    sessions_per_week = models.FloatField(verbose_name="Sessions per Week")  # Number of learning sessions per week
    night_activity_freq = models.FloatField(verbose_name="Night Activity Frequency")  # Frequency of learning activities at night
    forum_vs_task = models.FloatField(verbose_name="Forum vs Task Ratio")  # Ratio of forum activities to task activities
    learning_type = models.CharField(max_length=20, verbose_name="Learning Type")  # Learning type based on the cluster
    cluster = models.IntegerField(verbose_name="Cluster")  # Cluster assigned to the student during clustering
    
    class Meta:
        db_table = 'student_learning_learningprofile'  # Optional: Custom table name
        managed =  True
         
    def __str__(self):
        return f"{self.student_id} - {self.learning_type}"
