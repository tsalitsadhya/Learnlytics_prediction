from django.shortcuts import render
from django.db import connection


def home(request):
    # Fetching course names and average grades from the database
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT c.course_name, AVG(e.grade) AS avg_grade
            FROM enrollment e
            JOIN course c ON e.course_id = c.course_id
            GROUP BY c.course_name
        """)
        rows = cursor.fetchall()

    # Convert Decimal to float
    if rows:
        course_names = [row[0] for row in rows]
        avg_grades = [float(row[1]) for row in rows]
    else:
        course_names = []
        avg_grades = []

    return render(request, 'Learnlytics/home.html', {
        'course_names': course_names,
        'avg_grades': avg_grades
    })


def about(request):
    return render(request, "Learnlytics/about.html")
