import os
import re
import ast
import random
import joblib
import pickle
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as opy
from django.conf import settings
from django.shortcuts import render, redirect
from .forms import UserInputForm, PartnerForm
from django.db import connection
from Learnlytics.models import UserInput
from django.views.decorators.csrf import csrf_exempt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def categorize_grade(grade):
    if grade < 60:
        return 'Low'
    elif grade < 80:
        return 'Medium'
    else:
        return 'High'

def retrain_apriori_from_data(data_list):
    if not data_list:
        return None

    df = pd.DataFrame(data_list)

    def process_activity_list(act_list):
        processed = []
        for item in act_list:
            try:
                name, grade_str = item.rsplit(':', 1)
                grade = int(grade_str)
                grade_cat = categorize_grade(grade)
                processed.append(f"{name}_{grade_cat}")
            except Exception:
                processed.append(item)
        return processed

    df['activities_processed'] = df['activities'].apply(process_activity_list)

    te = TransactionEncoder()
    te_ary = te.fit(df['activities_processed']).transform(df['activities_processed'])
    df_tf = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_tf, min_support=0.01, use_colnames=True)
    if frequent_itemsets.empty:
        return None

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.03)
    return rules

def similarity_score_numeric(activities, user_items_dict):
    score = 0.0
    for a in activities:
        try:
            name, grade_str = a.rsplit(':', 1)
            grade = int(grade_str.strip())
            if name in user_items_dict:
                diff = abs(user_items_dict[name] - grade)
                sim = max(0, 1 - diff / 100)
                score += sim
        except Exception:
            continue
    return score

def find_partner(request):
    recommendations = []
    chart_divs = []

    if request.method == 'POST':
        form = PartnerForm(request.POST)
        if form.is_valid():
            user_name = form.cleaned_data['name']
            user_activities_str = form.cleaned_data['activity']
            user_grades_str = form.cleaned_data['grade']

            user_activities_list = [act.strip() for act in user_activities_str.split(',') if act.strip()]
            user_grades_list = [g.strip() for g in user_grades_str.split(',') if g.strip()]
            min_len = min(len(user_activities_list), len(user_grades_list))
            user_activities_list = user_activities_list[:min_len]
            user_grades_list = user_grades_list[:min_len]

            # Load old data from CSV
            df_csv = pd.read_csv('dummy_student_activities.csv')
            df_csv['activities'] = df_csv['activities'].apply(ast.literal_eval)
            data_old = df_csv[['student_name', 'activities']].to_dict(orient='records')

            # Load new data from pickle (if exists)
            try:
                with open('data_activities.pkl', 'rb') as f:
                    data_pkl = pickle.load(f)
            except FileNotFoundError:
                data_pkl = []

            # Add current input user data to pkl data (remove old if same name)
            data_pkl = [d for d in data_pkl if d['student_name'] != user_name]
            user_activities_with_grades = [f"{a}:{g}" for a, g in zip(user_activities_list, user_grades_list)]
            data_pkl.append({
                'student_name': user_name,
                'activities': user_activities_with_grades
            })

            # Save updated pkl
            with open('data_activities.pkl', 'wb') as f:
                pickle.dump(data_pkl, f)

            # Combine CSV data and pkl data
            combined_data = data_old + data_pkl

            # Remove duplicates by student_name, keep latest (data_pkl preferred)
            seen = set()
            combined_unique = []
            for d in reversed(combined_data):
                if d['student_name'] not in seen:
                    combined_unique.append(d)
                    seen.add(d['student_name'])
            combined_unique.reverse()

            rules_df = retrain_apriori_from_data(combined_unique)

            if rules_df is None or rules_df.empty:
                recommendations = []
                chart_divs = []
            else:
                user_items_dict = {act: int(gr) for act, gr in zip(user_activities_list, user_grades_list)}

                def similarity_score_wrapper(activities):
                    return similarity_score_numeric(activities, user_items_dict)

                df_all = pd.DataFrame(combined_unique)
                df_all['similarity'] = df_all['activities'].apply(similarity_score_wrapper)

                df_filtered = df_all[df_all['student_name'] != user_name]
                df_filtered = df_filtered[df_filtered['similarity'] > 0]
                df_filtered = df_filtered.sort_values(by='similarity', ascending=False)
                top5 = df_filtered.head(5)

                max_similarity = top5['similarity'].max() if not top5.empty else 0
                recommendations = []
                for _, row in top5.iterrows():
                    student_activities = row['activities']
                    filtered_activities = []
                    for act in student_activities:
                        try:
                            name, grade_str = act.rsplit(':', 1)
                            grade = int(grade_str.strip())
                            if name in user_items_dict:
                                filtered_activities.append(f"{name}:{grade}")
                        except Exception:
                            continue
                    activities_str = ', '.join(filtered_activities)
                    note = ''
                    if row['similarity'] == max_similarity:
                       note = f"Top Match for You! Highlighted student activities with grades: {activities_str}"
                    recommendations.append({
                        'student_name': row['student_name'],
                        'similarity': f"{row['similarity']:.2f}",
                        'note': note
                    })

                user_items = set(user_items_dict.keys())
                def is_rule_relevant(row):
                    antecedents = {a.split('_')[0] for a in row['antecedents']}
                    consequents = {c.split('_')[0] for c in row['consequents']}
                    return len(user_items.intersection(antecedents.union(consequents))) > 0

                relevant_rules = rules_df[rules_df.apply(is_rule_relevant, axis=1)]
                top_rules = relevant_rules.sort_values(by='confidence', ascending=False).head(7)

                hover_texts = []
                for _, row in top_rules.iterrows():
                    antecedents = ', '.join([a.split('_')[0] for a in row['antecedents']])
                    consequents = ', '.join([c.split('_')[0] for c in row['consequents']])
                    hover_text = (f"Rule:\nIf you do [{antecedents}]\n"
                                  f"Then you are likely to do [{consequents}]\n"
                                  f"Support: {row['support']*100:.1f}%\n"
                                  f"Confidence: {row['confidence']*100:.1f}%\n"
                                  f"Lift: {row['lift']:.2f}")
                    hover_texts.append(hover_text)

                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=top_rules['support'],
                    y=top_rules['confidence'],
                    mode='markers',
                    marker=dict(
                        size=top_rules['lift'] * 12,
                        color=top_rules['lift'],
                        colorscale=[[0, 'rgba(235, 243, 255, 0.9)'],
                                    [0.5, 'rgba(138, 180, 240, 0.8)'],
                                    [1, 'rgba(93, 100, 190, 0.8)']],
                        line=dict(width=1, color='white'),
                        showscale=True,
                        colorbar=dict(
                            title='Lift',
                            tickfont=dict(color='#5a2c8d'),
                            outlinecolor='rgba(90, 44, 141, 0.8)')
                    ),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>'
                ))

                fig_scatter.update_layout(
                    title=dict(
                        text='Scatter Plot of Relevant Association Rules<br><sup>Bubble size & color indicate lift value</sup>',
                        x=0.5,
                        xanchor='center',
                        font=dict(size=20, color='#5a2c8d')
                    ),
                    xaxis=dict(
                        title='Support',
                        gridcolor='rgba(200, 210, 230, 0.3)',
                        zeroline=False,
                        tickformat='.1%'
                    ),
                    yaxis=dict(
                        title='Confidence',
                        gridcolor='rgba(200, 210, 230, 0.3)',
                        zeroline=False,
                        tickformat='.1%'
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=500
                )

                chart_divs.append(opy.plot(fig_scatter, auto_open=False, output_type='div'))

    else:
        form = PartnerForm()

    context = {
        'form': form,
        'recommendations': recommendations,
        'chart_divs': chart_divs,
    }
    return render(request, 'Learnlytics/shalwa/find_partner.html', context)