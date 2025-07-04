import pandas as pd
import joblib
from Learnlytics.models import ModelInfo
from django.core.management.base import BaseCommand
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import ast

class Command(BaseCommand):
    help = 'Train apriori model, simpan ke file pkl, dan simpan info model ke DB'

    def handle(self, *args, **kwargs):
        csv_path = 'dummy_student_activities.csv'
        self.stdout.write(f'Membaca data dari {csv_path}...')

        df = pd.read_csv(csv_path)
        df['activities'] = df['activities'].apply(ast.literal_eval)

        def categorize_grade(grade):
            if grade < 60:
                return 'Low'
            elif grade < 80:
                return 'Medium'
            else:
                return 'High'

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

        self.stdout.write('Running apriori...')
        frequent_itemsets = apriori(df_tf, min_support=0.01, use_colnames=True)

        if frequent_itemsets.empty:
            self.stdout.write('No frequent itemsets found.')
            return

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.03)

        model_filename = 'association_rules_output.pkl'
        joblib.dump(rules, model_filename)
        self.stdout.write(self.style.SUCCESS(f'Model rules saved as {model_filename}'))

        model_info = ModelInfo.objects.create(
            model_name='AssociationRulesStudent',
            model_file=model_filename,
            training_data=csv_path,
            training_date=pd.Timestamp.now(),
            model_summary=f'Apriori rules count: {len(rules)}'
        )
        self.stdout.write(self.style.SUCCESS(f'Model info saved to DB: ID {model_info.id}'))
