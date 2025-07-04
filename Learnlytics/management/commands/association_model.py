#kezia
from django.core.management.base import BaseCommand
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class Command(BaseCommand):
    help = 'Run association rules analysis on student activity and grades with relaxed thresholds'

    def handle(self, *args, **kwargs):
        # Load transaksi CSV
        df = pd.read_csv('etl_output/activity_grade_transactions.csv')
        transactions = df.fillna('').values.tolist()
        transactions = [[item for item in row if item != ''] for row in transactions]

        # One-hot encoding
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

        # Apriori dengan min_support lebih rendah
        frequent_itemsets = apriori(df_onehot, min_support=0.1, use_colnames=True)
        frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) <= 3)]

        self.stdout.write(f"Jumlah frequent itemsets yang ditemukan setelah filter: {len(frequent_itemsets)}")

        # Association rules dengan confidence threshold rendah agar lebih banyak aturan
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

        self.stdout.write(f"Jumlah aturan asosiasi yang dihasilkan: {len(rules)}")

        # Filter aturan yang mengandung label nilai GRADE_
        rules_with_grade = rules[rules['consequents'].apply(lambda x: any('GRADE_' in item for item in x))]

        if rules_with_grade.empty:
            self.stdout.write("Tidak ada aturan asosiasi yang mengandung label nilai (GRADE_).")
            return

        # Ambil 10 aturan teratas berdasarkan confidence
        top_rules = rules_with_grade[['antecedents', 'consequents', 'confidence', 'lift']]\
            .sort_values(by='confidence', ascending=False).head(10)

        # Format dan tampilkan output
        self.stdout.write("| Antecedents | Consequents | Confidence | Lift |")
        self.stdout.write("|-------------|-------------|------------|------|")
        for _, row in top_rules.iterrows():
            antecedents = ', '.join(row['antecedents'])
            consequents = ', '.join(row['consequents'])
            confidence = f"{row['confidence']:.2f}"
            lift = f"{row['lift']:.2f}"
            self.stdout.write(f"| {antecedents} | {consequents} | {confidence} | {lift} |")