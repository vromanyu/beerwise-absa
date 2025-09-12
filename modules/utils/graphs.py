import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.style as style
import seaborn as sns
import pandas as pd
import numpy as np

def plot_models_f1_average():
    models = [
        "Multinomial Naive Bayes",
        "Logistic Regression",
        "Linear SVC",
        "Ridge Classifier",
        "BERT-mini",
        "DistilBERT"]

    f1_scores = [0.7905, 0.8905, 0.8929, 0.8804, 0.8766, 0.9134 ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, f1_scores, color='skyblue', edgecolor='black')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, f"{yval:.2f}")

    plt.title("Mέση F1 ανά μοντέλο", fontsize=14)
    plt.ylabel("Μέση F1", fontsize=12)
    plt.ylim(0.6, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def generate_heatmap():
    data = {
        "Multinomial Naive Bayes": [0.7941, 0.7869],
                "Logistic Regression": [0.8878, 0.8932],
                "Linear SVC": [0.8878, 0.8981],
                "Ridge Classifier": [0.8848, 0.8760],
                "BERT-mini": [0.8848, 0.8683],
                "DistilBERT": [0.9219, 0.9048]
        }

    aspects = ["Εμφάνιση", "Γευστικό προφίλ"]

    df = pd.DataFrame(data, index=aspects)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="YlGnBu", cbar_kws={'label': 'F1-Score'})

    plt.title("Απόδοση μοντέλων ανά όψη (F1-Score)", fontsize=14)
    plt.xlabel("Μοντέλα", fontsize=12)
    plt.ylabel("Όψεις", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def models_line_chart():
        # Εφαρμογή αισθητικού στυλ
    style.use('seaborn-v0_8')

    models = ['Multinomial NB', "Logistic Regression", "Linear SVC", 'Ridge Classifier', 'BERT-mini', 'DistilBERT']
    appearance_scores = [0.7941, 0.8878, 0.8878, 0.8848, 0.8848, 0.9219]
    palate_scores = [0.7869, 0.8932, 0.8981, 0.8760, 0.8683, 0.9048]

    aspects = ['Εμφάνιση', 'Γευστικό προφίλ']

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        plt.plot(aspects, [appearance_scores[i], palate_scores[i]], marker='o', label=model)

    plt.title('Μεταβολή F1-score ανά μοντέλο και όψη', fontsize=14)
    plt.xlabel('Όψη', fontsize=12)
    plt.ylabel('F1-score', fontsize=12)
    plt.ylim(0.75, 0.93)
    plt.legend(title='Μοντέλο', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def model_performance_based_on_time():

    models = [
        'Multinomial NB',
        'Logistic Regression',
        'Linear SVC',
        'Ridge Classifier',
        'BERT-mini',
        'DistilBERT'
    ]

    mean_f1 = [0.7905, 0.8905, 0.8929, 0.8804, 0.8766, 0.9134]
    training_time = [10, 10, 120, 20, 240, 400]

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(training_time, mean_f1, c=mean_f1, cmap='viridis', s=120, edgecolors='black')

    for i in range(len(models)):
        plt.text(training_time[i] + 2, mean_f1[i], models[i], fontsize=9)

    plt.title("Σχέση χρόνου εκπαίδευσης και μέσης F1", fontsize=14)
    plt.xlabel("Χρόνος εκπαίδευσης (λεπτά)", fontsize=12)
    plt.ylabel("Μέση F1", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Μέση F1-score')

    plt.tight_layout()
    plt.show()

def grouped_statistics_per_aspect_and_category():
    plt.style.use('seaborn-v0_8')

    categories = ['Negative', 'Neutral', 'Positive']
    models = ['Multinomial Naive Bayes', 'Logistic Regression', 'Linear SVC', 'Ridge Classifier', 'BERT-mini', 'DistilBERT']

    appearance_scores = {
        'Multinomial Naive Bayes': [0.91, 0.72, 0.76],
        'Logistic Regression': [1, 0.83, 0.84],
        'Linear SVC': [1, 0.84, 0.84],
        'Ridge Classifier': [0.99, 0.83, 0.83],
        'BERT-mini': [0.99, 0.82, 0.82],
        'DistilBERT': [1, 0.88, 0.88]
        }
    palate_scores = {
        'Multinomial Naive Bayes': [0.92, 0.74, 0.71],
        'Logistic Regression': [0.98, 0.84, 0.85],
        'Linear SVC': [0.99, 0.85, 0.85],
        'Ridge Classifier': [0.98, 0.83, 0.85],
        'BERT-mini': [0.98, 0.79, 0.83],
        'DistilBERT': [0.99, 0.86, 0.87]
        }
    def plot_grouped_bar(data, title):
        x = np.arange(len(categories))
        width = 0.13
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, model in enumerate(models):
            scores = data[model]
            ax.bar(x + i * width, scores, width, label=model)

        ax.set_xlabel('Κατηγορία Συναισθήματος')
        ax.set_ylabel('F1-score')
        ax.set_title(title)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(categories)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()
    plot_grouped_bar(appearance_scores, 'Απόδοση μοντέλων στην όψη \"εμφάνιση\" (F1-score)')
    plot_grouped_bar(palate_scores, 'Απόδοση μοντέλων στην όψη \"γευστικό προφίλ\" (F1-score)')

plot_models_f1_average()
