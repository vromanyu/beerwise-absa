import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.style as style
import seaborn as sns
import pandas as pd

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

    plt.title("Συνολική μέση F1 ανά μοντέλο", fontsize=14)
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

    # Δεδομένα
    models = ['Multinomial NB', "Logistic Regression", "Linear SVC", 'Ridge Classifier', 'BERT-mini', 'DistilBERT']
    appearance_scores = [0.7941, 0.8878, 0.8878, 0.8848, 0.8848, 0.9219]
    palate_scores = [0.7869, 0.8932, 0.8981, 0.8760, 0.8683, 0.9048]

    # Ετικέτες άξονα Χ
    aspects = ['Εμφάνιση', 'Γευστικό προφίλ']

    # Δημιουργία γραφήματος
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        plt.plot(aspects, [appearance_scores[i], palate_scores[i]], marker='o', label=model)

    # Τίτλος και άξονες
    plt.title('Μεταβολή F1-score ανά μοντέλο και όψη', fontsize=14)
    plt.xlabel('Όψη', fontsize=12)
    plt.ylabel('F1-score', fontsize=12)
    plt.ylim(0.75, 0.93)
    plt.legend(title='Μοντέλο', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

models_line_chart()