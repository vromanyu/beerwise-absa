import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from modules.utils.utilities import load_pre_processed_dataset

DATASET: str = "../../dataset/dataset_portion_pre_processed.xlsx"

def multinomial_nb() -> None:
    aspects = ["taste", "aroma", "appearance", "palate"]
    df: pd.DataFrame = load_pre_processed_dataset(DATASET)
    df.drop(columns=["appearance_similarity", "aroma_similarity", "palate_similarity", "taste_similarity"], inplace=True)
    for aspect in aspects:
        df[f"{aspect}_sentiment"] = df[aspect].apply(map_sentiment)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df["processed_text"].apply(lambda x: " ".join(x)))
    for aspect in aspects:
        X_train, X_test, y_train, y_test = train_test_split(X_text, df[f"{aspect}_sentiment"], test_size=0.5)
        mb = MultinomialNB()
        mb.fit(X_train, y_train)
        pred = mb.predict(X_test)
        print(f"Classification Report for {aspect} sentiment:")
        print(classification_report(y_test, pred))
        print("-" * 60)

def multi_multinomial_nb() -> None:
    aspects = ["taste", "aroma", "appearance", "palate"]
    df: pd.DataFrame = load_pre_processed_dataset(DATASET)
    df.drop(columns=["appearance_similarity", "aroma_similarity", "palate_similarity", "taste_similarity"], inplace=True)
    targets : list[str] = []
    for aspect in aspects:
        df[f"{aspect}_sentiment"] = df[aspect].apply(map_sentiment)
        targets.append(f"{aspect}_sentiment")
    X: pd.Series = df["processed_text"].apply(lambda x: " ".join(x))
    Y: pd.Series = df[targets]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2))),
        ("moc", MultiOutputClassifier(MultinomialNB()))
    ])
    pipeline.fit(X_train, Y_train)
    predictions = pipeline.predict(X_test)
    for i, target in enumerate(targets):
        print(f"classification report for {target}")
        print(classification_report(Y_test[target], predictions[:, i]))
        print("-"*60)

def map_sentiment(rating):
    if rating <= 3:
        return "negative"
    elif rating <= 4:
        return "neutral"
    elif rating <= 5:
        return "positive"
    else:
        return None

if __name__ == "__main__":
    multi_multinomial_nb()

