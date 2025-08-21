import logging
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from modules.utils.utilities import load_dataframe_from_database

LOGGER: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
MODELS_DIRECTORY: str = "./models/logistic_regression"


def logistic_regression_trainer(is_sample: bool = False) -> None:
    df = load_dataframe_from_database(is_sample)

    # Suggestion 1: Enhanced TF-IDF settings
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),          # Include trigrams
        sublinear_tf=True,           # Dampens overly frequent terms
        min_df=5,                    # Ignore rare terms
        max_df=0.8                   # Ignore overly common terms
    )

    for aspect in ['appearance', 'aroma', 'palate', 'taste']:
        train_logistic_model(df, aspect, vectorizer)

    vectorizer_path = f"{MODELS_DIRECTORY}/{'sample' if is_sample else 'full'}/logistic_regression_vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)



def train_logistic_model(df: pd.DataFrame, aspect: str, vectorizer: TfidfVectorizer, is_sample=False) -> None:
    df_aspect = df[df[f"{aspect}_sentiment"] != -2]

    max_size = df_aspect[f"{aspect}_sentiment"].value_counts().max()
    df_balanced = pd.concat([
        resample(df_aspect[df_aspect[f"{aspect}_sentiment"] == label],
                 replace=True, n_samples=max_size, random_state=42)
        for label in df_aspect[f"{aspect}_sentiment"].unique()
    ])

    X = vectorizer.fit_transform(df_balanced['processed_text'].str.join(" "))
    y = df_balanced[f"{aspect}_sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Suggestion 3: Tuned Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=0.5,                      # Stronger regularization
        solver="saga"               # Better for sparse data
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\nAspect: {aspect} — Upsampled")
    print(classification_report(y_test, y_pred))

    model_path = f"{MODELS_DIRECTORY}/{'sample' if is_sample else 'full'}/{aspect}_logistic_regression_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

