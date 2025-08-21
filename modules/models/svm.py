import logging
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils import resample

from modules.utils.utilities import load_dataframe_from_database

LOGGER: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
MODELS_DIRECTORY: str = "./models/svm"


def svm_trainer(is_sample: bool = False) -> None:
    df = load_dataframe_from_database(is_sample)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    for aspect in ['appearance', 'aroma', 'palate', 'taste']:
        train_svm_model(df, aspect, vectorizer)
    if is_sample:
        joblib.dump(vectorizer, f"{MODELS_DIRECTORY}/sample/svm_vectorizer.pkl")
    else:
        joblib.dump(vectorizer, f"{MODELS_DIRECTORY}/full/svm_vectorizer.pkl")


def train_svm_model(df: pd.DataFrame, aspect: str, vectorizer: TfidfVectorizer, is_sample=False) -> None:
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

    model = LinearSVC(class_weight="balanced", max_iter=10000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\nAspect: {aspect} — Upsampled")
    print(classification_report(y_test, y_pred))

    if is_sample:
        os.makedirs(f"{MODELS_DIRECTORY}/sample", exist_ok=True)
        joblib.dump(model, f"{MODELS_DIRECTORY}/sample/{aspect}_svm_model.pkl")
    else:
        os.makedirs(f"{MODELS_DIRECTORY}/full", exist_ok=True)
        joblib.dump(model, f"{MODELS_DIRECTORY}/full/{aspect}_svm_model.pkl")
