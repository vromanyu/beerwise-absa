import logging
import os

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import dump
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from modules.utils.utilities import load_dataframe_from_database

LOGGER: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
MODELS_DIRECTORY: str = "./models/logistic_regression"


def logistic_regression_trainer(is_sample: bool = False) -> None:
    df: pd.DataFrame = load_dataframe_from_database(is_sample)
    aspect_cols = ['appearance_sentiment', 'aroma_sentiment', 'palate_sentiment', 'taste_sentiment']

    df_filtered = df[
        (df['appearance_sentiment'] != -2) |
        (df['aroma_sentiment'] != -2) |
        (df['palate_sentiment'] != -2) |
        (df['taste_sentiment'] != -2)
        ].copy()

    df_negative = df_filtered[df_filtered[aspect_cols].apply(lambda row: -1 in row.values, axis=1)]
    df_negative_upsampled = df_negative.sample(n=len(df_filtered), replace=True, random_state=42)

    df_balanced = pd.concat([df_filtered, df_negative_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    y = df_balanced[aspect_cols].replace(-2, np.nan)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df_balanced['processed_text'].apply(lambda x: " ".join(x)))

    models = {}
    reports = {}

    for aspect in aspect_cols:
        # Drop rows where this aspect is NaN
        valid_idx = ~y[aspect].isna()
        X_aspect = X[valid_idx]
        y_aspect = y.loc[valid_idx, aspect]

        X_train, X_test, y_train, y_test = train_test_split(X_aspect, y_aspect, test_size=0.2, random_state=42)

        clf = LogisticRegression(class_weight='balanced', max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        models[aspect] = clf
        reports[aspect] = classification_report(y_test, y_pred, output_dict=False)
        print(f"\nAspect: {aspect}")
        print(classification_report(y_test, y_pred))

    os.makedirs(f"{MODELS_DIRECTORY}/sample", exist_ok=True)
    os.makedirs(f"{MODELS_DIRECTORY}/full", exist_ok=True)
    for aspect, model in models.items():
        if is_sample:
            joblib.dump(model, f"{MODELS_DIRECTORY}/sample/{aspect.split("_")[0]}_logistic_regression_model.pkl")
        else:
            joblib.dump(model, f"{MODELS_DIRECTORY}/full/{aspect.split("_")[0]}_logistic_regression_model.pkl")
    if is_sample:
        joblib.dump(vectorizer, f"{MODELS_DIRECTORY}/sample/logistic_regression_vectorizer.pkl")
    else:
        joblib.dump(vectorizer, f"{MODELS_DIRECTORY}/full/logistic_regression_vectorizer.pkl")
