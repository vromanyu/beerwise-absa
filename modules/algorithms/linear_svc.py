import logging
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
from modules.utils.utilities import load_dataframe_from_database

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
MODELS_DIRECTORY = "./models/linear_svc"


def upsample_to_match(df, label_col):
    grouped = df.groupby(label_col)
    target_size = grouped.size().max()
    balanced_groups = [
        resample(group, replace=True, n_samples=target_size, random_state=42)
        for _, group in grouped
    ]
    return (
        pd.concat(balanced_groups)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


def prepare_data(df):
    sentiment_map = {-1: 0, 0: 1, 1: 2}
    df["appearance_label"] = df["appearance_sentiment"].map(sentiment_map)
    df["palate_label"] = df["palate_sentiment"].map(sentiment_map)

    df["appearance_label"] = df["appearance_label"].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
    df["palate_label"] = df["palate_label"].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
    df["processed_text"] = df["processed_text"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else str(x)
    )

    df = df.dropna(subset=["appearance_label", "palate_label"])

    df["joint_label"] = (
        df["appearance_label"].astype(str) + "_" + df["palate_label"].astype(str)
    )
    df = upsample_to_match(df, "joint_label")

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_df = None
    test_df = None
    for train_idx, test_idx in strat_split.split(df, df["joint_label"]):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

    return train_df, test_df


def linear_svc_trainer() -> None:
    df = load_dataframe_from_database(is_target=True)
    train_df, test_df = prepare_data(df)

    vectorizer = TfidfVectorizer(
        max_features=120_000, ngram_range=(1, 2), sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(train_df["processed_text"])
    X_test = vectorizer.transform(test_df["processed_text"])

    y_train = train_df[["appearance_label", "palate_label"]]
    y_test = test_df[["appearance_label", "palate_label"]]

    base_model = LinearSVC(C=1, class_weight="balanced", max_iter=1000, random_state=42)
    model = MultiOutputClassifier(base_model)

    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    print("\nTest Classification Report (Appearance):")
    report_app = classification_report(
        y_test["appearance_label"], y_test_pred[:, 0], output_dict=True
    )
    print(classification_report(y_test["appearance_label"], y_test_pred[:, 0]))
    print(f"Macro avg F1-score (Appearance): {report_app['macro avg']['f1-score']:.4f}")

    print("\nTest Classification Report (Palate):")
    report_pal = classification_report(
        y_test["palate_label"], y_test_pred[:, 1], output_dict=True
    )
    print(classification_report(y_test["palate_label"], y_test_pred[:, 1]))
    print(f"Macro avg F1-score (Palate): {report_pal['macro avg']['f1-score']:.4f}")

    os.makedirs(MODELS_DIRECTORY, exist_ok=True)
    model_path = f"{MODELS_DIRECTORY}/multioutput_linear_svc_model.pkl"
    joblib.dump(model, model_path)
    vectorizer_path = f"{MODELS_DIRECTORY}/linear_svc_vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)
