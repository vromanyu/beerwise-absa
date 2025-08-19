import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils import resample

from modules.utils.utilities import load_dataframe_from_database


def logistic_regression_trainer() -> None:
    df: pd.DataFrame = load_dataframe_from_database(is_sample=False)

    aspects = ["taste", "aroma", "palate", "appearance"]
    labels = [-1, 0, 1]

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["processed_text"].apply(lambda x: " ".join(x)))

    Y = df[[f"{aspect}_sentiment" for aspect in aspects]]

    for aspect in aspects:
        y = Y[f"{aspect}_sentiment"]
        mask = y != -2

        X_aspect = X[mask]
        y_aspect = y[mask]

        df_aspect = pd.DataFrame(X_aspect.toarray())
        df_aspect["label"] = y_aspect.values

        dfs = []
        for label in labels:
            df_label = df_aspect[df_aspect["label"] == label]
            if not df_label.empty:
                dfs.append(resample(
                    df_label,
                    replace=True,
                    n_samples=max(len(df_label), df_aspect["label"].value_counts().max()),
                    random_state=42
                ))
        df_balanced = pd.concat(dfs)
        X_balanced = df_balanced.drop("label", axis=1)
        y_balanced = df_balanced["label"]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_balanced, y_balanced)

        # Evaluate on training data (optional)
        y_pred = model.predict(X_balanced)
        print(f"\nAspect: {aspect}")
        print(classification_report(y_balanced, y_pred, zero_division=0))





