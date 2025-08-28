import joblib

import ast

from sqlalchemy import Engine, create_engine
import logging
import os

import pandas as pd

from modules.processing.preprocessing import handle_pre_processing
import torch
from transformers import AutoTokenizer
from modules.algorithms.transformer_based import MultiAspectModel

PRE_PROCESSED_PREFIX = "dataset_portion_pre_processed"
DATASET_LOCATION = "dataset/"
DATABASE_URL = "sqlite:///dataset.db"
TARGET_DATABASE_URL = "sqlite:///target_dataset.db"
DATASET_TABLE_NAME = "BEER_ADVOCATE"
TARGET_DATASET_TABLE_NAME = "TARGET_BEER_ADVOCATE"
DATABASE_LOCATION = "dataset.db"
TARGET_DATABASE_LOCATION = "target_dataset.db"
ASPECTS_FILE_LOCATION = "most_common_aspects.txt"
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_preprocessed_dataset() -> pd.DataFrame:
    aggregated_df = pd.DataFrame()
    datasets = os.listdir(DATASET_LOCATION)
    for dataset in datasets:
        if dataset.startswith(PRE_PROCESSED_PREFIX):
            LOGGER.info(f"loading dataset: {dataset}")
            df = pd.read_excel(f"dataset/{dataset}")
            aggregated_df = pd.concat([aggregated_df, df])

    aggregated_df["processed_text"] = aggregated_df["processed_text"].apply(
        ast.literal_eval
    )
    aggregated_df = aggregated_df[
        aggregated_df["processed_text"].apply(lambda x: len(x) > 0)
    ]
    return aggregated_df


def create_database_engine(is_target: bool = False) -> Engine:
    if is_target:
        return create_engine(TARGET_DATABASE_URL, echo=True)
    else:
        return create_engine(DATABASE_URL, echo=True)


def dump_dataframe_to_sqlite(df: pd.DataFrame, is_target: bool = False) -> None:
    df["processed_text"] = df["processed_text"].apply(str)
    if is_target:
        engine = create_database_engine(is_target=True)
        df.to_sql(
            name=TARGET_DATASET_TABLE_NAME, con=engine, if_exists="replace", index=False
        )
        LOGGER.info(
            f"dataset was dump to sqlite({TARGET_DATABASE_LOCATION}/{TARGET_DATASET_TABLE_NAME})"
        )
    else:
        engine = create_database_engine(is_target=False)
        df.to_sql(name=DATASET_TABLE_NAME, con=engine, if_exists="replace", index=False)
        LOGGER.info(
            f"dataset was dump to sqlite({DATABASE_LOCATION}/{DATASET_TABLE_NAME})"
        )


def load_dataframe_from_database(is_target: bool = False) -> pd.DataFrame:
    if is_target:
        LOGGER.info(f"loading target dataset from {TARGET_DATABASE_LOCATION}")
        df = pd.read_sql_table(TARGET_DATASET_TABLE_NAME, TARGET_DATABASE_URL)
        df["processed_text"] = df["processed_text"].apply(ast.literal_eval)
        return df
    else:
        LOGGER.info(f"loading dataset from {DATABASE_LOCATION}")
        df = pd.read_sql_table(DATASET_TABLE_NAME, DATABASE_URL)
        df["processed_text"] = df["processed_text"].apply(ast.literal_eval)
        return df


def save_most_common_aspects(aspects: list[str]) -> None:
    with open(ASPECTS_FILE_LOCATION, "w", encoding="utf-8") as f:
        for aspect in aspects:
            f.write(f"{aspect}\n")


def read_most_common_aspects() -> list[str]:
    with open(ASPECTS_FILE_LOCATION, "r", encoding="utf-8") as f:
        aspects = [line.strip() for line in f]
    return aspects


def predict_sentiments_using_logistic_regression(user_input: str):
    vectorizer = joblib.load(
        "./models/logistic_regression/logistic_regression_vectorizer.pkl"
    )
    model = joblib.load(
        "./models/logistic_regression/multioutput_logistic_regression_model.pkl"
    )

    pre_processed_input = handle_pre_processing(user_input, lemmatize=False)
    X = vectorizer.transform([" ".join(pre_processed_input)])
    preds = model.predict(X)

    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    appearance_sentiment = sentiment_map[preds[0][0]]
    palate_sentiment = sentiment_map[preds[0][1]]
    print(f"Appearance: {appearance_sentiment}, Palate: {palate_sentiment}")


def predict_sentiments_using_linear_svc(user_input: str):
    vectorizer = joblib.load("./models/linear_svc/linear_svc_vectorizer.pkl")
    model = joblib.load("./models/linear_svc/multioutput_linear_svc_model.pkl")

    pre_processed_input = handle_pre_processing(user_input, lemmatize=False)
    X = vectorizer.transform([" ".join(pre_processed_input)])
    preds = model.predict(X)

    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    appearance_sentiment = sentiment_map[preds[0][0]]
    palate_sentiment = sentiment_map[preds[0][1]]
    print(f"Appearance: {appearance_sentiment}, Palate: {palate_sentiment}")


def predict_sentiments_using_naive_bayes(user_input: str):
    vectorizer = joblib.load("./models/naive_bayes/naive_bayes_vectorizer.pkl")
    model = joblib.load("./models/naive_bayes/multioutput_naive_bayes_model.pkl")

    pre_processed_input = handle_pre_processing(user_input, lemmatize=False)
    X = vectorizer.transform([" ".join(pre_processed_input)])
    preds = model.predict(X)

    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    appearance_sentiment = sentiment_map[preds[0][0]]
    palate_sentiment = sentiment_map[preds[0][1]]
    print(f"Appearance: {appearance_sentiment}, Palate: {palate_sentiment}")


def predict_sentiments_using_ridge_classifier(user_input: str):
    vectorizer = joblib.load("./models/ridge_classifier/ridge_classifier_vectorizer.pkl")
    model = joblib.load("./models/ridge_classifier/multioutput_ridge_classifier_model.pkl")

    pre_processed_input = handle_pre_processing(user_input, lemmatize=False)
    X = vectorizer.transform([" ".join(pre_processed_input)])
    preds = model.predict(X)

    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    appearance_sentiment = sentiment_map[preds[0][0]]
    palate_sentiment = sentiment_map[preds[0][1]]
    print(f"Appearance: {appearance_sentiment}, Palate: {palate_sentiment}")


def predict_sentiments_using_bert_mini(user_input: str):
    model_name = "prajjwal1/bert-mini"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_path = "./models/transformer/prajjwal1_bert-mini/prajjwal1_bert-mini.pt"

    model = MultiAspectModel(model_name)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()

    pre_processed_input = handle_pre_processing(user_input, lemmatize=False)
    text = " ".join(pre_processed_input)
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    with torch.no_grad():
        app_logits, pal_logits = model(
            encoding["input_ids"], encoding["attention_mask"]
        )
        app_pred = torch.argmax(app_logits, dim=1).item()
        pal_pred = torch.argmax(pal_logits, dim=1).item()
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    appearance_sentiment = sentiment_map[app_pred]
    palate_sentiment = sentiment_map[pal_pred]
    print(f"Appearance: {appearance_sentiment}, Palate: {palate_sentiment}")
