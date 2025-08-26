import joblib


import ast

from sqlalchemy import Engine, create_engine
from logging import Logger
import logging
import os

import pandas as pd

from modules.processing.preprocessing import handle_pre_processing

PRE_PROCESSED_PREFIX = "dataset_portion_pre_processed"
DATASET_LOCATION = "dataset/"
DATABASE_URL = "sqlite:///dataset.db"
TARGET_DATABASE_URL = "sqlite:///target_dataset.db"
DATASET_TABLE_NAME = "BEER_ADVOCATE"
TARGET_DATASET_TABLE_NAME = "TARGET_BEER_ADVOCATE"
DATABASE_LOCATION = "dataset.db"
TARGET_DATABASE_LOCATION = "target_dataset.db"
ASPECTS_FILE_LOCATION = "most_common_aspects.txt"
LOGGER: Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_pre_processed_dataset() -> pd.DataFrame:
    aggregated_df: pd.DataFrame = pd.DataFrame()
    datasets: list[str] = os.listdir(DATASET_LOCATION)
    for dataset in datasets:
        if dataset.startswith(PRE_PROCESSED_PREFIX):
            LOGGER.info(f"loading dataset: {dataset}")
            df: pd.DataFrame = pd.read_excel(f"dataset/{dataset}")
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
        engine: Engine = create_database_engine(is_target=True)
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
        df: pd.DataFrame = pd.read_sql_table(
            TARGET_DATASET_TABLE_NAME, TARGET_DATABASE_URL
        )
        df["processed_text"] = df["processed_text"].apply(ast.literal_eval)
        return df
    else:
        LOGGER.info(f"loading dataset from {DATABASE_LOCATION}")
        df: pd.DataFrame = pd.read_sql_table(DATASET_TABLE_NAME, DATABASE_URL)
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


def predict_sentiments_using_logistic_regression(input: str):
    vectorizer = joblib.load(
        "./models/logistic_regression/logistic_regression_vectorizer.pkl"
    )
    model = joblib.load(
        "./models/logistic_regression/multioutput_logistic_regression_model.pkl"
    )

    pre_processed_input = handle_pre_processing(input, lemmatize=False)
    X = vectorizer.transform([" ".join(pre_processed_input)])
    preds = model.predict(X)

    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    appearance_sentiment = sentiment_map[preds[0][0]]
    palate_sentiment = sentiment_map[preds[0][1]]
    print(f"Appearance: {appearance_sentiment}, Palate: {palate_sentiment}")
