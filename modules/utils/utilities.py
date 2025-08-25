import ast

from sqlalchemy import Engine, create_engine
from logging import Logger
import logging
import os

import pandas as pd

PRE_PROCESSED_PREFIX: str = "dataset_portion_pre_processed"
DATASET_LOCATION: str = "dataset/"
DATABASE_URL: str = "sqlite:///dataset.db"
TARGET_DATABASE_URL: str = "sqlite:///target_dataset.db"
DATASET_TABLE_NAME: str = "BEER_ADVOCATE"
TARGET_DATASET_TABLE_NAME: str = "TARGET_BEER_ADVOCATE"
DATABASE_LOCATION: str = "dataset.db"
TARGET_DATABASE_LOCATION: str = "target_dataset.db"
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

    aggregated_df["processed_text"] = aggregated_df["processed_text"].apply(ast.literal_eval)
    aggregated_df = aggregated_df[aggregated_df["processed_text"].apply(lambda x: len(x) > 0)]
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
        df.to_sql(name=TARGET_DATASET_TABLE_NAME, con=engine, if_exists="replace", index=False)
        LOGGER.info(f"dataset was dump to sqlite({TARGET_DATABASE_LOCATION}/{TARGET_DATASET_TABLE_NAME})")
    else:
        engine = create_database_engine(is_target=False)
        df.to_sql(name=DATASET_TABLE_NAME, con=engine, if_exists="replace", index=False)
        LOGGER.info(f"dataset was dump to sqlite({DATABASE_LOCATION}/{DATASET_TABLE_NAME})")


def load_dataframe_from_database(is_target: bool = False) -> pd.DataFrame:
    if is_target:
        LOGGER.info(f"loading target dataset from {TARGET_DATABASE_LOCATION}")
        df: pd.DataFrame = pd.read_sql_table(TARGET_DATASET_TABLE_NAME, TARGET_DATABASE_URL)
        df["processed_text"] = df["processed_text"].apply(ast.literal_eval)
        return df
    else:
        LOGGER.info(f"loading dataset from {DATABASE_LOCATION}")
        df: pd.DataFrame = pd.read_sql_table(DATASET_TABLE_NAME, DATABASE_URL)
        df["processed_text"] = df["processed_text"].apply(ast.literal_eval)
        return df
