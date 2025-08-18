from sqlalchemy import Engine, create_engine
from logging import Logger
import logging
import os

import pandas as pd

PRE_PROCESSED_PREFIX: str = "dataset_portion_pre_processed"
DATASET_LOCATION: str = "dataset/"
DATABASE_URL: str = "sqlite:///dataset.db"
SAMPLE_DATABASE_URL: str = "sqlite:///sample_dataset.db"
DATASET_TABLE_NAME: str = "BEER_ADVOCATE"
SAMPLE_DATASET_TABLE_NAME: str = "SAMPLE_BEER_ADVOCATE"
DATABASE_LOCATION: str = "dataset.db"
SAMPLE_DATABASE_LOCATION: str = "sample_dataset.db"
LOGGER: Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_pre_processed_dataset() -> pd.DataFrame:
    aggregated_df: pd.DataFrame = pd.DataFrame()
    # df: pd.DataFrame = pd.read_excel(dataset)
    datasets: list[str] = os.listdir(DATASET_LOCATION)
    for dataset in datasets:
        if dataset.startswith(PRE_PROCESSED_PREFIX):
            LOGGER.info(f"loading dataset: {dataset}")
            df: pd.DataFrame = pd.read_excel(f"dataset/{dataset}")
            aggregated_df = pd.concat([aggregated_df, df])

    return aggregated_df


def create_database_engine(is_sample: bool = False) -> Engine:
    if is_sample:
        return create_engine(SAMPLE_DATABASE_URL, echo=True)
    else:
        return create_engine(DATABASE_URL, echo=True)


def dump_dataframe_to_sqlite(df: pd.DataFrame, is_sample: bool = False) -> None:
    if is_sample:
        LOGGER.info("loaded sample dataset")
        engine: Engine = create_database_engine(is_sample=True)
        df.to_sql(name=SAMPLE_DATASET_TABLE_NAME, con=engine, if_exists="replace", index=False)
        LOGGER.info(f"dataset was dump to sqlite({SAMPLE_DATABASE_LOCATION}/{SAMPLE_DATASET_TABLE_NAME})")
    else:
        LOGGER.info("loaded whole dataset")
        engine = create_database_engine(is_sample=False)
        df.to_sql(name=DATASET_TABLE_NAME, con=engine, if_exists="replace", index=False)
        LOGGER.info(f"dataset was dump to sqlite({DATABASE_LOCATION}/{DATASET_TABLE_NAME})")


def load_dataframe_from_database(is_sample: bool = False) -> pd.DataFrame:
    if is_sample:
        LOGGER.info(f"loading sample dataset from {SAMPLE_DATABASE_LOCATION}")
        return pd.read_sql_table(SAMPLE_DATASET_TABLE_NAME, SAMPLE_DATABASE_URL)
    else:
        LOGGER.info(f"loading dataset from {DATABASE_LOCATION}")
        return pd.read_sql_table(DATASET_TABLE_NAME, DATABASE_URL)


def create_sample_dataset(sample_length: int) -> None:
    df: pd.DataFrame = load_dataframe_from_database()
    dump_dataframe_to_sqlite(df[:sample_length], is_sample=True)
