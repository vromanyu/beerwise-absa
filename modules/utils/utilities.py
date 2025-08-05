from sqlalchemy import Engine, create_engine
from logging import Logger
import logging
import os

import pandas as pd

PRE_PROCESSED_PREFIX: str = "dataset_portion_pre_processed"
DATASET_LOCATION: str = "dataset/"
DATABASE_URL: str = "sqlite:///dataset.db"
DATASET_TABLE_NAME: str = "BEER_ADVOCATE"
DATABASE_LOCATION: str = "dataset.db"
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


def create_database_engine() -> Engine:
    return create_engine(DATABASE_URL, echo=True)


def dump_dataframe_to_sqlite(df: pd.DataFrame) -> None:
    LOGGER.info("loaded whole dataset")
    engine: Engine = create_database_engine()
    df.to_sql(name=DATASET_TABLE_NAME, con=engine, if_exists="replace")
    LOGGER.info(f"dataset was dump to sqlite({DATABASE_URL}/{DATASET_TABLE_NAME})")


def load_dataframe_from_database() -> pd.DataFrame:
    LOGGER.info(f"loading dataset from {DATABASE_LOCATION}")
    return pd.read_sql_table(DATASET_TABLE_NAME, DATABASE_URL)
