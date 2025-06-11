import ast
import asyncio
import logging
import os
from logging import Logger
import sys

import pandas as pd
from filesplit.split import Split

from modules.processing.processor import handle_pre_processing

DATASET: str = "dataset/beeradvocate.json"
OUTPUT: str = "dataset/dataset_portion_pre_processed.xlsx"
CHUNKS: str = "dataset/chunks"
LINES_PER_CHUNK: int = 500_000
LOGGER: Logger = logging.getLogger(__name__)


async def create_processed_dataframe(file: str, limit: int = 0) -> pd.DataFrame:
    result = pd.DataFrame()
    counter: int = 0
    with open(f"{CHUNKS}/{file}") as f:
        for line in f:
            counter += 1
            if limit == counter:
                break
            LOGGER.info(f"reading line: {counter} --- {line[0:150]}...")
            dataset_json = ast.literal_eval(line)
            if not dataset_json:
                LOGGER.warning(f"parsed JSON at line {counter} was empty")
                continue
            try:
                LOGGER.info(f"processing line: {counter} -- dataset: {dataset_json}")
                data_json_transform = extract_keys_from_dataset(dataset_json)
                json_df = pd.DataFrame([data_json_transform])
                json_df["processed_text"] = json_df["text"].apply(handle_pre_processing)
                result = pd.concat([result, json_df], ignore_index=True)
            except KeyError:
                LOGGER.error(
                    f"error processing line: {counter} --- dataset: {dataset_json}"
                )
                continue
            except ValueError:
                LOGGER.error(
                    f"error converting at line: {counter} ---- dataset: {dataset_json}"
                )
                continue
    return result


def extract_keys_from_dataset(dataset: dict) -> dict:
    res: dict = {
        "beer_id": str(dataset["beer/beerId"]),
        "name": str(dataset["beer/name"]),
        "appearance": float(dataset["review/appearance"]),
        "aroma": float(dataset["review/aroma"]),
        "palate": float(dataset["review/palate"]),
        "taste": float(dataset["review/taste"]),
        "overall": float(dataset["review/overall"]),
        "text": str(dataset["review/text"]),
        "processed_text": [],
    }
    return res


def split_dataset_to_chunks() -> None:
    try:
        os.makedirs(CHUNKS)
    except FileExistsError:
        LOGGER.warning("chunks folder already exists")
    split = Split(DATASET, CHUNKS)
    split.bylinecount(LINES_PER_CHUNK)
    os.remove(f"{CHUNKS}/manifest")


def remove_chunks() -> None:
    chunks: list[str] = os.listdir(CHUNKS)
    for chunk in chunks:
        os.remove(f"{CHUNKS}/{chunk}")


def export_dataframe_to_excel(excel_file_name: str, df: pd.DataFrame) -> None:
    df.to_excel(excel_file_name, index=False, engine="openpyxl")


async def async_dataframe_creator(limit: int = 0):
    split_dataset_to_chunks()
    chunk_files = os.listdir(CHUNKS)
    tasks: list = []
    for chunk_file in chunk_files:
        tasks.append(asyncio.create_task(create_processed_dataframe(chunk_file, limit)))
    await asyncio.gather(*tasks, return_exceptions=True)
    return tasks


def generate_processed_dataframe(limit: int = 0):
    res = asyncio.run(async_dataframe_creator(limit))
    remove_chunks()
    return [res.result() for res in res]


def main(limit: int = 0):
    df: pd.DataFrame = pd.DataFrame()
    dataframes: list = generate_processed_dataframe(limit)
    for dataframe in dataframes:
        df = pd.concat([df, dataframe], ignore_index=True)
    df.reset_index(inplace=True, drop=True)
    export_dataframe_to_excel(OUTPUT, df)


if __name__ == "__main__":
    limit: int = 0
    try:
        limit = int(input("enter limit: "))
    except ValueError:
        limit = 0
    except EOFError:
        sys.exit()
    main()
