import ast
import concurrent
import json
import logging
import multiprocessing
import os
from typing import Optional

import pandas as pd
from filesplit.split import Split

from modules.processing.preprocessing import (
    handle_pre_processing,
)
from modules.utils.utilities import dump_dataframe_to_sqlite, load_preprocessed_dataset

DATASET = "dataset/beeradvocate.json"
PARSED_DATASET = "dataset/parsed_dataset.json"
OUTPUT_FILE_PREFIX = "dataset/dataset_portion_pre_processed_"
CHUNKS = "dataset/chunks"
LINES_PER_CHUNK = 500_000

LOG_FILE = "creator.log"
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

NUMBER_OF_CORES = multiprocessing.cpu_count()


def parse_json_dataset(file: str) -> None:
    counter = 0
    try:
        with open(file) as json_file:
            for line in json_file:
                counter += 1
                dataset_json: dict = ast.literal_eval(line)
                if not dataset_json:
                    LOGGER.warning(
                        f"parsed JSON at line: {counter} -- file: {json_file.name}  was empty"
                    )
                    continue
                with open(PARSED_DATASET, "a", encoding="utf-8") as parsed_dataset_json:
                    json.dump(dataset_json, parsed_dataset_json)
                    parsed_dataset_json.write("\n")
    except FileNotFoundError as e:
        print(e)


def create_preprocessed_excel_files_and_save_to_db() -> None:
    create_preprocessed_excel_files()
    df = load_preprocessed_dataset()
    dump_dataframe_to_sqlite(df)


def create_preprocessed_excel_files() -> None:
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=NUMBER_OF_CORES)
    split_dataset_to_chunks()
    chunk_files = os.listdir(CHUNKS)
    chunk_number = 1
    for chunk_file in chunk_files:
        pool.submit(process_chunk, chunk_file, chunk_number)
        chunk_number += 1
    pool.shutdown(wait=True)
    remove_chunks()
    LOGGER.info("finished parallel processing of all chunks")


def process_chunk(chunk_file: str, chunk_number: int) -> None:
    result = pd.DataFrame()
    with open(f"{CHUNKS}/{chunk_file}") as dataset:
        counter = 0
        for line in dataset:
            counter += 1
            processed_line: Optional[pd.DataFrame] = process_line(line, counter)
            if processed_line is not None:
                result = pd.concat([result, processed_line], ignore_index=True)
            else:
                LOGGER.warning(
                    f"line {counter} in chunk: {chunk_file} was not processed"
                )
        LOGGER.info(f"finished parallel processing chunk: {chunk_file}")
        result.reset_index(inplace=True, drop=True)
        export_dataframe_to_excel(f"{OUTPUT_FILE_PREFIX}_{chunk_number}.xlsx", result)
        LOGGER.info(f"finished exporting chunk: {chunk_file}")


def process_line(data: str, line: int) -> Optional[pd.DataFrame]:
    dataset_json: dict = ast.literal_eval(data)
    if not dataset_json:
        LOGGER.warning(f"line {line}: can't be processed")
        return None
    try:
        result = pd.DataFrame()
        data_json_transform = extract_keys_from_dataset(dataset_json)
        json_df = pd.DataFrame([data_json_transform])
        json_df["processed_text"] = json_df["text"].apply(handle_pre_processing)
        return pd.concat([result, json_df], ignore_index=True)
    except KeyError:
        LOGGER.error(f"error processing line: {line} and data: {data}")
        return None
    except ValueError:
        LOGGER.error(f"error converting line: {line} and data: {data}")
        return None


def extract_keys_from_dataset(dataset: dict):
    res = {
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
    split = Split(PARSED_DATASET, CHUNKS)
    split.bylinecount(LINES_PER_CHUNK)
    os.remove(f"{CHUNKS}/manifest")


def remove_chunks() -> None:
    chunks: list[str] = os.listdir(CHUNKS)
    for chunk in chunks:
        os.remove(f"{CHUNKS}/{chunk}")


def export_dataframe_to_excel(excel_file_name: str, df: pd.DataFrame) -> None:
    df.to_excel(excel_file_name, index=False, engine="openpyxl")


def clean_logs() -> None:
    with open(LOG_FILE, "w") as log_file:
        log_file.close()
