import ast
import asyncio
import logging
import os
from logging import Logger

import pandas as pd
from filesplit.split import Split

from modules.processing.processor import handle_pre_processing, handle_aspect_extraction

DATASET: str = "./dataset/beeradvocate.json"
DATASET_EXCEL_WITH_ALL_ROWS: str = "./dataset/dataset_as_excel_all_rows.xlsx"
DATASET_EXCEL_WITH_MANDATORY_ROWS: str = "./dataset/dataset_as_excel_mandatory_rows.xlsx"
LINES_PER_CHUNK: int = 1_000_000
LOGGER: Logger = logging.getLogger(__name__)


async def create_processed_dataframe_with_all_rows(file: str) -> pd.DataFrame:
    result = pd.DataFrame()
    counter: int = 0
    with open(f"./dataset/chunks/{file}") as f:
        for line in f:
            counter += 1
            if counter == 5:
                break
            LOGGER.info(f"reading line: {counter} --- {line[0:150]}...")
            dataset_json = ast.literal_eval(line)
            if not dataset_json:
                LOGGER.warning(f"parsed JSON at line {counter} was empty")
                continue
            try:
                data_json_transform = extract_all_keys_from_dataset(dataset_json)
                json_df = pd.DataFrame([data_json_transform])
                json_df["processed_text"] = json_df["text"].apply(
                    handle_pre_processing)
                json_df["extracted_aspects"] = json_df["processed_text"].apply(handle_aspect_extraction)
                result = pd.concat([result, json_df], ignore_index=True)
            except KeyError:
                LOGGER.error(
                    f"error processing line: {counter} --- dataset: {dataset_json}")
                continue
            except ValueError:
                LOGGER.error(
                    f"error converting at line: {counter} ---- dataset: {dataset_json}")
                continue
    return result


async def create_processed_dataframe_with_mandatory_rows(file: str) -> pd.DataFrame:
    result = pd.DataFrame()
    counter: int = 0
    with open(f"./dataset/chunks/{file}") as f:
        for line in f:
            counter += 1
            if counter == 5:
                break
            LOGGER.info(f"reading line: {counter} --- {line[0:150]}...")
            dataset_json = ast.literal_eval(line)
            if not dataset_json:
                LOGGER.warning(f"parsed JSON at line {counter} was empty")
                continue
            try:
                data_json_transform = extract_mandatory_keys_from_dataset(dataset_json)
                json_df = pd.DataFrame([data_json_transform])
                json_df["processed_text"] = json_df["text"].apply(
                    handle_pre_processing)
                json_df["extracted_aspects"] = json_df["processed_text"].apply(handle_aspect_extraction)
                result = pd.concat([result, json_df], ignore_index=True)
            except KeyError:
                LOGGER.error(
                    f"error processing line: {counter} --- dataset: {dataset_json}")
                continue
            except ValueError:
                LOGGER.error(
                    f"error converting at line: {counter} ---- dataset: {dataset_json}")
                continue
    return result


def extract_all_keys_from_dataset(dataset: dict) -> dict:
    res: dict = {"beer_id": str(dataset["beer/beerId"]), "name": str(dataset["beer/name"]),
                 "brewer_id": str(dataset["beer/brewerId"]),
                 "abv": float(dataset["beer/ABV"]),
                 "style": str(dataset["beer/style"]), "appearance": float(dataset["review/appearance"]),
                 "aroma": float(dataset["review/aroma"]), "palate": float(dataset["review/palate"]),
                 "taste": float(dataset["review/taste"]),
                 "overall": float(dataset["review/overall"]), "text": str(dataset["review/text"]),
                 "time": int(dataset["review/time"]),
                 "profile_name": str(dataset["review/profileName"]), "processed_text": [], "extracted_aspects": []}
    return res


def extract_mandatory_keys_from_dataset(dataset: dict) -> dict:
    res: dict = {"appearance": float(dataset["review/appearance"]),
                 "aroma": float(dataset["review/aroma"]), "palate": float(dataset["review/palate"]),
                 "taste": float(dataset["review/taste"]),
                 "overall": float(dataset["review/overall"]), "text": str(dataset["review/text"]),
                 "processed_text": [], "extracted_aspects": []}
    return res


def split_dataset_to_chunks() -> None:
    try:
        os.makedirs("./dataset/chunks")
    except FileExistsError:
        LOGGER.warning("chunks folder already exists")
    split = Split(DATASET, "./dataset/chunks")
    split.bylinecount(LINES_PER_CHUNK)
    os.remove("./dataset/chunks/manifest")


def remove_chunks() -> None:
    chunks: list[str] = os.listdir("./dataset/chunks")
    for chunk in chunks:
        os.remove(f"./dataset/chunks/{chunk}")


def export_dataframe_to_excel(excel_file_name: str, df: pd.DataFrame) -> None:
    df.to_excel(excel_file_name,
                index=False, engine="openpyxl")


async def async_dataframe_creator():
    split_dataset_to_chunks()
    chunk_files = os.listdir("./dataset/chunks")
    tasks: list = []
    for chunk_file in chunk_files:
        tasks.append(asyncio.create_task(
            create_processed_dataframe_with_mandatory_rows(chunk_file)))
    await asyncio.gather(*tasks, return_exceptions=True)
    return tasks


def generate_processed_dataframe_chunks():
    res = asyncio.run(async_dataframe_creator())
    remove_chunks()
    return [res.result() for res in res]
