import ast
import asyncio
import logging
import time
import os
from logging import Logger
from modules.processing.processor import handle_pre_processing, handle_aspect_extraction
from ast import literal_eval

import pandas as pd
from filesplit.split import Split

DATASET: str = "./dataset/beeradvocate.json"
# LINES_PER_CHUNK: int = 400_000
LINES_PER_CHUNK: int = 1_000_000

LOGGER: Logger = logging.getLogger(__name__)


async def create_processed_dataframe(file: str) -> pd.DataFrame:
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
                data_json_transform = extract_keys_from_dataset(dataset_json)
                json_df = pd.DataFrame([data_json_transform])
                json_df["processed_text"] = json_df["text"].apply(
                    handle_pre_processing)
                print(json_df.iloc[0])
                json_df["extracted_aspects"] = json_df["processed_text"].apply(
                    lambda x: print(type(x)))
                result = pd.concat([result, json_df], ignore_index=True)
            except KeyError:
                LOGGER.error(
                    f"error processing line: {counter} --- dataset: {dataset_json}")
                continue
            except ValueError:
                LOGGER.error(
                    f"error converting at line: {counter} ---- dataset: {dataset_json}")
                continue
            except Exception:
                LOGGER.error(Exception)
    return result


def extract_keys_from_dataset(dataset: dict) -> dict:
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


async def async_main():
    split_dataset_to_chunks()
    chunk_files = os.listdir("./dataset/chunks")
    tasks: list = []
    for chunk_file in chunk_files:
        tasks.append(asyncio.create_task(
            create_processed_dataframe(chunk_file)))
    await asyncio.gather(*tasks, return_exceptions=True)
    return tasks


def generate_processed_dataframe():
    res = asyncio.run(async_main())
    remove_chunks()
    return [res.result() for res in res]
