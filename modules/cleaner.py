import ast
import asyncio
import json
import logging
import os
import time

import pandas as pd
from filesplit.split import Split

import processor

DATASET: str = "../dataset/beeradvocate.json"
DATASET_OUTPUT: str = "../dataset/beeradvocate-final.json"
LOGGER: logging.Logger = logging.getLogger(__name__)
LINES_PER_CHUNK: int = 10_000


def reorganize_dataset() -> dict:
    result: dict = {}

    with open(DATASET) as f:
        counter: int = 0
        for line in f:
            json_transform: dict = {}
            invalid_json = ast.literal_eval(line)
            try:
                extract_keys_from_dataset(invalid_json)
            except KeyError:
                LOGGER.warning(f"Invalid Entry at line {counter}")
                continue
            result[counter] = json_transform.copy()
            counter += 1
    return result


def create_dataframe_from_dataset(file: str) -> pd.DataFrame:
    result = pd.DataFrame(
        columns=["name", "brewer_id", "abv", "style", "appearance", "aroma", "palate", "taste", "overall", "time",
                 "text", "profile_name", "processed_text"])
    # limit to 20 rows for development
    # counter: int = 0
    with open(f"../dataset/chunks/{file}") as f:
        for line in f:
            # if counter == 10:
            #     break
            dataset_json = ast.literal_eval(line)
            if not dataset_json:
                continue
            try:
                data_json_transform = extract_keys_from_dataset(dataset_json)
                json_df = pd.DataFrame([data_json_transform])
                result = pd.concat([result, json_df], ignore_index=True)
            except KeyError:
                continue
            # counter += 1
    return result


def extract_keys_from_dataset(dataset: dict) -> dict:
    res: dict = {"beer_id": dataset["beer/beerId"], "name": dataset["beer/name"], "brewer_id": dataset["beer/brewerId"],
                 "abv": dataset["beer/ABV"],
                 "style": dataset["beer/style"], "appearance": dataset["review/appearance"],
                 "aroma": dataset["review/aroma"], "palate": dataset["review/palate"], "taste": dataset["review/taste"],
                 "overall": dataset["review/overall"], "text": dataset["review/text"], "time": dataset["review/time"],
                 "profile_name": dataset["review/profileName"]}
    return res


def write_reorganized_json(result: dict) -> None:
    with open(DATASET_OUTPUT, "w") as f:
        f.write(json.dumps(result, indent=4))


def get_number_of_lines(dataset_file: str) -> int:
    count: int = 0
    with open(dataset_file) as f:
        for line in f:
            count += 1
    return count

def split_dataset_to_chunks() -> None:
    split = Split(DATASET, "../dataset/chunks")
    split.bylinecount(LINES_PER_CHUNK)
    os.remove("../dataset/chunks/manifest")

async def async_main():
    split_dataset_to_chunks()
    chunk_files = os.listdir("../dataset/chunks")
    dataframes: list = []
    tasks: list = []
    for chunk_file in chunk_files:
        dataframes.append(create_dataframe_from_dataset(chunk_file))
    for dataframe in dataframes:
        tasks.append(asyncio.create_task(processor.data_cleaning(dataframe)))
    await asyncio.gather(*tasks, return_exceptions=True)
    return tasks


def main():
    start = time.time()
    res = asyncio.run(async_main())
    print(f"calculation took: ${time.time() - start} seconds")
    print(res)


if __name__ == "__main__":
    main()


