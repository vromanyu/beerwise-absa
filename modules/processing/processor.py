import ast
import json
import logging
import multiprocessing
import re
import sys
from logging import Logger

import pandas as pd

from modules.processing.preprocessing import (
    handle_pre_processing,
    download_required_runtime_packages,
)

DATASET: str = "dataset/beeradvocate.json"
NORMALIZED_DATASET: str = "dataset/beeradvocate_normalized.json"
OUTPUT: str = "dataset/dataset_portion_pre_processed.xlsx"

LOG_FILE: str = "creator.log"
LOGGER: Logger = logging.getLogger(__name__)
logging.basicConfig(filename=LOG_FILE, filemode="a", level=logging.INFO)

NUMBER_OF_CORES: int = multiprocessing.cpu_count()


def normalize_json_dataset(file: str) -> None:
    counter: int = 0
    with open(file) as json_file:
        for line in json_file:
            counter += 1
            dataset_json: dict = ast.literal_eval(line)
            if not dataset_json:
                LOGGER.warning(
                    f"parsed JSON at line: {counter} -- file: {json_file.name}  was empty"
                )
                continue
            review_wrong_format_text: str = dataset_json["review/text"].lower().strip()
            removed_non_alpha_chars: str = re.sub(
                r"[^a-zA-Z0-9\s]", "", review_wrong_format_text
            )
            removed_digits: str = re.sub(r"\d+", "", removed_non_alpha_chars)
            remove_new_lines_characters: str = re.sub(r"\n", " ", removed_digits)
            remove_tab_characters: str = re.sub(r"\t", "", remove_new_lines_characters)
            remove_excessive_backslashes: str = remove_tab_characters.replace("\\", "")
            # escaped_characters_clean_text: str = bytes(remove_excessive_backslashes, "utf-8").decode("unicode_escape")
            remove_excessive_spaces: str = " ".join(
                re.split(r"\s+", remove_excessive_backslashes, flags=re.UNICODE)
            )
            dataset_json["review/text"] = remove_excessive_spaces
            with open(
                NORMALIZED_DATASET, "a", encoding="utf-8"
            ) as normalized_dataset_json:
                json.dump(dataset_json, normalized_dataset_json)
                normalized_dataset_json.write("\n")
                LOGGER.info(f"line {counter} written")


def create_processed_dataframe(
    file: str = NORMALIZED_DATASET, limit: int = 0
) -> pd.DataFrame:
    result: pd.DataFrame = pd.DataFrame()
    counter: int = 0
    with open(f"{file}") as dataset:
        for line in dataset:
            counter += 1
            if limit == counter:
                break
            processed_line: pd.DataFrame = process_line(line, counter)
            if processed_line is not None:
                result = pd.concat([result, processed_line], ignore_index=True)
            else:
                LOGGER.warning(f"line {counter} was not processed")
    LOGGER.info("finished parallel processing")
    return result


def process_line(data: str, line: int) -> pd.DataFrame | None:
    dataset_json = ast.literal_eval(data)
    if not dataset_json:
        LOGGER.warning(f"line {line}: can't be processed")
        return None
    try:
        result: pd.DataFrame = pd.DataFrame()
        LOGGER.info(f"processing line: {line}")
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


def export_dataframe_to_excel(excel_file_name: str, df: pd.DataFrame) -> None:
    df.to_excel(excel_file_name, index=False, engine="openpyxl")


def clean_logs() -> None:
    with open(LOG_FILE, "w") as log_file:
        log_file.close()


def main():
    menu()


def menu():
    print("1 - normalize_json_dataset\n2 - create_processed_dataframe")
    option: str = input("Enter your option: ")
    if option == "1":
        normalize_json_dataset(DATASET)
    elif option == "2":
        limit: int = 0
        try:
            limit = int(input("enter limit (default: 0): "))
        except ValueError:
            limit = 0
        except EOFError:
            sys.exit()
        download_required_runtime_packages()
        result: pd.DataFrame = create_processed_dataframe(NORMALIZED_DATASET, limit)
        result.reset_index(inplace=True, drop=True)
        export_dataframe_to_excel(OUTPUT, result)
    else:
        print("invalid option. Exiting...")


if __name__ == "__main__":
    main()
