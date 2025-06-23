import sys

import pandas as pd

from modules.processing.processor import normalize_json_dataset, DATASET, create_processed_dataframe, \
    NORMALIZED_DATASET, OUTPUT, export_dataframe_to_excel


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
        result: pd.DataFrame = create_processed_dataframe(NORMALIZED_DATASET, limit)
        result.reset_index(inplace=True, drop=True)
        export_dataframe_to_excel(OUTPUT, result)
    else:
        print("invalid option. Exiting...")


def main():
    menu()

if __name__ == "__main__":
    main()