import sys

from pandas import DataFrame

from modules.processing.processor import normalize_json_dataset, DATASET, create_processed_excel_files
from modules.utils.utilities import dump_dataframe_to_sqlite, load_dataframe_from_databaes, load_pre_processed_dataset


def menu():
    print("1 - normalize_json_dataset\n" +
          "2 - create_processed_excel_files\n" +
          "3 - load preprocessed dataset\n" +
          "4 - dump whole dataset to database\n" +
          "5 - load dataframe from database\n")
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
        create_processed_excel_files(limit)
    elif option == "3":
        load_pre_processed_dataset()
    elif option == "4":
        df: DataFrame = load_pre_processed_dataset()
        dump_dataframe_to_sqlite(df)
    elif option == "5":
        df: DataFrame = load_dataframe_from_databaes()
        print(df.head())
        print(df.info())
    else:
        print("invalid option. Exiting...")



def main():
    menu()

if __name__ == "__main__":
    main()