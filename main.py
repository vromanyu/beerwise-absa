import sys

from pandas import DataFrame

from modules.models.logistic_regression import logistic_regression_trainer
from modules.processing.processor import (
    parse_json_dataset,
    DATASET,
    create_processed_excel_files,
)
from modules.models.fast_text_training import fast_text_model_trainer, generate_similarity_scores_and_labels
from modules.utils.utilities import (
    dump_dataframe_to_sqlite,
    load_dataframe_from_database,
    load_pre_processed_dataset, create_sample_dataset,
)


def menu():
    print(""
          + "1 - parse_json_dataset\n"
          + "2 - create_processed_excel_files\n"
          + "3 - load preprocessed dataset\n"
          + "4 - dump whole dataset to database\n"
          + "5 - load dataframe from database\n"
          + "6 - train FastText model\n"
          + "7 - generate similarity scores and add sentiment labels\n"
          + "8 - create sample dataset\n"
          + "9 - train Logistic Regression model (multi-output classifier)\n"
          )
    option: str = input("Enter your option: ")
    if option == "1":
        parse_json_dataset(DATASET)
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
        is_sample: bool = False
        try:
            user_input = input("load the sample dataset (y/n): ")
            if user_input.lower() == "y":
                is_sample = True
        except EOFError:
            sys.exit()
        df: DataFrame = load_dataframe_from_database(is_sample)
        print(df.index)
        print(df.info())
        print(df.describe())
    elif option == "6":
        is_sample: bool = False
        try:
            user_input = input("use the sample dataset (y/n): ")
            if user_input.lower() == "y":
                is_sample = True
        except EOFError:
            sys.exit()
        fast_text_model_trainer(is_sample)
    elif option == "7":
        is_sample: bool = False
        try:
            user_input = input("use the sample dataset (y/n): ")
            if user_input.lower() == "y":
                is_sample = True
        except EOFError:
            sys.exit()
        generate_similarity_scores_and_labels(is_sample)
    elif option == "8":
        length: int = 50_000
        try:
            length = int(input("enter sample length (default: 50_000): "))
        except ValueError:
            length = 50_000
        except EOFError:
            sys.exit()
        create_sample_dataset(length)
    elif option == "9":
        is_sample: bool = False
        try:
            user_input = input("use the sample dataset (y/n): ")
            if user_input.lower() == "y":
                is_sample = True
        except EOFError:
            sys.exit()
        logistic_regression_trainer(is_sample)
    else:
        print("invalid option. Exiting...")
        sys.exit()


def main():
    menu()


if __name__ == "__main__":
    main()
