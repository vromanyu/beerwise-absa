import sys

from pandas import DataFrame

from modules.models.logistic_regression import logistic_regression_trainer
from modules.models.transformer_based import transformer_based_trainer
from modules.processing.processor import (
    parse_json_dataset,
    DATASET,
    create_processed_excel_files,
)
from modules.models.fast_text_training import (
    fast_text_model_trainer,
    generate_similarity_scores_and_labels,
    find_most_common_aspect_combination,
)
from modules.utils.utilities import (
    dump_dataframe_to_sqlite,
    load_dataframe_from_database,
    load_pre_processed_dataset,
)


def menu():
    print(
        ""
        + "1    - parse_json_dataset\n"
        + "2    - create_processed_excel_files\n"
        + "3    - load preprocessed dataset\n"
        + "4    - dump whole dataset to database\n"
        + "5    - load dataframe from database\n"
        + "6    - train FastText model\n"
        + "7    - generate similarity scores and add sentiment labels\n"
        + "8    - create target dataset by identifying mostly frequently mentioned aspects\n"
        + "9    - train Logistic Regression model\n"
        + "10   - train transformer-based model\n"
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
        df: DataFrame = load_dataframe_from_database()
        print(df.index)
        print(df.info())
        print(df.describe())
    elif option == "6":
        fast_text_model_trainer()
    elif option == "7":
        generate_similarity_scores_and_labels()
    elif option == "8":
        find_most_common_aspect_combination()
    elif option == "9":
        logistic_regression_trainer()
    elif option == "10":
        transformer_based_trainer()
    else:
        print("invalid option. Exiting...")
        sys.exit()


def main():
    menu()


if __name__ == "__main__":
    main()
