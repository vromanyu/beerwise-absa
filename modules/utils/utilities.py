from ast import literal_eval

import pandas as pd

DATASET_EXCEL: str = "dataset/dataset_portion_pre_processed.xlsx"


def load_pre_processed_dataset(dataset: str = DATASET_EXCEL) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_excel(dataset)
    # df["processed_text"] = df["processed_text"].apply(literal_eval)
    return df
