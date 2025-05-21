from ast import literal_eval

import pandas as pd


def load_pre_processed_dataset(dataset: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_excel(dataset)
    df["processed_text"] = df["processed_text"].apply(literal_eval)
    return df
