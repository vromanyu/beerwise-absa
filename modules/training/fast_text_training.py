import logging
from logging import Logger

import gensim.models
# import fasttext
import pandas as pd
from gensim.models import FastText

from modules.dataframe_creator import df_creator
from modules.processing.processor import load_pre_processed_dataset

# from fasttext import FastText

LOGGER: Logger = logging.getLogger(__name__)


def fast_text_model_trainer():
    df: pd.DataFrame = load_pre_processed_dataset("../../dataset/dataset_as_excel_all_rows_full.xlsx")
    data_words: list[list] = df["processed_text"].to_list()
    fasttext_model = FastText(data_words, vector_size=300, window=5, min_count=5, workers=20, sg=1)
    fasttext_model.save("../../models/fast_text_for_absa.bin")


def load_model(location: str):
    model: FastText = gensim.models.FastText.load(location)
    return model


def generate_similarity_scores() -> pd.DataFrame:
    df: pd.DataFrame = load_pre_processed_dataset("../../dataset/dataset_as_excel_all_rows_full.xlsx")
    model: FastText = load_model("../../models/fast_text_for_absa.bin")
    aspects: list = ["appearance", "aroma", "palate", "taste"]
    for aspect in aspects:
        df[f"{aspect}_similarity"] = df["processed_text"].apply(lambda x: get_similarity(x, aspect, model))
    return df


def get_similarity(text: list, aspect: str, model: FastText):
    try:
        text = " ".join(text)
        return model.wv.similarity(text, aspect)
    except Exception:
        LOGGER.error(f"error generating similarity scores for aspect {aspect}")
        return 0


if __name__ == "__main__":
    # fast_text_model_trainer()
    df: pd.DataFrame = generate_similarity_scores()
    df_creator.export_dataframe_to_excel("../../dataset/dataset_as_excel_all_rows_full_with_similarities.xlsx", df)
