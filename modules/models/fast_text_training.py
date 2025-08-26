import logging
import multiprocessing
import os

import gensim.models
import numpy as np
import pandas as pd
from gensim.models import FastText

from modules.utils.utilities import (
    dump_dataframe_to_sqlite,
    load_dataframe_from_database,
    save_most_common_aspects,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
MODELS_LOCATION = "./models"
FULL_MODEL_NAME = "fast_text_for_absa.bin"
NUMBER_OF_CORES = multiprocessing.cpu_count()
SIMILARITY_THRESHOLD = 0.33


def fast_text_model_trainer() -> None:
    df = load_dataframe_from_database()
    LOGGER.info("loading all 'processed_text'")
    data_words = df["processed_text"].to_list()
    LOGGER.info("initialize model training")
    fasttext_model = FastText(
        data_words,
        vector_size=100,
        window=5,
        min_count=5,
        workers=NUMBER_OF_CORES,
        sg=1,
    )
    os.makedirs(f"{MODELS_LOCATION}/fast_text", exist_ok=True)
    fasttext_model.save(f"{MODELS_LOCATION}/fast_text/{FULL_MODEL_NAME}")
    LOGGER.info("model saved")


def load_model() -> FastText | None:
    try:
        LOGGER.info("loading FastText model")
        return gensim.models.FastText.load(
            f"{MODELS_LOCATION}/fast_text/{FULL_MODEL_NAME}"
        )
    except FileNotFoundError:
        LOGGER.error("model was not found or trained")
        return None


def generate_similarity_scores_labels_and_filter() -> None:
    df = load_dataframe_from_database()

    model = load_model()
    if model is None:
        fast_text_model_trainer()
        model = load_model()

    aspects = ["appearance", "aroma", "palate", "taste"]
    for aspect in aspects:
        LOGGER.info(f"finding similarity score for aspect: {aspect}")
        df[f"{aspect}_similarity"] = df["processed_text"].apply(
            lambda x: get_similarity(x, aspect, model)
        )
        LOGGER.info(
            f"checking where '{aspect}' is mentioned using threshold: {SIMILARITY_THRESHOLD}"
        )
        df[f"{aspect}_mentioned"] = df.apply(
            lambda row: is_aspect_mentioned(row, aspect), axis=1
        )

        LOGGER.info(f"assigning sentiment label to aspect: {aspect}")
        df[f"{aspect}_sentiment"] = df.apply(
            lambda row: rating_to_sentiment(row[f"{aspect}"])
            if row[f"{aspect}_mentioned"]
            else np.NaN,
            axis=1,
        )

    find_most_common_aspect_combination(df)


def get_similarity(text: list[str], aspect: str, model: FastText):
    return model.wv.n_similarity(text, aspect)


def is_aspect_mentioned(row: pd.Series, aspect: str) -> bool:
    return row[f"{aspect}_similarity"] >= SIMILARITY_THRESHOLD


def rating_to_sentiment(rating: float) -> int:
    if rating >= 4.0:
        return 1
    elif rating >= 2.5:
        return 0
    else:
        return -1


def find_most_common_aspect_combination(df: pd.DataFrame) -> None:
    df = load_dataframe_from_database()
    aspects_mentioned = [
        "appearance_mentioned",
        "aroma_mentioned",
        "palate_mentioned",
        "taste_mentioned",
    ]

    df["aspect_combo"] = df[aspects_mentioned].apply(tuple, axis=1)

    aspect_counts = df["aspect_combo"].value_counts()
    most_common_combo = aspect_counts.idxmax()

    df_filtered = df[df["aspect_combo"] == most_common_combo]

    aspect_names = [
        aspect.split("_")[0]
        for aspect, mentioned in zip(aspects_mentioned, most_common_combo)
        if mentioned
    ]

    LOGGER.info(
        f"Most common aspect combination: {aspect_names} - {len(df_filtered)} rows"
    )

    base_columns = ["beer_id", "name", "text", "processed_text"]
    aspect_cols = []
    for aspect in aspect_names:
        aspect_cols.extend(
            [
                aspect,
                f"{aspect}_similarity",
                f"{aspect}_mentioned",
                f"{aspect}_sentiment",
            ]
        )
    keep_columns = base_columns + aspect_cols
    df_final = df_filtered[keep_columns].copy()

    dump_dataframe_to_sqlite(df_final, is_target=True)
    save_most_common_aspects()
