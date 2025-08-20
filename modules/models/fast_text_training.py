import logging
import multiprocessing
import os
from logging import Logger
from math import floor

import gensim.models
import pandas as pd
from gensim.models import FastText

from modules.utils.utilities import (
    dump_dataframe_to_sqlite,
    load_dataframe_from_database,
)

LOGGER: Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
MODELS_LOCATION: str = "./models"
FULL_MODEL_NAME: str = "fast_text_for_absa_full.bin"
SAMPLE_MODEL_NAME: str = "fast_text_for_absa_sample.bin"
NUMBER_OF_CORES: int = multiprocessing.cpu_count()
SIMILARITY_THRESHOLD: float = 0.25


def fast_text_model_trainer(is_sample: bool = False) -> None:
    df: pd.DataFrame = load_dataframe_from_database(is_sample)
    LOGGER.info("loading all 'processed_text'")
    data_words: list[list] = df["processed_text"].to_list()
    LOGGER.info("initialize model training")
    fasttext_model: FastText = FastText(
        data_words,
        vector_size=100,
        window=5,
        min_count=5,
        workers=NUMBER_OF_CORES,
        sg=1,
    )
    os.makedirs(f"{MODELS_LOCATION}/fast_text/sample", exist_ok=True)
    os.makedirs(f"{MODELS_LOCATION}/fast_text/full", exist_ok=True)
    if is_sample:
        fasttext_model.save(f"{MODELS_LOCATION}/fast_text/sample/{SAMPLE_MODEL_NAME}")
    else:
        fasttext_model.save(f"{MODELS_LOCATION}/fast_text/full/{FULL_MODEL_NAME}")
    LOGGER.info("model saved")


def load_model(is_sample: bool = False) -> FastText | None:
    try:
        if is_sample:
            LOGGER.info("loading sample FastText model")
            return gensim.models.FastText.load(f"{MODELS_LOCATION}/fast_text/sample/{SAMPLE_MODEL_NAME}")
        else:
            LOGGER.info("loading full FastText model")
            return gensim.models.FastText.load(f"{MODELS_LOCATION}/fast_text/full/{FULL_MODEL_NAME}")
    except FileNotFoundError:
        LOGGER.error("model was not found or trainer")
        return None


def generate_similarity_scores_and_labels(is_sample: bool = False) -> None:
    df: pd.DataFrame = load_dataframe_from_database(is_sample)

    model: FastText | None = load_model(is_sample)
    if model is None:
        return

    aspects: list[str] = ["appearance", "aroma", "palate", "taste"]
    for aspect in aspects:
        LOGGER.info(f"finding similarity score for aspect: {aspect}")
        df[f"{aspect}_similarity"] = df["processed_text"].apply(
            lambda x: get_similarity(x, aspect, model)
        )
        LOGGER.info(f"checking where '{aspect}' is mentioned using threshold: {SIMILARITY_THRESHOLD}")
        df[f"{aspect}_mentioned"] = df.apply(lambda row: is_aspect_mentioned(row, aspect), axis=1)

        LOGGER.info(f"assigning sentiment label to aspect: {aspect}")
        df[f"{aspect}_sentiment"] = df.apply(
            lambda row: rating_to_sentiment(row[f"{aspect}"]) if row[f"{aspect}_mentioned"] else -2, axis=1)

    dump_dataframe_to_sqlite(df, is_sample)


def get_similarity(text: list[str], aspect: str, model: FastText):
    return model.wv.n_similarity(text, aspect)


def is_aspect_mentioned(row: pd.Series, aspect: str) -> bool:
    return row[f"{aspect}_similarity"] >= SIMILARITY_THRESHOLD


def rating_to_sentiment(rating: float) -> int:
    if rating >= 4.0:
        return 1
    elif rating <= 2.0:
        return -1
    else:
        return 0


# Unused
# def generate_and_print_topics():
#     df: pd.DataFrame = load_pre_processed_dataset(DATASET)
#     data_words: list[list] = df["processed_text"].to_list()
#     id2word: Dictionary = corpora.Dictionary(data_words)
#     corpus: list[list[tuple[int, int]]] = [id2word.doc2bow(text) for text in data_words]
#     lda_model = LdaMulticore(
#         corpus=corpus, id2word=id2word, num_topics=10, iterations=400
#     )
#     pprint.pprint(lda_model.print_topics())
