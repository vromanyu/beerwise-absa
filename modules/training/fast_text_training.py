import logging
import multiprocessing
from logging import Logger

import gensim.models
import pandas as pd
from gensim.models import FastText

from modules.utils.utilities import (
    dump_dataframe_to_sqlite,
    load_dataframe_from_database,
)

LOGGER: Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
MODEL_LOCATION: str = "models/fast_text_for_absa.bin"
NUMBER_OF_CORES: int = multiprocessing.cpu_count()
SIMILARITY_THRESHOLD: float = 0.4


def fast_text_model_trainer():
    df: pd.DataFrame = load_dataframe_from_database()
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
    LOGGER.info("dumping model at /models")
    fasttext_model.save(f"{MODEL_LOCATION}")
    LOGGER.info("model saved")


def load_model() -> FastText | None:
    LOGGER.info("loading trained FastText model")
    try:
        model: FastText = gensim.models.FastText.load(MODEL_LOCATION)
        return model
    except FileNotFoundError as e:
        print(e)
        return None


def generate_similarity_scores_and_labels() -> None:
    df: pd.DataFrame = load_dataframe_from_database()

    model: FastText | None = load_model()
    if model is None:
        return

    aspects: list[str] = ["appearance", "aroma", "palate", "taste"]
    for aspect in aspects:
        LOGGER.info(f"finding similarity score for aspect: {aspect}")
        df[f"{aspect}_similarity"] = df["processed_text"].apply(
            lambda x: get_similarity(x, aspect, model)
        )
        df[f"{aspect}_mentioned"] = df.apply(lambda row: is_aspect_mentioned(row, aspect), axis=1)

    dump_dataframe_to_sqlite(df)


def get_similarity(text: list[str], aspect: str, model: FastText):
    return model.wv.n_similarity(text, aspect)

def is_aspect_mentioned(row: pd.Series, aspect: str) -> bool:
    return row[f"{aspect}_similarity"] >= SIMILARITY_THRESHOLD

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
