import logging
import multiprocessing
import pprint
from logging import Logger

import gensim.models
import pandas as pd
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import FastText, LdaMulticore

from modules.utils.utilities import load_dataframe_from_database, load_pre_processed_dataset

LOGGER: Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
DATASET: str = "../../dataset/dataset_portion_pre_processed.xlsx"
MODEL_LOCATION: str = "../../models/fast_text_for_absa.bin"
NUMBER_OF_CORES: int = multiprocessing.cpu_count()


def fast_text_model_trainer():
    df: pd.DataFrame = load_dataframe_from_database()
    LOGGER.info("loading all 'processed_text'")
    data_words: list[list] = df["processed_text"].to_list()
    LOGGER.info("initialize model training")
    fasttext_model: FastText = FastText(data_words, vector_size=100, window=5, min_count=5, workers=NUMBER_OF_CORES, sg=1)
    LOGGER.info("dumping model at /models")
    fasttext_model.save("../../models/fast_text_for_absa.bin")
    LOGGER.info("model saved")


def load_model(location: str):
    model: FastText = gensim.models.FastText.load(location)
    return model


# def generate_similarity_scores() -> None:
#     df: pd.DataFrame = load_pre_processed_dataset(DATASET)
#     model: FastText = load_model("../../models/fast_text_for_absa.bin")
#     aspects: list[str] = ["appearance", "aroma", "palate", "taste"]
#     for aspect in aspects:
#         df[f"{aspect}_similarity"] = df["processed_text"].apply(lambda x: get_similarity(x, aspect, model))
#     export_dataframe_to_excel(DATASET, df)


def generate_and_print_topics():
    df: pd.DataFrame = load_pre_processed_dataset(DATASET)
    data_words: list[list] = df["processed_text"].to_list()
    id2word: Dictionary = corpora.Dictionary(data_words)
    corpus: list[list[tuple[int, int]]] = [id2word.doc2bow(text) for text in data_words]
    lda_model = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=10, iterations=400)
    pprint.pprint(lda_model.print_topics())


def get_similarity(text: list, aspect: str, model: FastText):
    text = " ".join(text)
    return model.wv.similarity(text, aspect)


if __name__ == "__main__":
    # fast_text_model_trainer()
    # generate_similarity_scores()
    # df_creator.export_dataframe_to_excel("../../dataset/dataset_as_excel_all_rows_full_with_similarities.xlsx", df)
    # generate_and_print_topics()
    pass
