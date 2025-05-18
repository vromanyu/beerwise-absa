import re
import string
from ast import literal_eval

import demoji
import nltk
import pandas as pd
import unicodedata as uni
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import spacy

DATASET: str = "../../dataset/dataset_as_excel_mandatory_rows.xlsx"


def download_required_runtime_packages() -> None:
    nltk.download("stopwords")
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    spacy.cli.download("en_core_web_sm")


def drop_missing_values(df: pd.DataFrame) -> None:
    df.dropna(inplace=True, axis=0)


def handle_emojis(text: str) -> str:
    emojis = demoji.findall(text)
    for emoji in emojis:
        text = text.replace(emoji, "" + emojis[emoji].split(":")[0])
    return text


def handle_spellchecking(text: str) -> str:
    tokens = re.findall("[a-zA-Z]+", text)
    spell = SpellChecker(language="en")
    misspelled: set[str] = spell.unknown(tokens)
    for word in misspelled:
        correct_token = spell.correction(word)
        if correct_token is not None:
            text = text.replace(word, correct_token)
    return text


def handle_pre_processing(text: str) -> list[str]:
    punctuations: list = list(string.punctuation)
    lower_text: str = text.lower()
    remove_numbers: str = "".join(
        [word for word in lower_text if not word.isdigit()])
    escape_characters_and_urls_removed: str = remove_numbers.replace(
        r"\t", " ").replace(r"\n", " ").replace(r"http\S+", "").replace("\"", "").strip()
    normalized_text: str = uni.normalize(
        "NFKD", escape_characters_and_urls_removed)
    emojis_removed: str = handle_emojis(normalized_text)
    spellchecked: str = handle_spellchecking(emojis_removed)
    tokens = nltk.word_tokenize(spellchecked)
    tokens_with_no_stopwords = [
        str(token) for token in tokens if token not in set(stopwords.words("english"))]
    tokens_with_no_punctuation = [
        str(token) for token in tokens_with_no_stopwords if token not in punctuations]
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [str(wordnet_lemmatizer.lemmatize(token))
                         for token in tokens_with_no_punctuation]
    return lemmatized_tokens


def handle_aspect_extraction(tokens: list[str]) -> list[str]:
    nlp = spacy.load("en_core_web_sm")
    extracted_aspects: list[str] = []
    processed_token_string = " ".join(tokens)
    include_tag: str = "NN"
    exclude_shapes: list[str] = ["x", "xx", "xxx"]
    document = nlp(processed_token_string)
    for token in document:
        if token.tag_ == include_tag and token.shape_ not in exclude_shapes:
            extracted_aspects.append(token.lemma_)
    return extracted_aspects


def load_pre_processed_dataset(dataset: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_excel(dataset)
    df["processed_text"] = df["processed_text"].apply(literal_eval)
    df["extracted_aspects"] = df["extracted_aspects"].apply(literal_eval)
    return df
