import re
import unicodedata as uni

import demoji
import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from spellchecker import SpellChecker

DATASET = "../../dataset/dataset_as_excel_mandatory_rows.xlsx"

STOPWORDS = set(stopwords.words("english"))
CLEANUP_REGEX = re.compile(r"http\S+|\d+|[^a-z\s]", flags=re.UNICODE)
WHITESPACE_REGEX = re.compile(r"\s+", flags=re.UNICODE)


def download_required_runtime_packages() -> None:
    nltk.download("stopwords")
    nltk.download("punkt_tab")
    nltk.download("wordnet")


def drop_missing_values(df: pd.DataFrame) -> None:
    df.dropna(inplace=True, axis=0)


def handle_emojis(text: str) -> str:
    emojis = demoji.findall(text)
    for emoji in emojis:
        text = text.replace(emoji, "" + emojis[emoji].split(":")[0])
    return text


# Unused (very slow)
def handle_spellchecking(text: str) -> str:
    tokens = re.findall("[a-zA-Z]+", text)
    spell = SpellChecker(language="en")
    misspelled: set[str] = spell.unknown(tokens)
    for word in misspelled:
        correct_token = spell.correction(word)
        if correct_token is not None:
            text = text.replace(word, correct_token)
    return text


def handle_pre_processing(text: str, lemmatize: bool = False) -> list[str]:
    text = uni.normalize("NFKD", text.lower().strip())
    text = handle_emojis(text)
    text = CLEANUP_REGEX.sub("", text)
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    tokens = nltk.word_tokenize(text)
    processed_tokens: list[str] = [
        str(token) for token in tokens if token not in STOPWORDS
    ]
    if lemmatize:
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized_tokens: list[str] = [
            str(wordnet_lemmatizer.lemmatize(token)) for token in processed_tokens
        ]
        return lemmatized_tokens
    return processed_tokens
