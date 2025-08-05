import re
import string

import demoji
import nltk
import pandas as pd
import unicodedata as uni
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from spellchecker import SpellChecker

DATASET: str = "../../dataset/dataset_as_excel_mandatory_rows.xlsx"

PUNCTUATION: list = list(string.punctuation)
STOPWORDS: set = set(stopwords.words("english"))


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
    escape_characters_and_urls_removed: str = text.replace(r"http\S+", "")
    normalized_text: str = uni.normalize("NFKD", escape_characters_and_urls_removed)
    emojis_removed: str = handle_emojis(normalized_text)
    # spellchecked: str = handle_spellchecking(emojis_removed)
    tokens = nltk.word_tokenize(emojis_removed)
    processed_tokens = [
        str(token)
        for token in tokens
        if token not in STOPWORDS and token not in PUNCTUATION
    ]
    if lemmatize:
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [
            str(wordnet_lemmatizer.lemmatize(token)) for token in processed_tokens
        ]
        return lemmatized_tokens
    return processed_tokens
