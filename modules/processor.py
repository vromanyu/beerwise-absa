import nltk
import pandas as pd
import cleaner
import unicodedata as uni
import demoji
import re
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import string
import time
import asyncio


DATASET: str = "../dataset/beeradvocate.json"

def download_required_nltk_packages() -> None:
    nltk.download("stopwords")
    nltk.download('punkt_tab')
    nltk.download('wordnet')

def load_dataframe(dataset: str) -> pd.DataFrame:
    dataset_list: list = cleaner.load_dataset_as_list(dataset)
    df = pd.DataFrame(dataset_list)
    return df

def drop_missing_values(df: pd.DataFrame) -> None:
    df.dropna(inplace=True, axis=0)

def basic_escape_character_cleaning(df: pd.DataFrame) -> None:
    df["text"] = df["text"].str.replace(r"\t", "", regex=True).str.replace(r"\n", "", regex=True)

def url_cleaning(df: pd.DataFrame) -> None:
    df["text"] = df["text"].str.replace(r"http\S+", "", regex=True)

def unicode_normalization(df: pd.DataFrame) -> None:
    df["text"] = df["text"].apply(lambda x: uni.normalize('NFKD', x))

def remove_emojis(df: pd.DataFrame) -> None:
    df["text"] = df["text"].apply(handle_emojis)

def fix_spelling(df: pd.DataFrame) -> None:
    df["text"] = df["text"].apply(handle_spellchecking)

def pre_processing(df: pd.DataFrame) -> None:
    df["processed_text"] = df["text"].apply(handle_pre_processing)

def handle_emojis(text :str) -> str:
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
    punctuations = list(string.punctuation)
    lower_text = text.lower()
    tokens = nltk.word_tokenize(lower_text)
    tokens_with_no_stopwords = [token for token in tokens if token not in set(stopwords.words("english"))]
    tokens_with_no_punctuation = [token for token in tokens_with_no_stopwords if token not in punctuations]
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens_with_no_punctuation]
    return lemmatized_tokens


async def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    basic_escape_character_cleaning(df)
    url_cleaning(df)
    unicode_normalization(df)
    remove_emojis(df)
    fix_spelling(df)
    pre_processing(df)
    return df
