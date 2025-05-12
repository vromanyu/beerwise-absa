import re
import string

import demoji
import nltk
import pandas as pd
import unicodedata as uni
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from spellchecker import SpellChecker


def download_required_nltk_packages() -> None:
    nltk.download("stopwords")
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')


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
    remove_numbers: str = "".join([word for word in lower_text if not word.isdigit()])
    escape_characters_and_urls_removed: str = remove_numbers.replace(r"\t", " ").replace(r"\n", " ").replace(r"http\S+", "").replace("\"", "").strip()
    normalized_text: str = uni.normalize("NFKD", escape_characters_and_urls_removed)
    emojis_removed: str = handle_emojis(normalized_text)
    spellchecked: str = handle_spellchecking(emojis_removed)
    tokens = nltk.word_tokenize(spellchecked)
    tokens_with_no_stopwords = [str(token) for token in tokens if token not in set(stopwords.words("english"))]
    tokens_with_no_punctuation = [str(token) for token in tokens_with_no_stopwords if token not in punctuations]
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [str(wordnet_lemmatizer.lemmatize(token)) for token in tokens_with_no_punctuation]
    return lemmatized_tokens


async def pre_process_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df["processed_text"] = df["text"].apply(handle_pre_processing)
    return df
