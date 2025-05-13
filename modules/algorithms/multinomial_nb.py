import pandas as pd
from ast import literal_eval
import numpy as np
from nltk import flatten
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy

DATASET: str = "./dataset/sample_dataset_as_excel.xlsx"


def load_dataframe(file: str) -> pd.DataFrame:
    df = pd.read_excel(file)
    df["processed_text"] = df["processed_text"].apply(literal_eval)
    # df["extracted_aspects"] = df["extracted_aspects"].apply(literal_eval)
    return df


def create_recommended_column(df: pd.DataFrame) -> pd.DataFrame:
    df["recommended"] = df["overall"].apply(
        lambda value: 1 if value > 3 else 0)
    return df


def main(df: pd.DataFrame) -> None:
    df: pd.DataFrame = load_dataframe(f"{DATASET}")
    # aspect_extraction(df)
    # print(df)
    # df = create_recommended_column(df)
    # X = df["processed_text"].astype(str)
    # y = df["recommended"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    # vectorizer = TfidfVectorizer(stop_words="english")
    # X_train_tfidf = vectorizer.fit_transform(X_train)
    # X_test_tfidf = vectorizer.transform(X_test)
    # model = MultinomialNB()
    # model.fit(X_train_tfidf, y_train)
    # y_pred = model.predict(X_test_tfidf)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Model Accuracy: {accuracy:.2f}")
    # print(classification_report(y_test, y_pred))

    # train: pd.DataFrame = df.iloc[0:100]
    # test: pd.DataFrame  = df.iloc[100:]
    # train_unique_words = np.unique(np.hstack(train["processed_text"].to_numpy()))
    # train_number_of_unique_words = len(train_unique_words)
    # vectorizer = CountVectorizer(max_features=train_number_of_unique_words)
    # print(train["processed_text"].astype(str))
    # X_train_vectors = vectorizer.fit_transform(train["processed_text"].astype(str))
    # print(X_train_vectors)
    # print(vectorizer.get_feature_names_out()[20:40])
    # model = MultinomialNB()
    # model.fit(X_train_vectors, train["recommended"])
    # X_test_vectors = vectorizer.transform(test["processed_text"])
    # y_test_hat = model.predict(X_test_vectors)
    # print(y_test_hat)
    # unique_words, words_count = np.unique(list_of_words, return_counts=True)
    # freq, words = (list(i) for i in zip(*(sorted(zip(words_count, unique_words), reverse=True))))
    # f_o_w = []
    # n_o_w = []
    # for f in sorted(np.unique(freq), reverse=True):
    #     f_o_w.append(f)
    #     n_o_w.append(freq.count(f))
    # print(f_o_w, n_o_w)
    # print(words)
    # x = n_o_w
    # y = f_o_w
    # plt.xlabel("no of words")
    # plt.ylabel("freq of words")
    # plt.plot(x,y)
    # plt.grid()
    # plt.show()
    # n = 200
    # features = words[0:n]
    # dictionary = {}
    # doc_num = 1
    # for doc_words in

    # print(df["unique_tokens"])


if __name__ == "__main__":
    main()
