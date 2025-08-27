import sys

from modules.algorithms.fast_text_training import (
    generate_similarity_scores_labels_and_filter,
)
from modules.algorithms.linear_svc import linear_svc_trainer
from modules.algorithms.logistic_regression import logistic_regression_trainer
from modules.algorithms.naive_bayes import naive_bayes_trainer
from modules.algorithms.ridge_classifier import ridge_classifier_trainer
from modules.algorithms.transformer_based import transformer_based_trainer
from modules.processing.processor import (
    create_preprocessed_excel_files_and_save_to_db,
    parse_json_dataset,
    DATASET,
)
from modules.utils.utilities import (
    predict_sentiments_using_linear_svc,
    predict_sentiments_using_logistic_regression,
    predict_sentiment_using_naive_bayes, predict_sentiments_using_ridge_classifier
)


def menu():
    print(
        ""
        + "1    - Parse original dataset\n"
        + "2    - Preprocess dataset and save to database\n"
        + "3    - Identify aspects using FastText and filter dataset\n"
        + "4    - Train Logistic Regression model\n"
        + "5    - Train LinearSVC model\n"
        + "6    - Train MultinomialNB model\n"
        + "7    - Train Ridge Classifier model\n"
        + "8    - Train Transformer-based model\n"
        + "9    - Predict sentiments using Logistic Regression model\n"
        + "10   - Predict sentiments using LinearSVC model\n"
        + "11   - Predict sentiments using Naive Bayes model\n"
        + "12   - Predict sentiments using Ridge Classifier model\n"
    )
    option: str = input("Enter your option: ")
    if option == "1":
        parse_json_dataset(DATASET)
    elif option == "2":
        create_preprocessed_excel_files_and_save_to_db()
    elif option == "3":
        generate_similarity_scores_labels_and_filter()
    elif option == "4":
        logistic_regression_trainer()
    elif option == "5":
        linear_svc_trainer()
    elif option == "6":
        naive_bayes_trainer()
    elif option == "7":
        ridge_classifier_trainer()
    elif option == "8":
        transformer_based_trainer()
    elif option == "9":
        user_input = input("Enter beer review: ")
        predict_sentiments_using_logistic_regression(user_input)
    elif option == "10":
        user_input = input("Enter beer review: ")
        predict_sentiments_using_linear_svc(user_input)
    elif option == "11":
        user_input = input("Enter beer review: ")
        predict_sentiment_using_naive_bayes(user_input)
    elif option == "12":
        user_input = input("Enter beer review: ")
        predict_sentiments_using_ridge_classifier(user_input)
    else:
        print("invalid option. Exiting...")
        sys.exit()


def main():
    menu()


if __name__ == "__main__":
    main()
