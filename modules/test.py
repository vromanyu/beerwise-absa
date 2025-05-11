# import string
# import nltk
# from nltk.corpus import stopwords
# from nltk import WordNetLemmatizer
#
# text = "dogs, churches, aardwolves and abaci"
# punctuations = list(string.punctuation)
# lower_text = text.lower()
# tokens = nltk.word_tokenize(lower_text)
# tokens_with_no_stopwords = [token for token in tokens if token not in set(stopwords.words("english"))]
# tokens_with_no_punctuation = [token for token in tokens_with_no_stopwords if token not in punctuations]
# wordnet_lemmatizer = WordNetLemmatizer()
# print(f"before lemmas: {tokens_with_no_punctuation}")
# lemmatized_tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens_with_no_punctuation]
# print(f"after lemma: {lemmatized_tokens}")
#

