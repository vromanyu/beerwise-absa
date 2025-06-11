from pprint import pprint

import pandas as pd
import gensim.corpora as corpora
from gensim.models import LdaMulticore, FastText

from modules.utils.utilities import load_pre_processed_dataset

if __name__ == "__main__":
    pass
    # df: pd.DataFrame = load_pre_processed_dataset("./dataset/dataset_as_excel_all_rows.xlsx")
    # data_words: list[list] = df["processed_text"].to_list()
    # id2word = corpora.Dictionary(data_words)
    # corpus = [id2word.doc2bow(data_word) for data_word in data_words]
    # lda_model = LdaMulticore(corpus=corpus, num_topics=50, id2word=id2word, iterations=400)
    # fasttext_model = FastText(data_words, vector_size=10, window=5, min_count=5, workers=20, sg=1)
    # fasttext_model.save("./models/FastText-Model-For-ABSA.bin")
    # print(fasttext_model.wv.n_similarity("A lot of foam. But a lot.	In the smell some banana, and then lactic and tart. Not a good start.	Quite dark orange in color, with a lively carbonation (now visible, under the foam).	Again tending to lactic sourness.	Same for the taste. With some yeast and banana.", "taste"))
    # print(fasttext_model.wv.n_similarity("Dark red color, light beige foam, average.	In the smell malt and caramel, not really light.	Again malt and caramel in the taste, not bad in the end.	Maybe a note of honey in teh back, and a light fruitiness.	Average body.	In the aftertaste a light bitterness, with the malt and red fruit.	Nothing exceptional, but not bad, drinkable beer.", "taste"))
