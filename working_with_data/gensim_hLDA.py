import pandas as pd
import numpy as np

from gensim import corpora, models, similarities
from gensim.models.hdpmodel import HdpModel

url_names = ['cnn', 'abc', 'fox', 'nyt', 'reuters', 'wapo', 'huffpo', 'esquire', 'rollingstone', 'cbs', '538', 'washtimes']

# df = pd.read_csv('../data/rss_feeds_new_data.csv')#, parse_dates=['date_published'])
df = pd.read_csv('../data/wsj_articles_data.csv')
df_no_nan = df[pd.notnull(df['processed_text'])]
documents = df_no_nan['processed_text'].values.tolist()

texts = [[word for word in document.split()] for document in documents]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
for text in texts]

from pprint import pprint  # pretty-printer
# pprint(texts)

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

print('Working on HDP...')
hdp = HdpModel(corpus, dictionary)
