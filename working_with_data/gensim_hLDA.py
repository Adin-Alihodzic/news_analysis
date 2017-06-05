import pandas as pd
import numpy as np

from gensim import corpora, models, similarities
from gensim.models.hdpmodel import HdpModel

url_names = ['cnn', 'abc', 'fox', 'nyt', 'reuters', 'wapo', 'huffpo', 'esquire', 'rollingstone', 'cbs', '538', 'washtimes']

# df = pd.read_csv('../data/rss_feeds_new_data.csv')#, parse_dates=['date_published'])
df = pd.read_csv('../data/rss_feeds_new_data.csv')
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

print(len(hdp.print_topics(-1)))


def topic_prob_extractor(hdp=None, topn=None):
    topic_list = hdp.show_topics(-1, topn)
    topics = [x[1] for x in topic_list]
    split_list = [x[1] for x in topic_list]
    weights = []
    for lst in split_list:
        weights.append([float(x.split('*')[0]) for x in lst.split(' + ')])
    sums = [np.sum(x) for x in weights]
    return pd.DataFrame({'topic_id' : topics, 'weight' : sums})

topic_weights = topic_prob_extractor(hdp, 500)
