import numpy as np
import pandas as pd
import nltk
from nltk.corpus import sentiwordnet as swn
import warnings
import pickle
warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/IPython/.*')

import pyLDAvis
import pyLDAvis.sklearn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt

from dateutil import parser
import datetime

pyLDAvis.enable_notebook()

df = pd.read_csv('wsj_articles_data.csv', parse_dates=False)
del df['Unnamed: 0']
df = df[df['processed_text'].notnull()]
# df['date_published'] = df['date_published'].apply(lambda x: parser.parse(x.split('|')[0]))
fixed_date = []
for date in df['date_published']:
    try:
        fixed_date.append(parser.parse(date.split('|')[0]))
    except:
        df = df[df['date_published'] != date]
df['date_published'] = fixed_date
df_2016 = df[(df['date_published'] > datetime.date(2015, 12, 31)) & (df['date_published'] < datetime.date(2017,1,1))]

df_no_nan = df[pd.notnull(df['processed_text'])]
text = df_no_nan['processed_text'].values.tolist()

print('Working on tfidf...')
max_features = 10000
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=max_features,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(text)

print('Working on LDA...')
n_topics = 100
lda_model = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                      learning_method='online',
                                      learning_offset=50.,
                                      random_state=0)

lda_model.fit(tf)
vis_data = pyLDAvis.sklearn.prepare(lda_model,tf, tf_vectorizer, R=n_topics, n_jobs=-1)
pyLDAvis.show(vis_data)

filename = 'pickles/2016_wsj_data_model.pkl'
pickle.dump(lda_model, open(filename, 'wb'), protocol=4)
