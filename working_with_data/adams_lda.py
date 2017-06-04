# USE Python3
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


def make_scatter(fit,ax,pcX=0,pcY=1,font_size=10,font_name='sans serif',ms=20,leg=True,title=None):
    colors = ['k','cyan','r','orange','g','b','magenta']
    lines = []
    indices = np.arange(fit.shape[0])
    s = ax.scatter(fit[indices,pcX],fit[indices,pcY],s=ms,alpha=0.9)
    lines.append(s)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(font_size-2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(font_size-2)

    buff = 0.02
    bufferX = buff * (fit[:,pcX].max() - fit[:,pcX].min())
    bufferY = buff * (fit[:,pcY].max() - fit[:,pcY].min())
    ax.set_xlim([fit[:,pcX].min()-bufferX,fit[:,pcX].max()+bufferX])
    ax.set_ylim([fit[:,pcY].min()-bufferY,fit[:,pcY].max()+bufferY])
    ax.set_xlabel("D-%s"%str(pcX+1),fontsize=font_size,fontname=font_name)
    ax.set_ylabel("D-%s"%str(pcY+1),fontsize=font_size,fontname=font_name)
    plt.locator_params(axis='x',nbins=5)

def get_top_words(model, feature_names, n_top_words):
    top_words = {}
    for topic_idx, topic in enumerate(model.components_):
        _top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        top_words[str(topic_idx)] = _top_words
    return(top_words)

def get_sentiment(word):
    mean_pos = 0
    mean_neg = 0
    mean_obj = 0
    score = 0
    count = 0
    for similar_words in swn.senti_synsets(word):
        mean_pos += similar_words.pos_score()
        mean_neg += similar_words.neg_score()
        mean_obj += similar_words.obj_score()
    size = len(list(swn.senti_synsets(word)))
    if size != 0:
        mean_pos = mean_pos/size
        mean_neg = mean_neg/size
        mean_obj = mean_obj/size
        count += 1
        score += (mean_pos - mean_neg)*(1-mean_obj)
    return mean_pos, mean_neg, mean_obj

def sentiment_wordnet(content, topics_mat, tf_feature_names):
    relevant_types = ['JJ', 'VB', 'RB'] #adjectives, verbs, adverbs
    sentiment = {topic: (0, 0, 0, 0) for topic in range(topics_mat.shape[0])}
    for topic in range(topics_mat.shape[0]):
        s_pos = 0
        s_neg = 0
        s_obj = 0
        score = 0
        relevant_word_count = 0
        print('Working on topic: '+str(topic))
        for article in content:
            for word, word_type in nltk.pos_tag(article.split(' ')):
                if word_type in relevant_types and word in tf_feature_names:
                    try:
                        idx = list(tf_feature_names).index(word)
                        prob = topics_mat[topic, idx]
                        relevant_word_count += 1
                        pos, neg, obj = get_sentiment(word)
                        s_pos += pos
                        s_neg += neg
                        s_obj += obj
                        score = score + ((pos - neg) * (1 - obj) * prob) #weight subjective words
                    except:
                        print('Problem getting word probability!')
                        import pdb; pdb.set_trace()

        if relevant_word_count != 0:
            sentiment[topic] = (s_pos/relevant_word_count, s_neg/relevant_word_count, s_obj/relevant_word_count, score)
    return sentiment


print('Starting LDA...')
# df = pd.read_csv('npr_articles.csv', parse_dates=['date_published'])
df = pd.read_csv('../data/rss_feeds_new_data.csv')#, parse_dates=['date_published'])
df_no_nan = df[pd.notnull(df['processed_text'])]
text = df_no_nan['processed_text'].values.tolist()

max_features = 1000
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=max_features,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(text)

n_topics = 8
lda_model = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                      learning_method='online',
                                      learning_offset=50.,
                                      random_state=0)

lda_model.fit(tf)
vis_data = pyLDAvis.sklearn.prepare(lda_model,tf, tf_vectorizer, R=n_topics, n_jobs=-1)
pyLDAvis.show(vis_data)

filename = '../pickles/all_words_data_model.pkl'
pickle.dump(lda_model, open(filename, 'wb'), protocol=2)

# filename = 'wsj_articles_data_tf.pickle'
# pickle.dump(tf, open(filename, 'wb'))
#
filename = '../pickles/all_words_tf_vectorizer.pkl'
pickle.dump(tf_vectorizer, open(filename, 'wb'), protocol=2)



## get the token to topic matrix
word_topic = np.zeros((max_features,n_topics),)
print(n_topics)
lda_model.components_
for topic_idx, topic in enumerate(lda_model.components_):
    word_topic[:,topic_idx] = topic

print("token-topic matrix",word_topic.shape)

## create a matrix of the top words used to define each topic
# top_words = 15
# tf_feature_names = np.array(tf_vectorizer.get_feature_names())
# top_words = get_top_words(lda_model,tf_feature_names,top_words)
#
# for key,vals in top_words.items():
#     print(key," ".join(vals))
# print("total words: %s"%len(all_top_words))
#
# top_word_inds = [np.where(tf_feature_names == tw)[0][0] for tw in all_top_words]


# fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111)

# mat = word_topic
# matScaled = preprocessing.scale(mat.T)
# pca_fit = PCA(n_components=2).fit_transform(matScaled)

# make_scatter(pca_fit,ax)
# plt.show()


url_names = ['cnn', 'abc', 'fox', 'nyt', 'reuters', 'wapo', 'huffpo', 'esquire', 'rollingstone', 'cbs', '538', 'washtimes']

tf_feature_names = np.array(tf_vectorizer.get_feature_names())
topics_mat = lda_model.components_


sentiment_by_topic = {site: 0 for site in url_names}
for i,site in enumerate(url_names):
    print('Working on site: '+site)
    sentiment_by_topic[site] = sentiment_wordnet(df_no_nan.loc[df_no_nan['source'] == site]['processed_text'], topics_mat, tf_feature_names)

with open('../pickles/sentiment_by_topic_20_all_words.pkl', 'wb') as f:
    pickle.dump(sentiment_by_topic, f, pickle.HIGHEST_PROTOCOL)

bias = [0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
for i in range(len(sentiment_by_topic.keys())):
    plt.subplot(4,5,i+1)
    score = []
    for topic in range(topics_mat.shape[0]):
        score.append(sentiment_by_topic[url_names[i]][topic][0])
    # score = np.array(score)
    # score /= sum(np.abs(score))
    plt.bar(np.arange(len(score)), score, align='center')
    plt.ylabel('Score')
    plt.title('Score by Topic for '+url_names[i])
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()
