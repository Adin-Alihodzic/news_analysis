from textblob import TextBlob
import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pattern.en import parse, split, wordnet #must have sentiwordnet available
import nltk
from nltk.corpus import sentiwordnet as swn




from nltk.stem.wordnet import WordNetLemmatizer
wnl = WordNetLemmatizer()
def sentiment_wordnet(content):
    wordnet.sentiment.load()
    relevant_types = ['JJ', 'VB', 'RB'] #adjectives, verbs, adverbs
    s_pos = 0
    s_neg = 0
    s_obj = 0
    score = 0
    sentiment = (0, 0, 0, 0)
    sentences = split(parse(content, lemmata=True))
    relevant_word_count = 0
    for sentence in sentences:
        for word in sentence.words:
            if word.type in relevant_types:
                relevant_word_count += 1
                pos, neg, obj = wordnet.sentiment[wordnet.lemmatize(word)]
                s_pos += pos
                s_neg += neg
                s_obj += obj
                score = score + ((pos - neg) * (1 - obj)) #weight subjective words
    sentiment = (s_pos/relevant_word_count, s_neg/relevant_word_count, s_obj/relevant_word_count, score)
    return sentiment

import sys

reload(sys)
sys.setdefaultencoding('utf8')

def sentiment_wordnet2(content, top_words, word_probs):
    relevant_types = ['JJ', 'VB', 'RB'] #adjectives, verbs, adverbs
    sentiment = {topic: (0, 0, 0, 0) for topic in top_words.keys()}
    for topic in top_words.keys():
        s_pos = 0
        s_neg = 0
        s_obj = 0
        score = 0
        relevant_word_count = 0
        print('Working on topic: '+topic)
        for article in content:
            for word, word_type in nltk.pos_tag(article.split(' ')):
                if word_type in relevant_types and word in top_words[topic]:
                    try:
                        idx = list(top_words[topic]).index(word)
                        prob = word_probs[topic][idx]
                        relevant_word_count += 1
                        pos, neg, obj = wordnet.sentiment[wnl.lemmatize(word)]
                        s_pos += pos
                        s_neg += neg
                        s_obj += obj
                        score = score + ((pos - neg) * (1 - obj) * prob) #weight subjective words and prob of word
                    except:
                        import pdb; pdb.set_trace()
                        print('Problem getting word probability!')

        if relevant_word_count != 0:
            sentiment[topic] = (s_pos/relevant_word_count, s_neg/relevant_word_count, s_obj/relevant_word_count, score)
    return sentiment

df = pd.read_csv('data.csv')#, parse_dates=['date_published'])
df_no_nan = df[pd.notnull(df['processed_text'])]

# with open('pickles/all_articles.pkl', 'rb') as f:
#     articles_dict = pickle.load(f)k

with open('pickles/all_words_tf_vectorizer.pkl', 'rb') as f:
    tf_vectorizer = pickle.load(f)
with open('pickles/all_words_data_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)
# all_words = [key for key in tf_vectorizer.vocabulary_.keys()]
tf_feature_names = np.array(tf_vectorizer.get_feature_names())
all_words = {str(key): tf_feature_names for key in range(20)}
comps = lda_model.components_
word_probs = {str(key): comps[i,:]/max(comps[i,:]) for i,key in enumerate(range(20))}

url_names = ['cnn', 'abc', 'fox', 'nyt', 'ap', 'reuters', 'wapo', 'economist', 'huffpo', 'esquire', 'rollingstone', 'cbs', '538', 'vox', 'time', 'slate', 'washtimes']
url_names_missing = ['infowars', 'dailybeast']

sentiment_by_topic = {site: 0 for site in url_names}
for i,site in enumerate(url_names):
    print('Working on site: '+site)
    sentiment_by_topic[site] = sentiment_wordnet2(df_no_nan.loc[df_no_nan['source'] == site]['processed_text'], all_words, word_probs)

with open('pickles/sentiment_by_topic_pattern_20_all_words.pkl', 'wb') as f:
    pickle.dump(sentiment_by_topic, f, pickle.HIGHEST_PROTOCOL)

bias = [0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
score = {key: [] for key in sentiment_by_topic.keys()}
for i, key in enumerate(sentiment_by_topic.keys()):
    plt.subplot(4,5,i+1)
    score[key] = []
    for j, topic in enumerate(all_words.keys()):
        score[key].append(sentiment_by_topic[url_names[i]][topic][3]*bias[j])
    score[key] = np.array(score[key])
    score[key] /= sum(np.abs(score[key]))
    plt.bar(np.arange(len(score[key])), score[key], align='center')
    plt.ylabel('Score')
    plt.title('Score by Topic for '+url_names[i])
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()






# use_textblob = False
#
# sentiment_dict = dict({site: dict() for site in articles_dict.keys()})
# for site in articles_dict.keys():
# # for site in ['fox']:
#     print('Working on: '+site)
#     sentiment = dict()
#     for url in articles_dict[site].keys():
#         article = articles_dict[site][url]['article_text']
#
#         if use_textblob:
#             tb = TextBlob(article)
#             # first entry is sentiment of entire article
#             sentiment[url] = [tb.sentiment]
#             for sentence in tb.sentences:
#                 sentiment[url].append(sentence.sentiment)
#         else:
#             sentiment[url] = [sentiment_wordnet(article)]
#     sentiment_dict[site] = sentiment
#
# polarity_by_url = dict({site: [] for site in articles_dict.keys()})
# sentiment_by_url = dict({site: [] for site in articles_dict.keys()})
# for site in articles_dict.keys():
#     for url in sentiment_dict[site].keys():
#         polarity_by_url[site].append(sentiment_dict[site][url][1][0])
#         sentiment_by_url[site].append(sentiment_dict[site][url][1][1])
#
#
# for site in articles_dict.keys():
#     plt.hist(polarity_by_url[site], bins, alpha=0.5, label=site, normed=True)
# plt.xlim(-0.1,1.1)
# plt.legend(loc='upper right')
# plt.show()
