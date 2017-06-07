import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/IPython/.*')
import pickle

import numpy as np
import pandas as pd
import json

import nltk
from nltk.corpus import sentiwordnet as swn




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


def sentiment_of_words_wordnet(df):
    df = df[pd.notnull(df['processed_text'])]
    articles = df['processed_text'].values.tolist()
    relevant_types = ['JJ', 'VB', 'RB'] #adjectives, verbs, adverbs
    sentiment_of_words = []
    for i, article in enumerate(articles):
        sentiment_of_words.append(dict())
        try:
            for word, word_type in nltk.pos_tag(article.split(' ')):
                if word_type in relevant_types:
                    try:
                        pos, neg, obj = get_sentiment(word)
                        score = (pos - neg) * (1 - obj) #weight subjective words
                        if pos == 0 and neg == 0 and obj == 0:
                            pass
                        else:
                            sentiment_of_words[i][word] = [pos, neg, obj, score]
                    except:
                        print('Problem getting sentiment!')
        except:
            import pdb; pdb.set_trace()
    df['sentiment_of_words'] = sentiment_of_words

    return sentiment_of_words

def sentiment_by_topic_wordnet(df, topics_mat, feature_names):
    site_names = df['source'].unique()
    sentiment_by_topic = {site: 0 for site in site_names}
    for site in site_names:
        print('Working on site: '+site)
        site_df = df[df['source'] == site]
        sentiment = {topic: (0, 0, 0, 0) for topic in range(topics_mat.shape[0])}
        for topic in range(topics_mat.shape[0]):
            print('Working on topic: '+str(topic))
            s_pos = 0
            s_neg = 0
            s_obj = 0
            s_score = 0
            relevant_word_count = 0
            for sentiment_dict in site_df['sentiment_of_words']:
                for word in sentiment_dict.keys():
                    if word in feature_names:
                        try:
                            idx = list(feature_names).index(word)
                            prob = topics_mat[topic, idx]
                            relevant_word_count += 1
                            pos, neg, obj, score = sentiment_dict[word]
                            s_pos += pos
                            s_neg += neg
                            s_obj += obj
                            s_score += score * prob #weight word depending on prob for that topic
                        except:
                            import pdb; pdb.set_trace()
                            print('Problem getting word probability!')

            if relevant_word_count != 0:
                sentiment[topic] = (s_pos/relevant_word_count, s_neg/relevant_word_count, s_obj/relevant_word_count, s_score/relevant_word_count)
        sentiment_by_topic[site] = sentiment
    return sentiment_by_topic

def sentiment_by_article_wordnet(sentiment_of_words, topics_mat, feature_names, topics):
    # new_topics_mat = topics_mat[topics, :]
    sentiment_by_topic = {topic: (0, 0, 0, 0) for topic in topics}

    for topic in topics:
        s_pos = 0
        s_neg = 0
        s_obj = 0
        s_score = 0
        relevant_word_count = 0
        for word in sentiment_of_words.keys():
            if word in feature_names:
                try:
                    idx = list(feature_names).index(word)
                    prob = topics_mat[topic, idx]
                    relevant_word_count += 1
                    pos, neg, obj, score = sentiment_of_words[word]
                    s_pos += pos * prob
                    s_neg += neg * prob
                    s_obj += obj * prob
                    s_score += score * prob #weight word depending on prob for that topic
                except:
                    import pdb; pdb.set_trace()
                    print('Problem getting word probability!')

        if relevant_word_count != 0:
            sentiment_by_topic[topic] = (s_pos/relevant_word_count, s_neg/relevant_word_count, s_obj/relevant_word_count, s_score/relevant_word_count)

    return sentiment_by_topic


if __name__ == '__main__':
    df = pd.read_csv('../data/rss_feeds_new.csv')
    df = sentiment_of_words_wordnet(df)

    with open('../pickles/all_words_tf_vectorizer.pkl', 'rb') as f:
        tf_vectorizer = pickle.load(f)
    with open('../pickles/all_words_data_model.pkl', 'rb') as f:
        lda_model = pickle.load(f)

    feature_names = np.array(tf_vectorizer.get_feature_names())
    topics_mat = lda_model.components_

    sentiment_by_topic = sentiment_by_article_wordnet(df, topics_mat, feature_names, [0])
