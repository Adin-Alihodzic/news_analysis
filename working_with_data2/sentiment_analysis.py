import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import sentiwordnet as swn

from data_exploration import process_articles, article_topics_and_topic_coverage

def get_sentiment(word):
    mean_pos = 0
    mean_neg = 0
    mean_obj = 0
    score = 0
    bias = 0
    for similar_words in swn.senti_synsets(word):
        mean_pos += similar_words.pos_score()
        mean_neg += similar_words.neg_score()
        mean_obj += similar_words.obj_score()
    size = len(list(swn.senti_synsets(word)))
    if size != 0:
        mean_pos = mean_pos/size
        mean_neg = mean_neg/size
        mean_obj = mean_obj/size
        score = (mean_pos - mean_neg)*(1-mean_obj)
        bias = (mean_pos + mean_neg) * (1-mean_obj)
    return mean_pos, mean_neg, mean_obj, score, bias

def word_sentiment(lda_model, sentiment_texts):
    lda_topics = lda_model.show_topics(num_topics=-1, num_words=10000,formatted=False)

    # We only want adjectives, verbs, adverbs
    relevant_types = ['JJ', 'VB', 'RB']

    # Get all the unique words found in each topic
    topic_words = []
    for word_and_prob in lda_topics[0][1]:
        word = word_and_prob[0]
        for word, word_type in nltk.pos_tag([word]):
            if word_type in relevant_types:
                topic_words.append(word)

    # Get list of unique words found in sentiment texts created in data_exploration.py
    sentiment_texts_words = set()
    for i in range(len(sentiment_texts)):
        sentiment_texts_words = sentiment_texts_words | set(sentiment_texts[i])
    sentiment_texts_words = list(sentiment_texts_words)

    # Get the sentiment for all words from sentiment_texts
    sentiment_of_words = dict()
    for word in sentiment_texts_words:
        if word in topic_words:
            pos, neg, obj, score, bias = get_sentiment(word)
            if pos == 0 and neg == 0:
                pass
            else:
                sentiment_of_words[word] = [pos, neg, obj, score, bias]

    return sentiment_of_words

def article_sentiment(article, sentiment_of_words):
    s_pos = 0
    s_neg = 0
    s_obj = 0
    relevant_word_count = 0
    for word in article:
        if word in sentiment_of_words.keys():
            relevant_word_count += 1
            pos, neg, obj, score, bias = sentiment_of_words[word]
            s_pos += pos
            s_neg += neg
            s_obj += obj
    if relevant_word_count != 0:
         s_pos, s_neg, s_obj = s_pos/relevant_word_count, s_neg/relevant_word_count, s_obj/relevant_word_count

    return s_pos, s_neg, s_obj


def topic_values(df, lda_model):
    print('Processing Articles...')
    topic_texts, sentiment_texts = process_articles(df)
    
    num_topics = lda_model.num_topics

    print('Getting Sentiment...')
    sentiment_of_words = word_sentiment(lda_model, sentiment_texts)

    print('Getting article topics...')
    all_article_topics = article_topics_and_topic_coverage(lda_model, topic_texts, tokenized=False)

    print('Creating Dictionary...')
    topic_dict = {topic: {'pos': [], 'neg': [], 'obj': [], 'topic_prob': [], 'url': [], 'source': [], 'headline': []} for topic in range(num_topics+1)}
    for i in range(len(sentiment_texts)):
        # Make sure article is long enough
        if len(sentiment_texts[i]) > 100:
            pos, neg, obj = article_sentiment(sentiment_texts[i], sentiment_of_words)
            # Topic 0 includes all articles
            topic_dict[0]['pos'].append(pos)
            topic_dict[0]['neg'].append(neg)
            topic_dict[0]['obj'].append(neg)
            topic_dict[0]['topic_prob'].append(0.5)
            topic_dict[0]['url'].append(df['url'][i])
            topic_dict[0]['source'].append(df['source'][i])
            topic_dict[0]['headline'].append(df['headline'][i])

            for topic_and_prob in all_article_topics[i]:
                topic = topic_and_prob[0] + 1
                prob = topic_and_prob[1]
                if prob > 0.20:
                    topic_dict[topic]['pos'].append(pos)
                    topic_dict[topic]['neg'].append(neg)
                    topic_dict[topic]['obj'].append(neg)
                    topic_dict[topic]['topic_prob'].append(prob)
                    topic_dict[topic]['url'].append(df['url'][i])
                    topic_dict[topic]['source'].append(df['source'][i])
                    topic_dict[topic]['headline'].append(df['headline'][i])

    return topic_dict



if __name__ == '__main__':
    df = pd.read_csv('../data/rss_feeds_new_good.csv')
    df = df[pd.notnull(df['article_text'])]

    # print('Processing Articles...')
    # topic_texts, sentiment_texts = process_articles(df)

    print('Getting LDA model...')
    with open('../working_with_data/lda_model.pkl', 'rb') as f:
        lda_model = pickle.load(f)


    lda_topics = lda_model.show_topics(num_topics=-1, num_words=10000,formatted=False)

    # def wrapper_get_sentiment_by_article(article_topic):
    #     return get_sentiment_by_article(lda_topics, sentiment_of_words, article_topic)

    # with open('sentiment_by_article.pkl', 'rb') as f:
    #     sentiment_by_article = pickle.load(f)

    # print('Getting Sentiment by Article using MutiProcessing...')
    # pool = mp.Pool(50)
    # test = pool.map(wrapper_get_sentiment_by_article, all_article_topics[:100])


    # pickle.dump(sentiment_by_article, open('sentiment_by_article.pkl', 'wb'))

    topic_dict = topic_values(df, lda_model)
    plt.scatter(topic_dict[5]['pos'], topic_dict[5]['neg'])
    plt.show()
