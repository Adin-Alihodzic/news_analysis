import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import sentiwordnet as swn

# from data_exploration import process_articles, article_topics_and_topic_coverage

def get_sentiment(word):
    '''
    Determines sentiment of word by first finding all similar words
    then returns the mean of all words sentiment
    '''
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
    return mean_pos, mean_neg, mean_obj

def word_sentiment(lda_model, sentiment_texts):
    '''
    Creates dictionary of words with their sentiments.
    Only considers words from sentiment_texts.
    Also only considers adjectives, verbs and adverbs.
    '''
    lda_topics = lda_model.show_topics(num_topics=-1, num_words=10000,formatted=False)

    # We only want adjectives, verbs, adverbs
    relevant_types = ['JJ', 'VB', 'RB']

    # Get all the unique words found in each topic
    topic_words = []
    for word_and_prob in lda_topics[0][1]:
        word = word_and_prob[0]
        for word, word_type in nltk.pos_tag([word]):
            for type in relevant_types:
                if word_type.startswith(type):
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
            pos, neg, obj = get_sentiment(word)
            if pos == 0 and neg == 0:
                pass
            else:
                sentiment_of_words[word] = [pos, neg, obj]

    return sentiment_of_words

def article_sentiment(article, sentiment_of_words):
    '''
    Determines the Positive and Negative sentiment in an article, along with the Objectivity.
    Values determined from the SentiWordNet library.
    '''
    s_pos = 0
    s_neg = 0
    s_obj = 0
    relevant_word_count = 0
    for word in article:
        if word in sentiment_of_words.keys():
            relevant_word_count += 1
            pos, neg, obj = sentiment_of_words[word]
            s_pos += pos
            s_neg += neg
            s_obj += obj
    if relevant_word_count != 0:
         s_pos, s_neg, s_obj = s_pos/relevant_word_count, s_neg/relevant_word_count, s_obj/relevant_word_count

    return s_pos, s_neg, s_obj


def article_topics_and_topic_coverage(model, topic_texts, tokenized=False):
    """
    Function to get each articles top topics and how much a topic is covered.
    Following are the steps we take:

    1. Get Bag of Words representation depending on whether or not the input is tokenized
    2. Use model to get topics and probability of tht topic for each article
    3. Normalize
    4. Save coverage bar grah.

    Parameters:
    ----------
    tokenized: Whether the topic_texts is tokenized

    Returns:
    -------
    all_article_topics: list containing a dictionary of topic with probabilty article belongs in that topic.
    """
    all_article_topics = []
    topic_coverage = [0 for topic in range(model.num_topics)]
    for article in topic_texts:
        if tokenized:
            article_bow = model.id2word.doc2bow([word for word in article])
        else:
            article_bow = model.id2word.doc2bow(article)
        # Following gives a dictionary where key is topic and value is probability article belongs in that topic
        article_topics = model[article_bow]
        all_article_topics.append(article_topics)

        for coverage in article_topics:
            topic_coverage[coverage[0]] += coverage[1]

    # Normalize
    topic_coverage = topic_coverage/sum(topic_coverage)

    fig = plt.figure(figsize=(16,12), dpi=300)
    plt.bar(range(model.num_topics), topic_coverage)
    plt.xlabel('Topic')
    plt.ylabel('Coverage')
    plt.title('Coverage by Topic')
    # plt.savefig('../web_app/static/img/topic_coverage.png')

    return all_article_topics, fig

def topic_values(df, topic_texts, sentiment_texts, lda_model):
    """
    Function to get dictionary containing all elements of a topic
    Following are the steps we take:

    1. Get the sentiment of all words in sentiment_texts.
    2. Use function above to get topics for each article.
    3. Create dictionary for topics with values we will need later.

    Returns:
    -------
    topic_dict
    """
    num_topics = lda_model.num_topics

    print('Getting Sentiment...')
    sentiment_of_words = word_sentiment(lda_model, sentiment_texts)

    print('Getting article topics...')
    all_article_topics, fig = article_topics_and_topic_coverage(lda_model, topic_texts, tokenized=False)

    print('Creating Dictionary...')
    topic_dict = {topic: {'pos': [], 'neg': [], 'obj': [], 'topic_prob': [], 'url': [], 'source': [], 'headline': [], 'tones': [], 'length': []} for topic in range(num_topics+1)}

    # def fill_topic_dict(i):
    for i in range(len(sentiment_texts)):
        pos, neg, obj = article_sentiment(sentiment_texts[i], sentiment_of_words)
        # Topic 0 includes all articles
        topic_dict[0]['pos'].append(pos)
        topic_dict[0]['neg'].append(neg)
        topic_dict[0]['obj'].append(obj)
        topic_dict[0]['topic_prob'].append(0.5)
        topic_dict[0]['url'].append(df['url'][i])
        topic_dict[0]['source'].append(df['source'][i])
        topic_dict[0]['headline'].append(df['headline'][i])
        topic_dict[0]['tones'].append(df['tones'][i])
        topic_dict[0]['length'].append(len(sentiment_texts[i]))

        for topic_and_prob in all_article_topics[i]:
            topic = topic_and_prob[0] + 1
            prob = topic_and_prob[1]

            topic_dict[topic]['pos'].append(pos)
            topic_dict[topic]['neg'].append(neg)
            topic_dict[topic]['obj'].append(obj)
            topic_dict[topic]['topic_prob'].append(prob)
            topic_dict[topic]['url'].append(df['url'][i])
            topic_dict[topic]['source'].append(df['source'][i])
            topic_dict[topic]['headline'].append(df['headline'][i])
            topic_dict[topic]['tones'].append(df['tones'][i])
            topic_dict[topic]['length'].append(len(sentiment_texts[i]))

    # pool = mp.Pool(50)
    # pool.map(fill_topic_dict, list(range(len(sentiment_texts))))

    return topic_dict, all_article_topics, sentiment_of_words, fig



if __name__ == '__main__':
    df = pd.read_csv('../data/rss_feeds_new_good.csv')
    df = df[pd.notnull(df['article_text'])]

    # print('Processing Articles...')
    # topic_texts, sentiment_texts = process_articles(df)

    print('Getting LDA model...')
    with open('../pickles/lda_model.pkl', 'rb') as f:
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

    print('Processing Articles...')
    topic_texts, sentiment_texts, quote_texts, tweet_texts = process_articles(df)

    topic_dict, all_article_topics, sentiment_of_words, fig = topic_values(df, topic_texts, sentiment_texts, lda_model)
    plt.scatter(topic_dict[5]['pos'], topic_dict[5]['neg'])
    plt.show()
