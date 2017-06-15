import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import sentiwordnet as swn


# BEGIN of python-dotenv section
from os.path import join, dirname
from dotenv import load_dotenv
import os

from watson_developer_cloud import ToneAnalyzerV3

dotenv_path = join(dirname('__file__'), '.env')
load_dotenv(dotenv_path)

tone_analyzer = ToneAnalyzerV3(
   username=os.environ.get("TONE_USERNAME"),
   password=os.environ.get("TONE_PASSWORD"),
   version='2016-05-19')

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
    topic_dict = {topic: {'pos': [], 'neg': [], 'obj': [], 'topic_prob': [], 'url': [], 'source': [], 'headline': [], 'Anger': [], 'Fear': [], 'Disgust': [], 'Joy': [], 'Sadness': [], 'Analytical': [], 'length': [], 'date_published': []} for topic in range(num_topics+1)}

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
        topic_dict[0]['Anger'].append(df['Anger'][i])
        topic_dict[0]['Fear'].append(df['Fear'][i])
        topic_dict[0]['Disgust'].append(df['Disgust'][i])
        topic_dict[0]['Joy'].append(df['Joy'][i])
        topic_dict[0]['Sadness'].append(df['Sadness'][i])
        topic_dict[0]['Analytical'].append(df['Analytical'][i])
        topic_dict[0]['date_published'].append(df['date_published'][i])
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
            topic_dict[topic]['Anger'].append(df['Anger'][i])
            topic_dict[topic]['Fear'].append(df['Fear'][i])
            topic_dict[topic]['Disgust'].append(df['Disgust'][i])
            topic_dict[topic]['Joy'].append(df['Joy'][i])
            topic_dict[topic]['Sadness'].append(df['Sadness'][i])
            topic_dict[topic]['Analytical'].append(df['Analytical'][i])
            topic_dict[topic]['date_published'].append(df['date_published'][i])
            topic_dict[topic]['length'].append(len(sentiment_texts[i]))

    # pool = mp.Pool(50)
    # pool.map(fill_topic_dict, list(range(len(sentiment_texts))))

    return topic_dict, all_article_topics, sentiment_of_words, fig


def parse_toneanalyzer_response(json_data):
    """Parses the JSON response from ToneAnalyzer to return
    a dictionary of emotions and their corresponding score.

    Parameters
    ----------
    json_data: {dict} a json response from ToneAnalyzer (see Notes)

    Returns
    -------
    dict : a {dict} whose keys are emotion ids and values are their corresponding score.
    """
    emotion_tones = {}
    for entry in json_data['document_tone']['tone_categories']:
        if entry['category_id'] == 'emotion_tone':
            for emotion in entry['tones']:
                emotion_key = emotion['tone_name']
                emotion_value = emotion['score']
                emotion_tones[emotion_key] = emotion_value

    language_tones = {}
    for entry in json_data['document_tone']['tone_categories']:
        if entry['category_id'] == 'language_tone':
            for language in entry['tones']:
                language_key = language['tone_name']
                language_value = language['score']
                language_tones[language_key] = language_value

    social_tones = {}
    for entry in json_data['document_tone']['tone_categories']:
        if entry['category_id'] == 'social_tone':
            for social in entry['tones']:
                social_key = social['tone_name']
                social_value = social['score']
                social_tones[social_key] = social_value

    return emotion_tones, language_tones, social_tones

def get_new_tones(df, prev_df):
    '''Used when you've already got some tones for df'''
    anger = []
    disgust = []
    fear = []
    joy = []
    sadness = []
    analytical = []
    confident = []
    tentative = []
    openness = []
    conscientiousness = []
    extraversion = []
    agreeableness = []
    emotional_range = []
    for url, sentiment_texts in zip(df['url'], df['sentiment_texts']):
        tone_idx = [i for i in prev_df[prev_df['url'] == url].index]
        if tone_idx != []:
            tone_idx = tone_idx[0]
            anger.append(prev_df['Anger'][tone_idx])
            disgust.append(prev_df['Disgust'][tone_idx])
            fear.append(prev_df['Fear'][tone_idx])
            joy.append(prev_df['Joy'][tone_idx])
            sadness.append(prev_df['Sadness'][tone_idx])
            analytical.append(prev_df['Analytical'][tone_idx])
            confident.append(prev_df['Confident'][tone_idx])
            tentative.append(prev_df['Tentative'][tone_idx])
            openness.append(prev_df['Openness'][tone_idx])
            conscientiousness.append(prev_df['Conscientiousness'][tone_idx])
            extraversion.append(prev_df['Extraversion'][tone_idx])
            agreeableness.append(prev_df['Agreeableness'][tone_idx])
            emotional_range.append(prev_df['Emotional Range'][tone_idx])
        else:
            try:
                json_response_sentiment = tone_analyzer.tone(text=' '.join(sentiment_texts), sentences=False)
                temp = parse_toneanalyzer_response(json_response_sentiment)
                anger.append(temp[0]['Anger'])
                disgust.append(temp[0]['Disgust'])
                fear.append(temp[0]['Fear'])
                joy.append(temp[0]['Joy'])
                sadness.append(temp[0]['Sadness'])
                analytical.append(temp[1]['Analytical'])
                confident.append(temp[1]['Confident'])
                tentative.append(temp[1]['Tentative'])
                openness.append(temp[2]['Openness'])
                conscientiousness.append(temp[2]['Conscientiousness'])
                extraversion.append(temp[2]['Extraversion'])
                agreeableness.append(temp[2]['Agreeableness'])
                emotional_range.append(temp[2]['Emotional Range'])
            except:
                print('API not working')
                anger.append(np.nan)
                disgust.append(np.nan)
                fear.append(np.nan)
                joy.append(np.nan)
                sadness.append(np.nan)
                analytical.append(np.nan)
                confident.append(np.nan)
                tentative.append(np.nan)
                openness.append(np.nan)
                conscientiousness.append(np.nan)
                extraversion.append(np.nan)
                agreeableness.append(np.nan)
                emotional_range.append(np.nan)


    df['Anger'] = anger
    df['Fear'] = fear
    df['Disgust'] = disgust
    df['Joy'] = joy
    df['Sadness'] = sadness
    df['Analytical'] = analytical
    df['Confident'] = confident
    df['Tentative'] = tentative
    df['Openness'] = openness
    df['Conscientiousness'] = conscientiousness
    df['Extraversion'] = extraversion
    df['Agreeableness'] = agreeableness
    df['Emotional Range'] = emotional_range

    return df



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
