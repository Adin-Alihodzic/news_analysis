import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re
import ast

from gensim.summarization import summarize
from gensim.summarization import keywords

import warnings
# Gensim gives annoying warnings
warnings.filterwarnings('ignore')

import gensim
from gensim.models import LdaModel, HdpModel
from gensim.corpora import Dictionary
import ast

import pyLDAvis.gensim

def all_length_hist(df, topic_texts, sentiment_texts, quote_texts, tweet_texts):
    """
    Function to get make histograms from texts.

    Parameters:
    ----------
    df, topic_texts, sentiment_texts, quote_texts, tweet_texts: dataframe and texts in string (non-tokenized) form.

    Returns:
    -------
    fig1, fig2: histogram of topic and sentiment texts. historgram of quotes.
    """
    site_word_length = {source: {'topic': [], 'sentiment': [], 'quote': [], 'tweet': []} for source in df['source'].unique()}
    for source in df['source'].unique():
        new_df = df[df['source'] == source]
        for i in new_df.index.tolist():
            site_word_length[source]['topic'].append(len(topic_texts[i]))
            site_word_length[source]['sentiment'].append(len(sentiment_texts[i]))
            if type(quote_texts[i]) != float:
                site_word_length[source]['quote'].append(len(quote_texts[i]))
            else:
                site_word_length[source]['quote'].append(0)
            if type(tweet_texts[i]) != float:
                site_word_length[source]['tweet'].append(len(tweet_texts[i]))
            else:
                site_word_length[source]['tweet'].append(0)

    # import pdb; pdb.set_trace()
    fig1 = plt.figure(figsize=(16,12), dpi=300)
    for i, source in enumerate(site_word_length.keys()):
        plt.subplot(4,3,i+1)
        plt.hist(site_word_length[source]['topic'], normed=True, alpha=0.5, bins=100, label='topic words')
        plt.hist(site_word_length[source]['sentiment'], normed=True, alpha=0.5, bins=100, label='sentiment words')
        plt.title('Length of Sentiment and Topic words for '+source)
        plt.legend(loc='upper right')
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    fig2 = plt.figure(figsize=(16,12), dpi=300)
    for i, source in enumerate(site_word_length.keys()):
        plt.subplot(4,3,i+1)
        plt.hist(site_word_length[source]['quote'][1:], normed=True, alpha=0.5, bins=30, label='quote words')
        # plt.hist(site_word_length[source]['tweet'][1:], normed=True, alpha=0.5, bins=30, label='tweet words')
        plt.title('Length of Quote and Tweet words for '+source)
        plt.legend(loc='upper right')
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    return fig1, fig2

def article_length_hist(df):
    '''Gets article lengths by site and returns figure'''
    site_article_length = {source: [] for source in df['source'].unique()}
    for source in df['source'].unique():
        new_df = df[df['source'] == source]
        for article in new_df['article_text']:
            site_article_length[source].append(len(article.split()))

    fig = plt.figure(figsize=(16,12), dpi=300)
    for i, source in enumerate(site_article_length.keys()):
        plt.subplot(4,3,i+1)
        plt.hist(site_article_length[source], normed=True, bins=100)
        plt.title('Article Length '+source)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    return fig

def quote_length_hist(df):
    '''Gets article lengths by site and returns figure'''
    site_article_length = {source: [] for source in df['source'].unique()}
    for source in df['source'].unique():
        new_df = df[df['source'] == source]
        for article in new_df['article_text']:
            site_article_length[source].append(len(article.split()))

    fig = plt.figure()
    for i, source in enumerate(site_article_length.keys()):
        plt.subplot(4,3,i+1)
        plt.hist(site_article_length[source], normed=True, bins=100)
        plt.title('Article Length '+source)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    return fig

def tweet_length_hist(df):
    '''Gets article lengths by site and returns figure'''
    site_article_length = {source: [] for source in df['source'].unique()}
    for source in df['source'].unique():
        new_df = df[df['source'] == source]
        for article in new_df['article_text']:
            site_article_length[source].append(len(article.split()))

    fig = plt.figure(figsize=(16,12), dpi=300)
    for i, source in enumerate(site_article_length.keys()):
        plt.subplot(4,3,i+1)
        plt.hist(site_article_length[source], normed=True, bins=100)
        plt.title('Article Length '+source)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    return fig

def relevant_words_hist(df, topic_texts, sentiment_texts, quote_texts, tweet_texts):
    """
    Function to get make histograms from texts.

    Parameters:
    ----------
    df, topic_texts, sentiment_texts, quote_texts, tweet_texts: dataframe and texts in string (non-tokenized) form.

    Returns:
    -------
    fig1, fig2: histogram of topic and sentiment texts. historgram of quotes.
    """
    relevant_types = ['JJ', 'VB', 'RB']
    counts = {source: [0,0,0,0] for source in df['source'].unique()}

    for source in df['source'].unique():
        new_df = df[df['source'] == source]
        for i in new_df.index.tolist():
            counts[source][3] += len(topic_texts[i].split(' '))
            for word, word_type in nltk.pos_tag(topic_texts[i].split(' ')):
                if word_type.startswith(relevant_types[0]):
                    counts[source][0] += 1
                if word_type.startswith(relevant_types[1]):
                    counts[source][1] += 1
                if word_type.startswith(relevant_types[2]):
                    counts[source][2] += 1


    # import pdb; pdb.set_trace()
    fig = plt.figure(figsize=(16,12), dpi=300)
    for i, source in enumerate(counts.keys()):
        ind = np.arange(3)  # the x locations for the groups
        plt.subplot(4,3,i+1)
        plt.bar(ind, np.array(counts[source])[:3]/counts[source][3], label='counts')
        plt.title('Length of Sentiment and Topic words for '+source)
        plt.legend(loc='upper right')
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    return fig

def coverage_by_site_hist(model, df):
    fig = []
    for source in df['source'].unique():
        new_df = df[df['source'] == source]
        topic_coverage = [0 for topic in range(model.num_topics)]
        for i in new_df.index.tolist():
            for article in df['topic_texts']:
                article_bow = model.id2word.doc2bow([word for word in article])
                # Following gives a dictionary where key is topic and value is probability article belongs in that topic
                article_topics = model[article_bow]

                for coverage in article_topics:
                    topic_coverage[coverage[0]] += coverage[1]

        fig = plt.figure(figsize=(16,12), dpi=300)
        plt.bar(range(model.num_topics), topic_coverage)
        plt.xlabel('Topic')
        plt.ylabel('Coverage')
        plt.title('Coverage by Topic for '+source)

        figs.append(fig)

    return figs

def mood_plots(topic_dict):

    anger_tones = {topic: [] for topic in range(len(topic_dict))}
    disgust_tones = {topic: [] for topic in range(len(topic_dict))}
    fear_tones = {topic: [] for topic in range(len(topic_dict))}
    joy_tones = {topic: [] for topic in range(len(topic_dict))}
    sadness_tones = {topic: [] for topic in range(len(topic_dict))}
    analytical_score = {topic: [] for topic in range(len(topic_dict))}

    a = 1.0
    colors = [(1, 0, 0, a), (0, 1, 0, a), (128.0/255, 0, 128.0/255, a), (1, 1, 0, a), (0, 0 , 1, a), (0.5,0.5,0.5,a)]

    figs = []
    for topic in range(len(topic_dict)):
        for tone in topic_dict[topic]['tones']:
            tone = ast.literal_eval(tone)
            anger_tones[topic].append(tone[0]['Anger'])
            disgust_tones[topic].append(tone[0]['Disgust'])
            fear_tones[topic].append(tone[0]['Fear'])
            joy_tones[topic].append(tone[0]['Joy'])
            sadness_tones[topic].append(tone[0]['Sadness'])
            analytical_score[topic].append(tone[1]['Analytical'])

        idx = np.argsort([a+b+c+d+e for a,b,c,d,e in zip(anger_tones[topic],disgust_tones[topic],fear_tones[topic],joy_tones[topic],sadness_tones[topic])])
        sorted_anger_tones = np.array(anger_tones[topic])[idx]
        sorted_disgust_tones = np.array(disgust_tones[topic])[idx]
        sorted_fear_tones = np.array(fear_tones[topic])[idx]
        sorted_joy_tones = np.array(joy_tones[topic])[idx]
        sorted_sadness_tones = np.array(sadness_tones[topic])[idx]

        N = len(idx)
        ind = np.arange(N)    # the x locations for the groups
        width = 0.35       # the width of the bars: can also be len(x) sequence

        fig = plt.figure(figsize=(16,12), dpi=300)

        p1 = plt.bar(ind, sorted_sadness_tones, width, color=colors[4])
        p2 = plt.bar(ind, sorted_disgust_tones, width, color=colors[1], bottom=sorted_sadness_tones)
        p3 = plt.bar(ind, sorted_anger_tones, width, color=colors[0], bottom=sorted_sadness_tones+sorted_disgust_tones)
        p4 = plt.bar(ind, sorted_fear_tones, width, color=colors[2], bottom=sorted_sadness_tones+sorted_disgust_tones+sorted_anger_tones)
        p5 = plt.bar(ind, sorted_joy_tones, width, color=colors[3], bottom=sorted_sadness_tones+sorted_disgust_tones+sorted_anger_tones+sorted_fear_tones)

        plt.xlabel('Article')
        plt.ylabel('Scores')
        plt.title('Stacked Mood Scores by Article for Topic '+str(topic))
        plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('Sadness', 'Disgust', 'Anger', 'Fear', 'Joy'))
        figs.append(fig)

    return figs

def pos_neg_plot(topic_dict):
    a = 1.0
    colors = [(1, 0, 0, a), (0, 0, 1,a)]

    figs = []
    for topic in range(len(topic_dict)):
        N = len(topic_dict[topic]['pos'])
        ind = np.arange(N)    # the x locations for the groups
        width = 0.35       # the width of the bars: can also be len(x) sequence

        idx = np.argsort([p+n for p,n in zip(topic_dict[topic]['pos'], topic_dict[topic]['neg'])])
        sorted_pos = np.array(topic_dict[topic]['pos'])[idx]
        sorted_neg = -1*np.array(topic_dict[topic]['neg'])[idx]

        fig = plt.figure(figsize=(16,12), dpi=300)

        p1 = plt.bar(ind, sorted_pos, width, color='b')
        p2 = plt.bar(ind, sorted_neg, width, color='r')

        plt.xlabel('Article')
        plt.ylabel('Scores')
        plt.title('Stacked Positive/Negative Scores by Article for Topic '+str(topic))
        plt.legend((p1[0], p2[0]), ('Positive', 'Negative'))

        figs.append(fig)

    return figs

def get_summaries(df):
    '''Uses summarization function from gensim library to summarize each article'''
    summary = []
    for article in df['article_text']:
        try:
            summary.append(summarize(article))
        except:
            summary.append('')

    return summary


def dictionary_and_corpus(topic_texts, no_below=20, no_above=0.5):
    '''
    Function to get dictionary and corpus from texts.

    Parameters:
    ----------
    topic_text: Processed Tokenized Articles from process_articles.
    no_below, no_above: If integer - # of articles word must be in (below/above).
                        If Float   - % of articles word must be in (below/above).

    Returns:
    -------
    dictionary, corpus
    '''
    dictionary = Dictionary(topic_texts)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    corpus = [dictionary.doc2bow(text) for text in topic_texts]

    return dictionary, corpus

def topic_prob_extractor(hdp=None, topn=None):
    '''
    Function to find the weights of each topic.

    Parameters:
    ----------
    hdp, topn: Trained HDP model and the number of top words to consider.

    Returns:
    -------
    Dataframe containing topic words/word_probabilities and weights

    '''
    topic_list = hdp.show_topics(-1, topn)
    topics = [x[1] for x in topic_list]
    split_list = [x[1] for x in topic_list]
    weights = []
    words = []
    for lst in split_list:
        temp = [x.split('*') for x in lst.split(' + ')]
        weights.append([float(w[0]) for w in temp])
        words.append([w[0] for w in temp])
    sums = [np.sum(x) for x in weights]
    return pd.DataFrame({'topic_id' : topics, 'weight' : sums})


def run_lda(topic_texts, dictionary, corpus, topn=10000, num_topics=None, weight_threshold=0.25, K=15, T=150, passes=20, iterations=400):
    """
    Function to get LDA model. Following are the steps we take:

    1. Get HDP model
    2. Get topic weights
    3. Calculate # of topics if needed
    4. Get LDA model and save it
    5. Save pyLDAvis plot to web app

    Parameters:
    ----------
    topic_texts, dictionary, corpus: All determined from functions above.
    topn, num_topics: # Words to consider and number of topics.
    weight_threshold: If num_topics not given - Give threshold to determine topics.
    K=15, T=150: HDP hyperparameters
    passes:  How many times it calculates the LDA model
    iterations: How many times it iterates over the articles

    Returns:
    -------
    lda_model: Calculated LDA model
    """

    hdp_model = HdpModel(corpus=corpus, id2word=dictionary, chunksize=10000, K=K, T=T)

    topic_weights = topic_prob_extractor(hdp_model, topn=topn)['weight'].tolist()

    fig = plt.figure(figsize=(16,12), dpi=300)
    plt.plot(topic_weights)
    plt.title('Topic Probabilities')
    plt.xlabel('Topic')
    plt.ylabel('Probability')

    if num_topics == None:
        total_weight = sum(topic_weights)
        percent_of_weight = 0
        num_topics = 0
        while percent_of_weight < weight_threshold:
            num_topics += 1
            percent_of_weight = sum(topic_weights[:num_topics]) / total_weight

    num_topics = num_topics     # This can be entered manually or found from HDP
    chunksize = 10000   # How many articles can be used at once
    eval_every = None   # Don't evaluate model perplexity, takes too much time.

    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, chunksize=chunksize, \
                           alpha='auto', eta='auto', \
                           iterations=iterations, \
                           passes=passes, eval_every=eval_every)

    vis_data = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, R=20, sort_topics=False)

    return lda_model, vis_data, fig


if __name__ == '__main__':
    df = pd.read_csv('../data/rss_feeds_new_good_with_extra.csv')
    df = df[pd.notnull(df['article_text'])]

    article_length_hist(df)

    # print('Getting Summaries...')
    # summary = get_summaries(df)

    print('Getting Dictionary and Corpus...')
    dictionary, corpus = dictionary_and_corpus(topic_texts, no_below=20, no_above=0.5)

    print('Making LDA model. This will take awhile...')
    lda_model = run_lda(topic_texts, dictionary, corpus, num_topics=40)
    lda_topics = lda_model.show_topics(num_topics=40,formatted=False)

    print('Getting article topics...')
    all_article_topics = article_topics_and_topic_coverage(lda_model, topic_texts, tokenized=False)
