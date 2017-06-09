import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re

from gensim.summarization import summarize
from gensim.summarization import keywords

import warnings
# Gensim gives annoying warnings
warnings.filterwarnings('ignore')

import nltk
# Make sure we have nltk packages
nltk.download('stopwords')
nltk.download('wordnet')
from gensim.utils import lemmatize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

import gensim
from gensim.models import LdaModel, HdpModel
from gensim.corpora import Dictionary

import pyLDAvis.gensim

def get_article_length_hist(df):
    '''Gets article lengths by site and saves histogram'''
    site_article_length = {source: [] for source in df['source'].unique()}
    for source in df['source'].unique():
        new_df = df[df['source'] == source]
        for article in new_df['article_text']:
            site_article_length[source].append(len(article.split()))


    for i, source in enumerate(site_article_length.keys()):
        plt.subplot(4,3,i+1)
        plt.hist(site_article_length[source], normed=True, bins=100)
        plt.title('Article Length '+source)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.savefig('../web_app/static/img/article_length_hist.png', dpi=300)


def get_summaries(df):
    '''Uses summarization function from gensim library to summarize each article'''
    summary = []
    for article in df['article_text']:
        try:
            summary.append(summarize(article))
        except:
            summary.append('')

    return summary

def build_texts(text):
    '''Uses gensim to get preprocessed text'''
    for line in text:
        yield gensim.utils.simple_preprocess(line, deacc=True, min_len=3)



def process_article(headline_text, article_text, bigram, trigram):
    """
    Function to process texts. Following are the steps we take:

    1. Seperate quotes and tweets.
    2. Stopword and unwanted word Removal.
    3. Bigram, trigram, quadgram creation.
    4. Lemmatization (not stem since stemming can reduce the interpretability).

    Parameters:
    ----------
    headline_text, article_text: string.
    bigram, trigram: Already trained bigram and trigram from gensim

    Returns:
    -------
    headline, article, quotes, tweets: Pre-processed tokenized texts.
    """
#     article_text = [[word for word in line if word not in stops] for line in article_text]

    lemmatizer = WordNetLemmatizer()

    article_quotes1 = ' '.join(re.findall('“.*?”', article_text))
    article_quotes2 = ' '.join(re.findall('".*?"', article_text))
    headline_quotes1 = ' '.join(re.findall('“.*?”', headline_text))
    headline_quotes2 = ' '.join(re.findall('".*?"', headline_text))
    quotes = article_quotes1 + article_quotes2 + headline_quotes1 + headline_quotes2

    tweets = ' '.join(re.findall('\n\n.*?@', article_text))+' '+' '.join(re.findall('\n\n@.*?@', article_text))

    # remove tweets
    article_text = re.sub('\n\n.*?@', '', article_text)
    article_text = re.sub('\n\n@.*?@', '', article_text)
    headline_text = re.sub('\n\n.*?@', '', headline_text)
    headline_text = re.sub('\n\n@.*?@', '', headline_text)

    article_text = ' '.join([word for word in article_text.split(' ') if not word.startswith('(@') and not word.startswith('http')])

    # remove quotes
    article_text = re.sub('“.*?”', '', article_text)
    article_text = re.sub('".*?"', '', article_text)


    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    sw = set(stopwords.words('english'))
    wordnet = WordNetLemmatizer()

    article_text = article_text.lower()
    headline_text = headline_text.lower()
    quotes = quotes.lower()
    tweets = tweets.lower()

    article_text_tokens = tokenizer.tokenize(article_text)
    headline_text_tokens = tokenizer.tokenize(headline_text)
    quotes_tokens = tokenizer.tokenize(quotes)
    tweets_tokens = tokenizer.tokenize(tweets)

    # remove stop words and unwanted words
    words_to_remove = ['http', 'com', '_', '__', '___', 'mr']
    article_text_stopped_tokens = [i for i in article_text_tokens if i not in sw and i not in words_to_remove]
    headline_text_stopped_tokens = [i for i in headline_text_tokens if i not in sw and i not in words_to_remove]
    quotes_stopped_tokens = [i for i in quotes_tokens if not i in sw and i not in words_to_remove]
    tweets_stopped_tokens = [i for i in tweets_tokens if not i in sw and i not in words_to_remove]

    # Create bigrams
    article_text_stopped_tokens = bigram[article_text_stopped_tokens]
    headline_text_stopped_tokens = bigram[headline_text_stopped_tokens]
    quotes_stopped_tokens = bigram[quotes_stopped_tokens]
    tweets_stopped_tokens = bigram[tweets_stopped_tokens]

    # Create trigrams (and quadgrams)
    article_text_stopped_tokens = trigram[bigram[article_text_stopped_tokens]]
    headline_text_stopped_tokens = trigram[bigram[headline_text_stopped_tokens]]
    quotes_stopped_tokens = trigram[bigram[quotes_stopped_tokens]]
    tweets_stopped_tokens = trigram[bigram[tweets_stopped_tokens]]

    # stem token
    article_text = [wordnet.lemmatize(i) for i in article_text_stopped_tokens]
    headline_text = [wordnet.lemmatize(i) for i in headline_text_stopped_tokens]
    quotes = [wordnet.lemmatize(i) for i in quotes_stopped_tokens]
    tweets = [wordnet.lemmatize(i) for i in tweets_stopped_tokens]

    return headline_text, article_text, quotes, tweets

def process_articles(df):
    '''Uses process_article to process all articles in a dataframe'''
    articles = df['article_text'].values.tolist()

    # Used to train bigrams and trigrams
    train_texts = list(build_texts(articles))

    bigram = gensim.models.Phrases(train_texts)  # for bigram collocation detection
    trigram = gensim.models.Phrases(bigram[train_texts])  # for trigram collocation detection

    # topic_texts used to create topics (includes headlines, articles, tweets and quotes)
    topic_texts = []
    # sentiment_texts used to calculate sentiment (only includes headlines and articles)
    sentiment_texts = []
    for headline, article in zip(df['headline'].tolist(), df['article_text'].tolist()):
        all_texts = process_article(headline, article, bigram, trigram)
        topic_texts.append(all_texts[0] + all_texts[1] + all_texts[2] + all_texts[3])
        sentiment_texts.append(all_texts[0] + all_texts[1])

    return topic_texts, sentiment_texts

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


def run_lda(topic_texts, dictionary, corpus, topn=10000, num_topics=None, weight_threshold=0.7, K=15, T=150):
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

    Returns:
    -------
    lda_model: Calculated LDA model
    """

    hdp_model = HdpModel(corpus=corpus, id2word=dictionary, chunksize=10000, K=K, T=T)

    topic_weights = topic_prob_extractor(hdp_model, topn=topn)['weight'].tolist()
    plt.plot(topic_weights)
    plt.title('Topic Probabilities')
    plt.xlabel('Topic')
    plt.ylabel('Probability')
    plt.savefig('../web_app/static/img/hdp_topic_probabilities.png')

    if num_topics == None:
        total_weight = sum(topic_weights)
        percent_of_weight = 0
        num_topics = 0
        while percent_of_weight < weight_threshold:
            num_topics += 1
            percent_of_weight = topic_weights[:num_topics] / total_weight

    num_topics = num_topics     # This can be entered manually or found from HDP
    chunksize = 10000   # How many articles can be used at once
    passes = 20         # How many times it calculates the LDA model
    iterations = 400    # How many times it iterates over the articles
    eval_every = None   # Don't evaluate model perplexity, takes too much time.

    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, chunksize=chunksize, \
                           alpha='auto', eta='auto', \
                           iterations=iterations, \
                           passes=passes, eval_every=eval_every)

    pickle.dump(lda_model, open('../pickles/lda_model.pkl', 'wb'))

    vis_data = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, R=20, sort_topics=False)
    pyLDAvis.save_html(vis_data, '../web_app/plots/pyLDAvis_40_topics.html')

    return lda_model

def article_topics_and_topic_coverage(model, topic_texts, tokenized=False):
    """
    Function to get each articles top topics and how much a topic is covered.
    Following are the steps we take:

    1. Get Bag of Words representation depending on whether or not the input is tokenized
    2. Use model to get topics for each article
    3. Calculate # of topics if needed
    4. Get LDA model and save it
    5. Save pyLDAvis plot to web app

    Parameters:
    ----------
    topic_texts, dictionary, corpus: All determined from functions above.
    topn, num_topics: # Words to consider and number of topics.
    weight_threshold: If num_topics not given - Give threshold to determine topics.
    K=15, T=150: HDP hyperparameters

    Returns:
    -------
    lda_model: Calculated LDA model
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

    plt.bar(range(model.num_topics), topic_coverage)
    plt.savefig('../web_app/static/img/topic_coverage.png')

    return all_article_topics

if __name__ == '__main__':
    df = pd.read_csv('../data/rss_feeds_new_good.csv')
    df = df[pd.notnull(df['article_text'])]

    get_article_length_hist(df)

    # print('Getting Summaries...')
    # summary = get_summaries(df)

    print('Processing Articles...')
    topic_texts, sentiment_texts = process_articles(df)

    print('Getting Dictionary and Corpus...')
    dictionary, corpus = dictionary_and_corpus(topic_texts, no_below=20, no_above=0.5)

    print('Making LDA model. This will take awhile...')
    lda_model = run_lda(topic_texts, dictionary, corpus, num_topics=40)
    lda_topics = lda_model.show_topics(num_topics=40,formatted=False)

    print('Getting article topics...')
    all_article_topics = article_topics_and_topic_coverage(lda_model, topic_texts, tokenized=False)
