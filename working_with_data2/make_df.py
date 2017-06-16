import subprocess
import requests
from pymongo import MongoClient
import datetime
from datetime import date
from bs4 import BeautifulSoup
from unidecode import unidecode
from dateutil import parser
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import re

import nltk
# Make sure we have nltk packages
nltk.download('stopwords')
nltk.download('wordnet')
import gensim
from gensim.utils import lemmatize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

import warnings
warnings.filterwarnings('ignore')



def get_df(db_name):
    """
    Function to get create Pandas DataFrame from JSON files in Mongo database.
    Following are the steps we take:

    1. Get collections from Mongo
    2. Loop through collections and find which site it refers to
    3. Save variables from JSON file to Pandas DataFrame

    Parameters:
    ----------
    db_name: Name of Mongo database

    Returns:
    -------
    df: DataFrame from Mongo (not clean)
    """
    url_names = ['wsj', 'cnn', 'abc', 'fox', 'nyt', 'reuters', 'wapo', 'huffpo', 'esquire', 'rollingstone', 'cbs', '538', 'washtimes']
    client = MongoClient()
    db = client[db_name]

    collection_names = db.collection_names()
    my_collection_names = [name for name in collection_names]

    df = pd.DataFrame(columns=['article_text', 'author', 'date_published', 'headline', 'url', 'source'])
    for collection_name in my_collection_names:
        if collection_name != 'system.indexes':
            site = [name for name in url_names if collection_name.startswith(name)]
            if len(site) != 0:
                site = site[0]
                print('Working on '+collection_name)
                for article in db[collection_name].find():
                    # remove article that just have videos
                    # remove powerpost articles from wapo becaue they are 4000+ words
                    if 'video' not in article['url'] and 'powerpost' not in article['url']:
                        try:
                            url = article['url']
                            source = site
                            headline = article['headline']
                            date_published = article['date_published']
                            # If date is missing then use the collection name date
                            if type(date_published) == float:
                                date = collection_name.split('_')[1]
                                date_published = datetime.datetime.strptime(date, '%Y%m%d')
                            author = article['author']
                            article_text = article['article_text']
                            # df.append(pd.Series([article_text, author, date_published, headline, url, processed_text, source]), ignore_index=True)
                            df.loc[-1] = [article_text, author, date_published, headline, url, source]  # adding a row
                            df.index = df.index + 1  # shifting index
                            df = df.sort()  # sorting by index
                        except:
                            print('Problem with article in '+site)
    return df


def convert_date(df):
    new_date = []
    for i,date in enumerate(df['date_published']):
        try:
            temp = parser.parse(date)
            if temp.microsecond != 0:
                new_date.append(temp - timedelta(hours=6))
            else:
                new_date.append(temp)
        except:
            new_date.append(date)
    df['date_published'] = new_date
    df = df.reset_index(drop=True)

    return df

def check_str(x):
    '''Checks if string or unidecode'''
    if isinstance(x, str):
        return unidecode(x)
    else:
        return str(x)

def fix_cnn(df):
    """
    Function to fix CNN articles because the library Newspaper didn't scrape these correctly,
    so I use BeautifulSoup to fix them
    """
    for i in df.index:
        if df['source'][i] == 'cnn':
            url = df['url'][i]
            try:
                result = requests.get(url)
                soup = BeautifulSoup(result.content, 'html.parser')

                tag1 = soup.find('div', attrs={'class': 'el__leafmedia el__leafmedia--sourced-paragraph'}).text
                tag2 = soup.find_all('div', attrs={'class': 'zn-body__paragraph speakable'})
                tag3 = soup.find_all('div', attrs={'class': 'zn-body__paragraph'})
                new_article_text = tag1+' /n '+check_str(' \n '.join([line.text for line in tag2]))+check_str(' \n '.join([line.text for line in tag3]))
                df['article_text'][i] = new_article_text
            except:
                pass
    return df

# This isn't working
def fix_huffpo(df):
    """
    This was needed to fix Huffington Post articles before I got rid of tweets.
    Don't need and doesn't work now.
    """
    for i in df.index:
        if df['source'][i] == 'huffpo':
            url = df['url'][i]
            try:
                result = requests.get(url)
                soup = BeautifulSoup(result.content, 'html.parser')

                tag = soup.find_all('div', attrs={'class': 'content-list-component bn-content-list-text text'})
                new_article_text = parse_str(' \n '.join([line.text for line in tag]))
                df['article_text'][i] = new_article_text
            except:
                pass
    return df

def clean_df(df):
    """
    Function to clean dataframe. Following are the steps we take:

    1. Remove unwanted sites (because they weren't strictly political)
    2. Remove unwanted articles from sites (Some articles were just photos or transcripts)
    3. Remove null values from article_text
    """
    # Remove duplicates by url
    df = df.drop_duplicates(subset='url')

    # Remove date below the start of my scraping
    df = df[df['date_published'] >= date(2017,5,18)]
    # Below I get rid of large articles that could throw off my algorithms
    # Remove two article types that were very long
    # any url with speech was just a transcript of speeches
    df = df[(df['source'] != 'ap') & (df['source'] != 'economist') & (df['source'] != 'vox') & (df['source'] != 'time') & (df['source'] != 'slate')]

    df[df['source'] == 'wapo'] = df[(df['source'] == 'wapo') & (df['url'].str.contains('powerpost') == False) & (df['url'].str.contains('-speech-') == False)]
    df[df['source'] == 'economist'] = df[(df['source'] == 'economist') & (df['url'].str.contains('transcript') == False)]
    df[df['source'] == 'fox'] = df[(df['source'] == 'fox') & (df['article_text'].str.contains('Want FOX News Halftime Report in your inbox every day?') == False)]
    df[df['source'] == 'esquire'] = df[(df['source'] == 'esquire') & (df['url'].str.contains('-gallery-') == False)]

    # Can't have null values in text
    df = df[pd.notnull(df['article_text'])]
    df = df.dropna(how='all')
    df = df.reset_index(drop=True)

    return df

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
    quote_texts = []
    tweet_texts = []
    for headline, article in zip(df['headline'].tolist(), df['article_text'].tolist()):
        all_texts = process_article(headline, article, bigram, trigram)
        topic_texts.append(all_texts[0] + all_texts[1] + all_texts[2] + all_texts[3])
        sentiment_texts.append(all_texts[0] + all_texts[1])
        quote_texts.append(all_texts[2])
        tweet_texts.append(all_texts[3])

    df['topic_texts'] = [' '.join(text) for text in topic_texts]
    df['sentiment_texts'] = [' '.join(text) for text in sentiment_texts]
    df['quote_texts'] = [' '.join(text) for text in quote_texts]
    df['tweet_texts'] = [' '.join(text) for text in tweet_texts]

    return topic_texts, sentiment_texts, quote_texts, tweet_texts

# def watson_tone_analyzer(df, have_tones=True):
#     new_df = None
#     if have_tones:
#         new_df = df[df['tones'].isnull()]
#     else:
#         new_df = df
#     tones = df['tones']
#     for i in range(len(new_df.shape[0])):
#         json_response_sentiment = tone_analyzer.tone(text=' '.join(ast.literal_eval(new_df['sentiment_texts'][i])), sentences=False)
#         tones[i] = parse_toneanalyzer_response(json_response_sentiment)
#
#     df['tones'] = tones
#     return tones




if __name__ == '__main__':
    # df = get_df('rss_feeds_new')
    # df = fix_cnn(df)
    # # df = fix_huffpo(df
    # df = clean_df(df)
    # df = df[pd.notnull(df['article_text'])]
    #

    df = pd.read_csv('../data/rss_feeds_new_good.csv')
    df = df[pd.notnull(df['article_text'])]
    print('Processing Articles...')
    topic_texts, sentiment_texts, quote_texts, tweets_texts = process_articles(df)

    df.to_csv('rss_feeds_new_good.csv')
