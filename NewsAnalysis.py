import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from working_with_data2.make_df import get_df, fix_cnn, clean_df, process_articles
from working_with_data2.data_exploration import all_length_hist, article_length_hist, dictionary_and_corpus, run_lda
from working_with_data2.sentiment_analysis import topic_values
from working_with_data2.bokeh_plotting import make_plots, make_clouds

import pyLDAvis.gensim


class NewsAnalysis:
    def __init__(self):
        self.topic_texts = None
        self.sentiment_texts = None
        self.quote_texts = None
        self.tweet_texts = None

        self.dictionary = None
        self.corpus = None

        self.lda_model = None
        self.lda_topics = None

        self.sentiment_of_words = None
        self.all_article_topics = None
        self.topic_dict = None

    def from_mongo(self, db_name):
        """
        Function calls from make_df.py to get create Pandas DataFrame from JSON files in Mongo database.
        Following are the steps we take:

        1. Get collections from Mongo
        2. Loop through collections and find which site it refers to
        3. Save variables from JSON file to Pandas DataFrame
        4. Fix CNN articles using BeautifulSoup (Newspaper library didn't scrape this correctly)
        5. Clean df:
            Remove unwanted sites (because they weren't strictly political)
            Remove unwanted articles from sites (Some articles were just photos or transcripts)
            Remove null values from article_text

        Parameters:
        ----------
        db_name: Name of Mongo database

        Returns:
        -------
        df: DataFrame from Mongo (clean)
        """
        print('Getting articles from Mongo...')
        df = get_df(db_name)
        print('Fixing CNN articles...')
        df = fix_cnn(df)
        print('Cleaning df...')
        df = clean_df(df)

        return df

    def from_csv(self,filename):
        df = pd.read_csv(filename)

        self.topic_texts = [text.split(' ') for text in df['topic_texts']]
        self.sentiment_texts = [text.split(' ') for text in df['sentiment_texts']]
        self.quote_texts = [text.split(' ') if type(text) != float else '' for text in df['quote_texts']]
        self.tweet_texts = [text.split(' ') if type(text) != float else '' for text in df['tweet_texts']]

        return df

    def to_csv(self, df, filename):
        df.to_csv(filename, index=False)

    def process_texts(self, df):
        """
        Function call from make_df.py to process texts. Following are the steps we take:

        1. Seperate quotes and tweets.
        2. Stopword and unwanted word Removal.
        3. Bigram, trigram, quadgram creation.
        4. Lemmatization (not stem since stemming can reduce the interpretability).

        Parameters:
        ----------
        df: dataframe (clean)

        Returns:
        -------
        df (with headline, article, quotes, tweets as new columns): Pre-processed tokenized texts.
        """
        self.topic_texts, self.sentiment_texts, self.quote_texts, self.tweet_texts = process_articles(df)
        df['topic_texts'] = [' '.join(text) for text in self.topic_texts]
        df['sentiment_texts'] = [' '.join(text) for text in self.sentiment_texts]
        df['quote_texts'] = [' '.join(text) for text in self.quote_texts]
        df['tweet_texts'] = [' '.join(text) for text in self.tweet_texts]

        return df

    def make_hists(self, df):
        '''Gets article lengths by site and saves histogram'''
        fig1, fig2 = all_length_hist(df, self.topic_texts, self.sentiment_texts, self.quote_texts, self.tweet_texts)
        fig1.savefig('web_app/static/img/topic_sent_length_hist.png')
        fig2.savefig('web_app/static/img/quote_tweet_length_hist.png')

    def make_mood_plots():
        figs = mood_plots(self.topic_dict)
        for i,fig in enumerate(figs):
            fig.savefig('web_app/static/img/mood_plots/mood_plot_by_topic'+str(i)+'.png')

    def make_pos_neg_plots():
        figs = mood_plots(self.topic_dict)
        for i,fig in enumerate(figs):
            fig.savefig('web_app/static/img/pos_neg_plots/pos_neg_plot_by_topic'+str(i)+'.png')

    def run_lda_model(self, no_below=20, no_above=0.5, topn=10000, num_topics=None, weight_threshold=0.7, K=15, T=150, passes=20, iterations=400):
        """
        Function to get LDA model. Following are the steps we take:

        1. Get HDP model
        2. Get topic weights
        3. Calculate # of topics if needed
        4. Get LDA model and save it
        5. Save pyLDAvis plot to web app

        Parameters:
        ----------
        topn, num_topics: # of Words to consider and # of topics.
        weight_threshold: If num_topics not given - Give threshold to determine topics from HDP-LDA.
        K=15, T=150: HDP hyperparameters

        Returns:
        -------
        lda_model: Calculated LDA model
        """
        self.dictionary, self.corpus = dictionary_and_corpus(self.topic_texts, no_below=no_below, no_above=no_above)

        self.lda_model, vis_data, fig = run_lda(self.topic_texts, self.dictionary, self.corpus, topn=topn,
                            num_topics=num_topics, weight_threshold=weight_threshold, K=K, T=T, passes=20, iterations=400)

        pyLDAvis.save_html(vis_data, 'web_app/plots/pyLDAvis_'+lda_model.num_topics+'_topics.html')

        fig.savefig('web_app/static/img/hdp_topic_probabilities.png', dpi=fig.dpi)

        self.lda_topics = lda_model.show_topics(num_topics=-1, num_words=100000,formatted=False)

        return self.lda_model

    def get_lda_model(self):
        '''
        Returns precomputed LDA model from pickle
        '''
        with open('pickles/lda_model.pkl', 'rb') as f:
            self.lda_model = pickle.load(f)

        self.lda_topics = lda_model.show_topics(num_topics=-1, num_words=100000,formatted=False)

        return self.lda_model

    def get_topic_values(self, df):
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
        self.topic_dict, self.all_article_topics, self.sentiment_of_words = \
                        topic_values(df, self.topic_texts, self.sentiment_texts, self.lda_model)

        pickle.dump(self.topic_dict, open('pickles/topic_dict.pkl', 'wb'))

        return self.topic_dict

    def make_plots_and_clouds(self):
        '''
        Uses file bokeh_plotting.py to create Bokeh plots and word clouds
        '''
        self.components_dict = make_plots(self.topic_dict, self.lda_model.num_topics)
        make_clouds(self.topic_texts, self.lda_model)


if __name__ == '__main__':
    na = NewsAnalysis()
    #
    # df = na.from_mongo('rss_feeds_new')
    # df = df[pd.notnull(df['article_text'])]
    #
    # df = na.process_texts(df)
    #
    # na.to_csv(df, 'data/rss_feeds_new_from_NA.csv')

    df = na.from_csv('data/rss_feeds_with_tones.csv')

    # df = pd.read_csv('data/rss_feeds_new_good_with_extra.csv')


    # na.make_hists(df)

    print('Making LDA model. This will take awhile...')
    lda_model = na.run_lda_model(no_below=20, no_above=0.5, topn=10000, num_topics=None, weight_threshold=0.25, K=15, T=150, passes=1, iterations=10)
    pickle.dump(lda_model, open('pickles/lda_model.pkl', 'wb'))

    print('Making topic dictionary model. This will also take awhile...')
    topic_dict = get_topic_values()
    pickle.dump(topic_dict, open('pickles/topic_dict.pkl', 'wb'))

    print('Making Bokeh Plots and Word Clouds')
    make_plots_and_clouds()
