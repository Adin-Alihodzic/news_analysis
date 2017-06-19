import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import datetime

from working_with_data2.make_df import get_df, fix_cnn, clean_df, process_articles, convert_date
from working_with_data2.data_exploration import all_length_hist, article_length_hist, dictionary_and_corpus, run_lda, mood_plots, mood_plots_by_site, pos_neg_plot, pos_neg_by_site_plot, coverage_by_site_by_topic
from working_with_data2.sentiment_analysis import topic_values, get_new_tones
from working_with_data2.bokeh_plotting import make_bokeh_plot, make_clouds

import pyLDAvis.gensim

# Used to distinguish new models from past
# identifier = datetime.datetime.now().date().isoformat()+'_55_topics_400_passes'
identifier = '2017-06-18_55_topics_400_passes'


class NewsAnalysis:
    def __init__(self):
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
        You want to run import_database.py before this to get the database.
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

        df = df[pd.notnull(df['article_text'])]
        df = df.reset_index(drop=True)

        return df

    def from_csv(self,filename):
        df = pd.read_csv(filename)
        df = df[pd.notnull(df['article_text'])]
        df = df.reset_index(drop=True)

        return df

    def to_csv(self, df, filename):
        df = df[pd.notnull(df['article_text'])]
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
        topic_texts, sentiment_texts, quote_texts, tweet_texts = process_articles(df)
        df['topic_texts'] = [' '.join(text) for text in topic_texts]
        df['sentiment_texts'] = [' '.join(text) for text in sentiment_texts]
        df['quote_texts'] = [' '.join(text) for text in quote_texts]
        df['tweet_texts'] = [' '.join(text) for text in tweet_texts]

        return df

    def get_tones(self, df, prev_df=None):
        """
        Function call from make_df.py to get tone from IBM Watson's ToneAnalyzerV3.
        Following are the steps we take:

        1. If no previous df given with tones then make empty df to pass in so we get tones for all rows in df.
        2. If given previous array then just get tones for new values so we don't overuse API.
        3. Get new tones from function and return.

        Parameters:
        ----------
        df: dataframe (clean)
        prev_df: (optional) If already gotten tones then give previous so we don't overuse API.

        Returns:
        -------
        df (with tones): Return same df with extra column for tones
        """
        # if prev_df == None:
        #     prev_df = df.drop(df.index)
        df = get_new_tones(df, prev_df)

        return df

    def make_plots(self, df):
        '''Makes all plots used in web app'''
        # # Length Histograms
        # topic_length_hist, quote_length_hist = all_length_hist(df)
        # topic_length_hist.savefig('web_app/static/img/topic_sent_length_hist_'+identifier+'.png')
        # quote_length_hist.savefig('web_app/static/img/quote_tweet_length_hist_'+identifier+'.png')
        #
        # Mood Bar Graphs
        # mood_figs = mood_plots(self.topic_dict)
        # for i,mood_fig in enumerate(mood_figs):
        #     mood_fig.savefig('web_app/static/img/mood_plots/mood_plot_by_topic'+str(i)+'_'+identifier+'.png')

        mood_by_site_figs = mood_plots_by_site(self.topic_dict)
        for i,mood_by_site_fig in enumerate(mood_by_site_figs):
            mood_by_site_fig.savefig('web_app/static/img/mood_plots/mood_by_site_plot_by_topic'+str(i)+'_'+identifier+'.png')

        # # Positive/Negative Bar Charts
        # pos_neg_figs = pos_neg_plot(self.topic_dict)
        # for i,pos_neg_fig in enumerate(pos_neg_figs):
        #     pos_neg_fig.savefig('web_app/static/img/pos_neg_plots/pos_neg_plot_by_topic'+str(i)+'_'+identifier+'.png')

        # pos_neg_by_site_figs = pos_neg_by_site_plot(self.topic_dict)
        # for i,pos_neg_by_site_fig in enumerate(pos_neg_by_site_figs):
        #     pos_neg_by_site_fig.savefig('web_app/static/img/pos_neg_plots/pos_neg_by_site_plot_by_topic'+str(i)+'_'+identifier+'.png')

        # coverage_figs = coverage_by_site_by_topic(df,self.topic_dict)
        # for i,coverage_fig in enumerate(coverage_figs):
        #     coverage_fig.savefig('web_app/static/img/coverage_plots/coverage_plot_by_topic'+str(i)+'_'+identifier+'.png')
        #
        # # Bokeh plots
        # components_dict = [{'script': None, 'div': None} for topic in range(self.lda_model.num_topics)]
        # for topic in range(self.lda_model.num_topics):
        #     components_dict[topic]['script'], components_dict[topic]['div'] = make_bokeh_plot(self.topic_dict, topic)
        # pickle.dump(components_dict, open('web_app/static/img/bokeh_plots/components_dict_'+identifier+'.pkl', 'wb'))

        # # Word Clouds
        # cloud_figs = make_clouds(df, self.lda_model)
        # for i,cloud_fig in enumerate(cloud_figs):
        #     cloud_fig.savefig('web_app/static/img/wordclouds/wordcloud_topic'+str(i)+'_'+identifier+'.png')

    def run_lda_model(self,df, no_below=20, no_above=0.5, topn=10000, num_topics=None, weight_threshold=0.25, K=15, T=150, passes=20, iterations=400):
        """
        Function to get LDA model. Following are the steps we take:

        1. Get HDP model
        2. Get topic weights
        3. Calculate # of topics if needed
        4. Get LDA model and save it
        5. Save pyLDAvis plot to web app

        Parameters:
        ----------
        topn, num_topics: # of Words to consider and # of topics (if None given then it will determine how many topics using HDP-LDA)
        weight_threshold: If num_topics not given - Give threshold to determine topics from HDP-LDA.
        K=15, T=150: HDP hyperparameters

        Returns:
        -------
        lda_model: Calculated LDA model
        """
        topic_texts = [text.split(' ') for text in df['topic_texts']]

        self.dictionary, self.corpus = dictionary_and_corpus(topic_texts, no_below=no_below, no_above=no_above)

        self.lda_model, vis_data, fig = run_lda(topic_texts, self.dictionary, self.corpus, topn=topn,
                            num_topics=num_topics, weight_threshold=weight_threshold, K=K, T=T, passes=20, iterations=400)


        pickle.dump(self.lda_model, open('pickles/lda_model_'+identifier+'.pkl', 'wb'))

        pyLDAvis.save_html(vis_data, 'web_app/static/img/pyLDAvis_graphs/pyLDAvis_'+identifier+'.html')

        fig.savefig('web_app/static/img/hdp_topic_probabilities_'+identifier+'.png', dpi=fig.dpi)

        self.lda_topics = self.lda_model.show_topics(num_topics=-1, num_words=100000,formatted=False)

        return self.lda_model

    def get_lda_model(self, lda_model):
        '''
        Returns precomputed LDA model from pickle
        '''
        self.lda_model = lda_model

        self.lda_topics = lda_model.show_topics(num_topics=-1, num_words=100000,formatted=False)


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
        self.topic_dict, self.all_article_topics, self.sentiment_of_words, fig = \
                        topic_values(df, self.lda_model)

        pickle.dump(self.topic_dict, open('pickles/topic_dict_'+identifier+'.pkl', 'wb'))

        fig.savefig('web_app/static/img/coverage_by_topics_'+identifier+'.png', dpi=fig.dpi)

        return self.topic_dict

    def get_topic_dict(self, topic_dict):
        self.topic_dict = topic_dict


if __name__ == '__main__':
    make_csv = False

    na = NewsAnalysis()
    df = None

    if make_csv:
        df = na.from_mongo('rss_feeds_new')
        df = df[pd.notnull(df['article_text'])]
        df = df.reset_index(drop=True)

        df = na.process_texts(df)

        df = convert_date(df)
        df = df.reset_index(drop=True)

        df_tones = pd.read_csv('data/rss_with_tones_in_df_fixed_time.csv')
        df = na.get_tones(df, df_tones)
        na.to_csv(df, 'data/rss_feeds_newest_with_tones.csv')

    else:
        df = na.from_csv('data/rss_feeds_final.csv')


    df = convert_date(df)
    df = df.reset_index(drop=True)

    df = df[pd.notnull(df['Anger'])]
    df = df.reset_index(drop=True)

    # print('Making LDA model. This will take awhile...')
    # lda_model = na.run_lda_model(df, no_below=20, no_above=0.5, topn=10000, num_topics=55, weight_threshold=0.25, K=15, T=150, passes=1000, iterations=10000)

    with open('pickles/lda_model_'+identifier+'.pkl', 'rb') as f:
        lda_model = pickle.load(f)

    na.get_lda_model(lda_model)
    #
    # print('Making topic dictionary model. This will also take awhile...')
    # topic_dict = na.get_topic_values(df)
    with open('pickles/topic_dict_'+identifier+'.pkl', 'rb') as f:
        topic_dict = pickle.load(f)
    na.get_topic_dict(topic_dict)
    #
    print('Making Plots. This takes a really long time...')
    na.make_plots(df)
