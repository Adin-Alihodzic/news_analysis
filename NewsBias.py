import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from working_with_data.clean_data import fix_cnn, fix_huffpo, get_df, clean_df, get_processed_text
from working_with_data.import_database import from_bucket
from working_with_data.sentiment_analysis import sentiment_of_words_wordnet, sentiment_by_topic_wordnet, sentiment_by_article_wordnet

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import pyLDAvis
import pyLDAvis.sklearn

class NewsBias:
    def __init__(self):
        self.tf_vectorizer = []
        self.tf = []
        self.lda_model = []
        self.feature_names = []
        self.topics_mat = []
        self.sentiment_by_topic = []

    def fix_sites(mongo_db):
        fix_cnn(mongo_db)
        fix_huffpo(mongo_db)

    def from_mongo(self, db_name):
        df = get_df(db_name)
        df = clean_df(df)
        df = df[pd.notnull(df['processed_text'])]
        df = df[df['processed_text'] != '']

        return df

    def from_csv(self, csv_name):
        try:
            df = pd.read_csv('data/'+csv_name, parse_dates=False)
            return df
        except:
            print('CSV file does not exist!')
            print('Make sure CSV file is in data folder.')
            return False

    def to_csv(self, df, filename):
        filename = 'data/'+filename
        df.to_csv(filename, index=False)
        print('CSV file saved to: '+filename)

    def update_from_bucket(self, filename):
        path = os.getcwd()
        # Example filename: 'dsiprojectdata/rss_feeds_new.tar'
        result = from_bucket(filename, path)
        if not result:
            print('Error updating data from bucket!')
            print('Make sure you include folder and file in filename from bucket.')

    def update_to_bucket(self, filename, bucketname, mongo_db=False):
        # If mongo database then just give database name as filename
        if mongo_db:
            cwd = os.getcwd()
            # Give permission to bash file then run
            p1 = subprocess.Popen('chmod', '+x', 'backup.sh', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out1, err1 = p1.communicate()
            p2 = subprocess.Popen(cwd+'/backup.sh', filename, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out2, err2 = p2.communicate()
        else:
            p = subprocess.Popen('/usr/bin/aws', 's3', 'cp', filename, 's3://'+bucketname+'/', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()

    def run_lda(self, df, max_features=1000, n_topics=20):
        df = df[pd.notnull(df['processed_text'])]
        processed_text = df['processed_text'].values.tolist()
        # Inclued quotes in LDA
        processed_quote = df['processed_quote'].values.tolist()
        processed_tweet = df['processed_tweet'].values.tolist()
        processed_all = []
        for text, quote, tweet in zip(processed_text, processed_quote):
            # Check if quote is nan
            if type(quote) == float:
                quote = ''
            if type(tweet) == float:
                tweet = ''
            processed_all.append(text + quote + tweet)
        try:
            self.tf_vectorizer = CountVectorizer(max_df=0.95, min_df=0.05,
                                            max_features=max_features,
                                            stop_words='english')
            self.tf = self.tf_vectorizer.fit_transform(processed_all)
        except:
            import pdb; pdb.set_trace()
        self.lda_model = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                              learning_method='online',
                                              learning_offset=50.,
                                              random_state=0,
                                              n_jobs=-1)

        self.lda_model.fit(self.tf)

        self.feature_names = np.array(self.tf_vectorizer.get_feature_names())
        self.topics_mat = self.lda_model.components_

        return self.lda_model

    def run_gensim_lda(self, df, n_topics=20):
        self.lda_model = gensim_lda(df, n_topics)

    def get_top_word_by_topic(topic, n_words):
        return self.feature_names[np.argsort(self.topics_mat[topic,:])[::-1]][:n_words]

    def visualize_lda(self, df, display=False):
        if self.lda_model == []:
            self.run_lda(df)
        max_features = self.tf_vectorizer.get_params()['max_features']
        n_topics = self.lda_model.get_params()['n_topics']
        vis_data = pyLDAvis.sklearn.prepare(self.lda_model, self.tf, self.tf_vectorizer, R=n_topics, n_jobs=-1)
        pyLDAvis.save_html(vis_data, 'plots/pyLDAvis_'+str(max_features)+'feats_'+str(n_topics)+'topics.html')
        if display:
            pyLDAvis.show(vis_data)

    def get_sentiment_of_words(self, df):
        sentiment_of_words = sentiment_of_words_wordnet(df)

        return sentiment_of_words

    def get_sentiment_by_topic(self, df, display=False):
        n_topics = self.lda_model.get_params()['n_topics']

        self.sentiment_by_topic = sentiment_by_topic_wordnet(df, self.topics_mat, self.feature_names)

        if display:
            for i, site in enumerate(sentiment_by_topic.keys()):
                plt.subplot(3,4,i+1)
                score = []
                for topic in range(n_topics):
                    score.append(sentiment_by_topic[site][topic][3])
                score = np.array(score)
                score /= sum(np.abs(score))
                plt.bar(np.arange(len(score)), score, align='center')
                plt.ylabel('Score')
                plt.title('Score by Topic for '+site)
            plt.subplots_adjust(hspace=0.4, wspace=0.4)
            plt.show()

        return self.sentiment_by_topic

    def length_of_articles_hist(self, df):
        for i, site in enumerate(df['source'].unique()):
            plt.subplot(3,4,i+1)
            new_df = df[df['source'] == site]
            article_len = [len(article.split(' ')) for article in new_df['article_text']]
            plt.hist(article_len, normed=True)
            plt.xlabel('Length of Article')
            plt.ylabel('# of Articles')
            plt.title('Length of articles for '+site)
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.show()

    def pickle_everything(self):
        filename = '../pickles/lda_model.pkl'
        pickle.dump(self.lda_model, open(filename, 'wb'), protocol=2)

        filename = '../pickles/tf_vectorizer.pkl'
        pickle.dump(self.tf_vectorizer, open(filename, 'wb'), protocol=2)

if __name__ == '__main__':
    nb = NewsBias()
    df = nb.from_csv('rss_feeds_new.csv')
    lda_model = nb.run_lda(df, max_features=1000, n_topics=20)
    nb.visualize_lda(df)
    sentiment_of_words = nb.get_sentiment_of_words(df)
    df['sentiment_of_words'] = sentiment_of_words
    if False:
        # May need this
        sentiment_of_words = []
        for s in df['sentiment_of_words']:
            json_acceptable_string = s.replace("'", "\"")
            sentiment_of_words.append(json.loads(json_acceptable_string))
        sentiment_by_topic = sentiment_by_topic_wordnet(df, topics_mat, feature_names)


    topics_mat = lda_model.components_
    feature_names = np.array(nb.tf_vectorizer.get_feature_names())
    article_sent_by_topic = []
    for text in df['processed_text']:
        try:
            total = [0 for i in range(topics_mat.shape[0])]
            words = text.split(' ')
            idx = []
            for i, word in enumerate(feature_names):
                if word in words:
                    for topic in range(topics_mat.shape[0]):
                        total[topic] += topics_mat[topic, i]
            for topic in range(topics_mat.shape[0]):
                total[topic] /= sum(total)
            article_sent_by_topic.append(total)
        except:
            import pdb; pdb.set_trace()

    df['probability_of_topic'] = article_sent_by_topic
    # top_3_topics = []

    import ast
    sentiment = []
    count = 0
    for prob, sentiment_of_words in zip(df['probability_of_topic'], df['sentiment_of_words']):
        #prob = ast.literal_eval(prob)
        print(count)
        count += 1
        json_acceptable_string = sentiment_of_words.replace("'", "\"")
        sentiment_of_words = json.loads(json_acceptable_string)
        top_3_topics = np.argsort(prob)[::-1][:3]
        sentiment.append(sentiment_by_article_wordnet(sentiment_of_words, topics_mat, feature_names, top_3_topics))

    df['sentiment'] = sentiment

    import multiprocessing as mp
    pool = mp.Pool(processes=10)
    results = [pool.apply(get_sentiment, args=(prob,)) for prob in probs]


    max_score = 0
    max_i = 0
    for i in range(len(sentiment)):
        if sentiment[i][list(sentiment[i].keys())[0]][3] > max_score and list(sentiment[i].keys())[2] != 15 and len(df['processed_text'][i].split(' ')) > 300:
            max_score = sentiment[i][list(sentiment[i].keys())[2]][3]
            max_i = i

    min_score = 0
    min_i = 0
    for i in range(len(sentiment)):
        if sentiment[i][list(sentiment[i].keys())[0]][3] < min_score and list(sentiment[i].keys())[2] != 15 and len(df['processed_text'][i].split(' ')) > 300:
            min_score = sentiment[i][list(sentiment[i].keys())[2]][3]
            min_i = i
