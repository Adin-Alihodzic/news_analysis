import numpy as np
import pandas as pd

from working_with_data.clean_data import get_df, clean_df, get_processed_text
from working_with_data.import_database import from_bucket
from working_with_data.sentiment_analysis import sentiment_of_words_wordnet, sentiment_by_topic_wordnet

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import pyLDAvis
import pyLDAvis.sklearn

class NewsBias:
    def __init__(self):
        self.tf_vectorizer = []
        self.tf = []
        self.lda_model = []
        self.sentiment_by_topic = []

    def from_mongo(self, db_name):
        df = get_df(db_name)
        df = clean_df(df)

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
        # Can only run on non-nan values
        df_no_nan = df[pd.notnull(df['processed_text'])]
        text = df_no_nan['processed_text'].values.tolist()
        self.tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=max_features,
                                        stop_words='english')
        self.tf = self.tf_vectorizer.fit_transform(text)

        self.lda_model = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                              learning_method='online',
                                              learning_offset=50.,
                                              random_state=0,
                                              n_jobs=-1)

        self.lda_model.fit(self.tf)

        return self.lda_model

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
        df = sentiment_of_words_wordnet(df)

        return df

    def get_sentiment_by_topic(self, df):
        feature_names = np.array(self.tf_vectorizer.get_feature_names())
        topics_mat = self.lda_model.components_

        self.sentiment_by_topic = sentiment_by_topic_wordnet(df, topics_mat, feature_names)

        return self.sentiment_by_topic

    def pickle_everything(self):
        pass
