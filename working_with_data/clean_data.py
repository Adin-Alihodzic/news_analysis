import subprocess
import nltk
import pandas as pd
from pymongo import MongoClient
from nltk.corpus import stopwords
from string import printable
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sys import argv
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from dateutil import parser
import datetime


def get_processed_text(article_text):
    tokenizer = RegexpTokenizer(r'\w+')
    raw = article_text.lower()
    tokens = tokenizer.tokenize(raw)

    # create English stop words list
    sw = set(stopwords.words('english'))
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in sw]

    wordnet = WordNetLemmatizer()
    # stem token
    texts = " ".join([wordnet.lemmatize(i) for i in stopped_tokens])
    return texts


# takes in mongo database name and if you want to restore as arg
db_name, restore = argv[1], argv[2]

if restore in ['True', 'true', 'Yes', 'yes']:
    print('Downloading mongo database from S3 Bucket')
    p1 = subprocess.Popen(['s3cmd', 'sync', 's3://dsiprojectdata/'+db_name+'.tar', '/home/ian/Galvanize/project/working_with_data/'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out1, err1 = p1.communicate()

    print('Unzipping file')
    p2 = subprocess.Popen(['tar', '-xvf', db_name+'.tar'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out2, err2 = p2.communicate()

    print('Saving as mongo database')
    p3 = subprocess.Popen(['mongorestore', '--db', db_name, './'+db_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out3, err3 = p3.communicate()


url_names = ['wsj', 'cnn', 'abc', 'fox', 'nyt', 'ap', 'reuters', 'wapo', 'economist', 'huffpo', 'esquire', 'rollingstone', 'cbs', '538', 'vox', 'time', 'slate', 'washtimes']

client = MongoClient()
db = client[db_name]

collection_names = db.collection_names()
my_collection_names = [name for name in collection_names]

# df_prev = pd.read_csv(str(db_name)+'_data.csv', parse_dates=False)
# df_prev['date_published'] = df_prev['date_published'].apply(lambda x: parser.parse(x.split('|')[0]))
# min_date_saved = min(df_prev['date_published']).to_pydatetime()
#
# min_date = datetime.date(2017, 5, 13)
# while min_date > min_date_saved:
#     min_date -= datetime.timedelta(days=7)
# print(min_date)

site_dict = dict({site: dict() for site in url_names})
df = pd.DataFrame(columns=['article_text', 'author', 'date_published', 'headline', 'url', 'processed_text', 'source'])
for collection_name in my_collection_names:
    if collection_name != 'system.indexes':
        site = [name for name in url_names if collection_name.startswith(name)]
        if len(site) != 0:
            site = site[0]
            print('Working on '+collection_name)
            for article in db[collection_name].find():
                if 'video' not in article['url']:
                    try:
                        url = article['url']
                        source = article['source']
                        headline = article['headline']
                        date_published = article['date_published']
                        author = article['author']
                        article_text = article['article_text']
                        processed_text = get_processed_text(article_text)
                        # df.append(pd.Series([article_text, author, date_published, headline, url, processed_text, source]), ignore_index=True)
                        df.loc[-1] = [article_text, author, date_published, headline, url, processed_text, source]  # adding a row
                        df.index = df.index + 1  # shifting index
                        df = df.sort()  # sorting by index

                        site_dict[site][url] = article
                    except:
                        print('Problem with article in '+site)
            print('Len of '+site+' is '+str(len(site_dict[site].keys())))

        #toks[name] = []
        # good_article_count = 0
        # bad_article_count = 0
        # for article in db[name].find():
        #     try:
        #         c = article['article_text']
        #         if len(c) != 0:
        #             c = ''.join([l for l in c if l in printable])
        #             wt = word_tokenize(c)
        #             c = [w for w in wt if w.lower() not in stopwords]
        #             lemmatized = [wordnet.lemmatize(i) for i in c]
        #             #toks[name].append(' '.join(lemmatized))
        #             site_dict[site].append(' '.join(lemmatized))
        #             # toks[name][len(toks[name])-1].replace("''","").replace("``","").replace("( CNN )", "").replace(",","").replace("The Situation Room with Wolf Blitzer")
        #             good_article_count += 1
        #     except:
        #         # print('Problem with article in '+site)
        #         # import pdb; pdb.set_trace()
        #         bad_article_count += 1
        # print('# of good articles: ' + str(good_article_count))
        # print('# of bad articles: ' + str(bad_article_count))

df.to_csv(str(db_name)+'_data.csv')

for site in url_names:
    print(site+': '+str(len(site_dict[site].keys())))

for site in url_names:
    for url in site_dict[site].keys():
        if site_dict[site][url]['date_published'] == None:
            print(site+' date: None')
        else:
            print(site+' date: '+str(site_dict[site][url]['date_published']))

with open('pickles/all_articles.pkl', 'wb') as f:
    pickle.dump(site_dict, f, pickle.HIGHEST_PROTOCOL)
