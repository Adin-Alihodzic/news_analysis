from textstat.textstat import textstat
import pickle
import numpy as np

with open('pickles/all_articles.pkl', 'rb') as f:
    articles_dict = pickle.load(f)

for site in articles_dict.keys():
    fre = []
    for url in articles_dict[site].keys():
        fre.append(textstat.flesch_reading_ease(articles_dict[site][url]['article_text']))
    print(site+': '+str(np.array(fre).mean()))
