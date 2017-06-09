from flask import Flask, render_template, send_from_directory
import pickle
import requests
import pandas as pd
from bs4 import BeautifulSoup
import sys
from sklearn import metrics
from pymongo import MongoClient
import random
import codecs
from newspaper import Article
from gensim.summarization import summarize
import datetime

from flask import Flask, render_template, request
import webbrowser, threading, os

from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8

import nltk
from nltk.corpus import sentiwordnet as swn

sys.path.append('/home/ian/Galvanize/news_bias/working_with_data2')
from data_exploration import process_articles
from sentiment_analysis import article_sentiment
sys.path.append('/home/ian/Galvanize/news_bias/web_app')

app = Flask(__name__)

# Create MongoClient
# client = MongoClient()
# # Initialize the Database
# db = client['events']
# tab = db['predicted_events']


def get_components(topic):
    with open('bokeh_plots/components_dict.pkl', 'rb') as f:
        components_dict = pickle.load(f)
    script = components_dict[topic]['script']
    div = components_dict[topic]['div']
    return script, div

def get_article(url):
    try:
        a = Article(url)
        attempt = 0
        while a.html == '' and attempt < 10:
            a = Article(url)
            a.download()
            attempt += 1
        if attempt >= 10:
            print('Article would not download!')
            return False, ()
        if a.is_downloaded:
            a.parse()
        else:
            print('Article would not download!')
            return False, ()
    except:
        return 'Article would not download!'
    try:
        headline = a.title
    except:
        return False, ()
    try:
        date_published = a.publish_date
        if date_published == '' or date_published == None:
            date_published = datetime.datetime.now()
    except:
        date_published = datetime.datetime.now()
    try:
        author = a.authors
    except:
        author = None
    try:
        article_text = a.text
    except:
        return False, ()

    return True, (article_text, headline, author, date_published)

def get_summary(article_text):
    '''Uses summarization function from gensim library to summarize each article'''
    summary = ''
    try:
        summary = summarize(article_text)
    except:
        summary = ''

    return summary

def get_sentiment(word):
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

def get_article_sentiment(topic_texts, sentiment_texts):
    sentiment_texts_words = set()
    for i in range(len(sentiment_texts)):
        sentiment_texts_words = sentiment_texts_words | set(sentiment_texts[i])
    sentiment_texts_words = list(sentiment_texts_words)

    relevant_types = ['JJ', 'VB', 'RB']

    s_pos = 0
    s_neg = 0
    s_obj = 0
    relevant_word_count = 0
    for word in sentiment_texts_words:
        for word, word_type in nltk.pos_tag([word]):
            if word_type in relevant_types:
                relevant_word_count += 1
                pos, neg, obj = get_sentiment(word)
                if pos == 0 and neg == 0:
                    pass
                else:
                    s_pos += pos
                    s_neg += neg
                    s_obj += obj
    if relevant_word_count != 0:
         s_pos, s_neg, s_obj = s_pos/relevant_word_count, s_neg/relevant_word_count, s_obj/relevant_word_count

    return s_pos, s_neg, s_obj


# home page
@app.route('/')
def index():
    return render_template('home.html')

# Button for prediction page
@app.route('/input')
def results():
    return render_template('input.html')

# predict page
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        url = str(request.form['url'])
        result = get_article(url)
        if result[0] != False:
            article_text, headline, author, date_published = result[1]
            summary = get_summary(article_text)

            data = {'article_text': article_text, 'headline': headline}
            df = pd.DataFrame(data, index=[0])

            topic_texts, sentiment_texts = process_articles(df)

            pos, neg, obj = get_article_sentiment(topic_texts, sentiment_texts)
            with open('../working_with_data/lda_model.pkl', 'rb') as f:
                lda_model = pickle.load(f)

            article_bow = lda_model.id2word.doc2bow(topic_texts[0])
            article_topics = lda_model[article_bow]

            max_topic = 0
            max_prob = 0
            for topic_and_prob in article_topics:
                topic = topic_and_prob[0]
                prob = topic_and_prob[1]
                if prob > max_prob:
                    max_topic = topic
                    max_prob = prob



            return render_template('prediction_worked.html', article_text=article_text, headline=headline, author=author, date_published=date_published, summary=summary, pos=pos, neg=neg, obj=obj, topic=max_topic, topic_prob=max_prob)
        else:

            return render_template('prediction_failed.html')

# graphs page
@app.route('/graphs')
def graphs():
    f = codecs.open("plots/pyLDAvis_40_topics.html", 'r')
    pyLDAvis_html = f.read()

    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    html = render_template('graphs.html',
                            js_resources=js_resources,
                            css_resources=css_resources,
                            pyLDAvis_html=pyLDAvis_html)
    return encode_utf8(html)

# graphs page
@app.route('/graphs_input', methods=['POST','GET'])
def graphs_input():
    if request.method == 'POST':
        topic = int(request.form['topic'])
        f = codecs.open("plots/pyLDAvis_40_topics.html", 'r')
        pyLDAvis_html = f.read()
        # return render_template('graphs.html', plot='bokeh_plots/topic'+str(selectedValue)+'.html')
        script, div = get_components(topic=topic)
    # return render_template('graphs.html', plot='./bokeh_plots/topic0.html')

        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        html = render_template('graphs_input.html',
                                js_resources=js_resources,
                                css_resources=css_resources,
                                script=script,
                                div=div,
                                topic_num=topic,
                                word_cloud='<img src="../static/img/wordclouds/wordcloud_topic'+str(topic)+'.png" style="width: 100%; display:inline-block;"/>')
        return encode_utf8(html)

# about page
@app.route('/about')
def about():
    return render_template('about.html')

# contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# @app.route('/more/')
# def more():
#     return render_template('starter_template.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8105, debug=True)
