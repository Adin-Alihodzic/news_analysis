import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt

from data_exploration import process_articles
from sentiment_analysis import topic_values

from bokeh.models import ColumnDataSource, OpenURL, TapTool, WheelZoomTool, HoverTool, LassoSelectTool, PanTool
from bokeh.plotting import figure, output_file, show
from bokeh.models.widgets import Panel, Tabs
from bokeh.embed import components

from wordcloud import WordCloud


def make_plots(topic_dict, num_topics):
    components_dict = {topic: {'script': '', 'div': ''} for topic in range(num_topics+1)}
    for topic in range(num_topics+1):
    # for topic in range(1):
        output_file("../web_app/bokeh_plots/topic"+str(topic)+".html")

        hover = HoverTool(
            tooltips=[
                ("source", "@site"),
                ("(pos,neg)", "(@pos, @neg)"),
                ("Headline", "@headline")
            ]
        )

        p = figure(plot_width=1200, plot_height=800,
                    tools=["tap, pan, wheel_zoom",hover], title="Topic: "+str(topic),
                  toolbar_location="right")

        # p.toolbar.active_drag = 'auto'

        a = 0.6
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0,100,0), (0, 0, 0), (100, 25, 200), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 128, 128), (0, 0, 128), (240,230,140)]
        # sources = ['cnn', 'abc', 'fox', 'nyt', 'reuters', 'wapo', 'huffpo', 'esquire', 'rollingstone', 'cbs', '538', 'washtimes']
        sources = np.unique(topic_dict[topic]['source'])

        pos_by_site = {site: [] for site in sources}
        neg_by_site = {site: [] for site in sources}
        size_by_site = {site: [] for site in sources}
        url_by_site = {site: [] for site in sources}
        headline_by_site = {site: [] for site in sources}
        for site in sources:
            indices = [j for j, s in enumerate(topic_dict[topic]['source']) if s == site]
            if indices == []:
                pass
            else:
                pos_by_site[site] = np.array(topic_dict[topic]['pos'])[indices]
                neg_by_site[site] = np.array(topic_dict[topic]['neg'])[indices]
                size_by_site[site] = [50*topic for topic in np.array(topic_dict[topic]['topic_prob'])[indices]]
                url_by_site[site] = np.array(topic_dict[topic]['url'])[indices]
                headline_by_site[site] = np.array(topic_dict[topic]['headline'])[indices]

        for site, color in zip(sources, colors):
            source = ColumnDataSource(data=dict(
                pos=pos_by_site[site],
                neg=neg_by_site[site],
        #         color=["navy", "orange", "olive", "firebrick", "gold"],
                size=size_by_site[site],
                site=[site for i in range(len(pos_by_site[site]))],
                headline=headline_by_site[site],
                url=url_by_site[site]
            ))

            p.circle('pos', 'neg', color=color, alpha=a, size='size', source=source, legend=site)

            p.xaxis.axis_label = "Positive Sentiment"
            p.yaxis.axis_label = "Negative Sentiment"

            url = "@url"
            taptool = p.select(type=TapTool)
            taptool.callback = OpenURL(url=url)

        script, div = components(p)
        components_dict[topic]['script'] = script
        components_dict[topic]['div'] = div

    pickle.dump(components_dict, open('../web_app/bokeh_plots/components_dict.pkl', 'wb'))

    return components_dict

def make_clouds(topic_texts, lda_model):
    plt.imshow(WordCloud(background_color="white", width=1200, height=800).generate(' '.join([' '.join(text) for text in topic_texts])), interpolation="bilinear")
    plt.axis("off")
    plt.title("Topic #0")
    plt.savefig('../web_app/static/img/wordclouds/wordcloud_topic0.png', dpi=300)

    for t in range(1, lda_model.num_topics+1):
        topic_word_probs = dict()
        lda_topics = lda_model.show_topics(num_topics=-1, num_words=100000,formatted=False)
        for word_prob in lda_topics[t][1]:
            topic_word_probs[word_prob[0]] = word_prob[1]
        plt.imshow(WordCloud(background_color="white", width=1200, height=800).fit_words(topic_word_probs), interpolation="bilinear")
        plt.axis("off")
        plt.title("Topic #" + str(t))
        plt.savefig('../web_app/static/img/wordclouds/wordcloud_topic'+str(t)+'.png', dpi=300)



if __name__ == '__main__':
    df = pd.read_csv('../data/rss_feeds_new_good.csv')
    df = df[pd.notnull(df['article_text'])]

    with open('../working_with_data/lda_model.pkl', 'rb') as f:
        lda_model = pickle.load(f)

    num_topics = lda_model.num_topics

    print('Making topics dictionary...')
    topic_dict = topic_values(df, lda_model)

    print('Making Plots...')
    components_dict = make_plots(topic_dict, num_topics)

    print('Processing Articles...')
    topic_texts, sentiment_texts = process_articles(df)

    print('Making WordClouds...')
    make_clouds(topic_texts, lda_model)
