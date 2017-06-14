import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt
import ast

# from data_exploration import process_articles
# from sentiment_analysis import topic_values

from bokeh.models import ColumnDataSource, OpenURL, TapTool, WheelZoomTool, HoverTool, LassoSelectTool, PanTool, CustomJS, Slider
from bokeh.plotting import figure, output_file, show
from bokeh.models.widgets import Panel, Tabs
from bokeh.embed import components
from bokeh.layouts import row, widgetbox

from wordcloud import WordCloud


def make_plot(topic_dict, topic, topic_prob_threshold=0.4):
    """
    Function to get Bokeh plots to be used in web app.
    Following are the steps we take:

    1. Loop through topics and create plot and determine variables for sites.
    2. Get script and div components to use in web app.
    3. Pickle components.
    """
    # analytical_score = {topic: [] for topic in range(len(topic_dict))}
    # for topic in range(len(topic_dict)):
    analytical_score= []
    for tone in topic_dict[topic]['tones']:
        tone = ast.literal_eval(tone)
        # analytical_score[topic].append(tone[1]['Analytical'])
        analytical_score.append(tone[1]['Analytical'])

    # components_dict = {topic: {'script': '', 'div': ''} for topic in range(num_topics+1)}
    # for topic in range(num_topics+1):
    # for topic in range(1):
    output_file("../web_app/bokeh_plots/topic"+str(topic)+".html")

    hover1 = HoverTool(
        tooltips=[
            ("source", "@site"),
            ("(pos,neg)", "(@pos, @neg)"),
            ("Headline", "@headline")
        ]
    )

    hover2 = HoverTool(
        tooltips=[
            ("source", "@site"),
            ("(pos,neg)", "(@pos, @neg)")
        ]
    )

    title="Analytical Score by Overall Score for Topic "+str(topic)

    p1 = figure(plot_width=1200, plot_height=800,
                tools=["tap, pan, wheel_zoom",hover1], title=title,
              toolbar_location="right")

    p2 = figure(plot_width=1200, plot_height=800,
                tools=["tap, pan, wheel_zoom",hover2], title=title,
              toolbar_location="right")

    # p.toolbar.active_drag = 'auto'

    if topic == 0:
        a = 0.2
    else:
        a = 0.6
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0,100,0), (0, 0, 0), (100, 25, 200), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 128, 128), (0, 0, 128), (240,230,140)]
    # sources = ['cnn', 'abc', 'fox', 'nyt', 'reuters', 'wapo', 'huffpo', 'esquire', 'rollingstone', 'cbs', '538', 'washtimes']
    sources = np.unique(topic_dict[topic]['source'])

    pos_by_site = {site: [] for site in sources}
    neg_by_site = {site: [] for site in sources}
    score_by_site = {site: [] for site in sources}
    analytical_by_site = {site: [] for site in sources}
    size_by_site = {site: [] for site in sources}
    url_by_site = {site: [] for site in sources}
    headline_by_site = {site: [] for site in sources}

    for site in sources:
        indices = [j for j, s in enumerate(topic_dict[topic]['source']) if s == site and analytical_score[j] != 0 and topic_dict[topic]['length'][j] > 200]
        if indices == []:
            pass
        else:
            pos_by_site[site] = np.array(topic_dict[topic]['pos'])[indices]
            neg_by_site[site] = np.array(topic_dict[topic]['neg'])[indices]
            score_by_site[site] = (np.array(pos_by_site[site]) + np.array(pos_by_site[site])) * (1 - np.array(topic_dict[topic]['obj'])[indices])
            analytical_by_site[site] = np.array(analytical_score)[indices]
            size_by_site[site] = [50*topic for topic in np.array(topic_dict[topic]['topic_prob'])[indices]]
            url_by_site[site] = np.array(topic_dict[topic]['url'])[indices]
            headline_by_site[site] = np.array(topic_dict[topic]['headline'])[indices]

    sources = []
    for site, color in zip(sources, colors):
        source = ColumnDataSource(data=dict(
            pos=score_by_site[site],
            neg=analytical_by_site[site],
    #         color=["navy", "orange", "olive", "firebrick", "gold"],
            size=size_by_site[site],
            site=[site for i in range(len(pos_by_site[site]))],
            headline=headline_by_site[site],
            url=url_by_site[site]
        ))

        sources.append(source)

        p1.circle('pos', 'neg', color=color, alpha=a, size='size', source=source, legend=site)

        # p1.xaxis.axis_label = "Positive Sentiment"
        # p1.yaxis.axis_label = "Negative Sentiment"

        p1.xaxis.axis_label = "Overall Score"
        p1.yaxis.axis_label = "Analytical Score"

        url = "@url"
        taptool = p1.select(type=TapTool)
        taptool.callback = OpenURL(url=url)

        source21 = ColumnDataSource(data=dict(
            pos=[np.mean(score_by_site[site])],
            neg=[np.mean(analytical_by_site[site])],
    #         color=["navy", "orange", "olive", "firebrick", "gold"],
            size=[np.mean(size_by_site[site])],
            site=[site]
        ))

        source22 = ColumnDataSource(data=dict(
            pos=[np.mean(score_by_site[site])],
            neg=[np.mean(analytical_by_site[site])],
    #         color=["navy", "orange", "olive", "firebrick", "gold"],
            size=[np.var(size_by_site[site])],
            site=[site]
        ))

        p2.circle('pos', 'neg', color=color, alpha=a, size='size', source=source21, legend=site)
        p2.circle('pos', 'neg', color=color, alpha=a, size='size', source=source22, legend=site)


        # p1.xaxis.axis_label = "Positive Sentiment"
        # p1.yaxis.axis_label = "Negative Sentiment"

        p2.xaxis.axis_label = "Overall Score"
        p2.yaxis.axis_label = "Analytical Score"
        p2.legend.location = "bottom_right"

    callback = CustomJS(args=dict(source1=sources[1], source2=sources[2], source3=sources[3], source4=sources[4], source5=sources[5], source6=sources[6], source7=sources[7], source8=sources[8], source9=sources[9], source10=sources[10], source11=sources[11], source12=sources[12]), code="""
        var data = [source1.data, source2.data, source3.data, source4.data, source5.data, source6.data, source7.data, source8.data, source9.data, source10.data, source11.data, source12.data];
        var A = prob.value;
        size = [data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11]]
        for (i = 0; i < )
            for (i = 0; i < size.length; i++) {
                if (size[i] < A) {
                    size[i] = 0;
                }
                else {
                    size[i] = 1;
                }
            }
        source.trigger('change');
    """)

    prob_slider = Slider(start=0.0, end=1.0, value=0.2, step=.1,
                        title="Topic Probability", callback=callback)
    callback.args["prob"] = dict(l0=l0, l1=l1, l2=l2, slider=prob_slider)

    layout = row(
        p1,
        widgetbox(prob_slider),
    )

    tab1 = Panel(child=layout, title="By Article")
    tab2 = Panel(child=p2, title="By Site")

    tabs = Tabs(tabs=[ tab1, tab2 ])

    script, div = components(tabs)
    # script, div = components(p)
    # components_dict[topic]['script'] = script
    # components_dict[topic]['div'] = div

    # return components_dict
    show(tabs)
    return script, div

def make_clouds(topic_texts, lda_model):
    """
    Function to get Word Clouds. Following are the steps we take:

    1. Get Word Cloud of all articles.
    2. Get Word Clouds of each topic.
    3. Save all word clouds to use in web app.
    """
    # Topic 0 refers to all articles
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

    with open('../pickles/lda_model.pkl', 'rb') as f:
        lda_model = pickle.load(f)

    num_topics = lda_model.num_topics

    print('Making topics dictionary...')
    topic_dict = topic_values(df, lda_model)

    print('Making Plots...')
    components_dict = make_plots(topic_dict, num_topics)
    pickle.dump(components_dict, open('../web_app/bokeh_plots/components_dict.pkl', 'wb'))

    print('Processing Articles...')
    topic_texts, sentiment_texts = process_articles(df)

    print('Making WordClouds...')
    make_clouds(topic_texts, lda_model)
