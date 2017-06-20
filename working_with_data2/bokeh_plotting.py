import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt
import ast

from bokeh.models import ColumnDataSource, OpenURL, TapTool, WheelZoomTool, HoverTool, LassoSelectTool, PanTool, CustomJS, Slider
from bokeh.plotting import figure, output_file, show
from bokeh.models.widgets import Panel, Tabs, Dropdown, MultiSelect, CheckboxGroup
from bokeh.embed import components
from bokeh.layouts import row, widgetbox
from bokeh.models.glyphs import Ellipse

from wordcloud import WordCloud


def make_bokeh_plot(topic_dict, topic, new_article=None):
    """
    Function to get Bokeh plots to be used in web app.
    Following are the steps we take:

    1. Loop through topics and create plot and determine variables for sites.
    2. Get script and div components to use in web app.
    3. Pickle components.
    """
    # output_file("web_app/bokeh_plots/topic"+str(topic)+".html")

    hover1 = HoverTool(
        tooltips=[
            ("Source", "@site"),
            ("Headline", "@headline"),
            ("Analytical Score", "@y{1.11}"),
            ("Sentiment Score", "@x{1.11}"),
            ("Positive Score", "@pos{1.11}"),
            ("Negative Score", "@neg{1.11}"),
            ("Objective Score", "@obj{1.11}"),
        ]
    )

    hover2 = HoverTool(
        tooltips=[
            ("Source", "@site"),
            ("Analytical Score", "@y{1.11}"),
            ("Sentiment Score", "@x{1.11}"),
            ("Positive Score", "@pos{1.11}"),
            ("Negative Score", "@neg{1.11}"),
            ("Objective Score", "@obj{1.11}"),
        ]
    )

    title="Analytical Score by Sentiment Score for Topic "+str(topic)

    p1 = figure(plot_width=1200, plot_height=800,
                tools=["tap, pan, wheel_zoom",hover1], title=title,
              toolbar_location="above")
    p1.xaxis.axis_label_text_font_size = "20pt"
    p1.yaxis.axis_label_text_font_size = "20pt"
    # p1.title.label_text_font_size("20pt")

    p2 = figure(plot_width=1200, plot_height=800,
                tools=["tap, pan, wheel_zoom",hover2], title=title,
              toolbar_location="above")
    p2.xaxis.axis_label_text_font_size = "20pt"
    p2.yaxis.axis_label_text_font_size = "20pt"
    # p2.title.label_text_font_size("20pt")
    # p.toolbar.active_drag = 'auto'

    a = 0.6
    if topic == 0:
        a = 0.2
    else:
        a = 0.6
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0,100,0), (0, 0, 0), (100, 25, 200), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 128, 128), (0, 0, 128), (240,230,140)]
    # colors = ['#ff0000', '#00ff00', '#0000ff', '#006400', '#000000', '#6419c8', '#ffff00', '#00ffff', '	#ff00ff', '#808080', '#000080', '#f0e68c']
    # colors = ['Red', 'Lime', 'Blue', 'DarkGreen', 'Black', 'No Name', 'Yellow', 'Aqua', 'Fuchsia', 'Grey', 'Navy', 'Khaki']
    sources = ['cnn', 'abc', 'fox', 'nyt', 'reuters', 'wapo', 'huffpo', 'esquire', 'rollingstone', 'cbs', '538', 'washtimes']
    colors = {'538': '#ff0000', 'abc': '#00ff00', 'cbs': '#0000ff', 'cnn': '#006400', 'esquire': '#000000', 'fox': '#6419c8', 'huffpo': '#ffff00', 'nyt': '#00ffff', 'reuters': '#ff00ff', 'rollingstone': '#808080', 'wapo': '#000080', 'washtimes': '#f0e68c'}

    # sources = np.unique(topic_dict[topic]['source'])

    pos_by_site = {site: [] for site in sources}
    neg_by_site = {site: [] for site in sources}
    obj_by_site = {site: [] for site in sources}
    score_by_site = {site: [] for site in sources}
    analytical_by_site = {site: [] for site in sources}
    size_by_site = {site: [] for site in sources}
    url_by_site = {site: [] for site in sources}
    headline_by_site = {site: [] for site in sources}
    site_by_site = {site: [] for site in sources}
    color_by_site = {site: [] for site in sources}

    for site in sources:
        indices = [j for j, s in enumerate(topic_dict[topic]['source']) if s == site and topic_dict[topic]['Analytical'][j] != 0 and topic_dict[topic]['length'][j] > 200]
        if indices == []:
            pos_by_site[site] = [0]
            neg_by_site[site] = [0]
            obj_by_site[site] = [0]
            score_by_site[site] = [0]
            analytical_by_site[site] = [0]
            size_by_site[site] = [0]
            url_by_site[site] = ['']
            headline_by_site[site] = ['']
            site_by_site[site] = [site]
            color_by_site[site] = [colors[site]]
        else:
            pos_by_site[site] = np.array(topic_dict[topic]['pos'])[indices]
            neg_by_site[site] = np.array(topic_dict[topic]['neg'])[indices]
            obj_by_site[site] = np.array(topic_dict[topic]['obj'])[indices]
            score_by_site[site] = (np.array(pos_by_site[site]) + np.array(pos_by_site[site])) * (1 - np.array(topic_dict[topic]['obj'])[indices])
            analytical_by_site[site] = np.array(topic_dict[topic]['Analytical'])[indices]
            size_by_site[site] = [50*topic for topic in np.array(topic_dict[topic]['topic_prob'])[indices]]
            url_by_site[site] = np.array(topic_dict[topic]['url'])[indices]
            headline_by_site[site] = np.array(topic_dict[topic]['headline'])[indices]
            site_by_site[site] = [site for i in range(len(indices))]
            color_by_site[site] = np.array([colors[site] for i in range(len(indices))])


    source = ColumnDataSource(data=dict(
        x=np.concatenate([np.array(score_by_site[x]) for x in sorted(score_by_site)]),
        y=np.concatenate([analytical_by_site[x] for x in sorted(analytical_by_site)]),
        pos=np.concatenate([pos_by_site[x] for x in sorted(pos_by_site)]),
        neg=np.concatenate([neg_by_site[x] for x in sorted(neg_by_site)]),
        obj=np.concatenate([obj_by_site[x] for x in sorted(obj_by_site)]),
        size=np.concatenate([size_by_site[x] for x in sorted(size_by_site)]),
        site=np.concatenate([site_by_site[x] for x in sorted(site_by_site)]),
        color=np.concatenate([color_by_site[x] for x in sorted(color_by_site)]),
        headline=np.concatenate([headline_by_site[x] for x in sorted(headline_by_site)]),
        url=np.concatenate([url_by_site[x] for x in sorted(url_by_site)])
    ))

    original_source = ColumnDataSource(data=dict(
        x=np.concatenate([score_by_site[x] for x in sorted(score_by_site)]),
        y=np.concatenate([analytical_by_site[x] for x in sorted(analytical_by_site)]),
        size=np.concatenate([size_by_site[x] for x in sorted(size_by_site)]),
        site=np.concatenate([site_by_site[x] for x in sorted(site_by_site)]),
        color=np.concatenate([color_by_site[x] for x in sorted(color_by_site)]),
        headline=np.concatenate([headline_by_site[x] for x in sorted(headline_by_site)]),
        url=np.concatenate([url_by_site[x] for x in sorted(url_by_site)])
    ))

    checkbox_source = ColumnDataSource(data=dict(
        x=np.concatenate([score_by_site[x] for x in sorted(score_by_site)]),
        y=np.concatenate([analytical_by_site[x] for x in sorted(analytical_by_site)]),
        size=np.concatenate([size_by_site[x] for x in sorted(size_by_site)]),
        site=np.concatenate([site_by_site[x] for x in sorted(site_by_site)]),
        color=np.concatenate([color_by_site[x] for x in sorted(color_by_site)]),
        headline=np.concatenate([headline_by_site[x] for x in sorted(headline_by_site)]),
        url=np.concatenate([url_by_site[x] for x in sorted(url_by_site)])
    ))

    slider_source = ColumnDataSource(data=dict(
        size=np.concatenate([size_by_site[x] for x in sorted(size_by_site)]),
        prob_slider=[0]
    ))


    p1.circle('x', 'y', color='color', alpha=a, size='size', source=source, legend='site')
    if new_article != None:
        article_legend=['Your Article']
        p1.diamond(x=new_article[0], y=new_article[1], size=20,
            color="#000000", line_width=2, legend=article_legend)

    # p1.xaxis.axis_label = "Positive Sentiment"
    # p1.yaxis.axis_label = "Negative Sentiment"

    p1.xaxis.axis_label = "Sentiment Score"
    p1.yaxis.axis_label = "Analytical Score"

    url = "@url"
    taptool = p1.select(type=TapTool)
    taptool.callback = OpenURL(url=url)

    code="""
        var indexOf = [].indexOf || function(item) { for (var i = 0, l = this.length; i < l; i++) { if (i in this && this[i] === item) return i; } return -1; };
        var data = source.data;
        var original_data = original_source.data;
        var checkbox_data = checkbox_source.data;
        var slider_data = slider_source.data;
        size = data['size']
        site = data['site']
        original_size = original_data['size']
        checkbox_size = checkbox_data['size']
        slider_size = slider_data['size']
        original_site = original_data['site']
        prob_slider = slider_data['prob_slider']
        for (i = 0; i < size.length; i++) {
            if (site[i] == '538') {
                if (indexOf.call(checkbox.active,0)>=0) {
                    if (original_size[i] < 50*prob_slider) {
                        size[i] = 0;
                    } else {
                        size[i] = original_size[i];
                        checkbox_size[i] = original_size[i];
                    }
                } else {
                    size[i] = 0;
                    checkbox_size[i] = 0;
                }
            }else if (site[i] == 'abc') {
                if (indexOf.call(checkbox.active,1)>=0) {
                    if (original_size[i] < 50*prob_slider) {
                        size[i] = 0;
                    } else {
                        size[i] = original_size[i];
                        checkbox_size[i] = original_size[i];
                    }
                } else {
                    size[i] = 0;
                    checkbox_size[i] = 0;
                }
            }else if (site[i] == 'cbs') {
                if (indexOf.call(checkbox.active,2)>=0) {
                    if (original_size[i] < 50*prob_slider) {
                        size[i] = 0;
                    } else {
                        size[i] = original_size[i];
                        checkbox_size[i] = original_size[i];
                    }
                } else {
                    size[i] = 0;
                    checkbox_size[i] = 0;
                }
            }else if (site[i] == 'cnn') {
                if (indexOf.call(checkbox.active,3)>=0) {
                    if (original_size[i] < 50*prob_slider) {
                        size[i] = 0;
                    } else {
                        size[i] = original_size[i];
                        checkbox_size[i] = original_size[i];
                    }
                } else {
                    size[i] = 0;
                    checkbox_size[i] = 0;
                }
            }else if (site[i] == 'esquire') {
                if (indexOf.call(checkbox.active,4)>=0) {
                    if (original_size[i] < 50*prob_slider) {
                        size[i] = 0;
                    } else {
                        size[i] = original_size[i];
                        checkbox_size[i] = original_size[i];
                    }
                } else {
                    size[i] = 0;
                    checkbox_size[i] = 0;
                }
            }else if (site[i] == 'fox') {
                if (indexOf.call(checkbox.active,5)>=0) {
                    if (original_size[i] < 50*prob_slider) {
                        size[i] = 0;
                    } else {
                        size[i] = original_size[i];
                        checkbox_size[i] = original_size[i];
                    }
                } else {
                    size[i] = 0;
                    checkbox_size[i] = 0;
                }
            }else if (site[i] == 'huffpo') {
                if (indexOf.call(checkbox.active,6)>=0) {
                    if (original_size[i] < 50*prob_slider) {
                        size[i] = 0;
                    } else {
                        size[i] = original_size[i];
                        checkbox_size[i] = original_size[i];
                    }
                } else {
                    size[i] = 0;
                    checkbox_size[i] = 0;
                }
            }else if (site[i] == 'nyt') {
                if (indexOf.call(checkbox.active,7)>=0) {
                    if (original_size[i] < 50*prob_slider) {
                        size[i] = 0;
                    } else {
                        size[i] = original_size[i];
                        checkbox_size[i] = original_size[i];
                    }
                } else {
                    size[i] = 0;
                    checkbox_size[i] = 0;
                }
            }else if (site[i] == 'reuters') {
                if (indexOf.call(checkbox.active,8)>=0) {
                    if (original_size[i] < 50*prob_slider) {
                        size[i] = 0;
                    } else {
                        size[i] = original_size[i];
                        checkbox_size[i] = original_size[i];
                    }
                } else {
                    size[i] = 0;
                    checkbox_size[i] = 0;
                }
            }else if (site[i] == 'rollingstone') {
                if (indexOf.call(checkbox.active,9)>=0) {
                    if (original_size[i] < 50*prob_slider) {
                        size[i] = 0;
                    } else {
                        size[i] = original_size[i];
                        checkbox_size[i] = original_size[i];
                    }
                } else {
                    size[i] = 0;
                    checkbox_size[i] = 0;
                }
            }else if (site[i] == 'wapo') {
                if (indexOf.call(checkbox.active,10)>=0) {
                    if (original_size[i] < 50*prob_slider) {
                        size[i] = 0;
                    } else {
                        size[i] = original_size[i];
                        checkbox_size[i] = original_size[i];
                    }
                } else {
                    size[i] = 0;
                    checkbox_size[i] = 0;
                }
            }else if (site[i] == 'washtimes') {
                if (indexOf.call(checkbox.active,11)>=0) {
                    if (original_size[i] < 50*prob_slider) {
                        size[i] = 0;
                    } else {
                        size[i] = original_size[i];
                        checkbox_size[i] = original_size[i];
                    }
                } else {
                    size[i] = 0;
                    checkbox_size[i] = 0;
                }
            }
        }
        source.trigger('change');
        checkbox_source.trigger('change');
        """

    # callback_checkbox = CustomJS(args=dict(source=source, original_source=original_source))

    # values = ["all", "538", "abc", "cbs", "cnn", "esquire", "fox", "huffpo", "nyt", "reuters", "rollingstone", "wapo", "washtimes"]
    options = ["FiveThirtyEight", "ABC", "CBS", "CNN", "Esquire", "FOX", "Huffington Post", "NY Times", "Reuters", "Rolling Stone", "Washington Post", "Washington Times"]
    # multi_select = MultiSelect(title="Option:", value=values, options=options, callback=callback_dropdown, size=13)
    # callback_dropdown.args["site"] = multi_select

    callback_checkbox = CustomJS(code=code, args={})
    checkbox = CheckboxGroup(labels=options, active=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], callback=callback_checkbox, width=100)
    callback_checkbox.args = dict(source=source, original_source=original_source, checkbox_source=checkbox_source, slider_source=slider_source, checkbox=checkbox)

    callback_slider = CustomJS(args=dict(source=source, checkbox_source=checkbox_source, slider_source=slider_source), code="""
        var data = source.data;
        var checkbox_data = checkbox_source.data;
        var slider_data = slider_source.data;
        var s = prob.value;
        size = data['size']
        prob_slider = slider_data['prob_slider']
        prob_slider = s
        checkbox_size = checkbox_data['size']
        for (i = 0; i < size.length; i++) {
            if (checkbox_size[i] < 50*s) {
                size[i] = 0;
            }
            else {
                size[i] = checkbox_size[i];
            }
        }
        slider_data['prob_slider'] = s;
        source.trigger('change');
        slider_source.trigger('change');
    """)

    prob_slider = Slider(start=0.0, end=1.0, value=0.0, step=.01,
                        title="Topic Probability", callback=callback_slider)
    callback_slider.args["prob"] = prob_slider

    layout = row(
        p1,
        widgetbox(prob_slider, checkbox),
    )

    source21 = ColumnDataSource(data=dict(
        x=[np.mean(score) for score in score_by_site.values()],
        y=[np.mean(analytical) for analytical in analytical_by_site.values()],
        pos=[np.mean(pos) for pos in pos_by_site.values()],
        neg=[np.mean(neg) for neg in neg_by_site.values()],
        obj=[np.mean(obj) for obj in obj_by_site.values()],
        size=[np.mean(size) for size in size_by_site.values()],
        color=[c for c in colors.values()],
        site=sources
    ))

    source22 = ColumnDataSource(data=dict(
        x=[np.mean(score) for score in score_by_site.values()],
        y=[np.mean(analytical) for analytical in analytical_by_site.values()],
        pos=[np.mean(pos) for pos in pos_by_site.values()],
        neg=[np.mean(neg) for neg in neg_by_site.values()],
        obj=[np.mean(obj) for obj in obj_by_site.values()],
        size=[np.mean(size)*(np.std(score)+np.std(analytical)) for size,score,analytical in zip(size_by_site.values(),score_by_site.values(),analytical_by_site.values())],
        color=[c for c in colors.values()],
        site=sources
    ))

    p2.circle('x', 'y', color='color', alpha=1.0, size='size', source=source21, legend='site')
    if new_article != None:
        article_legend=['Your Article']
        p2.diamond(x=new_article[0], y=new_article[1], size=20,
            color="#000000", line_width=2, legend=article_legend)

    source23 = ColumnDataSource(data=dict(
        x=[np.mean(score) for score in score_by_site.values()],
        y=[np.mean(analytical) for analytical in analytical_by_site.values()],
        pos=[np.mean(pos) for pos in pos_by_site.values()],
        neg=[np.mean(neg) for neg in neg_by_site.values()],
        obj=[np.mean(obj) for obj in obj_by_site.values()],
        size=[np.mean(size)*(np.std(score)+np.std(analytical)) for size,score,analytical in zip(size_by_site.values(),score_by_site.values(),analytical_by_site.values())],
        w=[np.std(score) for score in score_by_site.values()],
        h=[np.std(analytical) for analytical in analytical_by_site.values()],
        color=[c for c in colors.values()],
        site=sources
    ))
    glyph = Ellipse(x="x", y="y", width="w", height="h", line_alpha=0.0, fill_alpha=0.2, fill_color="color")
    p2.add_glyph(source23, glyph)


    p2.xaxis.axis_label = "Sentiment Score"
    p2.yaxis.axis_label = "Analytical Score"
    p2.legend.location = "bottom_right"


    tab1 = Panel(child=layout, title="By Article")
    tab2 = Panel(child=p2, title="By Site")

    tabs = Tabs(tabs=[ tab1, tab2 ])

    # show(tabs)

    script, div = components(tabs)

    return script, div

def make_clouds(topic_texts, lda_model):
    """
    Function to get Word Clouds. Following are the steps we take:

    1. Get Word Cloud of all articles.
    2. Get Word Clouds of each topic.
    3. Save all word clouds to use in web app.
    """
    topic_texts = [text.split(' ') for text in df['topic_texts']]
    # Topic 0 refers to all articles
    figs = []

    fig = plt.figure(figsize=(16,12), dpi=300)
    plt.imshow(WordCloud(background_color="white", width=1200, height=800).generate(' '.join([' '.join(text) for text in topic_texts])), interpolation="bilinear")
    plt.axis("off")
    plt.title("Topic #0")
    figs.append(fig)

    for t in range(0, lda_model.num_topics):
        fig = plt.figure(figsize=(16,12), dpi=300)
        topic_word_probs = dict()
        lda_topics = lda_model.show_topics(num_topics=-1, num_words=100000,formatted=False)
        for word_prob in lda_topics[t][1]:
            topic_word_probs[word_prob[0]] = word_prob[1]
        plt.imshow(WordCloud(background_color="white", width=1200, height=800).fit_words(topic_word_probs), interpolation="bilinear")
        plt.axis("off")
        plt.title("Topic #" + str(t+1))
        plt.rcParams.update({'font.size': 22})
        figs.append(fig)

    return figs

if __name__ == '__main__':
    df = pd.read_csv('../data/rss_feeds_new_good.csv')
    df = df[pd.notnull(df['article_text'])]

    with open('../pickles/lda_model.pkl', 'rb') as f:
        lda_model = pickle.load(f)

    with open('../pickles/topic_dict.pkl', 'rb') as f:
        topic_dict = pickle.load(f)

    num_topics = lda_model.num_topics

    # print('Making topics dictionary...')
    # topic_dict = topic_values(df, lda_model)

    print('Making Plots...')
    components_dict = [{'script': None, 'div': None} for topic in range(self.lda_model.num_topics)]
    for topic in range(lda_model.num_topics):
        components_dict[topic]['script'], components_dict[topic]['div'] = make_bokeh_plot(self.topic_dict, topic)
    pickle.dump(components_dict, open('../web_app/bokeh_plots/components_dict.pkl', 'wb'))

    print('Processing Articles...')
    topic_texts, sentiment_texts = process_articles(df)

    print('Making WordClouds...')
    figs = make_clouds(topic_texts, lda_model)
    for i,fig in enumerate(cloud_figs):
        fig.savefig('../web_app/static/img/wordclouds/wordcloud_topic'+str(t)+'.png')
