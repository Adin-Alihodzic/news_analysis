import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode, plot, iplot

a = 0.9
colors = ['rgba(255, 0, 0, '+str(a)+')', 'rgba(0, 255, 0, '+str(a)+')', 'rgba(0, 0, 255, '+str(a)+')', 'rgba(255, 255, 255, '+str(a)+')', 'rgba(0, 0, 0, '+str(a)+')', 'rgba(, , , '+str(a)+')', 'rgba(255, 255, 0, '+str(a)+')', 'rgba(0, 255, 255, '+str(a)+')', 'rgba(255, 0, 255, '+str(a)+')', 'rgba(128, 128, 128, '+str(a)+')', 'rgba(0, 0, 128, '+str(a)+')', 'rgba(128, 0, 128, '+str(a)+')']
sources = ['cnn', 'abc', 'fox', 'nyt', 'reuters', 'wapo', 'huffpo', 'esquire', 'rollingstone', 'cbs', '538', 'washtimes']

traces = []
for i, site in enumerate(sources):
    indices = [j for j, t in enumerate(source) if t == site]
    if indices == []:
        print('Not: '+site)
    else:
        print(site)
        trace = go.Scatter(
            x = np.array(x)[indices],
            y = np.array(y)[indices],
            name = site,
#           text = ['<a href=\{0}>link to article</a>'.format(link) for link in np.array(url)[indices]],
            mode = 'markers',
            marker = dict(
                size = [50*topic for topic in np.array(topic_prob)[indices]],
                color = colors[i],
                line = dict(
                    width = 1,
                )
            )
        )
        traces.append(trace)

# plotAnnotes = []

# plotAnnotes.append(dict(x=x,
#                         y=y,
#                         text=['<a href="{0}">link to article</a>'.format(link) for link in url],
#                         showarrow=False,
#                         xanchor='center',
#                         yanchor='center',
#                         ))
# layout = go.Layout(
#     showlegend=True,
#     annotations=plotAnnotes
# )
data = [trace for trace in traces]
fig = go.Figure(data=data)#, layout=layout)


# Plot and embed in ipython notebook!
plotly.offline.init_notebook_mode()
from plotly.offline import iplot
# plotly_url = plotly.offline.plot(fig, filename='basic-scatter.html', auto_open=True)
# plotly_url = iplot(fig)

from plotly.widgets import GraphWidget
import pathlib
url = pathlib.Path('/home/ian/Galvanize/news_bias/working_with_data/basic-scatter.html').as_uri()

# graph = GraphWidget(HTML(filename='/home/ian/Galvanize/news_bias/working_with_data/basic-scatter.html'))
graph = GraphWidget(url)
g = graph
from IPython.display import display
display(graph)
# print(HTML(filename='/home/ian/Galvanize/news_bias/working_with_data/basic-scatter.html'))
from IPython.display import Image
from IPython.display import HTML
# display(HTML(filename='/home/ian/Galvanize/news_bias/working_with_data/basic-scatter.html'))
def message_handler(widget, msg):
    clear_output()
    print(idget._graph_url)
    display(msg)

g.on_click(message_handler)
# help(GraphWidget)

# from ipywidgets import Output
# plotly.offline.init_notebook_mode()
# ow = Output()
# with ow:
#     plotly.offline.plot(fig, filename='basic-scatter.html')
# ow
