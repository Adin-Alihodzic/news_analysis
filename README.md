# Determining Bias in On-line News Journalism
## Goals
The goal of this project was to determine if there is bias in on-line American political news journalism and if there is, how much and in what direction (liberal or conservative). I measured the bias using sentiment analysis paired with topic modeling. I extract the topics of a weeks worth of news articles and determine their bias score using the formula: `(postive_sentiment - negative_sentiment)*(1-objective_score)*(word_probability)`, where the positive, negative and objective values are determined by the sentiment library and the word probability is the probability that word pertains to that topic and is determined by LDA.

## Data
My data consists of articles gather from the Rich Site Summaries (RSS) feeds of 17 different sites. Those sites with their associated RSS links are [CNN][1], [ABC][2], [FOX][3], [NYT][4], [AP][5], [Reuters][6], [Washing Post][7], [The Economist][8], [Huffington Post][9], [Esquire][10], [Rolling Stone][11], [CBS][12], [FiveThirtyEight][13], [VOX][14], [Time][15], [Slate][16], [The Washington Times][17]. Every hour the articles linked from each RSS feed is scraped and saved to a Mongo database on an Amazon Web Services (AWS). The data is then converted to a CSV file and stored on an S3 bucket.

All past articles from the Wall Street Journal (WSJ) were available, so they were scraped as well. 100 gigabytes of data was collected, but a only the article pertaining to politics were kept. These will be used to show how sentiment/bias changes over time.

## Sentiment
Two sentiment libraries are used in this project; sentiwordnet and pattern. Sentiwordnet is used in python 2, while pattern is still in python 2.

## Ideas
* Post bias calculations to reddit
* Post bias calculations to twitter
* Web site that displays results and description of models
* Have web page that takes in data from user.
  * Ex: display a topic to user and user inputs name of topic and whether it is liberal of conservative and to what degree.
  * Ex: let user input url with their opinion of what the bias is.
  * Ex: Display article to user and ask them what they think the bias is.








[1]: http://rss.cnn.com/rss/cnn_allpolitics.rss
[2]: http://feeds.abcnews.com/abcnews/politicsheadlines
[3]: http://feeds.foxnews.com/foxnews/politics'
[4]: http://rss.nytimes.com/services/xml/rss/nyt/Politics.xml
[5]: http://hosted2.ap.org/atom/APDEFAULT/89ae8247abe8493fae24405546e9a1aa
[6]: http://feeds.reuters.com/Reuters/PoliticsNews'
[7]: http://feeds.washingtonpost.com/rss/politics
[8]: http://www.economist.com/sections/united-states/rss.xml
[9]: http://www.huffingtonpost.com/feeds/verticals/politics/index.xml
[10]: http://www.esquire.com/rss/news-politics.xml
[11]: http://www.rollingstone.com/politics/rss
[12]: http://www.cbsnews.com/latest/rss/politics
[13]: https://fivethirtyeight.com/politics/feed/
[14]: https://www.vox.com/rss/index.xml
[15]: http://feeds.feedburner.com/timeblogs/swampland
[16]: http://feeds.slate.com/slate-101526
[17]: http://www.washingtontimes.com/rss/headlines/news/politics/
