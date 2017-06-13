# Analyzing On-line News Journalism
## Goals
The goal of this project was to determine if there is bias in on-line American political news journalism and if there is, how much and in what direction (liberal or conservative). I measured the bias using sentiment analysis paired with topic modeling. I extract the topics of a weeks worth of news articles and determine their bias score using the formula: `(postive_sentiment - negative_sentiment)*(1-objective_score)*(word_probability)`, where the positive, negative and objective values are determined by the sentiment library and the word probability is the probability that word pertains to that topic and is determined by LDA.

## Data
My data consists of articles gather from the Rich Site Summaries (RSS) feeds of 12 different sites. Those sites with their associated RSS links are [CNN][1], [ABC][2], [FOX][3], [NYT][4], [Reuters][5], [Washing Post][6], [Huffington Post][7], [Esquire][8], [Rolling Stone][9], [CBS][10], [FiveThirtyEight][11], [The Washington Times][12]. Every hour the articles linked from each RSS feed are scraped and saved to a Mongo database on an Amazon Web Services (AWS) server. The data is then converted to a CSV file and stored on an S3 bucket.

All past articles from the Wall Street Journal (WSJ) were available, so they were scraped as well. 100 gigabytes of data was collected, but a only the article pertaining to politics were kept. These will be used to show how sentiment/bias changes over time.

## Sentiment
Two sentiment libraries are used in this project; sentiwordnet and pattern. Sentiwordnet is used in python 3, while pattern is still in python 2.

## Cleaning Data
For sentiment analysis we only want to look at words the author wrote. Therefore, we have to remove quotes and tweets that are included in the article. However, quotes and tweets can be useful in determining what the article is talking about, so we don't want to ignore them entirely. I decided to split the article into words that had to do with the topic and words that had to do with sentiment.


## Ideas
* How to determine how many features and topics to use in LDA
* Determine whether the article is political
* Use LDA in Spark to find topics of larger WSJ dataset
  * Get topics week to week over 2016 and maybe even further
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
[5]: http://feeds.reuters.com/Reuters/PoliticsNews'
[6]: http://feeds.washingtonpost.com/rss/politics
[7]: http://www.huffingtonpost.com/feeds/verticals/politics/index.xml
[8]: http://www.esquire.com/rss/news-politics.xml
[9]: http://www.rollingstone.com/politics/rss
[10]: http://www.cbsnews.com/latest/rss/politics
[11]: https://fivethirtyeight.com/politics/feed/
[12]: http://www.washingtontimes.com/rss/headlines/news/politics/
