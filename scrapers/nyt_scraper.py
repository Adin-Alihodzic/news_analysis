from pymongo import MongoClient
from bs4 import BeautifulSoup
from requests import get
import json
from unidecode import unidecode
from load_data import load_urls, get_file_name
from sys import argv
import datetime


def add_to_mongo(tab, url):
    if already_exists(tab, url):
        print('here')
        return False
    try:
        html = get(url)
    except:
        return url

    soup = BeautifulSoup(html.content, 'html.parser')

    try:
        headline = unidecode(soup.find('h1', attrs={'itemprop': 'headline'}).contents[0])
    except:
        return url
    try:
        date_published = soup.find('time', attrs={'class': 'dateline'}).get('datetime')
    except:
        return url
    try:
        authors = soup.findAll('span', attrs={'class': 'byline-author'})
        author = ' and '.join([a.text for a in authors])
    except:
        author = None
    try:
        paragraphs = soup.findAll('p', attrs={'class': 'story-body-text story-content'})
        paragraphs = [unidecode(p.text) for p in paragraphs]
        article_text = ' \n '.join(paragraphs)
    except:
        return url

    insert = {'url': url,
              'source': 'foxnews',
              'headline': headline,
              'date_published': date_published,
              'author': author,
              'article_text': article_text}
    tab.insert_one(insert)
    return False


def already_exists(tab, url):
    return bool(tab.find({'url': url}).count())


if __name__=='__main__':
    ''' This script should be called in the following way:
    $ python fox_scraper.py 'startdate' 'enddate' 'table (optional)'
    '''
    start_date, end_date = argv[1], argv[2]

    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    # Create MongoClient
    client = MongoClient()
    # Initialize the Database
    db = client['election_analysis']
    while True:
        # Initialize table
        # If a table name has been provided use that, otherwise initialize 'articles' table
        if len(argv) > 3:
            tab = db[argv[3]]
        else:
            tab = db['nyt_'+start_datetime.strftime('%Y%m%d')+'_'+end_datetime.strftime('%Y%m%d')]

        print('Scraping NYT URLs from {0} to {1}'.format(start_date, end_date))

        file_path = '../url_files/{0}'.format(get_file_name('nyt', start_date, end_date))
        urls = load_urls(file_path)

        bad_urls = []
        for url in urls:
            result = add_to_mongo(tab, url)
            if result:
                bad_urls.append(result)

        print('NYT Scraping Done...')
        print('Number of Bad Extractions = {0}'.format(bad_urls))


        start_datetime = start_datetime - datetime.timedelta(days=7)
        end_datetime = end_datetime - datetime.timedelta(days=7)
        start_date = start_datetime.strftime('%Y-%m-%d')
        end_date = end_datetime.strftime('%Y-%m-%d')
