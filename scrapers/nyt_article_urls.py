from selenium import webdriver
from bs4 import BeautifulSoup
import json
import threading
from load_data import get_keywords_2016, get_week_tuples, get_file_name
from sys import argv
import pdb
import numpy as np
import datetime
import sys
import re


def get_urls_from_search(date, attempt=0):
    driver = webdriver.PhantomJS()

    url = 'https://query.nytimes.com/search/sitesearch/#/*/from{0}to{1}/document_type%3A%22article%22/3/allauthors/relevance/U.S./'.format(date[0], date[1])

    # Get the html from the site and create a BeautifulSoup object from it
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    # import pdb; pdb.set_trace()
    try:
        # Get the number of search results.  If greater than 10 we'll need to
        # loop through subsequent pages to get the additional urls
        num_found = soup.find('div', attrs={'id': 'totalResultsCount'})
        num_found = [int(s.replace(',','')) for s in str(num_found).split() if s.replace(',', '').isdigit()][0]
        print(num_found)
        if num_found <= 10:
            articles = soup.findAll('div', class_='element2')
            urls = set()
            for tag in articles:
                if tag.find('a') != None:
                    article = str(tag.find('a').get('href'))
                    if 'politics' in article:
                        urls.add(article)
            return True, urls
        else:
            urls = set()
            num_pages = num_found / 10
            articles = soup.findAll('div', class_='element2')
            for tag in articles:
                if tag.find('a') != None:
                    article = str(tag.find('a').get('href'))
                    if 'politics' in article:
                        urls.add(article)

            # print(searchterm, num_found, len(urls))

            pagination = soup.find('div', attrs={'class': 'searchPagination'})
            # import pdb; pdb.set_trace()
            while 'next' in str(pagination):
                try:
                    #import pdb; pdb.set_trace()
                    page = re.findall(r'\d+', str(pagination))[0]
                    print 'Finished page '+str(page)
                    link = driver.find_element_by_xpath('//*[@id="searchPagination"]/a[2]')
                    link.click()
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    articles = soup.findAll('div', class_='element2')
                    for tag in articles:
                        if tag.find('a') != None:
                            article = str(tag.find('a').get('href'))
                            if 'politics' in article:
                                urls.add(article)
                    pagination = soup.find('div', attrs={'class': 'searchPagination'})
                except:
                    pass
            return True, urls
    except:
        print "Unexpected error:", sys.exc_info()[0]
        # import pdb; pdb.set_trace()
        if attempt < 3:
            attempt += 1
            return get_urls_from_search(date, attempt)
        else:
            return False, url




if __name__ == '__main__':
    ''' This script should be called in the following way:
    $ python nyt_article_urls.py 'startdate' 'enddate'
    '''

    start_date, end_date = argv[1], argv[2]
    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    while True:
        print 'Scraping NYT from {0} to {1}'.format(start_date, end_date)

        # Get dates to search over
        dates = tuple((start_datetime.strftime('%Y%m%d'), end_datetime.strftime('%Y%m%d')))

        # Initialize empty lists for urls to be appended to

        did_it_work, urls = get_urls_from_search(dates)

        print 'NYT Scraping Done...'
        print 'There were a total of {0} urls'.format(len(urls))

        if did_it_work:
            # Convert good_urls set to a list and write to a txt file
            file_path = '../url_files/{0}'.format(get_file_name('nyt', start_date, end_date))
            f = open(file_path, 'w')
            f.write(json.dumps(list(urls)))
            f.close()
        else:
            print "Didn't work"

        start_datetime = start_datetime - datetime.timedelta(days=7)
        end_datetime = end_datetime - datetime.timedelta(days=7)
        start_date = start_datetime.strftime('%Y-%m-%d')
        end_date = end_datetime.strftime('%Y-%m-%d')
