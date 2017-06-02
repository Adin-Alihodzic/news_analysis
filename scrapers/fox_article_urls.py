from selenium import webdriver
from bs4 import BeautifulSoup
import json
import threading
from load_data import get_keywords_2016, get_week_tuples, get_file_name
from sys import argv
import pdb
import numpy as np
import datetime
import re


def get_urls_from_search(driver, searchterm, date, attempt=0):
    # driver = webdriver.PhantomJS()
    searchterm = searchterm.replace(' ', '%20')
    start = 0
    url = 'http://www.foxnews.com/search-results/search?q={0}&ss=fn&section.path=fnc/politics&type=story&min_date={1}&max_date={2}&start={3}'.format(searchterm, date[0], date[1], start)

    # Get the html from the site and create a BeautifulSoup object from it
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    # import pdb; pdb.set_trace()
    try:
        # Get the number of search results.  If greater than 10 we'll need to
        # loop through subsequent pages to get the additional urls
        # num_found = soup.find('div', attrs={'ng-model': 'numFound'})
        # num_found = int(re.findall(r'\d+', str(num_found))[0])
        # print(num_found)
        # if num_found == 0:
        #     return True, []
        # elif num_found <= 10:
        #     articles = soup.findAll('div', class_='search-article ng-scope')
        #     return True, [str(tag.find('a').get('href')) for tag in articles]
        # else:
        #     urls = []
        #     num_pages = num_found / 10
        #     articles = soup.findAll('div', class_='search-article ng-scope')
        #     for tag in articles:
        #         urls.append(str(tag.find('a').get('href')))
        #
        #     # print(searchterm, num_found, len(urls))
        #
        #     #pagination = soup.find('div', attrs={'id': 'pagination-container'})
        #     page = int(soup.find('span', attrs={'ng-if': 'item === currentNum'}).text)
        #     arrow = soup.findAll('li', attrs={'class': 'arrow'})[1]
        #     # import pdb; pdb.set_trace()
        #     while 'next' in str(arrow):
        #         try:
        #             #import pdb; pdb.set_trace()
        #             page = int(soup.find('span', attrs={'ng-if': 'item === currentNum'}).text)
        #             print('Finished page '+str(page))
        #             link = driver.find_element_by_xpath('//*[@id="pagination-container"]/ul/li[11]/a')
        #             link.click()
        #             soup = BeautifulSoup(driver.page_source, 'html.parser')
        #             articles = soup.findAll('div', class_='search-article ng-scope')
        #             for tag in articles:
        #                 urls.append(str(tag.find('a').get('href')))
        #             arrow = soup.findAll('li', attrs={'class': 'arrow'})[1]
        #         except:
        #             pass


        num_found = int(
            soup.find('span', attrs={'ng-bind': 'numFound'}).contents[0])
        if num_found <= 10:
            articles = soup.findAll('div', class_='search-article ng-scope')
            return True, [str(tag.find('a').get('href')) for tag in articles]
        else:
            # import pdb; pdb.set_trace()
            urls = []
            num_pages = num_found / 10
            articles = soup.findAll('div', class_='search-article ng-scope')
            for tag in articles:
                urls.append(str(tag.find('a').get('href')))

            attempt_url = 0
            #print(searchterm, len(urls))
            for i in range(int(num_pages)):
                start += 10
                url = 'http://www.foxnews.com/search-results/search?q={0}&ss=fn&section.path=fnc/politics&type=story&min_date={1}&max_date={2}&start={3}'.format(searchterm, date[0], date[1], start)

                driver.get(url)
                soup = BeautifulSoup(driver.page_source, 'html.parser')

                articles = soup.findAll('div', class_='search-article ng-scope')
                for tag in articles:
                    urls.append(str(tag.find('a').get('href')))
                #print(searchterm, len(urls))

            if num_found % 10 != 0:
                start += 10
                url = 'http://www.foxnews.com/search-results/search?q={0}&ss=fn&section.path=fnc/politics&type=story&min_date={1}&max_date={2}&start={3}'.format(searchterm, date[0], date[1], start)

                driver.get(url)
                soup = BeautifulSoup(driver.page_source, 'html.parser')

                articles = soup.findAll('div', class_='search-article ng-scope')
                for tag in articles:
                    urls.append(str(tag.find('a').get('href')))
                if len(articles) == 0 and attempt_url < 3:
                    attempt_url +=1
                    start -= 10
            # print(searchterm, num_found, len(urls))
            # if num_found % 10 == 0:
            #     for page in range(2, 1 + num_pages):
            #         #import pdb; pdb.set_trace()
            #         link = driver.find_element_by_link_text(str(page))
            #         link.click()
            #         soup = BeautifulSoup(driver.page_source, 'html.parser')
            #         articles = soup.findAll(
            #             'div', class_='search-article ng-scope')
            #         for tag in articles:
            #             urls.append(str(tag.find('a').get('href')))
            # else:
            #     for page in range(2, 2 + num_pages):
            #         link = driver.find_element_by_link_text(str(page))
            #         link.click()
            #         soup = BeautifulSoup(driver.page_source, 'html.parser')
            #         articles = soup.findAll(
            #             'div', class_='search-article ng-scope')
            #         for tag in articles:
            #             urls.append(str(tag.find('a').get('href')))
            return True, urls
    except:
        print('error')
        if attempt < 3:
            attempt += 1
            return get_urls_from_search(driver, searchterm, date, attempt)
        else:
            return False, url


def get_urls(driver, searchterm, dates, good_urls, bad_searches):
    for date in dates:
        response = get_urls_from_search(driver, searchterm, date)
        if response[0]:
            for url in response[1]:
                good_urls.add(url)
        else:
            bad_searches.add((searchterm, date))

        #print(searchterm, len(good_urls), len(np.unique(response[1])))
    return good_urls, bad_searches


def thread_get_urls(driver, searchterm, dates, good_urls, bad_searches):
    self = threading.current_thread()
    self.result = get_urls(driver, searchterm, dates, good_urls, bad_searches)


def concurrent_get_urls(searchterms, dates, good_urls, bad_searches):
    threads = []
    drivers = [webdriver.PhantomJS() for i in range(len(searchterms))]
    print(searchterms)
    for idx, searchterm in enumerate(searchterms):
        thread = threading.Thread(target=thread_get_urls,
                                  args=(drivers[idx], searchterm, dates, good_urls, bad_searches))
        threads.append(thread)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    results = []
    for thread in threads:
        try:
            results.append(thread.result)
        except:
            pdb.set_trace()

    for result in results:
        good_urls = good_urls.union(result[0])
        bad_searches = bad_searches.union(result[1])
    for driver in drivers:
        driver.close()
    print('# of good urls: '+str(len(good_urls)))
    return good_urls, bad_searches


if __name__ == '__main__':
    ''' This script should be called in the following way:
    $ python fox_article_urls.py 'startdate' 'enddate'
    '''
    # Get all the keywords to search for
    searchterms = get_keywords_2016()

    start_date, end_date = argv[1], argv[2]
    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    while True:
        print('Scraping Fox News from {0} to {1}'.format(start_date, end_date))

        # Get dates to search over
        dates = get_week_tuples(start_date, end_date)

        # Initialize empty lists for urls to be appended to
        good_urls, bad_searches = set(), set()

        # for i in range(int(len(searchterms)/4)):
            # did_it_run, urls = get_urls_from_search('a', dates[0])
            # good_urls, bad_urls = concurrent_get_urls(
            #     'a', dates, good_urls, bad_searches)
            # good_urls, bad_urls = concurrent_get_urls(
            #     searchterms[4*i:4*i+4], dates, good_urls, bad_searches)
        st = searchterms[25:30]
        st.append('trump')
        st = ['a', 'i']
        for i in range(2):
            good_urls, bad_urls = concurrent_get_urls(st[i], dates, good_urls, bad_searches)

        print('Fox Scraping Done...')
        print('There were a total of {0} failed searches'.format(len(bad_searches)))

        # If there were any bad searchs we should try and make some attempts to redo the searchs in a non-threaded way
        # attempt = 0
        # while attempt < 3 and len(bad_searches) > 0:
        #     # This will give us a tuple of (searchterm, date tuple) to research over
        #     searchterms_and_dates = list(bad_searches)
        #     # Reset our bad_searches to an empty set
        #     bad_searches = set()
        #     # Create a Firefox driver
        #     driver = webdriver.Firefox()
        #     for searchterm, date in searchterms_and_dates:
        #         good_urls, bad_searches = get_urls(driver, searchterm, [date], good_urls, bad_searches)
        #     driver.close()
        #     attempt += 1
        #     print('Total of {0} failed searches after attempt {1}'.format(len(bad_searches), attempt))

        # Convert good_urls set to a list and write to a txt file
        file_path = '../url_files/{0}'.format(get_file_name('fox', start_date, end_date))
        with open(file_path, 'w') as f:
            f.write(json.dumps(list(good_urls)))
            f.close()

        # If there are any bad URLs, print how many there were and write them to a file for review
        print('Number of Bad Searches = {0}'.format(len(bad_searches)))
        if len(bad_searches):
            file_path = '../url_files/{0}'.format(get_file_name('fox', start_date, end_date, bad=True))
            with open(file_path, 'w') as f:
                f.write(json.dumps(list(bad_searches)))
                f.close()

        start_datetime = start_datetime - datetime.timedelta(days=7)
        end_datetime = end_datetime - datetime.timedelta(days=7)
        start_date = start_datetime.strftime('%Y-%m-%d')
        end_date = end_datetime.strftime('%Y-%m-%d')



# driver = webdriver.PhantomJS()
# driver.cookiesEnabled = 'True'
# driver.set_window_size(1366,768)
#
# max_wait = 10
# url = 'http://www.foxnews.com/search-results/search?q={0}&ss=fn&section.path=fnc/politics&type=story&min_date={1}&max_date={2}&start=0'.format('a', dates[0][0], dates[0][1])
# driver.get(url)
# driver.set_page_load_timeout(max_wait)
#
# element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.LINK_TEXT, "2")))
# element.click()
