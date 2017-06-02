import requests
import numpy as np
import time
import datetime

while True:
    now = datetime.datetime.now()

    #all top
    url = 'https://politics.alltop.com/'
    response = requests.get(url)
    f = open('../rss_feeds/all_top_'+now.strftime('%Y%m%d%H'), 'w')
    f.write(response.text)
    f.close()

    time.sleep(60*60*5)
