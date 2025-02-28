#%%
import pandas as pd
import time
import urllib
from selenium import webdriver
from bs4 import BeautifulSoup
import numpy as np
import os
#%%
# %%
# PART 1 - GET LINKS

chrome_path = '/usr/bin/chromedriver'
driver = webdriver.Chrome(chrome_path)

for exam_type in np.arange(1, 2):
    lims = [53, 71, 35, 14, 6] # Number of pages for each exam_type (got manually)
    article_list = []
    for page in np.arange(1, lims[exam_type]):
        pg = str(1 + ((page - 1) * 250))
        if exam_type == 0:
            url = r'https://www.diva-portal.org/smash/resultList.jsf?&p=' + pg + r'&dswid=-1112&language=sv&searchType=UNDERGRADUATE&query=&af=%5B%5D&aq=%5B%5B%5D%5D&aq2=%5B%5B%7B%22dateIssued%22%3A%7B%22from%22%3A%221990%22%2C%22to%22%3A%22%22%7D%7D%2C%7B%22categoryId%22%3A%2211742%22%7D%2C%7B%22thesisLevel%22%3A%22H3%22%7D%2C%7B%22isArtisticWork%22%3A%22false%22%7D%5D%5D&aqe=%5B%5D&noOfRows=250&sortOrder=author_sort_asc&sortOrder2=title_sort_asc&onlyFullText=false&sf=all'
        elif exam_type == 1:
            url = r'https://www.diva-portal.org/smash/resultList.jsf?&p=' + pg + r'&dswid=-1112&fs=false&language=sv&searchType=UNDERGRADUATE&query=&af=%5B%5D&aq=%5B%5B%5D%5D&aq2=%5B%5B%7B%22dateIssued%22%3A%7B%22from%22%3A%221990%22%2C%22to%22%3A%22%22%7D%7D%2C%7B%22categoryId%22%3A%2211742%22%7D%2C%7B%22thesisLevel%22%3A%22M3%22%7D%2C%7B%22isArtisticWork%22%3A%22false%22%7D%5D%5D&aqe=%5B%5D&noOfRows=250&sortOrder=author_sort_asc&sortOrder2=title_sort_asc&onlyFullText=false&sf=all'
        elif exam_type == 2:
            url = r'https://www.diva-portal.org/smash/resultList.jsf?dswid=-1112&p=' + pg + r'&fs=false&language=sv&searchType=UNDERGRADUATE&query=&af=%5B%5D&aq=%5B%5B%5D%5D&aq2=%5B%5B%7B%22dateIssued%22%3A%7B%22from%22%3A%221990%22%2C%22to%22%3A%22%22%7D%7D%2C%7B%22categoryId%22%3A%2211742%22%7D%2C%7B%22thesisLevel%22%3A%22M2%22%7D%2C%7B%22isArtisticWork%22%3A%22false%22%7D%5D%5D&aqe=%5B%5D&noOfRows=250&sortOrder=author_sort_asc&sortOrder2=title_sort_asc&onlyFullText=false&sf=all'
        elif exam_type == 3:
            url = r'https://www.diva-portal.org/smash/resultList.jsf?dswid=-1112&p=' + pg + r'&fs=false&language=sv&searchType=UNDERGRADUATE&query=&af=%5B%5D&aq=%5B%5B%5D%5D&aq2=%5B%5B%7B%22dateIssued%22%3A%7B%22from%22%3A%221990%22%2C%22to%22%3A%22%22%7D%7D%2C%7B%22categoryId%22%3A%2211742%22%7D%2C%7B%22thesisLevel%22%3A%22H1%22%7D%2C%7B%22isArtisticWork%22%3A%22false%22%7D%5D%5D&aqe=%5B%5D&noOfRows=250&sortOrder=author_sort_asc&sortOrder2=title_sort_asc&onlyFullText=false&sf=all'
        elif exam_type == 4:
            url = r'https://www.diva-portal.org/smash/resultList.jsf?dswid=-1112&p=' + pg + r'&fs=false&language=sv&searchType=UNDERGRADUATE&query=&af=%5B%5D&aq=%5B%5B%5D%5D&aq2=%5B%5B%7B%22dateIssued%22%3A%7B%22from%22%3A%221990%22%2C%22to%22%3A%22%22%7D%7D%2C%7B%22categoryId%22%3A%2211742%22%7D%2C%7B%22thesisLevel%22%3A%22H2%22%7D%2C%7B%22isArtisticWork%22%3A%22false%22%7D%5D%5D&aqe=%5B%5D&noOfRows=250&sortOrder=author_sort_asc&sortOrder2=title_sort_asc&onlyFullText=false&sf=all'
        driver.get(url)
        driver.implicitly_wait(10)
        soup = BeautifulSoup(driver.page_source)
        articles = soup.find_all('a', class_='titleLink singleRow linkcolor')
        articles = [article['href'] for article in articles]
        article_list.extend(articles)
        time.sleep(10)

    pd.DataFrame(data={'links': article_list}).to_csv(f'./data/links/article_links_{exam_type}.csv')

driver.close()

#%%
# Part 2 - Get CSVs per link
start_from_i = 0
start_from_e = 0

for e in range(5):
    if e < start_from_e:
        continue
    links = pd.read_csv(f'./data/links/article_links_{e}.csv', index_col=0)
    # Go in chunks of 250
    for i in range(0, len(links), 250):
        if i < start_from_i:
            continue
        links_chunk = links['links'][i:i+250]
        pid_chunk = [l.split('pid=')[1].split('&')[0] for l in links_chunk]
        pid_chunk = ',%20'.join(pid_chunk)
        url = f'https://www.diva-portal.org/smash/references?referenceFormat=CSVALL2&pids=[{pid_chunk}]&fileName={e}_{i}.csv'
        urllib.request.urlretrieve(url, f'./data/get_csv/{e}_{i}.csv')
        time.sleep(30)

