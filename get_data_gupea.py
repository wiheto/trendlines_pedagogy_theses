#%%
import urllib 
from bs4 import BeautifulSoup
import time
import numpy as np 
import pandas as pd
import os
#%%
# rpp={limit}&sort_by=2&type=dateissued&etal=-1&order=ASC
# Define the base urls
base_urls = [f'https://gupea.ub.gu.se/handle/2077/24912/browse?',
             f'https://gupea.ub.gu.se/handle/2077/24914/browse?',
             f'https://gupea.ub.gu.se/handle/2077/24913/browse?',
             f'https://gupea.ub.gu.se/handle/2077/19452/browse?']

max_hits = [1028, 337, 525, 166] # Number of articles pertopic
pagelim = 200 # Number articles per page
#%%
# PART 1 - GET LINKS
url_collect = []
for bu, burl in enumerate(base_urls):
    print(burl)
    for pi, _ in enumerate(np.arange(0, max_hits[bu], pagelim)):
        print(pi)
        url = f'{burl}rpp={pagelim}&offset={pi*pagelim}'
        r = urllib.request.urlopen(url)
        soup = BeautifulSoup(r, 'html.parser')
        urls = soup.findAll('h4', {'class': 'artifact-title'})
        print(len(urls))
        for url in urls: 
            url_collect.append(url.a['href'])
        time.sleep(5) # Wait so gupea does not get mad
            
           
# %%
# PART 2  - GET HTML OF EACH LINKS
df = pd.DataFrame({'url': url_collect})
df.to_csv('./data/gupea_urls.csv')
# %%
df = pd.read_csv('./data/gupea_urls.csv', index_col=[0])
base_url = 'https://gupea.ub.gu.se'
for i, row in df.iterrows():
    print(i)
    if os.path.exists(f'./data/gupea_data/{i}.html'):
        continue
    url = base_url + row['url']
    r = urllib.request.urlopen(url)
    soup = BeautifulSoup(r, 'html.parser')
    # Save soup to html
    with open(f'./data/gupea_data/{i}.html', 'w') as file:
        file.write(str(soup))
    time.sleep(5)
# %%
# PART 3  - EXTRACT DATA
data = {}
for i, _ in df.iterrows():
    print(i)
    # Open html
    with open(f'./data/gupea_data/{i}.html', 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')
    abstract = soup.find('div', {'class': 'simple-item-view-description'})
    if abstract:
        abstract = abstract.find('div').text
    title = soup.find('h2').text
    uri = soup.find('div', {'class': 'simple-item-view-uri'}).find('a')['href']
    collections = soup.find('div', {'class': 'simple-item-view-collections'}).find('ul').text.replace('\n'  , '')
    rcol = soup.find('div', {'class': 'col-xs-6'})
    pdf_link = soup.find('div', {'class': 'col-xs-6'}).find('a')
    if pdf_link:
        pdf_link = pdf_link['href']
    date = soup.find('div', {'class': 'simple-item-view-date'}).text.replace('\n', '').replace('Date', '')
    year = int(date.split('-')[0])
    keywords = soup.find('div', {'class': 'simple-item-view-keywords'})
    if keywords:
        keywords = keywords.text.replace('\n', ';').replace(';Keywords;', '')
    language = soup.find('div', {'class': 'simple-item-view-language'})
    if language:
        language = language.find('div').text
    series = soup.find('div', {'class': 'simple-item-view-ispartofseries'})
    if series:
        series = series.text.replace('\nSeries/Report no.\n', '').replace('\n', ' ')
    # combine data
    data[i] = {'abstract': abstract, 'title': title, 'uri': uri, 'collections': collections, 'pdf_link': pdf_link, 'date': date, 'year': year, 'keywords': keywords, 'language': language, 'series': series}
# %%
df = pd.DataFrame(data).T
# %%
# drop with no abstraact
df = df.dropna(subset=['abstract'])
# Only take swedish
df = df[df['language'] == 'swe']
# %%
df.to_csv('./data/gupea_abstracts.csv')
# %%
