from bs4 import BeautifulSoup
import os
import requests
import bs4
import json
import urllib
import pickle
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, CollectionInvalid
import datetime as dt
import pandas as pd

# Query the NYT API once
def single_query(link, payload):
    response = requests.get(link, params=payload)
    if response.status_code != 200:
        print ('WARNING', response.status_code)
    else:
        return response.json()

# Scrape the meta data (link to article and put it into Mongo)
def scrape_meta(table, days=1):
    nyt_key = os.environ['NYT_API_KEY']
    # The basic parameters for the NYT API
    link = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
    payload = {'api-key': nyt_key }

    today = dt.datetime(2017, 11, 11)
    for day in range(days):
        payload['end_date'] = str(today).replace('-','')
        yesterday = today - dt.timedelta(days=1)
        payload['begin_date'] = str(yesterday).replace('-','')
        print ('Scraping period: %s - %s ' % (str(yesterday), str(today)))

        today -= dt.timedelta(days=1)

        return single_query(link, payload)

# For NYT
def table_setup (content):
    if content == None:
        with open('temp.pkl','rb') as f:
            content = pickle.load(f)
    df = pd.DataFrame(content['response']['docs'])
    df.drop(['document_type','blog','byline','keywords','multimedia','new_desk','print_page','score','snippet','source','type_of_material','uri'],axis=1,inplace = True)
    content_list = []
    for d in content['response']['docs']:
        d_id = d['_id']
        link = d['web_url']
        r = requests.get(link)
        html = r.content
        soup = bs4.BeautifulSoup(html, 'html.parser')
        print(r.status_code, link)
        article_content = '\n'.join([i.text for i in soup.select('p.story-body-text')])
        # if not article_content:
        #     article_content = '\n'.join([i.text for i in soup.select('story-body-text story-content')])
        # if not article_content:
        #     article_content = '\n'.join([i.text for i in soup.select('p.story-body-text')])
        # if not article_content:
        #     article_content = '\n'.join([i.text for i in soup.select('.caption-text')])
        # if not article_content:
        #     article_content = '\n'.join([i.text for i in soup.select('[itemprop="description"]')])
        # if not article_content:
        #     article_content = '\n'.join([i.text for i in soup.select('#nytDesignBody')])
        # else:
        #     article_content = ''
        #     print('No content found ', d_id)

        content_list.append(article_content)
        # table.update({'_id': uid}, {'$set': {'raw_html': html}})
        # table.update({'_id': uid}, {'$set': {'content_txt': article_content}})
    df['content'] = content_list
    df['news_source'] = 'NYT'
    return df

if __name__ == '__main__':

    # username = urllib.parse.quote_plus('admin')
    # password = urllib.parse.quote_plus('admin123')
    # db_cilent = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))
    # db = db_cilent['nyt']
    # table = db['meta']
    # content = scrape_meta(table)
    # get_articles(table)
    df = table_setup(None)
