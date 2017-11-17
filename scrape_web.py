import os
import requests
import bs4
import json
import urllib
import datetime as dt
import pandas as pd
import time

# Query the NYT API once
def single_query(link, payload):
    response = requests.get(link, params=payload)
    if response.status_code != 200:
        print ('WARNING', response.status_code)
    else:
        return response.json()

# Scrape the meta data (link to article and put it into Mongo)
def nyt_scrape_meta(scrape_date = dt.date.today()):
    """ Grabs a query containing NYT articles from that date and stores those in the specified location

    Parameters
    ----------
    scrape_date: specified date to request data on

    Returns
    -------
    JSON object from a single query for the specified data containing urls, headlines, and timestamps
    """

    nyt_key = os.environ['NYT_API_KEY']
    # The basic parameters for the NYT API
    link = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
    payload = {'api-key': nyt_key }
    today = scrape_date
    payload['end_date'] = str(today).replace('-','')
    yesterday = today - dt.timedelta(days=1)
    payload['begin_date'] = str(yesterday).replace('-','')
    print ('Scraping period: %s - %s ' % (str(yesterday), str(today)))
    content = single_query(link, payload)
    df = nyt_table_setup(content)
    df_tot = pd.read_csv('temp_data1.csv',index_col=0)
    temp_cols = ['_id', 'content', 'headline', 'news_source', 'pub_date', 'section_name', 'web_url', 'word_count']
    df_tot = df_tot[temp_cols]
    df_tot = df_tot.append(df[temp_cols])
    for col in df_tot.columns:
        if col not in temp_cols:
            print('dropping', col)
            df_tot.drop(col, inplace=True,axis=1)
    df_tot.dropna(subset=['content'],axis=0, inplace=True)
    # df_tot.drop_duplicates(subset=['headline'], inplace=True) WTF???????
    df_tot.to_csv('temp_data1.csv')


def nyt_scrape_meta_continuous(days=1, end_date = dt.date.today()):
    """ Updates a dataframe containing information on all articles.
    Will save the updated csv file with the scraped data
    Is scraping NYT

    Parameters
    ----------
    days: how many days backwards from that date your want to go
    end_date: specified date to request data on

    Returns
    -------
    None
    """
    nyt_key = os.environ['NYT_API_KEY']
    link = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
    payload = {'api-key': nyt_key }

    # TODO: replace with actual source
    df_tot = pd.read_csv('temp_data1.csv',index_col=0)
    temp_cols = ['_id', 'content', 'headline', 'news_source', 'pub_date', 'section_name', 'web_url', 'word_count']
    df_tot = df_tot[temp_cols]
    today = end_date
    for day in range(days):
        payload['end_date'] = str(today).replace('-','')
        yesterday = today - dt.timedelta(days=1)
        payload['begin_date'] = str(yesterday).replace('-','')
        print ('Scraping period: %s - %s ' % (str(yesterday), str(today)))

        today -= dt.timedelta(days=1)
        content = single_query(link, payload)
        df = nyt_table_setup(content)
        '''Have issue with appending where it creates an extra column. This removes that column '''
        df_tot = df_tot.append(df[temp_cols])
        # May not need
        for col in df_tot.columns:
            if col not in temp_cols:
                print('dropping', col)
                df_tot.drop(col, inplace=True,axis=1)

        ''' Sleeps so that NYT doesn't consider the requests spam (429 error)'''
        time.sleep(5)

    ''' Cleans out duplicates and any rows where there is no content in the article (e.g. Visualization articles with just images) '''
    df_tot.dropna(subset=['content'],axis=0, inplace=True)
    # df_tot.drop_duplicates(subset=['headline'], inplace=True) WTF???????
    df_tot.to_csv('temp_data1.csv')

# For NYT
def nyt_table_setup (content):
    """ For NYT, converts from a NYT API requests object lacking the article text, gathers that text and puts all info into a new dataframe. Lastly will return that dataframe

    Parameters
    ----------
    content: NYT API article requests JSON object, a standard response to a request under article search

    Returns
    -------
    df: a dataframe containing article headlines, urls, content, source (NYT), section name, and publication date
    """

    if content == None:
        return None

    df = pd.DataFrame(content['response']['docs'])
    content_list = []
    for d in content['response']['docs']:
        d_id = d['_id']
        link = d['web_url']
        r = requests.get(link)
        html = r.content
        soup = bs4.BeautifulSoup(html, 'html.parser')
        print(r.status_code, link)
        article_content = ' '.join([i.text for i in soup.select('p.story-body-text')])
        content_list.append(article_content)
    df['content'] = content_list
    df['news_source'] = 'NYT'
    return df

# Do batch time saves
def big_batch_nyt(month, year):
    """ Generates the NYT articles from a given month and year and saves them to disk

    Parameters
    ----------
    month: the numerical version of the month to grab articles from
    year: the year to grab articles from

    Returns
    -------
    df_tot: the dataframe now containing all articles from requested time period
    """

    nyt_key = os.environ['NYT_API_KEY']
    link = 'http://api.nytimes.com/svc/archive/v1/'
    link = link + str(year) + '/' + str(month) + '.json'
    payload = {'api-key': nyt_key }
    content = single_query(link, payload)
    df_tot = pd.read_csv('temp_data1.csv',index_col=0)
    temp_cols = ['_id', 'content', 'headline', 'news_source', 'pub_date', 'section_name', 'web_url', 'word_count']
    df_tot = df_tot[temp_cols]
    save_rate = 25
    for i in range(len(content['response']['docs']) // save_rate):
        df_tot = nyt_batch_save(content['response']['docs'][i * save_rate : (i+1) * save_rate], df_tot)
    return nyt_batch_save(content['response']['docs'][-(len(content['response']['docs']) % save_rate):], df_tot)

def nyt_batch_save (jsony, df_tot):
    """ Given a batch from a json response from a NYT API request done in big_batch_nyt, generates the content and writes it to disk

    Parameters
    ----------
    jsony: the batch of json objects of NYT articles provided by big_batch_nyt
    df_tot: the dataframe that stores the article info

    Returns
    -------
    df_tot: the dataframe now updated with the information from the NYT articles in jsony
    """

    temp_cols = ['_id', 'content', 'headline', 'news_source', 'pub_date', 'section_name', 'web_url', 'word_count']
    ''' Looks to remove duplicate articles based on '_id' '''
    jsony = [d for d in jsony if d['_id'] not in df_tot['_id']]
    df = pd.DataFrame(jsony)
    df['news_source'] = 'NYT'
    content_list = []
    for d in jsony:
        r = requests.get(d['web_url'])
        print(r.status_code, d['web_url'])
        soup = bs4.BeautifulSoup(r.content, 'html.parser')
        article_content = ' '.join([i.text for i in soup.select('p.story-body-text')])
        content_list.append(article_content)
    df['content'] = content_list
    df_tot = df_tot.append(df[temp_cols])
    print('saving')
    df_tot.to_csv('temp_data1.csv')
    return df_tot

def news_thingy_scrape_meta(source, sortBy):
    """ Similar to nyt_scrape_meta, but now is using NEWS API as a requests source. Returns the response for a single request given a news source and a sorting method

    Parameters
    ----------
    source: news source to request. Visit https://newsapi.org/sources for names
    sortBy: specifed sorting method. Valid types are 'top', 'latest', or 'popular'

    Returns
    -------
    The response to the specified single query. See https://newsapi.org/#documentation for more information
    """
    n_key = os.environ['NEWS_API_KEY']
    link = 'https://newsapi.org/v1/articles'
    payload = { 'apiKey' : n_key, 'source' : source, 'sortBy' : sortBy }
    return single_query(link, payload)


# works for washington post
def news_thingy_table_setup(content, source):
    """ Similar to nyt_table_setup, converts from a News API requests object lacking the article text, gathers that text and puts all info into a new dataframe. Lastly will return that dataframe

    Parameters
    ----------
    content: NYT API article requests JSON object, a standard response to a request under article search
    source: news source to request. Visit https://newsapi.org/sources for names

    Returns
    -------
    df: a dataframe containing article headlines, urls, content, source, section name, and publication date
    """

    ''' Given a web source, picks the method that will pull from a BeautifulSoup object the article content '''
    if source == 'the-washington-post':
        soup_to_text = wa_post_text
    elif source == 'bbc-news':
        soup_to_text = bbc_text
    elif source == 'cnn':
        soup_to_text = cnn_text
    elif source == 'breitbart-news':
        soup_to_text = brtbr_text
    else:
        print('Could not handle source')
        return
    df = pd.DataFrame(content['articles'])
    df.drop(['author', 'description', 'urlToImage'], axis = 1, inplace = True)
    content_list = []
    topic_list = []
    for d in content['articles']:
        link = d['url']
        r = requests.get(link)
        print(r.status_code, link)
        html = r.content
        soup = bs4.BeautifulSoup(html, 'html.parser')
        soup.a.decompose()
        text, section = soup_to_text(soup, link)
        content_list.append(text)
        topic_list.append(section)

    df.columns = ['pub_date','headline','web_url']
    df['content'] = content_list
    df['section_name'] = topic_list
    df['news_source'] = source
    df = df[df['content'] != '']
    return df

''' The following methods take in a BeautifulSoup object and the link it is from and returns the text of that article and it's section name. The difference between the methods are the differences between web sites (news source) '''
def wa_post_text (soup, link):
    """ Digs through the BeautifulSoup object to extract the article text. In addition, extracts the section name if available

    Parameters
    ----------
    soup: BeautifulSoup object for the desired article
    link: the website url used to generate 'soup'

    Returns
    -------
    text: article text extracted from the BeautifulSoup object
    topic: the section name that this article falls under
    """
    tempy = [t.text for t in soup.findAll('p') if '<p ' not in t]
    while len(tempy) > 0 and 'email address' not in tempy[0]:
        tempy = tempy[1:]
    tempy = tempy[1:]
    topic = link[link.find('.com/')+5:]
    return ' '.join(tempy), topic[:topic.find('/')]

def bbc_text (soup, link):
    """ Digs through the BeautifulSoup object to extract the article text. In addition, extracts the section name if available

    Parameters
    ----------
    soup: BeautifulSoup object for the desired article
    link: the website url used to generate 'soup'

    Returns
    -------
    text: article text extracted from the BeautifulSoup object
    topic: the section name that this article falls under
    """
    tempy = [t.text for t in soup.findAll('p') if '<p ' not in t]
    while len(tempy) > 0 and 'external links' not in tempy[0]:
        tempy = tempy[1:]
    tempy=tempy[1:]
    if len(tempy) > 0 and 'newsletter' in tempy[-1]:
        tempy = tempy[:-1]
    topic = link[link.find('.co.uk/')+7:]
    return ' '.join(tempy), topic[:topic.find('/')]

def cnn_text (soup, link):
    """ Digs through the BeautifulSoup object to extract the article text. In addition, extracts the section name if available

    Parameters
    ----------
    soup: BeautifulSoup object for the desired article
    link: the website url used to generate 'soup'

    Returns
    -------
    text: article text extracted from the BeautifulSoup object
    topic: the section name that this article falls under
    """
    tempy = [t.text for t in soup.findAll("div", {"class" : ["zn-body__paragraph speakable", "zn-body__paragraph"]})]
    topic = link[link.find('.com/')+16:]
    return ' '.join(tempy), topic[:topic.find('/')]

def brtbr_text (soup, link):
    """ Digs through the BeautifulSoup object to extract the article text. In addition, extracts the section name if available

    Parameters
    ----------
    soup: BeautifulSoup object for the desired article
    link: the website url used to generate 'soup'

    Returns
    -------
    text: article text extracted from the BeautifulSoup object
    topic: the section name that this article falls under
    """
    tempy = [t.text for t in soup.select('p')]
    tempy = tempy[1:-5]
    topic = link[link.find('.com/')+5:]
    return ' '.join(tempy), topic[:topic.find('/')]

def tot_newsy (sources):
    """ Updates the dataframe with articles from each sorting order from all in sources
    All sources have top, but may not have latest or popular
    Will print out following for when that sorting order is not available from that news service:
            WARNING 400
            {Sorting Order} not available for {News Source}

    Parameters
    ----------
    sources: List of news sources. Visit https://newsapi.org/sources for names.

    Returns
    -------
    None
    """

    df_tot = pd.read_csv('temp_data1.csv', index_col=0)
    temp_cols = ['_id', 'content', 'headline', 'news_source', 'pub_date', 'section_name', 'web_url', 'word_count']
    df_tot = df_tot[temp_cols]
    orders = ['top', 'latest', 'popular']
    for o in orders:
        for s in sources:
            time.sleep(5)
            content = news_thingy_scrape_meta(s, o)
            if content == None:
                print('{0} not available for {1}'.format(o,s))
            else:
                df = news_thingy_table_setup(content, s)
                df_tot = df_tot.append(df)
                # May not need
                for col in df_tot.columns:
                    if col not in temp_cols:
                        print('dropping', col)
                        df_tot.drop(col, inplace=True, axis=1)

    df_tot.dropna(subset=['content'],axis=0, inplace=True)
    # df_tot.drop_duplicates(subset=['headline'], inplace=True) WTF???????
    df_tot.to_csv('temp_data1.csv')

def foxy_news(start_date = dt.date(2017,8,1),end_date = dt.date.today()):
    link = 'http://www.foxnews.com/search-results/search?q=a&ss=fn&min_date={0}&max_date={1}&start=0'.format(start_date, end_date)
    r = requests.get(link)
    html = r.content
    soup = bs4.BeautifulSoup(html, 'html.parser')
    # objs = soup.findAll("div", {"class" : ["search-info responsive-image"])
    # objs1 = soup.findAll('a.ng-binding')
    links = []
    titles = []
    sections = []
    dates = []
    # http://www.foxnews.com/us/2017/11/15/texas-sheriff-concerned-about-truck-with-anti-trump-message.html
    for obj in soup.findAll('a.ng-binding'):
        t = obj.get('href')
        links.append(t)
        l = len(t) - t[::-1].find('/')
        f = t[23:l-1]
        e = t[l:-5]
        titles.append(e.replace('-',' '))
        sections.append(f[:f.find('/')])
        dates.append(f[f.find('/')+1:-1].replace('/','-'))
    temp_cols = ['_id', 'content', 'headline', 'news_source', 'pub_date', 'section_name', 'web_url', 'word_count']
    df = pd.DataFrame(columns=temp_cols)
    df['headline']=titles
    df['web_url']=links
    df['news_source']='Fox News'
    df['pub_date'] = dates
    df['section_name']=sections
    return df

def foxy_table(links,titles,sections):
    save_rate = 100
    df_tot = pd.read_csv('temp_data1.csv',index_col=0)
    temp_cols = ['_id', 'content', 'headline', 'news_source', 'pub_date', 'section_name', 'web_url', 'word_count']
    df_tot = df_tot[temp_cols]
    i = 0
    while (i+1)*save_rate < len(links):
        df = pd.DataFrame(columns=temp_cols)
        content, dates = [fxc(link) for link in links[i*save_rate:(i+1)*save_rate]]
        df['content'] = content
        df['pub_date'] = dates
        df['headline'] = titles[i*save_rate:(i+1)*save_rate]
        df['section_name'] = sections[i*save_rate:(i+1)*save_rate]
        df_tot.append(df,inplace=True)
        df_tot.to_csv('temp_data1.csv')
        i += 1
    '''Have issue with appending where it creates an extra column. This removes that column '''
    df_tot = df_tot.append(df[temp_cols])


def clean_up_df(df):
    ''' Given a dataframe of articles, drops articles with no contents, drops duplcate articles, and converts pub_data to a datetime object '''
    df.dropna(subset=['content'], axis=0, inplace=True)
    df.drop_duplicates(subset='headline', inplace=True)
    df['pub_date'] = pd.to_datetime(df['pub_date'])
    df.sort('pub_date')
    df.to_csv('temp_data1.csv')

 return ' '.join(lems)

if __name__ == '__main__':
    sources = ['the-washington-post','bbc-news','cnn','breitbart-news']
    big_batch_nyt(11, 2017)
    # The three general things you can run (Pick 1)
    # nyt_scrape_meta_continuous(days=16, end_date=dt.datetime(2017, 7, 25))    r = requests.get(link)
    # html = r.content
    # soup = bs4.BeautifulSoup(html, 'html.parser')
    # soup.a.decompose()
    # tot_newsy(sources)
    # nyt_scrape_meta() # Good for getting today's nyt news




            #
            #
            # ''' Given a web source, will pull from a BeautifulSoup object the article content '''
            # if source == 'the-washington-post':
            #     tempy = [t.text for t in soup.findAll('p') if '<p ' not in t]
            #     while len(tempy) > 0 and 'email address' not in tempy[0]:
            #         tempy = tempy[1:]
            #     tempy=tempy[1:]
            #     content_list.append(' '.join(tempy))
            #     topic = link[link.find('.com/')+5:]
            #     topic_list.append(topic[:topic.find('/')])
            # elif source == 'bbc-news':
            #     tempy = [t.text for t in soup.findAll('p') if '<p ' not in t]
            #     while len(tempy) > 0 and 'external links' not in tempy[0]:
            #         tempy = tempy[1:]
            #     tempy=tempy[1:]
            #     if len(tempy) > 0 and 'newsletter' in tempy[-1]:
            #         tempy = tempy[:-1]
            #     content_list.append(' '.join(tempy))
            #     topic = link[link.find('.co.uk/')+7:]
            #     topic_list.append(topic[:topic.find('/')])
            # elif source == 'cnn':
            #     tempy = [t.text for t in soup.findAll("div", {"class" : ["zn-body__paragraph speakable", "zn-body__paragraph"]})]
            #     content_list.append(' '.join(tempy))
            #     topic = link[link.find('.com/')+16:]
            #     topic_list.append(topic[:topic.find('/')])
            # elif source == 'breitbart-news':
            #     tempy = [t.text for t in soup.select('p')]
            #     tempy = tempy[1:-5]
            #     content_list.append(' '.join(tempy))
            #     topic = link[link.find('.com/')+5:]
            #     topic_list.append(topic[:topic.find('/')])
            # else:
            #     print('Could not handle source')
