from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from words_to_vals import NMF_Time
from work_with_counts import Count_Worker
import json
import matplotlib.pyplot as plt
import pickle
from io import BytesIO
import mpld3
from mpld3 import plugins

app = Flask(__name__)

#TODO: grab images to replace template images
#TODO: center forms on index page

# Want to say that topic 0 is cw.trending_order[0]
@app.route('/')
def index():
    return render_template('index.html', total_topics=TOPIC_LIST, tot_topics=TOPIC_ARTICLE)

@app.route('/topic_counts/')
def topic_counts():
    c_topic = request.args.get('topic_select')
    if c_topic == None:
        c_topic = TOPIC_LIST[0]
        cw.web_index = -1
        topic_word = 'All'
        list_index = -1
    if type(c_topic) == int:
        cw.web_index = cw.trending_order[c_topic]
        c_topic = TOPIC_LIST[c_topic+1]
        list_index = cw.web_index
    else:
        cw.web_index = TOPIC_LIST.index(c_topic) - 1
        list_index = cw.web_index
        if cw.web_index >= 0:
            cw.web_index = cw.trending_order[cw.web_index]
            topic_word = cw.topics[cw.web_index][0]
        else:
            topic_word = 'All'

    return render_template('topic_counts.html', plot_title="Counts for Topic {1} ('{0}')".format(topic_word, cw.web_index), current_topic = list_index, total_topics = TOPIC_LIST)


@app.route('/word_cloud_list')
def word_cloud_words():
    '''
        Given a topic by its indece, generates a json list of tokens and their weights in that topic for jQCloud to work with. Possibly will add links given time
    '''
    word_list = [{}]
    if cw.web_index >= 0:
        #TODO: looking at better scaling the weights so that they look better
        #TODO: way to define the url to this web app through flask attribute
        word_list = [{ 'text': k, 'weight': 5*(1 - np.exp(-v)), 'link' : 'http://0.0.0.0:8080/token_topics/?c_token={}'.format(k) } for k, v in cw.dc[cw.web_index].items()]
    print(len(word_list))
    return json.dumps(word_list)

@app.route('/word_plot.png')
def word_plot():
    '''
        Given a topic by its indece, generates a png of a plot containing the recent activity of that topic. Contains counts, smoothed counts, predicted counts/acceleration
        With no topic, shows just the smoothed counts (including predicted) of the n most active topics
    '''
    #TODO: limit data to be in range of 'recent' news
    #TODO: generate subplots for acceleration on one and counts on the other
    if cw.web_index >= 0:
        plt.figure(figsize=(12,5))
        # topic_index = request.args.get('topic_index')
        pb = 42 # One week period back
        pa = 6 # one day ahead
        counts_back = cw.smooth_data[cw.web_index]
        counts_back = counts_back[-pb:]
        back_times = cw.times[-pb:]
        counts_fwd = cw.predicted_values[cw.web_index]
        plt.plot(back_times, counts_back,'r',label='Current')
        plt.plot(cw.predicted_times, counts_fwd,'r--',label='Predicted')
        plt.plot(cw.trend_times, cw.trend_points[cw.web_index], 'g-.',alpha=0.3, linewidth=3.0, label='Trend')
        plt.axvline(x=cw.predicted_times[0], c='k', ls=':',label='Today')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Topic Article Counts')
        plt.grid(True, alpha=0.6)
        # plt.plot(cw.times, cw.topic_counts[cw.web_index], '--', alpha=0.6, label = 'Counts')
        # plt.plot(cw.times, cw.smooth_data[cw.web_index], label = 'Counts Smoothed')
        # plt.plot(cw.times, cw._s[cw.web_index], ':', label = 'S')
        # plt.plot(cw.times, cw._b[cw.web_index], '-.', label = 'B')
        # plt.plot(cw.times, cw.pos_accel[cw.web_index], label = 'Acceleration')
        # plt.legend()
        image = BytesIO()
        plt.savefig(image)
        return image.getvalue(), 200, {'Content-Type': 'image/png'}
    else:
        print(type(cw.web_index))
        print(cw.web_index)
        return ''' '''

    return ''' '''


@app.route('/token_topics/')
def related_topics():
    ''' This page will look at a selected token and show all the topics with that token and the significance of the token in that topic (in top n words). The topics will have links so that you can visit the word cloud plot page to see recent activity of that topic '''
    c_token = request.args.get('c_token')
    if c_token == None or type(c_token) != str:
        return ''' Invalid token provided '''
    c_token = c_token.lower()
    t_list = [["Topic {}".format(key), 1 + np.argwhere(topic == c_token)[0][0], ', '.join(topic[:5])] for key, topic in cw.topics.items() if c_token in topic]
    return render_template('topics_from_token.html', token=c_token, topic_list=t_list)
    # return '''page under development'''

@app.route('/topic_articles/')
def topic_articles():
    '''
        Given a provided topic, shows a list of the articles that make up that topic
    '''
    c_topic = request.args.get('topic_select')
    topic_index = TOPIC_ARTICLE.index(c_topic)
    columns = ['web_url','headline','pub_date']
    if topic_index >= 0:
        df_topic = df.iloc[cw.article_relates[topic_index],:]
        df_topic = df_topic[columns]
        headlines = df_topic.headline.values
        hf = [h[h.find(':')+3:] for h in headlines]
        hf = [h[:h.find("'")] for h in hf]
        df_topic['headline']= hf
        article_list = df_topic.to_dict(orient='records')
        return render_template('topic_articles.html', columns = ['Website', 'Headline', 'Publication Date'], article_list = article_list, topic=c_topic)
    else:
        return ''' No Topic Selected '''


if __name__ == '__main__':
    #TODO: grab 'current' articles (define current)
    model = NMF_Time(load_model=True)
    with open('app_model/output_data.pkl', 'rb') as od:
        cw = pickle.load(od)
    df = pd.read_csv('../temp_data1.csv', index_col=0)
    df = df[df['news_source'] == 'NYT']
    TOPIC_LIST = ['Topic {}'.format(i) for i in cw.trending_order]
    TOPIC_ARTICLE = TOPIC_LIST.copy()
     #cw.topics.shape[0])]
    TOPIC_LIST.insert(0, 'All Topics')
    cw.web_index = -1
    app.run(host='0.0.0.0', port=8080, debug=True)
