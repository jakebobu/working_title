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

# Want to say that topic 0 is cw.trending_order[0]
@app.route('/')
def index():
    '''
        If a topic is selected, shows a plot of recent activity of that topic along with a word cloud of the significant words in that topic, else shows recent activity of the topics with the greatest recent activity going on.

        May pull this page out of the index/landing page and onto its own as this web app gets more functionality
    '''
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

    return render_template('index.html', plot_title="Counts for Topic {1} ('{0}')".format(topic_word, cw.web_index), current_topic = list_index, total_topics = TOPIC_LIST)

@app.route('/word_cloud_list')
def word_cloud_words():
    '''
        Given a topic by its indece, generates a json list of tokens and their weights in that topic for jQCloud to work with. Possibly will add links given time
    '''
    word_list = [{}]
    if cw.web_index >= 0:
        #TODO: looking at better scaling the weights so that they look better
        #TODO: way to define the url to this web app through flask attribute
        word_list = [{ 'text': k, 'weight': 5*(1 - np.exp(-v)), 'link' : 'http://0.0.0.0:8080/topics/?c_token={}'.format(k) } for k, v in cw.dc[cw.web_index].items()]
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
        # fig, ax = plt.figure(figsize=(12,5))
        # for i in range(len(TOPIC_LIST)):
        #     ax.plot(cw.times, cw.smooth_data[i], label=TOPIC_LIST[i])
        # handles, labels = ax.get_legend_handles_labels()
        # interactive_legend = plugins.InteractiveLegendPlugin(zip(handles,
        #                                                  ax.collections),
        #                                              labels,
        #                                              alpha_unsel=0.5,
        #                                              alpha_over=1.5,
        #                                              start_visible=True)
        # plugins.connect(fig, interactive_legend)
        # return mpld3.fig_to_html(fig)
        return ''' '''

    return ''' '''


@app.route('/topics/')
def related_topics():
    ''' This page will look at a selected token and show all the topics with that token and the significance of the token in that topic (in top n words). The topics will have links so that you can visit the word cloud plot page to see recent activity of that topic '''
    c_token = request.args.get('c_token')
    if c_token == None or type(c_token) != str:
        return ''' Invalid token provided '''
    c_token = c_token.lower()
    t_list = [["Topic {}".format(key), 1 + np.argwhere(topic == c_token)[0][0], ', '.join(topic[:5])] for key, topic in cw.topics.items() if c_token in topic]
    return render_template('topics.html', token=c_token, topic_list=t_list)
    # return '''page under development'''

@app.route('/topic_articles/')
def topic_articles():
    topic_index = request.args.get('topic_index')
    columns = ['web_url','headline','pub_date']
    if topic_index >= 0:
        df_topic = df.iloc[cw.article_relates[topic_index],columns]
        return render_template('topic_articles.html', columns = columns, article_list = df_topic.to_json())
    else:
        return ''' No Topic Selected '''




if __name__ == '__main__':
    #TODO: grab 'current' articles (define current)
    model = NMF_Time(load_model=True)
    with open('app_model/output_data.pkl', 'rb') as od:
        cw = pickle.load(od)
    df = pd.read_csv('../temp_data1.csv', index_col=0)
    df = df[df['news_source'] == 'NYT']
    TOPIC_LIST = ['Topic {}'.format(i) for i in cw.trending_order] #cw.topics.shape[0])]
    TOPIC_LIST.insert(0, 'All Topics')
    cw.web_index = -1
    app.run(host='0.0.0.0', port=8080, debug=True)
