from flask import Flask, request, render_template
import numpy as np
from words_to_vals import NMF_Time
from work_with_counts import Count_Worker
import json
import matplotlib.pyplot as plt
import pickle
from io import BytesIO

app = Flask(__name__)

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
    else:
        cw.web_index = TOPIC_LIST.index(c_topic) - 1
        if cw.web_index >=0:
            topic_word = cw.topics[cw.web_index][0]
        else:
            topic_word = 'All'

    return render_template('index.html', plot_title="Counts for '{0}' Topic {1}".format(topic_word, cw.web_index), current_topic = c_topic, total_topics = TOPIC_LIST)

@app.route('/word_cloud_list')
def word_cloud_words():
    '''
        Given a topic by its indece, generates a json list of tokens and their weights in that topic for jQCloud to work with. Possibly will add links given time
    '''
    word_list = [{}]
    if type(cw.web_index) == int:
        #TODO: looking at better scaling the weights so that they look better
        word_list = [{ 'text': k, 'weight': 1 - np.exp(-v), 'link' : 'http://0.0.0.0:8080/topics/?c_token={}'.format(k) } for k, v in cw.dc[cw.web_index].items()]
    return json.dumps(word_list)

@app.route('/word_plot.png')
def word_plot():
    '''
        Given a topic by its indece, generates a png of a plot containing the recent activity of that topic. Contains counts, smoothed counts, predicted counts/acceleration
        With no topic, shows just the smoothed counts (including predicted) of the n most active topics
    '''
    #TODO: limit data to be in range of 'recent' news
    plt.figure(figsize=(12,5))
    if type(cw.web_index) == int and cw.web_index >= 0:
        plt.plot(cw.times, cw.topic_counts[cw.web_index], '--', alpha=0.6, label = 'Counts')
        plt.plot(cw.times, cw.smooth_data[cw.web_index], label = 'Counts Smoothed')
        plt.plot(cw.times, cw._s[cw.web_index], ':', label = 'S')
        plt.plot(cw.times, cw._b[cw.web_index], '-.', label = 'B')
        plt.plot(cw.times, cw.pos_accel[cw.web_index], label = 'Acceleration')
        plt.legend()
    else:
        for i in range(len(TOPIC_LIST)):
            plt.plot(cw.times, cw.smooth_data[i], label=TOPIC_LIST[i])
        plt.legend()
    image = BytesIO()
    plt.savefig(image)
    return image.getvalue(), 200, {'Content-Type': 'image/png'}

@app.route('/topics/')
def related_topics():
    ''' This page will look at a selected token and show all the topics with that token and the significance of the token in that topic (in top n words). The topics will have links so that you can visit the word cloud plot page to see recent activity of that topic '''
    c_token = request.args.get('c_token')
    if c_token == None or type(c_token) != str:
        return ''' Invalid token provided '''
    t_list = [[key, 1 + np.argwhere(topic == c_token)[0][0], ', '.join(topic[:5])] for key, topic in cw.topics.items() if c_token in topic]
    return render_template('topics.html', token=c_token, topic_list=t_list)
    # return '''page under development'''

if __name__ == '__main__':
    model = NMF_Time(load_model=True)
    with open('app_model/output_data.pkl', 'rb') as od:
        cw = pickle.load(od)
    TOPIC_LIST = ['Topic {}'.format(i) for i in range(10)] #cw.topics.shape[0])]
    TOPIC_LIST.insert(0, 'All Topics')
    app.run(host='0.0.0.0', port=8080, debug=True)
