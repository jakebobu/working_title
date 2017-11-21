from flask import Flask, request, render_template
import numpy as np
from words_to_vals import NMF_Time
from work_with_counts import Count_Worker
import json
# from build_model import TextClassifier

app = Flask(__name__)


# selection of topic list by indece or by leading word by significance
jqcloud_page = "http://mistic100.github.io/jQCloud/index.html"

# Need to think of time period for which to trend on (last week, 5 days, 2 weeks?)
# Show divide going from current to predicted trends based on modeling
def generate_plot(counts, topics, select_topic=None):
    if type(select_topic) == str:
        topic_index = TOPIC_LIST.index(select_topic)

        # Highlight one topic, have rest be 1 color/translucent alpha
    else:
        # Show all equally (may only be subset of all depending on how many topics are currently trending)
        # show first n


@app.route('/')
def index():
    '''
        Main page that shows last time database was updated.
        Has button to view data
    '''
    c_topic = request.args.get('topic_select')
    if c_topic == None:
        c_topic = TOPIC_LIST[0]
        cw.web_index = None
    else:
        cw.web_index = TOPIC_LIST.index(c_topic) - 1
    return render_template('index.html', current_topic = c_topic, total_topics = TOPIC_LIST)

@app.route('/word_cloud_list')
def word_cloud_words():
    word_list = [{}]
    if type(cw.web_index) == int:
        #TODO: looking at scaling the weights so that they look better
        word_list = [{ 'text': k, 'weight': v } for k, v in cw.dc[cw.web_index].items()]
    return json.dumps(word_list)

@app.route('/topics')
def topic_growths():
    return '''<h1> currently under progress </h1> '''
if __name__ == '__main__':
    model = NMF_Time(load_model=True)
    cw = Coun(model)
    TOPIC_LIST = ['Topic {}'.format(i) for i in range(10)] #cw.topics.shape[0])]
    TOPIC_LIST.insert(0, 'Select Topic')
    app.run(host='0.0.0.0', port=8080, debug=True)
