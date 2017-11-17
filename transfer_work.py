# rolling_means = np.convolve(testy.counts[:,topic_index], np.ones((N,))/N, mode='same')
# plt.close('all')
# plt.plot(testy.times,testy.counts[:,topic_index], label = 'Counts')
# plt.plot(testy.times,rolling_means, '--', label = 'Counts Smoothed')
# plt.plot(testy.times,testy.pos_accel[:,topic_index], label = 'Acceleration')
# plt.legend()
# plt.show()


# import curses
# import time
#
# def report_progress(filename, progress):
#     """progress: 0-10"""
#     stdscr.addstr(0, 0, "I Model")
#     stdscr.addstr(1, 0, "Moving file: {0}".format(filename))
#     stdscr.addstr(2, 0, "Total progress: [{1:10}] {0}%".format(progress * 10, "#" * progress))
#     stdscr.refresh()
#
# if __name__ == "__main__":
#     stdscr = curses.initscr()
#     curses.noecho()
#     curses.cbreak()
#
#     try:
#         for i in range(10):
#             report_progress("file_{0}.txt".format(i), i+1)
#             time.sleep(0.5)
#     finally:
#         curses.echo()
#         curses.nocbreak()
#         curses.endwin()

import pickle
import pandas as pd
import numpy as np
from words_to_vals import NMF_Time, _tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
#
# def make_a_model(testy):
#     new_save = NMF_Time_Save(testy)
#     with open ('model.pkl', 'wb') as f:
#         pickle.dump(new_save, f)
#
# def _tokenize(text):
#      return [stemmer(i) for i in word_tokenize(text)]
#
# def simple_test():
#     tf_idf = TfidfVectorizer()
#     tf_idf.vocabulary_ = {}
#     tf_idf.idf_ = None
#     tf_idf.stop_words_ = set([])
#     with open('temp.pkl', 'wb') as f:
#         pickle.dump(tf_idf,f)
#
#
if __name__ == '__main__':
    # stemmer = SnowballStemmer('english').stem
    # simple_test()
    df = pd.read_csv('temp_data1.csv',index_col=0)
    df = df[df['news_source'] == 'NYT'][:100]
    testy = NMF_Time()
    testy.generate_topics(df['content'].values, _tokenize)
    testy.perform_time_counting_self(df)
