'''Used to take in a csv or other dataframe like file with the content of the articles under 'content' and then outputs another dataframe now only containing the tf_idf matrix

What I can customize:
- TfidfVectorizer
    - tokenize function
    - max_df and min_df
    - max_features
- NMF
    - init
        ‘random’: non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        ‘nndsvd’: Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        ‘nndsvda’: NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        ‘nndsvdar’: NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa for when sparsity is not desired)

    - n_components
    - solver 'cd' or 'mu'
    - alpha (regularization)
    - l1_ratio


In Counting:
    dt: time interval for grouping counts of articles by topic
    threshold: minimum similarity between an article and a topic to count as being about that topic

'''

import pickle
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import boto3
from post_to_s3 import get_client_bucket
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from string import punctuation
import matplotlib.pyplot as plt
import pyflux as pf
from work_with_counts import Count_Worker
# class NMF_Time_Save ():
#     def __init__(self, obj):
#         self.top_n_words = obj.top_n_words
#         # self.t_vect = obj.t_vect
#         self.nmf = obj.nmf
#         self.counts = obj.counts
#         self.total_counts = obj.total_counts
#         self.times = obj.times
#         self.topics = obj.topics

    # def load_save(self, file_location):
    #     with open (file_location, 'rb') as f:
    #         save_obj = pickle.load(f)
    #     # self.t_vect = save_obj.t_vect
    #     self.nmf = save_obj.nmf
    #     self.top_n_words = save_obj.top_n_words
    #     self.topics = save_obj.topics
    #     self.counts = save_obj.counts
    #     self.total_counts = save_obj.total_counts
    #     self.times = save_obj.times


class NMF_Time():
    """docstring for NMF_Time."""
    def __init__(self, top_n_words=10):
        """ Initialization of class

        Parameters
        ----------
        top_n_words: the number of words you want to show per topic

        """

        print('Intializing Class')
        self.top_n_words = top_n_words
        self.t_vect = None
        self.nmf = None
        self.counts = None
        self.total_counts = None
        self.times = None
        self.topics = None
        self.nlp = spacy.load('en')
        self._spacy_tokenizer = English().Defaults.create_tokenizer(self.nlp)
        self._c = 0
        self._train_length = 0
        self._punctuation = punctuation + '’' + '--' + '’s'

    def _tokenize(self, doc):
        '''
            tokenizer function to use for the TfidfVectorizer class
            Currently using spacy, can replace with nltk stuff as well for comparison
        '''
        self._c += 1
        print('Tokenizing ({0}/{1})'.format(self._c, self._train_length), end="\r")
        wList = [t.text if t.lemma_ == '-PRON-' else t.lemma_ for t in [token for token in self._spacy_tokenizer(doc) if token.is_alpha]]
        return [token for token in wList if token not in self._punctuation and '@' not in token]


    # Need to cap my dictionary or modify min_df
    def generate_topics (self, content, **kwargs):
        """ Converts a list of str containing the article text into a feature matrix
        Allows for ability to permutate the functional components to form comparisons

        Parameters
        ----------
        content: article contents as a list of str

        kwargs
        ---------
        max_df: In the vectorizer, the maximum allowable frequency of the words (default = 1.0)
        min_df: In the vectorizer, the minimum allowable frequency of the words (default = 0.0)
        max_features: The maximum number of words to use in the vectorized vocabulary (default = None)
        n_components: Number of topics you want to work with (default = 10)
        init: Method of performing the NMF (default = 'nndsvd')
        solver: solver for the NMF algorithm (default = 'cd')
        alpha: Reqularization of the algorithm (default = 0)
        l1_ratio: ratio between l1 and l2 Reqularization (default = 0)

        Returns
        -------
        W: np.array docs v topics
        H: np.array topics v words
        """

        if kwargs == None:
            kwargs = dict()
        self._train_length = content.shape[0]
        t_vect = TfidfVectorizer(stop_words = 'english', tokenizer = self._tokenize, max_df = kwargs.get('max_df', 1.0), min_df = kwargs.get('min_df', 0.0), max_features = kwargs.get('max_features', None))
        nmf = NMF(n_components = kwargs.get('n_components', 10), init = kwargs.get('init', 'nndsvd'), solver = kwargs.get('solver', 'cd'), random_state = 2, alpha = kwargs.get('alpha', 0), l1_ratio = kwargs.get('l1_ratio', 0), shuffle = True, verbose = True)
        print('Starting Vectorizer')
        self._c = 0
        print('Tokenizing (1/{0})'.format(self._train_length), end="\r")
        t_mat = t_vect.fit_transform(content)
        print('Tokenizing completed ({0}/{0})'.format(self._train_length))
        print('Starting NMF')
        self.W = nmf.fit_transform(t_mat)
        print('Topics Generated')
        self.t_vect = t_vect
        self.nmf = nmf
        H = nmf.components_
        vocab = { v: k for k, v in t_vect.vocabulary_.items()}
        top_words = []
        ordering = H.argsort(axis=1)[:,:-self.top_n_words-1:-1]
        for i in range(H.shape[0]):
            tp = [vocab[ordering[i,j]] for j in range(self.top_n_words)]
            top_words.append(tp)
        self.topics = np.array(top_words)


        # TODO: when delta is set to 1 day, it creates a weird effect, might look to modify the time to 0:00:00 to correct for that
    def perform_time_counting_self (self, df, delta=dt.timedelta(days=1), threshold=0.1):
        """ Takes in a dataframe of data, and returns across time the percentage of total articles that are part of topics
        This assumes that df content is that the model was fitted on

        Parameters
        ----------
        df (pandas.dataframe) : DataFrame of articles containing article content and article publication date. Can be completely new data
        dt (datetime.timedelta) : timespan for which to bin articles into (default = 3 days)
        threshold : the value at which equal to or above an article is considered counted as of that topic (default = 0.1)

        Returns
        -------
        topic_counts : counts of articles that are pertaining to a topic, across time
        total_counts : total number of articles in that time period
        time_periods : the periods of time relating to topic_counts
        """

        df['pub_date'] = pd.to_datetime(df['pub_date'])
        start_time = df['pub_date'].min()
        end_time = start_time + delta
        ending_point = df['pub_date'].max()
        topic_counts = []
        period_counts = []
        time_periods = []
        print("Starting time analysis")
        while start_time <= ending_point:
            sub_W = self.W[(df['pub_date'] < end_time) & (df['pub_date'] >= start_time),:]
            topic_pick = np.sum(1*(sub_W >= threshold),axis=0)
            topic_counts.append(topic_pick)
            period_counts.append(sub_W.shape[0])
            time_periods.append(start_time)
            start_time = end_time
            end_time = start_time + delta
        self.counts = np.array(topic_counts)
        self.total_counts = np.array(period_counts)
        self.times = np.array(time_periods)
        print('Time Counts is Complete')


    def perform_time_counting_new (self, df, delta=dt.timedelta(days=1), threshold=0.1):
        """ Takes in a dataframe of data, and returns across time the percentage of total articles that are part of topics

        Parameters
        ----------
        df (pandas.dataframe) : DataFrame of articles containing article content and article publication date. Can be completely new data
        dt (datetime.timedelta) : timespan for which to bin articles into (default = 3 days)
        threshold : the value at which equal to or above an article is considered counted as of that topic (default = 0.1)

        Returns
        -------
        topic_counts : counts of articles that are pertaining to a topic, across time
        total_counts : total number of articles in that time period
        time_periods : the periods of time relating to topic_counts
        """

        if 'content' not in df.columns or 'pub_date' not in df.columns:
            print('Provided dataframe of Invalid type')
            return
        elif self.t_vect == None or self.nmf == None:
            content = df['content'].values
            generate_topics(content)
        df['pub_date'] = pd.to_datetime(df['pub_date'])
        start_time = df['pub_date'].min()
        end_time = start_time + delta
        ending_point = df['pub_date'].max()
        topic_counts = []
        period_counts = []
        time_periods = []
        print("Starting time analysis")
        # May instead utilize spacy similarity to determine similarity between article and topics
        while start_time <= ending_point:
            print('Time period left (days): {}'.format((ending_point-start_time).days))
            df_dt = df[(df['pub_date'] < end_time) & (df['pub_date'] >= start_time)]
            dt_content = df_dt['content'].values
            self._c = 0
            self._train_length = dt_content.shape[0]
            topic_vals = self.nmf.transform(self.t_vect.transform(dt_content))
            topic_pick = np.sum(1*(topic_vals >= threshold),axis=0)
            topic_counts.append(topic_pick)
            period_counts.append(dt_content.shape[0])
            time_periods.append(start_time)
            start_time = end_time
            end_time = start_time + delta
        self.counts = np.array(topic_counts)
        self.total_counts = period_counts
        self.times = np.array(time_periods)
        print('Time Counts is Complete')


if __name__ == '__main__':
    # from words_to_vals import NMF_Time
    df = pd.read_csv('temp_data1.csv',index_col=0)
    df = df[df['news_source'] == 'NYT']
    testy = NMF_Time()
    testy.generate_topics(df['content'].values, min_df = 0.01, max_features = 10000, n_components=100)
    testy.perform_time_counting_self(df)
    CW = Count_Worker(testy)
    with open('model.pkl', 'wb') as f:
        pickle.dump(CW,f)
