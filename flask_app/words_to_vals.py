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
# from post_to_s3 import get_client_bucket
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from string import punctuation
import matplotlib.pyplot as plt
import pyflux as pf
from work_with_counts import Count_Worker


nlp = spacy.load('en')
spacy_tokenizer = English().Defaults.create_tokenizer(nlp)
punct = punctuation + '’' + '--' + '’s'

def _tokenize(doc):
    '''
        tokenizer function to use for the TfidfVectorizer class
        Currently using spacy, can replace with nltk stuff as well for comparison
    '''
    # ITER_COUNT += 1
    # print('Tokenizing ({0}/{1})'.format(ITER_COUNT, ITER_LENGTH), end="\r")
    wList = [t.text if t.lemma_ == '-PRON-' else t.lemma_ for t in [token for token in spacy_tokenizer(doc) if token.is_alpha]]
    return [token for token in wList if token not in punct and '@' not in token]


class NMF_Time():
    """docstring for NMF_Time."""
    def __init__(self, top_n_words=10, load_model=False, verbose=False):
        """ Initialization of class

        Parameters
        ----------
        top_n_words: the number of words you want to show per topic
        load_model: whether or not to load the previously saved model under 'model/'
        """

        print('Intializing Class')
        self.top_n_words = top_n_words
        self.t_vect = None
        self.nmf = None
        self.counts = None
        self.total_counts = None
        self.times = None
        self.topics = None
        self.topic_dc = None
        self.verbose = verbose
        if load_model:
            self.load_model()

    # Need to cap my dictionary or modify min_df
    def generate_topics (self, content, tok, **kwargs):
        """ Converts a list of str containing the article text into a feature matrix
        Allows for ability to permutate the functional components to form comparisons

        Parameters
        ----------
        content: article contents as a list of str
        tok: the tokenizer function to use within the vectorizer

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
        self._fit_vectorizer(content=content,tok=tok,**kwargs)
        self._fit_factorizer(**kwargs)
        self._generate_topics()

    def _fit_vectorizer (self, content, tok, **kwargs):
        self.t_vect = TfidfVectorizer(stop_words = 'english', tokenizer = tok, max_df = kwargs.get('max_df', 1.0), min_df = kwargs.get('min_df', 0.0), max_features = kwargs.get('max_features', None))
        self.t_mat = self.t_vect.fit_transform(content)

    def _fit_factorizer (self, **kwargs):
        self.nmf = NMF(n_components = kwargs.get('n_components', 10), init = kwargs.get('init', 'nndsvd'), solver = kwargs.get('solver', 'cd'), random_state = 2, alpha = kwargs.get('alpha', 0), l1_ratio = kwargs.get('l1_ratio', 0), shuffle = True, verbose = self.verbose)
        self.W = self.nmf.fit_transform(self.t_mat)

    def _generate_topics (self):
        H = self.nmf.components_
        vocab = { v: k for k, v in self.t_vect.vocabulary_.items()}
        top_words = []
        temp_dict = []
        ordering = H.argsort(axis=1)[:,:-self.top_n_words-1:-1]
        for i in range(H.shape[0]):
            tdict = {vocab[ordering[i,j]] : H[i, ordering[i,j]] for j in range(self.top_n_words)}
            temp_dict.append(tdict)
            tp = [vocab[ordering[i,j]] for j in range(self.top_n_words)]
            top_words.append(tp)
        self.topics = np.array(top_words)
        self.topic_dc = np.array(temp_dict)

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
        self.topic_threshold = threshold
        print('Time Counts is Complete')

    def comprehensive_time_count_self (self):
        """ Generates plots to look at the counts of articles to topics based on thresholds.

        Parameters
        ----------
        None

        Returns
        -------
        None
        A figure containing plots of article counts against thresholds
        - Plot of % of articles in corpus that meet the threshold of a topic
        - Plot of average number of topics per article
        - Plot of topics with atleast a critical number of related articles
        """

        threshold = np.logspace(-2,-.7,100)
        percent_counts = np.zeros(100)
        average_topics = np.zeros(100)
        n = [3,5,7,10]
        valid_topics = np.zeros((100, len(n)))
        for i, t in enumerate(threshold):
            percent_counts[i] = np.sum(1*(np.max(self.W, axis=1) >= t))/self.W.shape[0]
            average_topics[i] = np.sum(1*(self.W>=t))/self.W.shape[0]
            for j, nn in enumerate(n):
                valid_topics[i,j] = np.sum(1*(np.sum(1*(self.W >=t), axis=0)>=nn))/self.W.shape[1]
        plt.subplot(2,2,1)
        plt.title('Percentage of Articles with a Topic above Threshold')
        plt.plot(threshold,percent_counts)
        plt.xlabel('Threshold')
        plt.ylabel('% of Articles')
        plt.subplot(2,2,2)
        plt.title('Average # of Topics per Article above Threshold')
        plt.plot(threshold,average_topics)
        plt.xlabel('Threshold')
        plt.ylabel('Average # of Topics')
        plt.subplot(2,1,2)
        plt.title('Percentage of Topics with # of related Articles >= n')
        for i in range(len(n)):
            plt.plot(threshold, valid_topics[:,i],label='n={}'.format(n[i]))
        plt.legend()
        plt.xlabel('Threshold')
        plt.ylabel('% of Topics with Articles')
        plt.show()

    def perform_time_counting_new (self, df, delta=dt.timedelta(days=1), threshold=0.1):
        """ Takes in a dataframe of data, and returns across time the percentage of total articles that are part of topics

        Parameters
        ----------
        df (pandas.dataframe) : DataFrame of articles containing article content and article publication date. Can be completely new data
        dt (datetime.timedelta) : timespan for which to bin articles into (default = 3 days)
        threshold : the value at which equal to or above an article is considered counted as of that topic (default = 0.1)

        Returns
        -------
        None, but assigns to self:
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
        t_mat = self.t_vect.transform(df['content'].values)
        self.W = self.nmf.transform(t_mat)
        self.perform_time_counting_self(df, delta, threshold)
        # df['pub_date'] = pd.to_datetime(df['pub_date'])
        # start_time = df['pub_date'].min()
        # end_time = start_time + delta
        # ending_point = df['pub_date'].max()
        # topic_counts = []
        # period_counts = []
        # time_periods = []
        # print("Starting time analysis")
        # # May instead utilize spacy similarity to determine similarity between article and topics
        # while start_time <= ending_point:
        #     print('Time period left (days): {}'.format((ending_point-start_time).days))
        #     df_dt = df[(df['pub_date'] < end_time) & (df['pub_date'] >= start_time)]
        #     dt_content = df_dt['content'].values
        #     topic_vals = self.nmf.transform(self.t_vect.transform(dt_content))
        #     topic_pick = np.sum(1*(topic_vals >= threshold),axis=0)
        #     topic_counts.append(topic_pick)
        #     period_counts.append(dt_content.shape[0])
        #     time_periods.append(start_time)
        #     start_time = end_time
        #     end_time = start_time + delta
        # self.counts = np.array(topic_counts)
        # self.total_counts = period_counts
        # self.times = np.array(time_periods)
        # print('Time Counts is Complete')

    def save_model(self):
        """ Pickle dumps the object into relevant files under directory 'model/'. Requires fitting of model and time counting to be done.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        cw = Count_Worker(self)
        cw.setup_work()
        with open('app_model/output_data.pkl', 'wb') as od:
            pickle.dump(cw, od)
        with open('app_model/vectorizer.pkl', 'wb') as vc:
            pickle.dump(self.t_vect, vc)
        with open('app_model/factorization.pkl', 'wb') as fc:
            pickle.dump(self.nmf, fc)

    def load_model(self):
        """ Pickle loads this class from relevant files under directory 'model/'.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        with open('app_model/output_data.pkl', 'rb') as od:
            cw = pickle.load(od)
        with open('app_model/vectorizer.pkl', 'rb') as vc:
            self.t_vect = pickle.load(vc)
        with open('app_model/factorization.pkl', 'rb') as fc:
            self.nmf = pickle.load(fc)
        self.topics = cw.all_topics
        self.counts = cw.all_counts.T
        self.total_counts = cw.total_counts
        self.topic_dc = cw.all_dc
        self.times = cw.times
        self.W = cw.W
        self.top_n_words = len(cw.dc[0].keys())
        self.topic_threshold = cw.topic_threshold
        
