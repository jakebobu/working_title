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


class NMF_Time(object):
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
        self.times = None
        self.topics = None
        self.pos_accel = None
        self.nlp = spacy.load('en')
        self._spacy_tokenizer = English().Defaults.create_tokenizer(self.nlp)
        self._c = 0
        self._train_length = 0
        self._punctuation = punctuation + '’' + '--' + '’s'
        self._vel = None
        self._accel = None


    # TODO: remove just numbers and email addresses and
    def _tokenize(self, doc):
        '''
            tokenizer function to use for the TfidfVectorizer class
            Currently using spacy, can replace with nltk stuff as well for comparison
        '''
        self._c += 1
        print('Tokenizing ({0}/{1})'.format(self._c,self._train_length), end="\r")
        wList = [t.text if t.lemma_ == '-PRON-' else t.lemma_ for t in [token for token in self._spacy_tokenizer(doc) if token.is_alpha]]
        return [token for token in wList if token not in self._punctuation and '@' not in token]

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
        nmf = NMF(n_components = kwargs.get('n_components', 10), init = kwargs.get('init', 'nndsvd'), solver = kwargs.get('solver', 'cd'), random_state = 2, alpha = kwargs.get('alpha', 0), l1_ratio = kwargs.get('l1_ratio', 0), shuffle = True)
        print('Starting Vectorizer')
        self._c = 0
        print('Tokenizing (1/{0})'.format(self._train_length), end="\r")
        t_mat = t_vect.fit_transform(content)
        print('Tokenizing completed ({0}/{0})'.format(self._train_length))
        print('Starting NMF')
        nmf.fit(t_mat)
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

    def perform_time_counting (self, df, delta=dt.timedelta(days=3), threshold=0.1):
        """ Takes in a dataframe of data, and returns across time the percentage of total articles that are part of topics

        Parameters
        ----------
        df (pandas.dataframe) : DataFrame of articles containing article content and article publication date. Can be completely new data
        dt (datetime.timedelta) : timespan for which to bin articles into (default = 3 days)
        threshold : the value at which equal to or above an article is considered counted as of that topic (default = 0.1)

        Returns
        -------
        topic_counts : counts of articles that are pertaining to a topic, across time
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
            topic_pick = np.sum(1*(topic_vals >= threshold),axis=0)/dt_content.shape[0]
            topic_counts.append(topic_pick)
            time_periods.append(start_time)
            start_time = end_time
            end_time = start_time + delta
        self.counts = np.array(topic_counts)
        self.times = np.array(time_periods)
        print('Time Counts is Complete')

    def calc_accel(self):
        """ Using the calculated counts of articles in the topics, finds the velocity and acceleration of those counts across all topics

        Assigns the velocity to self._vel
        Assigns the acceleration to self._accel

        Returns
        -------
        None
        """

        if type(self.counts) != np.ndarray or type(self.times) != np.ndarray:
            print("Requires 'perform_time_counting' to be done first")
            return
        rolling_means = np.zeros_like(self.counts)
        N = 3
        for i in range(self.counts.shape[1]):
            rolling_means[:,i] = np.convolve(self.counts[:,i], np.ones((N,))/N, mode='same')

        vel = np.zeros_like(rolling_means)
        accel = np.zeros_like(rolling_means)

        for i in range(self.counts.shape[1]):
            vel[:,i] = np.convolve(rolling_means[:,i], np.array([1,0,-1]),mode='same')
            accel[:,i]=np.convolve(rolling_means[:,i], np.array([1,-2,1]),mode='same')

        self._vel = vel
        self._accel = accel
        self.pos_accel = accel*(vel > 0)*(accel > 0)

    def plot_topics_across_time(self, top_n_topics=None):
        """ Plots the counts of the desired topics across time

        Parameters
        ----------
        top_n_topics: None shows all topics, type(int) returns the top topics of that number, type(list/numpy.array) returns those topics if they exist

        Returns
        -------
        A plot of those topics in top_n_topics
        """

        if type(self.counts) != np.ndarray or type(self.times) != np.ndarray:
            print("Requires 'perform_time_counting' to be done first")
            return
        plt.close('all')
        if type(top_n_topics) == int:
            if top_n_topics > self.counts.shape[1]:
                top_n_topics = self.counts.shape[1]
            for i in range(top_n_topics):
                plt.plot(self.times, self.counts[:,i], label=i)
        elif type(top_n_topics) == np.array or type(top_n_topics) == list:
            for t in top_n_topics:
                if t in range(self.counts.shape[1]):
                    plt.plot(self.times, self.counts[:,t], label=t)
        else:
            for i in range(self.counts.shape[1]):
                plt.plot(self.times, self.counts[:,i], label=i)
        plt.legend()
        plt.show()

    def plot_count_accel(self, topic_index=5):
        """ Given a selected topic, plots the counts, averaged counts, and acceleration of that topic

        Parameters
        ----------
        topic_index: The index of the desired topic to see its counts and acceleration

        Returns
        -------
        A plot of the topic at topic_index containing acceleration and counts
        """

        N = 3
        rolling_means = np.convolve(self.counts[:,topic_index], np.ones((N,))/N, mode='same')
        plt.close('all')
        plt.plot(self.times,self.counts[:,topic_index], label = 'Counts')
        plt.plot(self.times,rolling_means, '--', label = 'Counts Smoothed')
        plt.plot(self.times,self.pos_accel[:,topic_index], label = 'Acceleration')
        plt.legend()
        plt.show()

    # TODO: look at statsmodels to see what they can offer http://www.statsmodels.org/stable/vector_ar.html#module-statsmodels.tsa.vector_ar
    # http://www.statsmodels.org/stable/vector_ar.html#module-statsmodels.tsa.vector_ar.var_model

    # TODO: look at the capabilities of http://www.pyflux.com/

    #TODO: cut counts data off early a couple time periods and use this to go back in and predict those values
    # Vary alpha and beta to see if there are optimal values for predictions
    def double_exp_smoothing(self):
        """ Applies double exponential smoothing to the counts of articles for topics

        Parameters
        ----------
        None, requires counts to be created before this step

        Returns
        -------
        None, assigns self._s and self._b which can be used for predictions
        """

        s = self.counts.copy()
        b = s.copy()
        alpha = 0.62
        beta = 0.5
        b[0,:] = (s[5,:]-s[0,:])/5
        b[1,:] -= s[0,:]
        for i in range(2, s.shape[0]):
            s[i,:] = alpha * s[i,:] + (1 - alpha) * (s[i-1,:] + b[i-1,:])
            b[i,:] = beta * (s[i,:] - s[i-1,:]) + (1 - beta) * b[i-1,:]
        self._s = s
        self._b = b

    def predict_ahead(self, topic_index=0, periods_ahead=1):
        """ Predicts topic counts a desired periods ahead

        Parameters
        ----------
        topic_index: The index of the desired topic to see its counts and acceleration
        periods_ahead: how many periods ahead to predict

        Returns
        -------
        A predicted count of articles for that topic
        """

        return self._s[topic_index,-1] + periods_ahead * self._b[topic_index,-1]

    # # Doesn't like pickling something about this class
    # def to_pkl(filename='model.pkl'):
    #     with open (filename, 'wb') as f:
    #         pickle.dumb(self, f)

if __name__ == '__main__':
    from words_to_vals import NMF_Time
    df = pd.read_csv('temp_data1.csv',index_col=0)
    df = df[df['news_source'] == 'NYT']
    testy = NMF_Time()
    testy.generate_topics(df['content'].values)
    testy.perform_time_counting(df)
    testy.calc_accel()
    with open ('model.pkl', 'wb') as f:
        pickle.dumb(testy, f)
