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
'''

import pickle
import spacy
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
        self._c = 0
        self._train_length = 0
        self._punctuation = punctuation + '’' + '--' + '’s'
        self._vel = None
        self._accel = None

    def _tokenize(self, doc):
        '''
            tokenizer function to use for the TfidfVectorizer class
            Currently using spacy, can replace with nltk stuff as well for comparison
        '''
        self._c += 1
        print('Tokenizing ({0}/{1})'.format(self._c,self._train_length), end="\r")
        wList = [token.text if token.lemma_ == '-PRON-' else token.lemma_ for token in self.nlp(doc)]
        return [token for token in wList if token not in self._punctuation]

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
        ordering = H.argsort(axis=1)[:,:-top_n_words-1:-1]
        for i in range(H.shape[0]):
            tp = [vocab[ordering[i,j]] for j in range(top_n_words)]
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
        self.times = time_periods
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

        # Look to see if you can utilize np.convolve instead of for loops
        # thinking for vel [-1,0,1]
        # thinking for accel [1,-2,1]
        # Compare results
        vel = np.zeros_like(rolling_means)
        accel = np.zeros_like(rolling_means)
        for i in range(1, vel.shape[0]-1):
            vel[i,:]=rolling_means[i+1,:] - rolling_means[i-1,:]
            accel[i,:] = rolling_means[i + 1,:] + rolling_means[i-1,:] - 2*rolling_means[i,:] # / dt^2
        vel[0,:] = rolling_means[1,:] - rolling_means[i,:]
        vel[-1,:] = rolling_means[-1,:] - rolling_means[-2,:]
        accel[0,:] = rolling_means[1,:] - rolling_means[i,:]
        accel[-1,:] = rolling_means[-2,:] - rolling_means[-1,:]
        self._vel = vel
        self._accel = accel
        self.pos_accel = accel*(vel > 0)*(accel > 0)

    def do_some_plotting(self):
        ''' Currently doing flat counts '''
        plt.close('all')
        for i in range(self.counts.shape[1]):
            plt.plot(self.times, self.counts[:,i], label=i)
        plt.legend()
        plt.show()

    def plot_count_accel(self, topic_index=5):
        plt.close('all')
        plt.plot(self.times,self.counts[:,topic_index], label = 'Counts')
        plt.plot(self.times,self.pos_accel[:,topic_index], label = 'Acceleration')
        plt.legend()
        plt.show()

    def to_pkl(filename='model.pkl'):
        with open (filename, 'wb') as f:
            pickle.dumb(self, f)

if __name__ == '__main__':
    # nlp = spacy.load('en')
    df = pd.read_csv('temp_data1.csv')
    df = df[df['news_source'] == 'NYT']
    testy = NMF_Time()
    testy.generate_topics(df['content'].values)
    testy.perform_time_counting(df)
    testy.calc_accel()
    testy.to_pkl()
    # testy.do_some_plotting()
    # W, H = do_local()


    # nlp = spacy.load('en')
    # corpus_vocab = []
    # vocab = []
    # for i, c in enumerate(content):
    #     print('Working on', i)
    #     vocab_set = {}
    #     doc = nlp(c)
    #     for token in doc:
    #         vocab_set[token.lemma_] = 1 + vocab_set.get(token.lemma_, 0)
    #         vocab.append(token.lemma_)
    #     corpus_vocab.append(vocab_set)
    # vocab = list(set(vocab))
    # mat = np.zeros((len(content), len(vocab)))
    # for i in range(mat.shape[0]):
    #     for j in range(mat.shape[1]):
    #         mat[i,j] = corpus_vocab[i].get(vocab[j],0.0)
    # return mat
