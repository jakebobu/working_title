import numpy as np
import matplotlib.pyplot as plt
import pyflux as pf


# TODO: remove all topics when there counts given the threshold < 10

class Count_Worker(object):
    """docstring for Count_Worker."""
    def __init__(self, obj):
        self.topic_counts = obj.counts.T
        self.total_counts = obj.total_counts
        self.topics = obj.topics
        self.times = obj.times
        self.calc_accel()
        self.double_exp_smoothing()

    def drop_useless_topics (self, useful_count = 10):
        tsum = np.sum(self.topic_counts,axis=1)
        self.topic_counts = self.topic_counts[tsum >= useful_count,:]
        self.topics = self.topics[tsum >= useful_count,:]


    def calc_accel(self):
        """ Using the calculated counts of articles in the topics, finds the velocity and acceleration of those counts across all topics

        Assigns the velocity to self._vel
        Assigns the acceleration to self._accel

        Returns
        -------
        None
        """

        if type(self.topic_counts) != np.ndarray or type(self.times) != np.ndarray:
            print("Requires 'perform_time_counting' to be done first")
            return
        rolling_means = np.zeros_like(self.topic_counts)
        N = 5
        for i in range(self.topic_counts.shape[0]):
            rolling_means[i] = np.convolve(self.topic_counts[i], np.ones((N,))/N, mode='same')

        vel = np.zeros_like(rolling_means)
        accel = np.zeros_like(rolling_means)

        for i in range(self.topic_counts.shape[0]):
            vel[i] = np.convolve(rolling_means[i], np.array([1,0,-1]),mode='same')
            accel[i] = np.convolve(rolling_means[i], np.array([1,-2,1]),mode='same')

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

        if type(self.topic_counts) != np.ndarray or type(self.times) != np.ndarray:
            print("Requires 'perform_time_counting' to be done first")
            return
        plt.close('all')
        if type(top_n_topics) == int:
            if top_n_topics > self.topic_counts.shape[0]:
                top_n_topics = self.topic_counts.shape[0]
            for i in range(top_n_topics):
                plt.plot(self.times, self.topic_counts[i], label=i)
        elif type(top_n_topics) == np.array or type(top_n_topics) == list:
            for t in top_n_topics:
                if t in range(self.topic_counts.shape[0]):
                    plt.plot(self.times, self.topic_counts[t], label=t)
        else:
            for i in range(self.topic_counts.shape[0]):
                plt.plot(self.times, self.topic_counts[i], label=i)
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

        N = 5
        rolling_means = np.convolve(self.topic_counts[topic_index], np.ones((N,))/N, mode='same')
        plt.close('all')
        plt.plot(self.times, self.topic_counts[topic_index], '--', alpha=0.6, label = 'Counts')
        plt.plot(self.times, rolling_means, label = 'Counts Smoothed')
        plt.plot(self.times, self._s[topic_index], ':', label = 'S')
        plt.plot(self.times, self._b[topic_index], '-.', label = 'B')
        plt.plot(self.times, self.pos_accel[topic_index], label = 'Acceleration')
        plt.legend()
        plt.show()

    # TODO: look at statsmodels to see what they can offer http://www.statsmodels.org/stable/vector_ar.html#module-statsmodels.tsa.vector_ar
    # http://www.statsmodels.org/stable/vector_ar.html#module-statsmodels.tsa.vector_ar.var_model

    # TODO: look at the capabilities of http://www.pyflux.com/

    #TODO: cut counts data off early a couple time periods and use this to go back in and predict those values
    # Vary alpha and beta to see if there are optimal values for predictions
    def double_exp_smoothing(self, alpha=0.5, beta=0.5):
        """ Applies double exponential smoothing to the counts of articles for topics

        Parameters
        ----------
        None, requires counts to be created before this step

        Returns
        -------
        None, assigns self._s and self._b which can be used for predictions
        """

        s = self.topic_counts.copy()
        b = s.copy()
        b[0,:] = (s[5,:]-s[0,:])/5
        b[1,:] -= s[0,:]
        for i in range(2, s.shape[1]):
            s[:,i] = alpha * s[:,i] + (1 - alpha) * (s[:,i-1] + b[:,i-1])
            b[:,i] = beta * (s[:,i] - s[:,i-1]) + (1 - beta) * b[:,i-1]
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

    def bte (self, topic_index=0):
        model = pf.EGARCH(self.topic_counts[topic_index],p=1,q=1)
        x = model.fit()
        x.summary()
        model.plot_fit()
