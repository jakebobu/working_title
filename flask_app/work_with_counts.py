import numpy as np
import matplotlib.pyplot as plt
import pyflux as pf


# TODO: remove all topics when there counts given the threshold < 10

class Count_Worker(object):
    """docstring for Count_Worker."""
    def __init__(self, obj):
        self.all_counts = obj.counts.T
        self.topic_counts = self.all_counts.copy()
        self.total_counts = obj.total_counts
        self.all_topics = obj.topics
        self.topics = self.all_topics.copy()
        self.all_dc = obj.topic_dc
        self.dc = self.all_dc.copy()
        self.times = obj.times
        # if type(obj.W) == np.array:
        #     self.W = obj.W
        self.H = obj.nmf.components_
        self.web_index = None

    def setup_work(self):
        self.drop_useless_topics()
        self.calc_accel()
        self.double_exp_smoothing()
        self.data_smoothing()

    def drop_useless_topics (self, useful_count = 3):
        tsum = np.sum(self.all_counts,axis=1)
        self.topic_counts = self.all_counts[tsum >= useful_count,:]
        self.topics = self.all_topics[tsum >= useful_count,:]
        self.topics = {i : self.topics[i,:] for i in range(self.topics.shape[0])}
        self.dc = self.all_dc[tsum >= useful_count]

    def calc_accel(self):
        """ Using the calculated counts of articles in the topics, finds the velocity and acceleration of those counts across all topics

        Assigns the velocity to self._vel
        Assigns the acceleration to self._accel

        Returns
        -------
        None
        """

        # if type(self.topic_counts) != np.ndarray or type(self.times) != np.ndarray:
        #     print("Requires 'perform_time_counting' to be done first")
        #     return
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

    def exp_smooth_range(self, topic_index=0):
        alphas = [0.1,0.5,0.9]
        betas = [0.1,0.5,0.9]
        s_vals = dict()
        b_vals = dict()
        for al in alphas:
            for bt in betas:
                s = self.topic_counts[topic_index].copy()
                b = s.copy()
                b[0] = (s[5]-s[0])/5
                b[1] -= s[0]
                for i in range(2,s.shape[0]):
                    s[i] = al*s[i]+(1-al)*(s[i-1]+b[i-1])
                    b[i] = bt*(s[i]-s[i-1])+(1-bt)*b[i-1]
                s_vals[(al,bt)] = s
                b_vals[(al,bt)] = b
        plt.close('all')
        plt.plot(self.times,self.topic_counts[topic_index], alpha=0.5, label='Counts')
        for k, v in s_vals.items():
            plt.plot(self.times, v, '--', label='a:{} , b:{}'.format(k[0],k[1]))
        plt.legend()
        plt.show()

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

    def gasy (self, topic_index=0):
        model = pf.GAS(ar=2, sc=2, data=self.topic_counts[topic_index], family=pf.Poisson())
        x = model.fit()
        x.summary()
        model.plot_fit()

    def data_smoothing(self):
        import scipy.stats as scs
        N = 13
        kernel = np.ones((N,))
        for i in range (N):
            kernel[i] = scs.norm(scale=4).pdf(i - N//2)
        kernel = kernel/np.sum(kernel)
        self.smooth_data = 1.0 * self.topic_counts.copy()
        # self.tot_sm = np.convolve(1.0*self.total_counts, kernel, mode='same')
        for i in range(self.topic_counts.shape[0]):
            self.smooth_data[i] = np.convolve(self.topic_counts[i],kernel,mode='same')

    def plot_smoothing_techniques(self):
        plt.close('all')
        plt.subplot(3,1,1)
        plt.title('Simple Counts')
        for i in range(self.topic_counts.shape[0]):
            plt.plot(self.times, self.topic_counts[i], label=i)
        plt.subplot(3,1,2)
        plt.title('Smoothed Counts (Avg)')
        for i in range(self.topic_counts.shape[0]):
            plt.plot(self.times, self.smooth_data[i], label=i)
        plt.subplot(3,1,3)
        plt.title('Smoothed Counts (Exp)')
        for i in range(self.topic_counts.shape[0]):
            plt.plot(self.times, self._s[i], label=i)
        plt.show()

    # def triple_exp_smoothing(self):
    #     s = self.smooth_data.copy()
    #     b = s.copy()
    #     c = b.copy()
    #     alpha = 0.5
    #     beta = 0.5
    #     gamma = 0.5
    #     L = 7*24/4 # weekly, sampling rate is 4 hours

    # Finds alpha/beta with lowest error
    def dc (self, periods_ahead = 1):
        N = 11
        alphas = np.linspace(0,1,N)
        betas = alphas.copy()
        errors = np.zeros(N)
        for i in range(N):
            for j in range(N):
                errors[i,j] = double_exp_comparions(alpha = alphas[i], beta = betas[j],periods_ahead = periods_ahead)
        e_min = np.argmin(errors)
        return alphas[e_min // N], betas[e_min % N]

    def double_exp_comparions(self, alpha=0.5, beta=0.5, periods_ahead=1):
        error = 0
        s = self.smooth_data.copy()
        b = s.copy()
        b[0,:] = (s[5,:]-s[0,:])/5
        b[1,:] -= s[0,:]
        for i in range(2, s.shape[1]):
            s[:,i] = alpha * s[:,i] + (1 - alpha) * (s[:,i-1] + b[:,i-1])
            b[:,i] = beta * (s[:,i] - s[:,i-1]) + (1 - beta) * b[:,i-1]
        for i in range(s.shape[0]):
            for j in range(1, s.shape[1]):
                error += np.abs(self.smooth_data[i,j] - (s[i,j-1] + periods_ahead * b[i,j-1]))/s.shape[1]
        return error









# END
