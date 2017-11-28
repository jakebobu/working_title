import numpy as np
import pyflux as pf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
from itertools import product

# TODO: remove all topics when there counts given the threshold < 10

class Count_Worker(object):
    """docstring for Count_Worker."""
    def __init__(self, obj):
        #TODO: determine how many periods ahead I want to predict (thinking 6 periods (1 day))
        self.all_counts = obj.counts.T
        self.topic_counts = self.all_counts.copy()
        self.total_counts = obj.total_counts
        self.all_topics = obj.topics
        self.topics = self.all_topics.copy()
        self.all_dc = obj.topic_dc
        self.dc = self.all_dc.copy()
        self.times = obj.times
        self.article_relates = obj.article_relates
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
            self.smooth_data[i] = 6.0*np.convolve(self.topic_counts[i],kernel,mode='same')

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

    def triple_exp_smoothing(self, alpha = 0.5, beta = 0.5, gamma = 0.5):
        """ Generates the variables for which to show and predict triple exponential smoothing
        Applies them to self under self.s_e3, self.b_e3, self.c_e3
        Requires this to run 'triple_exp_predict'

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        s = self.smooth_data.copy()
        b = np.zeros_like(s)
        c = b.copy()
        L = 42 # weekly, sampling rate is 4 hours -> 7 days/week * 24 hours/day / 4 hours/sample = 42 samples/week
        n_cycles = s.shape[1] // L
        c_0 = np.zeros((s.shape[0],L))
        avgs = [np.sum(s[:,i*L:(i+1)*L],axis=1)/L for i in range(n_cycles)]
        for i in range(L):
            b[:,0] += (s[:,i+L]-s[:,i])/(L*L)
            c_0[:,i] = sum([s[:,L*j + i]-avgs[j] for j in range(n_cycles)])/n_cycles
        c[:,0]=c_0[:,0]
        for i in range(1, s.shape[0]):
            if i < L:
                s[:,i]=alpha*(self.smooth_data[:,i]-c_0[:,i])+(1-alpha)*(s[:,i-1] + b[:,i-1])
                b[:,i]=beta*(s[:,i]-s[:,i-1])+(1-beta)*b[:,i-1]
                c[:,i]=gamma*(self.smooth_data[:,i]-s[:,i])+(1-gamma)*c_0[:,i]
            else:
                s[:,i]=alpha*(self.smooth_data[:,i]-c[:,i-L])+(1-alpha)*(s[:,i-1] + b[:,i-1])
                b[:,i]=beta*(s[:,i]-s[:,i-1])+(1-beta)*b[:,i-1]
                c[:,i]=gamma*(self.smooth_data[:,i]-s[:,i])+(1-gamma)*c[:,i-L]
        self.s_e3 = s
        self.b_e3 = b
        self.c_e3 = c

    def triple_exp_predict(self, topic_index=0, periods_ahead=1, at_time=-1):
        """ Triple Exponential Prediction. Requires the generation of s,b,c first.

        Parameters
        ----------
        topic_index: The index of the desired topic to see predict
        periods_ahead: how many periods ahead to predict
        at_time: at which point in time you want to predict

        Returns
        -------
        A predicted count of articles for that topic
        """

        L = 42
        predictions = np.zeros(periods_ahead)
        while at_time < 0:
            at_time += self.s_e3.shape[1]
        for p in range(periods_ahead):
            predictions[p] = self.s_e3[topic_index,at_time]+(p+1)*self.b_e3[topic_index,at_time]+self.c_e3[topic_index,(at_time+p+1) % L]
        return predictions

    def triple_exp_error(self, alpha=0.5, beta=0.5, gamma=0.5, periods_ahead=1):
        """ Given hyper-parameters and a number of periods ahead to predict, returns the RMSE
        Utilizes triple exponential smoothing to produce predictions

        Parameters
        ----------
        alpha,beta,gamma: the specified values of these hyper-parameters for the triple exponential
        periods_ahead: number of periods ahead to predict

        Returns
        -------
        error: the RMSE from the provided parameters
        """

        s = self.smooth_data.copy()
        b = np.zeros_like(s)
        c = np.zeros_like(s)
        L = 42 # weekly, sampling rate is 4 hours -> 7 days/week * 24 hours/day / 4 hours/sample = 42 samples/week
        n_cycles = s.shape[1] // L
        c_0 = np.zeros((s.shape[0],L))
        avgs = [np.sum(s[:,i*L:(i+1)*L],axis=1)/L for i in range(n_cycles)]
        for i in range(L):
            b[:,0] += (s[:,i+L]-s[:,i])/(L*L)
            c_0[:,i] = sum([s[:,L*j + i]-avgs[j] for j in range(n_cycles)])/n_cycles
        c[:,0]=c_0[:,0]
        for i in range(1, s.shape[0]):
            if i < L:
                s[:,i]=alpha*(self.smooth_data[:,i]-c_0[:,i])+(1-alpha)*(s[:,i-1] + b[:,i-1])
                b[:,i]=beta*(s[:,i]-s[:,i-1])+(1-beta)*b[:,i-1]
                c[:,i]=gamma*(self.smooth_data[:,i]-s[:,i])+(1-gamma)*c_0[:,i]
            else:
                s[:,i]=alpha*(self.smooth_data[:,i]-c[:,i-L])+(1-alpha)*(s[:,i-1] + b[:,i-1])
                b[:,i]=beta*(s[:,i]-s[:,i-1])+(1-beta)*b[:,i-1]
                c[:,i]=gamma*(self.smooth_data[:,i]-s[:,i])+(1-gamma)*c[:,i-L]
        error = 0
        for i in range(s.shape[0]): # For each topic
            for j in range(s.shape[1]-periods_ahead): #for all times that can be predicted ahead
                for m in range(1,periods_ahead+1): # for that specified ahead prediction rate
                    error += (self.topic_counts[i,j+m]-(s[i,j]+(m+1)*b[i,j]+c[i,(j+m)%L]))**2/periods_ahead
        return (error/(periods_ahead*(s.shape[1]-periods_ahead)*s.shape[0]))**0.5


# (0.0, 0.80000000000000004, 1.0)  3.066492

    def triple_exp_select (self, N=3, start_new=False):
        """ Triple Exponential hyper-parameter Selection using a temporary storage pickle file
        Looks through an N set of values in range 0-1 for alpha, beta, gamma
        Finds the parameters with the lowest RMSE

        Parameters
        ----------
        topic_index: The index of the desired topic to see its counts and acceleration
        periods_ahead: how many periods ahead to predict

        Returns
        -------
        A predicted count of articles for that topic
        """

        r = np.linspace(0.01,0.99,N)
        if start_new:
            df = pd.DataFrame(columns=['hyperparams','error'])
        else:
            df = pd.read_csv('error_data1.csv',index_col=0)
        for c in product(r,repeat=3):
            if c not in df.hyperparams.values:
                print("Working on", c)
                error = self.triple_exp_error(alpha=c[0],beta=c[1],gamma=c[2],periods_ahead=6)
                df.loc[len(df.index)]=[c,error]
                df.to_csv('error_data1.csv')

    def plot_triple_exp_from_df(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        df = pd.read_csv('error_data1.csv',index_col=0)
        hp = np.array([t.strip('(').strip(')').split(',') for t in df.hyperparams.values]).astype(float)
        df['alpha']= hp[:,0]
        df['beta'] = hp[:,1]
        df['gamma'] = hp[:,2]
        for g in df['gamma'].unique():
            df_min = df[df.gamma == g]
            # df_min = df[df['gamma']==df[df['error']==df.error.min()].gamma.values[0]]
            X, Y = np.meshgrid(df_min.alpha.unique(), df_min.beta.unique())
            Z = np.zeros_like(X)
            for i in range(Z.shape[0]):
                Z[i,:]=df_min[df_min.alpha == X[0,i]].error.values
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        e_min = np.argmin(df.error.values)
        ax.scatter(hp[e_min,0], hp[e_min,1], df.error.min(), c='k',s=50)
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlim(df.error.min()-.1,df.error.max()+.1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


    # Finds alpha/beta with lowest error
    def exp_dc (self, periods_ahead = 1):
        N = 11
        alphas = np.linspace(0,1,N)
        betas = alphas.copy()
        errors = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                errors[i,j] = self.double_exp_comparions(alpha = alphas[i], beta = betas[j],periods_ahead = periods_ahead)
        e_min = np.argmin(errors)
        return alphas[e_min // N], betas[e_min % N], errors

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
            for j in range(periods_ahead, s.shape[1]):
                error += np.abs(self.topic_counts[i,j] - (s[i,j-periods_ahead] + periods_ahead * b[i,j-periods_ahead]))/s.shape[1]
        return error

# 0.37 1.0 24.2570408562

    def plot_double_exp_vars(self, periods_ahead=1, N=11):
        alphas = np.linspace(0.1,1,N)
        betas = np.linspace(0.1,1,N)
        errors = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                errors[i,j] = self.double_exp_comparions(alpha = alphas[i], beta = betas[j],periods_ahead = periods_ahead)

        X, Y = np.meshgrid(alphas,betas)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, errors, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        e_min = np.argmin(errors)
        print(alphas[e_min // N], betas[e_min % N], np.min(errors))
        ax.scatter([alphas[e_min // N]], [betas[e_min % N]], np.min(errors), c='k',s=50)
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlim(np.min(errors)-.1,np.max(errors)+.1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def predict_all(self, periods_ahead=6):
        self.triple_exp_smoothing(alpha=0.5, beta=0.5, gamma=1.0)
        predicted_values = np.zeros((self.smooth_data.shape[0],periods_ahead+1))
        predicted_values[:,0]= self.smooth_data[:,-1]
        for i in range(predicted_values.shape[0]):
            predicted_values[i,1:]=self.triple_exp_predict(topic_index=i, periods_ahead=periods_ahead)
        self.predicted_values = predicted_values
        delta_time = self.times[1] - self.times[0]
        self.predicted_times = np.array([self.times[-1] + delta_time*i for i in range(periods_ahead + 1)])

    def plot_predicted_thginy(self,topic_index=0):
        pb = 42 # One week period back
        pa = 6 # one day ahead
        counts_back = self.smooth_data[topic_index,-pb:]
        back_times = self.times[-pb:]
        counts_fwd = self.predicted_values[topic_index]
        plt.plot(back_times, counts_back,'r',label='Current')
        plt.plot(self.predicted_times, counts_fwd,'r--',label='Predicted')
        plt.axvline(x=self.predicted_times[0], c='k', ls=':',label='Today')
        plt.legend()
        plt.show()

    def find_trending_topics(self):
        '''
            Creates a sorted order of topics in descending order of trendingness
        '''
        recent_counts = np.zeros((self.smooth_data.shape[0], 48))
        recent_counts[:,:42] = self.smooth_data[:,-42:]
        recent_counts[:,41:] = self.predicted_values
        trend = np.zeros((self.smooth_data.shape[0],3))
        points = np.zeros_like(recent_counts)
        for i in range(recent_counts.shape[0]):
            if np.sum(recent_counts[i]) >= 3:
                trend[i], points[i] = self.recent_trend_strength(recent_counts[i])
            else:
                trend[i,:] = np.array([0.0,0.0,0.0])
                points[i] = np.zeros_like(recent_counts[i])
        self.trend_points = points
        self.trend_times = np.array([self.times[-42] + i*(self.times[1] - self.times[0]) for i in range(recent_counts.shape[1])])
        trend = trend[:,0]*(1.0*trend[:,1] >= 0.0)
        self.trending_order = np.argsort(trend)[::-1]

    def recent_trend_strength(self, counts):
        # vel = np.convolve(counts, np.array([1,0,-1]), mode='same')
        # accel = np.convolve(counts, np.array([1,-2,1]), mode='same')
        vals = np.polyfit(np.arange(counts.shape[0]), counts, deg=2)
        points = np.zeros_like(counts)
        for i in range(len(points)):
            points[i] = vals[0]*i**2 + vals[1]*i + vals[2]
        return np.array(vals), points

# import pickle
# with open('app_model/output_data.pkl','rb') as f:
#     cw = pickle.load(f)
# cw.plot_double_exp_vars(N=3)


# END
