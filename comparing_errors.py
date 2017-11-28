
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def triple_exp_me(data, alpha=0.5, beta=0.5, gamma=0.5):
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

    s = data.copy()
    b = np.zeros_like(s)
    c = np.zeros_like(s)
    L = 42 # weekly, sampling rate is 4 hours -> 7 days/week * 24 hours/day / 4 hours/sample = 42 samples/week
    n_cycles = s.shape[0] // L
    c_0 = np.zeros((s.shape[0],L))
    avgs = [np.sum(s[:,i*L:(i+1)*L],axis=1)/L for i in range(n_cycles)]
    for i in range(L):
        b[:,0] += (s[:,i+L]-s[:,i])/(L*L)
        c_0[:,i] = sum([s[:,L*j + i]-avgs[j] for j in range(n_cycles)])/n_cycles
    c[:,0]=c_0[:,0]
    for i in range(1, s.shape[0]):
        if i < L:
            s[:,i]=alpha*(data[:,i]-c_0[:,i])+(1-alpha)*(s[:,i-1] + b[:,i-1])
            b[:,i]=beta*(s[:,i]-s[:,i-1])+(1-beta)*b[:,i-1]
            c[:,i]=gamma*(data[:,i]-s[:,i])+(1-gamma)*c_0[:,i]
        else:
            s[:,i]=alpha*(data[:,i]-c[:,i-L])+(1-alpha)*(s[:,i-1] + b[:,i-1])
            b[:,i]=beta*(s[:,i]-s[:,i-1])+(1-beta)*b[:,i-1]
            c[:,i]=gamma*(data[:,i]-s[:,i])+(1-gamma)*c[:,i-L]
    return c + s + b


def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals

def initial_trend(series, slen):
    ssum = 0.0
    for i in range(slen):
        ssum += float(series[i+slen] - series[i]) / slen
    return ssum / slen

def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result


if __name__ == '__main__':
    with open('flask_app/app_model/output_data.pkl','rb') as f:
        cw = pickle.load(f)
    other_vals = triple_exponential_smoothing(cw.smooth_data[5],42,0.5,0.5,0.5,0)
    my_vals = triple_exp_me(cw.smooth_data)
    plt.plot(cw.smooth_data[5],'b',label='counts')
    plt.plot(other_vals,'y--',label='other')
    plt.plot(my_vals[5],'g-.',label='mine')
    plt.legend()
    plt.show()
