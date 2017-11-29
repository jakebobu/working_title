from words_to_vals import NMF_Time, _tokenize
from work_with_counts import Count_Worker
import pickle
from scipy.optimize import minimize, differential_evolution
import numpy as np
import pandas as pd
import datetime as dt

def minimize_error(weights):
    """ Minimization function, the y values are stored in the main block

    Parameters
    ----------
    weights: a tuple of current alpha, beta, gamma values

    Returns
    -------
    error: Sum of squared errors value looking to be minimized
    """

    periods_ahead = 6
    m = 7
    alpha = weights[0]
    beta = weights[1]
    gamma = weights[2]
    s = Y.copy()
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
            s[:,i]=alpha*(Y[:,i]-c_0[:,i])+(1-alpha)*(s[:,i-1] + b[:,i-1])
            b[:,i]=beta*(s[:,i]-s[:,i-1])+(1-beta)*b[:,i-1]
            c[:,i]=gamma*(Y[:,i]-s[:,i])+(1-gamma)*c_0[:,i]
        else:
            s[:,i]=alpha*(Y[:,i]-c[:,i-L])+(1-alpha)*(s[:,i-1] + b[:,i-1])
            b[:,i]=beta*(s[:,i]-s[:,i-1])+(1-beta)*b[:,i-1]
            c[:,i]=gamma*(Y[:,i]-s[:,i])+(1-gamma)*c[:,i-L]
    error = 0
    # for i in range(s.shape[0]): # For each topic
    for j in range(s.shape[1]-periods_ahead): #for all times that can be predicted ahead
        error += np.sum(Y[:,j+m-1]-(s[:,j]+m*b[:,j]+c[:,(j+m)%L]))**2
    return error

def minimize_start():
    """ Utilizes scipy optimization to mimimize the Holt-Winter hyper-parameters

    Parameters
    ----------
    None

    Returns
    -------
    The response object of the scipy optimization
    """

    with open('app_model/output_data.pkl','rb') as f:
        cw = pickle.load(f)
    w0 = np.array([0.2,0.2,0.2])
    Y = cw.smooth_data
    r = (0.001,0.999)
    # return minimize(minimize_error, w0, bounds=[(0.0,1.0),(0.0,1.0),(0.0,1.0)])
    # return differential_evolution(minimize_error,bounds=[r,r,r])

def generate_model(data_location, save_model=True):
    """ Generates a model for the flask app to utilize as data source

    Parameters
    ----------
    data_location: the relative location of the data to generate the model
    save_model: if you want to save this generated model for the flask app to use

    Returns
    -------
    nmf_model: the generated model
    """

    nmf_model = NMF_Time(top_n_words=25, verbose=True)
    df = pd.read_csv(data_location, index_col=0)
    df = df[df['news_source'] == 'NYT'] # Currently due to not enough from other sources
    nmf_model.generate_topics(df['content'].values, tok=_tokenize, min_df = 0.01, max_features = 10000, n_components=500)
    nmf_model.perform_time_counting_self(df, delta=dt.timedelta(hours=4), threshold=0.05)
    if save_model:
        nmf_model.save_model()
    return nmf_model

def load_prior_model():
    ''' Loads and returns the currently saved pickled model found under '/app_model' '''
    return NMF_Time(load_model=True)

if __name__ == '__main__':
    obj = generate_model('../temp_data1.csv',save_model=False)
