from work_with_counts import Count_Worker
import pickle
from scipy.optimize import minimize, differential_evolution
import numpy as np

def minimize_error(weights):
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

if __name__ == '__main__':
    with open('app_model/output_data.pkl','rb') as f:
        cw = pickle.load(f)
    w0 = np.array([0.2,0.2,0.2])
    Y = cw.smooth_data
    r = (0.001,0.999)
    # response = minimize(minimize_error, w0, bounds=[(0.0,1.0),(0.0,1.0),(0.0,1.0)])
    # response = differential_evolution(minimize_error,bounds=[r,r,r])
