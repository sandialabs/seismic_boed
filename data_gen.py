import numpy as np
import like_models as lm
import time
import warnings
warnings.filterwarnings('error')
import sys
import os
import signal

def gen_arrival_normal(theta, sensors, ndata):
    # Variance is combination of arrival time and general sensor variance
    #compute tt mean, model std, measruement std
    tt_data = lm.compute_tt(theta, sensors)
    mean_tt = tt_data[:,0]
    stdmodel = tt_data[:,1]
    measurenoise = tt_data[:,2]


    #compute corr matrix
    corr = lm.compute_corr(theta, sensors)
    cov = np.multiply(np.outer(stdmodel,stdmodel),corr) + np.diag(measurenoise**2.0)
    min_eig = np.min(np.real(np.linalg.eigvals(cov)))

    try:
        return np.random.multivariate_normal(mean_tt, cov, ndata, tol=1e-5)
    except RuntimeWarning as r:
        print(cov)
        print('--------------------------------------------------------------')
        print(mean_tt)
        np.save('bad_data.npy', theta)
        np.save('tt.npy', mean_tt)
        np.save('new_psd_arr3.npy', cov)
        os.kill(os.getpid(), signal.SIGINT)

def generate_data(theta,sensors,ndata):
    #compute detection probablity
    probs = lm.detection_probability(theta,sensors)
    
    #make probs bigger
    fullprobs = np.outer(np.ones(ndata),probs)
    u_mat = np.random.uniform(size = fullprobs.shape)
    
    #sample arrival times
    atimes = gen_arrival_normal(theta, sensors, ndata)
    print(np.sum(u_mat<fullprobs))
    #get data[probs arrivaltimes]
    data = np.concatenate((atimes,u_mat<fullprobs),axis=1)
    return data
