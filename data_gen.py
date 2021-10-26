import numpy as np
import like_models as lm

def gen_arrival_normal(theta, sensors, ndata):
    # Variance is combination of arrival time and general sensor variance
    #compute mean
    mean_tt = lm.compute_mean_tt(theta, sensors)
    
    #compute corr matrix
    corr = lm.compute_corr(theta, sensors)
    
    measurenoise = sensors[:,2]
    stdmodel = (2.75758229e-02)*mean_tt + (-5.57985096e-04)*(mean_tt**2.0) + (1.63610033e-05)*(mean_tt**3.0)
    
    cov = np.multiply(np.outer(stdmodel,stdmodel),corr) + np.diag(measurenoise**2.0)
    np.random.seed(0)
    return np.random.multivariate_normal(mean_tt, cov, ndata)


def generate_data(theta,sensors,ndata):
    #compute detection probablity
    probs = lm.detection_probability(theta,sensors)
    
    #make probs bigger
    fullprobs = np.outer(np.ones(ndata),probs)
    np.random.seed(0)
    u_mat = np.random.uniform(size = fullprobs.shape)
    
    #sample arrival times
    atimes = gen_arrival_normal(theta, sensors, ndata)
    
    #get data[probs arrivaltimes]
    data = np.concatenate((atimes,u_mat<fullprobs),axis=1)
    return data