import numpy as np
import sobol_seq as sq
from scipy import stats

#Sample prior some how to generate events that we will use to generate data
def generate_theta_data(lat_range,long_range, depth_range, mag_range, nsamp, skip):
    
    #sbvals = sq.i4_sobol_generate(4, 1*nsamp)
    #Change so seed can be set
    dim_num = 4
    sbvals = np.full((nsamp, dim_num), np.nan)
    for j in range(nsamp):
        sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1+skip+j)

    # Calculate min and max value for magnitutde range
    max_mag = 1 - 10**(-mag_range[1])
    min_mag = 1 - 10**(-mag_range[0])

    sbvals[:,0] = sbvals[:,0]*(lat_range[1] - lat_range[0])+lat_range[0]
    sbvals[:,1] = sbvals[:,1]*(long_range[1] - long_range[0])+long_range[0]
    sbvals[:,2] = sbvals[:,2]*(depth_range[1] - depth_range[0])+depth_range[0]
    sbvals[:,3] = sbvals[:,3]*(max_mag - min_mag) + min_mag
    sbvals[:, 3] = -np.log(1 - sbvals[:,3]) / np.log(10)
    
    return sbvals


#Define the event space descritization
#We assume that these are uniformly sampled from the prior so they all have equal aprior likelihood.
#This may be something that we should change in the future to add a prior likleihood associated with each sample
#so that we dont have to consider them to be uniform.

def sample_theta_space(lat_range, long_range, depth_range, mag_range, nsamp, skip):
    # sbvals = sq.i4_sobol_generate(4, 1*nsamp)
    # Change so seed can be set
    dim_num = 4
    sbvals = np.full((nsamp, dim_num), np.nan)
    for j in range(nsamp):
        sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1+skip+j)

    # Calculate min and max value for magnitutde range
    max_mag = 1 - 10**(-mag_range[1])
    min_mag = 1 - 10**(-mag_range[0])

    sbvals[:,0] = sbvals[:,0]*(lat_range[1] - lat_range[0])+lat_range[0]
    sbvals[:,1] = sbvals[:,1]*(long_range[1] - long_range[0])+long_range[0]
    sbvals[:,2] = sbvals[:,2]*(depth_range[1] - depth_range[0])+depth_range[0]
    sbvals[:,3] = sbvals[:,3]*(max_mag - min_mag) + min_mag
    sbvals[:, 3] = -np.log(1 - sbvals[:,3]) / np.log(10)
    
    return sbvals
    # width = .5
    # lat_interval = np.abs(lat_range[1]-lat_range[0])
    # long_interval = np.abs(long_range[1] - long_range[0])
    # depth_interval = np.abs(depth_range[1] - depth_range[0])
    # mag_interval = np.abs(mag_range[1] - mag_range[0])

    # lat_norm = stats.norm(loc=lat_range[0] + lat_interval/2, scale=width)
    # long_norm = stats.norm(loc=long_range[0]+ long_interval/2, scale=width)
    # depth_norm = stats.norm(loc=depth_range[0] + depth_interval/2, scale=width)
    # mag_norm = stats.norm(loc=mag_range[0] + mag_interval/2, scale=width)
    
    # min_lat = lat_norm.cdf(lat_range[0])
    # max_lat = lat_norm.cdf(lat_range[1])
    # min_long = long_norm.cdf(long_range[0])
    # max_long = long_norm.cdf(long_range[1])
    # min_depth = depth_norm.cdf(depth_range[0])
    # max_depth = depth_norm.cdf(depth_range[1])
    # min_mag = mag_norm.cdf(mag_range[0])
    # max_mag = mag_norm.cdf(mag_range[1])
    
    # dim_num = 4
    # sbvals = np.full((nsamp, dim_num), np.nan)
    
    # for j in range(nsamp):
    #     sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1+skip+j)
        
    # sbvals[:,0] = sbvals[:,0]*(max_lat - min_lat) + min_lat 
    # sbvals[:,1] = sbvals[:,1]*(max_long - min_long) + min_long
    # sbvals[:,2] = sbvals[:,2]*(max_depth - min_depth) + min_depth
    # sbvals[:,3] = sbvals[:,3]*(max_mag - min_mag) + min_mag 
    
    # sbvals[:,0] = lat_norm.ppf(sbvals[:,0])
    # sbvals[:,1] = long_norm.ppf(sbvals[:,1])
    # sbvals[:,2] = depth_norm.ppf(sbvals[:,2])
    # sbvals[:,3] = mag_norm.ppf(sbvals[:,3])
    
    # return sbvals

def eval_theta_prior(thetas, lat_range, long_range, depth_range, mag_range):
    if len(thetas.shape) == 1:
        thetas = thetas.reshape((1,-1))

    lat_prob = 1/np.abs(lat_range[1]-lat_range[0])
    long_prob = 1/np.abs(long_range[1] - long_range[0])
    depth_prob = 1/np.abs(depth_range[1] - depth_range[0])
    mag_prob = (np.log(10)/10**thetas[:,3]) / ((1 - 10**(-mag_range[1])) - (1 - 10**(-mag_range[0])))

    return lat_prob*long_prob*depth_prob*mag_prob 



def eval_importance(thetas, lat_range, long_range, depth_range, mag_range):
    if len(thetas.shape) == 1:
        thetas = thetas.reshape((1,-1))
    
    # lat_interval = np.abs(lat_range[1]-lat_range[0])
    # long_interval = np.abs(long_range[1] - long_range[0])
    # depth_interval = np.abs(depth_range[1] - depth_range[0])
    # mag_interval = np.abs(mag_range[1] - mag_range[0])

    # width = .5
    
    # lat_norm = stats.norm(loc=lat_range[0] + lat_interval/2, scale=width)
    # long_norm = stats.norm(loc=long_range[0]+ long_interval/2, scale=width)
    # depth_norm = stats.norm(loc=depth_range[0] + depth_interval/2, scale=width)
    # mag_norm = stats.norm(loc=mag_range[0] + mag_interval/2, scale=width)

    # lat_prob = lat_norm.pdf(thetas[:,0])/(lat_norm.cdf(lat_range[1]) - lat_norm.cdf(lat_range[0]))
    # long_prob = long_norm.pdf(thetas[:,1])/(long_norm.cdf(long_range[1]) - long_norm.cdf(long_range[0]))
    # depth_prob = depth_norm.pdf(thetas[:,2])/(depth_norm.cdf(depth_range[1]) - depth_norm.cdf(depth_range[0]))
    # mag_prob = mag_norm.pdf(thetas[:,3])/(mag_norm.cdf(mag_range[1]) - mag_norm.cdf(mag_range[0]))

    # return lat_prob * long_prob * depth_prob * mag_prob
    if len(thetas.shape) == 1:
        thetas = thetas.reshape((1,-1))

    lat_prob = 1/np.abs(lat_range[1]-lat_range[0])
    long_prob = 1/np.abs(long_range[1] - long_range[0])
    depth_prob = 1/np.abs(depth_range[1] - depth_range[0])
    mag_prob = (np.log(10)/10**thetas[:,3]) / ((1 - 10**(-mag_range[1])) - (1 - 10**(-mag_range[0])))

    return lat_prob*long_prob*depth_prob*mag_prob 


#Generate psuedo random sensor distribution for initial OED
def sample_sensors(lat_range,long_range, nsamp,skip):
    #sbvals = sq.i4_sobol_generate(4, 1*nsamp)
    #Change so seed can be set
    dim_num = 2
    sbvals = np.full((nsamp, dim_num), np.nan)
    for j in range(nsamp):
        sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1+skip+j)    
    
    sbvals[:,0] = sbvals[:,0]*(lat_range[1] - lat_range[0])+lat_range[0]
    sbvals[:,1] = sbvals[:,1]*(long_range[1] - long_range[0])+long_range[0]
    
    return sbvals
