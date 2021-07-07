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

# def sample_theta_space(lat_range,long_range, depth_range, mag_range, nsamp, skip):
#     #sbvals = sq.i4_sobol_generate(4, 1*nsamp)
#     #Change so seed can be set
#     dim_num = 4
#     sbvals = np.full((nsamp, dim_num), np.nan)
#     for j in range(nsamp):
#         sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1+skip+j)    

#     sbvals[:,0] = sbvals[:,0]*(lat_range[1] - lat_range[0])+lat_range[0]
#     sbvals[:,1] = sbvals[:,1]*(long_range[1] - long_range[0])+long_range[0]
#     sbvals[:,2] = sbvals[:,2]*(depth_range[1] - depth_range[0])+depth_range[0]
#     sbvals[:, 3] = -np.log(1 - sbvals[:,3]) / np.log(10)
#     sbvals[:, 3] += 0.5
    
#     return sbvals
def sample_theta_space(lat_range,long_range, depth_range, mag_range, nsamp, skip):
    print(f'lat range:{lat_range}')
    print(f'long_range: {long_range}')
    print(f'depth_range: {depth_range}')
    print(f'mag_range: {mag_range}')
    
    lat_interval = np.abs(lat_range[1]-lat_range[0])
    long_interval = np.abs(long_range[1] - long_range[0])
    depth_interval = np.abs(depth_range[1] - depth_range[0])
    mag_interval = np.abs(mag_range[1] - mag_range[0])

    lat_norm = stats.norm(loc=lat_range[0] + lat_interval/2, scale=lat_interval/3)
    long_norm = stats.norm(loc=long_range[0]+ long_interval/2, scale=long_interval/3)
    depth_norm = stats.norm(loc=depth_range[0] + depth_interval/2, scale=depth_interval/3)
    mag_norm = stats.norm(loc=mag_range[0] + mag_interval/2, scale=mag_interval/3)

    dim_num = 4
    sbvals = np.full((nsamp, dim_num), np.nan)
    
    sbvals[:,0] = np.clip(lat_norm.rvs(nsamp),*lat_range)
    sbvals[:,1] = np.clip(long_norm.rvs(nsamp), *long_range)
    sbvals[:,2] = np.clip(depth_norm.rvs(nsamp), *depth_range)
    sbvals[:,3] = np.clip(mag_norm.rvs(nsamp), *mag_range)

    print(sbvals[:,3].min(), sbvals[:,3].max())

    return sbvals
    

def eval_theta_prior(thetas, lat_range, long_range, depth_range, mag_range):
    if len(thetas.shape) == 1:
        thetas = thetas.reshape((1,-1))

    lat_prob = 1/np.abs(lat_range[1]-lat_range[0])
    long_prob = 1/np.abs(long_range[1] - long_range[0])
    depth_prob = 1/np.abs(depth_range[1] - depth_range[0])
    mag_prob = (np.log(10)/10**thetas[:,3])

    return lat_prob*long_prob*depth_prob*mag_prob


def eval_importance(thetas, lat_range, long_range, depth_range, mag_range):
    if len(thetas.shape) == 1:
        thetas = thetas.reshape((1,-1))
    
    lat_interval = np.abs(lat_range[1]-lat_range[0])
    long_interval = np.abs(long_range[1] - long_range[0])
    depth_interval = np.abs(depth_range[1] - depth_range[0])
    mag_interval = np.abs(mag_range[1] - mag_range[0])

    lat_norm = stats.norm(loc=lat_range[0] + lat_interval/2, scale=lat_interval/3)
    long_norm = stats.norm(loc=long_range[0]+ long_interval/2, scale=long_interval/3)
    depth_norm = stats.norm(loc=depth_range[0] + depth_interval/2, scale=depth_interval/3)
    mag_norm = stats.norm(loc=mag_range[0] + mag_interval/2, scale=mag_interval/3)

    return lat_norm.pdf(thetas[:,0])*long_norm.pdf(thetas[:,1])*depth_norm.pdf(thetas[:,2])*mag_norm.pdf(thetas[:,3])

    
# def eval_theta_prior(thetas, lat_range, long_range, depth_range):
#     # compute log prior likelihood
#     if len(thetas.shape) == 1:
#         thetas = thetas.reshape((1,-1))
#     # Compute p(lat)
#     # lat_probs = 1/(lat_range[1] - lat_range[0])
#     fault_min_lat = 41.0
#     fault_max_lat = 41.25
#     lat_probs = np.zeros(len(thetas))
    
#     lat_mask = (thetas[:,0] >= fault_min_lat) & (thetas[:,0] <= fault_max_lat)
    
#     fault_prob_lat = stats.norm(loc=41.125,scale=.2)
#     outside_prob_lat = 1/(np.abs(fault_min_lat - lat_range[0]) + np.abs(lat_range[1]-fault_max_lat))
    
#     lat_probs[:] = outside_prob_lat
#     lat_probs[lat_mask] = fault_prob_lat.pdf(thetas[lat_mask][:,0])
    
    
#     # Compute p(long)
#     fault_min_long = -110.25
#     fault_max_long = -110.
#     long_probs = np.zeros(len(thetas))
    
#     long_mask = (thetas[:,1] >= fault_min_long) & (thetas[:,1] <= fault_max_long)
    
#     fault_prob = stats.norm(loc=-110.125,scale=.2)
#     outside_prob = 1/(np.abs(fault_min_long - long_range[0]) + np.abs(long_range[1]-fault_max_long))
    
#     long_probs[:] = outside_prob
#     long_probs[long_mask] = fault_prob.pdf(thetas[long_mask][:,1])
    
#     # Compute p(depth)
#     depth_prob = 1/(depth_range[1]-depth_range[0])
    
#     # Compute p(mag)
#     mag_probs = np.log(10)/(10**(thetas[:,3]))
    
#     # p(lat,long,depth,mag)
#     return long_probs*mag_probs*lat_probs*depth_prob


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