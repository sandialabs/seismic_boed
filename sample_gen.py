import numpy as np
import sobol_seq as sq

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

def sample_theta_space(lat_range,long_range, depth_range, nsamp, skip):
    #sbvals = sq.i4_sobol_generate(4, 1*nsamp)
    #Change so seed can be set
    dim_num = 4
    sbvals = np.full((nsamp, dim_num), np.nan)
    for j in range(nsamp):
        sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1+skip+j)    

    sbvals[:,0] = sbvals[:,0]*(lat_range[1] - lat_range[0])+lat_range[0]
    sbvals[:,1] = sbvals[:,1]*(long_range[1] - long_range[0])+long_range[0]
    sbvals[:,2] = sbvals[:,2]*(depth_range[1] - depth_range[0])+depth_range[0]
    sbvals[:, 3] = -np.log(1 - sbvals[:,3]) / np.log(10)
    sbvals[:, 3] += 0.5
    
    return sbvals

def eval_theta_prior(thetas):
    # compute log prior likelihood
    # Compute p(location)

    loc_probs = np.zeros(len(thetas))

    # Events within fault line zone
    mask = (thetas[:,1] >= -111.) & (thetas[:,1] <= -110.)
    
    num_in_fault = sum(mask)
    
    fault_prob = stats.norm(loc=-110.5,scale=1)
    fault_density = fault_prob.cdf(-110) - fault_prob.cdf(-111)
    outside_prob = (1-fault_density)/(len(thetas)-num_in_fault)

    loc_probs[:] = outside_prob
    loc_probs[mask] = fault_prob.pdf(thetas[mask][:,1])
    
    # Compute p(mag)
    mag_probs = np.log(10)/(10**(thetas[:,3]))
    
    # p(location,mag)
    return loc_probs*mag_probs

def eval_importance(thetas, nlpts_space):
    return (1/nlpts_space) * (1-(1/(10*np.exp(thetas[:,3]))))

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