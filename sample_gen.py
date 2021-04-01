import numpy as np
import sobol_seq as sq

#Sample prior some how to generate events that we will use to generate data
def sample_prior(lat_range,long_range, depth_range, nsamp):
    
    sbvals = sq.i4_sobol_generate(4, 1*nsamp)
    sbvals[:,0] = sbvals[:,0]*(lat_range[1] - lat_range[0])+lat_range[0]
    sbvals[:,1] = sbvals[:,1]*(long_range[1] - long_range[0])+long_range[0]
    sbvals[:,2] = sbvals[:,2]*(depth_range[1] - depth_range[0])+depth_range[0]
    sbvals[:, 3] = -np.log(1 - sbvals[:,3]) / np.log(10)
    sbvals[:, 3] += 0.5
    
    return sbvals

#Define the event space descritization
def descritize_space(lat_range,long_range, depth_range, nsamp):
    sbvals = sq.i4_sobol_generate(4, 1*nsamp)
    sbvals[:,0] = sbvals[:,0]*(lat_range[1] - lat_range[0])+lat_range[0]
    sbvals[:,1] = sbvals[:,1]*(long_range[1] - long_range[0])+long_range[0]
    sbvals[:,2] = sbvals[:,2]*(depth_range[1] - depth_range[0])+depth_range[0]
    sbvals[:, 3] = -np.log(1 - sbvals[:,3]) / np.log(10)
    sbvals[:, 3] += 0.5
    
    return sbvals