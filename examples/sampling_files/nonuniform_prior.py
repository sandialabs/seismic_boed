import numpy as np
import sobol_seq as sq
from scipy import stats

#Sample prior to get events that we will use to generate data
def generate_theta_data(lat_range,long_range, depth_range, mag_range, nsamp, skip):
    width = .5
    lat_interval = np.abs(lat_range[1]-lat_range[0])
    long_interval = np.abs(long_range[1] - long_range[0])
    depth_interval = np.abs(depth_range[1] - depth_range[0])
    mag_interval = np.abs(mag_range[1] - mag_range[0])

    lat_norm = stats.norm(loc=lat_range[0] + lat_interval/2, scale=width)
    long_norm = stats.norm(loc=long_range[0]+ long_interval/2, scale=width)
    depth_norm = stats.norm(loc=depth_range[0] + depth_interval/2, scale=width)
    mag_norm = stats.norm(loc=mag_range[0] + mag_interval/2, scale=width)
    
    min_lat = lat_norm.cdf(lat_range[0])
    max_lat = lat_norm.cdf(lat_range[1])
    min_long = long_norm.cdf(long_range[0])
    max_long = long_norm.cdf(long_range[1])
    min_depth = depth_norm.cdf(depth_range[0])
    max_depth = depth_norm.cdf(depth_range[1])
    min_mag = mag_norm.cdf(mag_range[0])
    max_mag = mag_norm.cdf(mag_range[1])
    
    dim_num = 4
    sbvals = np.full((nsamp, dim_num), np.nan)
    
    for j in range(nsamp):
        sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1+skip+j)
        
    sbvals[:,0] = sbvals[:,0]*(max_lat - min_lat) + min_lat 
    sbvals[:,1] = sbvals[:,1]*(max_long - min_long) + min_long
    sbvals[:,2] = sbvals[:,2]*(max_depth - min_depth) + min_depth
    sbvals[:,3] = sbvals[:,3]*(max_mag - min_mag) + min_mag 
    
    sbvals[:,0] = lat_norm.ppf(sbvals[:,0])
    sbvals[:,1] = long_norm.ppf(sbvals[:,1])
    sbvals[:,2] = depth_norm.ppf(sbvals[:,2])
    sbvals[:,3] = mag_norm.ppf(sbvals[:,3])
    
    return sbvals


#Define the event space descritization
def sample_theta_space(lat_range, long_range, depth_range, mag_range, nsamp, skip):
    width = .5
    lat_interval = np.abs(lat_range[1]-lat_range[0])
    long_interval = np.abs(long_range[1] - long_range[0])
    depth_interval = np.abs(depth_range[1] - depth_range[0])
    mag_interval = np.abs(mag_range[1] - mag_range[0])

    lat_norm = stats.norm(loc=lat_range[0] + lat_interval/2, scale=width)
    long_norm = stats.norm(loc=long_range[0]+ long_interval/2, scale=width)
    depth_norm = stats.norm(loc=depth_range[0] + depth_interval/2, scale=width)
    mag_norm = stats.norm(loc=mag_range[0] + mag_interval/2, scale=width)
    
    min_lat = lat_norm.cdf(lat_range[0])
    max_lat = lat_norm.cdf(lat_range[1])
    min_long = long_norm.cdf(long_range[0])
    max_long = long_norm.cdf(long_range[1])
    min_depth = depth_norm.cdf(depth_range[0])
    max_depth = depth_norm.cdf(depth_range[1])
    min_mag = mag_norm.cdf(mag_range[0])
    max_mag = mag_norm.cdf(mag_range[1])
    
    dim_num = 4
    sbvals = np.full((nsamp, dim_num), np.nan)
    
    for j in range(nsamp):
        sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1+skip+j)
        
    sbvals[:,0] = sbvals[:,0]*(max_lat - min_lat) + min_lat 
    sbvals[:,1] = sbvals[:,1]*(max_long - min_long) + min_long
    sbvals[:,2] = sbvals[:,2]*(max_depth - min_depth) + min_depth
    sbvals[:,3] = sbvals[:,3]*(max_mag - min_mag) + min_mag 
    
    sbvals[:,0] = lat_norm.ppf(sbvals[:,0])
    sbvals[:,1] = long_norm.ppf(sbvals[:,1])
    sbvals[:,2] = depth_norm.ppf(sbvals[:,2])
    sbvals[:,3] = mag_norm.ppf(sbvals[:,3])
    
    return sbvals

def eval_theta_prior(thetas, lat_range, long_range, depth_range, mag_range):
    def loc_pdf(x, lat_range,long_range):
        # Mixture weights
        unif_pi = .02
        fault_pi = .49
        box_pi = .49

        # Parameters to define box of bivariate normal
        box_lower = np.array([lat_range[0], long_range[0]])
        box_upper = np.array([lat_range[1], long_range[1]])
        box_means = np.array([40.25,-109])
        box_cov = np.array([[.125,0],[0,.125]])

        # Parameters to define normal distribution along fault line
        fault_mean = (long_range[1] - long_range[0])/2 + long_range[0]
        fault_std = .125

        # Define distributions
        box_dist = stats.multivariate_normal(mean=box_means, cov=box_cov, allow_singular=False)
        fault_long = stats.norm(loc=fault_mean, scale=fault_std) # Fault line is normally distributed along longitude
        unif_lat = stats.uniform(loc=lat_range[0],scale=lat_range[1] - lat_range[0])
        unif_long = stats.uniform(loc=long_range[0],scale=long_range[1] - long_range[0])

        # Constants for normalizing truncated distributions
        fault_const = fault_long.cdf(long_range[1]) - fault_long.cdf(long_range[0])
        box_const, _ = stats.mvn.mvnun(box_lower, box_upper, box_means, box_cov)

        # Get probabilities for each mixture component
        unif_lat_prob = unif_lat.pdf(x[:,0])
        unif_long_prob = unif_long.pdf(x[:,1])
        unif_prob = unif_lat_prob * unif_long_prob
        
        fault_long_prob = fault_long.pdf(x[:,1]) / fault_const
        fault_lat_prob = unif_lat.pdf(x[:,0])
        fault_prob = fault_lat_prob * fault_long_prob
        
        box_prob = box_dist.pdf(x[:,:2]) / box_const

        return unif_pi*unif_prob + fault_pi*fault_prob + box_pi*box_prob

    def depth_pdf(x, depth_range):
        dist = stats.uniform(loc=depth_range[0], scale=depth_range[1])
        return dist.pdf(x)

    def mag_pdf(x, mag_range):
        mag_prob = (np.log(10)/10**x)
        mag_const =  ((1 - 10**(-mag_range[1])) - (1 - 10**(-mag_range[0])))

        return mag_prob/mag_const

    if len(thetas.shape) == 1:
            thetas = thetas.reshape((1,-1))

    loc_prob = loc_pdf(thetas[:,:2], lat_range, long_range)
    depth_prob = depth_pdf(thetas[:,2], depth_range)
    mag_prob = mag_pdf(thetas[:,3], mag_range)
    
    return loc_prob * depth_prob * mag_prob



def eval_importance(thetas, lat_range, long_range, depth_range, mag_range):
    if len(thetas.shape) == 1:
        thetas = thetas.reshape((1,-1))
    
    lat_interval = np.abs(lat_range[1]-lat_range[0])
    long_interval = np.abs(long_range[1] - long_range[0])
    depth_interval = np.abs(depth_range[1] - depth_range[0])
    mag_interval = np.abs(mag_range[1] - mag_range[0])

    width = .5
    
    lat_norm = stats.norm(loc=lat_range[0] + lat_interval/2, scale=width)
    long_norm = stats.norm(loc=long_range[0]+ long_interval/2, scale=width)
    depth_norm = stats.norm(loc=depth_range[0] + depth_interval/2, scale=width)
    mag_norm = stats.norm(loc=mag_range[0] + mag_interval/2, scale=width)

    lat_prob = lat_norm.pdf(thetas[:,0])/(lat_norm.cdf(lat_range[1]) - lat_norm.cdf(lat_range[0]))
    long_prob = long_norm.pdf(thetas[:,1])/(long_norm.cdf(long_range[1]) - long_norm.cdf(long_range[0]))
    depth_prob = depth_norm.pdf(thetas[:,2])/(depth_norm.cdf(depth_range[1]) - depth_norm.cdf(depth_range[0]))
    mag_prob = mag_norm.pdf(thetas[:,3])/(mag_norm.cdf(mag_range[1]) - mag_norm.cdf(mag_range[0]))

    return lat_prob * long_prob * depth_prob * mag_prob


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
