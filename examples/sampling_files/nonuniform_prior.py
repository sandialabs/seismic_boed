import numpy as np
import sobol_seq as sq
from scipy import stats


def check_valid(bounds, points):
    masks = []
    if not isinstance(bounds, np.ndarray):
        bounds = np.array(bounds)
    if len(bounds.shape) == 2:
        bounds = bounds.reshape((1,*bounds.shape))

    for polygon in bounds:
        valid_region = mpltPath.Path(polygon)
        valid_pts_idx = valid_region.contains_points(points)
        masks.append(valid_pts_idx)
    point_is_valid = np.any(masks, axis=0)

    return point_is_valid


def compute_sample_bounds(input_bounds):
    if not isinstance(input_bounds, np.ndarray):
        input_bounds = np.array(input_bounds)
    if len(input_bounds.shape) == 2:
        input_bounds = input_bounds.reshape((1,*input_bounds.shape))

    bounds = []
    
    for i in range(len(input_bounds)):
        bounds.append(input_bounds[i].copy())

    min_x = bounds[0][:,0].min()
    min_y = bounds[0][:,1].min()
    max_x = bounds[0][:,0].max()
    max_y = bounds[0][:,1].max()

    for i in range(1,len(bounds)):
        curr_minx = bounds[i][:,0].min()
        curr_miny = bounds[i][:,1].min()
        curr_maxx = bounds[i][:,0].max()
        curr_maxy = bounds[i][:,1].max()

        if curr_minx < min_x:
            min_x = curr_minx
        if curr_miny < min_y:
            min_y = curr_miny
        if curr_maxx > max_x:
            max_x = curr_maxx
        if curr_maxy > max_y:
            max_y = curr_maxy

    sample_bounds = np.array([[min_x, max_x], 
                              [min_y, max_y]])
    
    return sample_bounds

def calc_area(bounds, nsamp=1000, skip=0):
    sample_bounds = compute_sample_bounds(bounds)
    long_range = sample_bounds[0]
    lat_range = sample_bounds[1]
    
    dim_num = 2
    
    count = 0
    sbvals = np.full((nsamp, dim_num), np.nan)
    for j in range(nsamp):
        sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1+skip+j)
    
    count += (j+1)

    sbvals[:,0] = sbvals[:,0]*(long_range[1] - long_range[0])+long_range[0]
    sbvals[:,1] = sbvals[:,1]*(lat_range[1] - lat_range[0])+lat_range[0]
    
    valid_idx = check_valid(bounds, sbvals)
    return sum(valid_idx)/nsamp
    
    
    
#Sample prior some how to generate events that we will use to generate data
def generate_theta_data(bounds, depth_range, mag_range, nsamp, skip):
    """ 
    Rejection sample uniformly for location, exponentially for mag, uniformly for depth
    """
    sample_bounds = compute_sample_bounds(bounds)
    long_range = sample_bounds[0]
    lat_range = sample_bounds[1]
#    print(f'LONG RANGE: {long_range}')
#    print(f'LAT RANGE: {lat_range}')
    
    #sbvals = sq.i4_sobol_generate(4, 1*nsamp)
    #Change so seed can be set
    dim_num = 4
    
    count = 0
    sbvals = np.full((nsamp, dim_num), np.nan)
    for j in range(nsamp):
        sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1+skip+j)
    
    count += (j+1)

    # Calculate min and max value for magnitutde range
    max_mag = 1 - 10**(-mag_range[1])
    min_mag = 1 - 10**(-mag_range[0])
    
    # Modify depth bound for shallow events ~5km(1km?)

    sbvals[:,0] = sbvals[:,0]*(long_range[1] - long_range[0])+long_range[0]
    sbvals[:,1] = sbvals[:,1]*(lat_range[1] - lat_range[0])+lat_range[0]
    sbvals[:,2] = sbvals[:,2]*(depth_range[1] - depth_range[0])+depth_range[0]
    sbvals[:,3] = sbvals[:,3]*(max_mag - min_mag) + min_mag
    sbvals[:, 3] = -np.log(1 - sbvals[:,3]) / np.log(10)
    
#    print(sbvals)
    # Only accept points inside bounds
    valid_test_pts_idx = check_valid(bounds, sbvals[:,:2])
    valid_test_pts = sbvals[valid_test_pts_idx]
    # return sbvals    

    while valid_test_pts.shape[0] < nsamp:
        # print(valid_test_pts.shape[0])
        for j in range(nsamp - valid_test_pts.shape[0]):
            addon_pts[j, :], _ = sq.i4_sobol(dim_num, seed=count+1+skip+j)
        
        count += (j+1)

        # Calculate min and max value for magnitutde range
        max_mag = 1 - 10**(-mag_range[1])
        min_mag = 1 - 10**(-mag_range[0])

        addon_pts[:,0] = addon_pts[:,0]*(long_range[1] - long_range[0])+long_range[0]
        addon_pts[:,1] = addon_pts[:,1]*(lat_range[1] - lat_range[0])+lat_range[0]
        addon_pts[:,2] = addon_pts[:,2]*(depth_range[1] - depth_range[0])+depth_range[0]
        addon_pts[:,3] = addon_pts[:,3]*(max_mag - min_mag) + min_mag
        addon_pts[:, 3] = -np.log(1 - addon_pts[:,3]) / np.log(10)

        valid_addon_pts_idx = check_valid(bounds, addon_pts[:,:2])
        valid_test_pts = np.vstack((valid_test_pts, addon_pts[valid_addon_pts_idx]))

    return valid_test_pts[:,[1,0,2,3]].copy()


#Define the event space descritization
def sample_theta_space(bounds, depth_range, mag_range, nsamp, skip):
    """ 
    Rejection sample uniformly for location, exponentially for mag, uniformly for depth
    """
    sample_bounds = compute_sample_bounds(bounds)
    long_range = sample_bounds[0]
    lat_range = sample_bounds[1]
    # print(f'LONG RANGE: {long_range}')
    # print(f'LAT RANGE: {lat_range}')
    
    #sbvals = sq.i4_sobol_generate(4, 1*nsamp)
    #Change so seed can be set
    dim_num = 4
    
    count = 0
    sbvals = np.full((nsamp, dim_num), np.nan)
    for j in range(nsamp):
        sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1+skip+j)
    
    count += (j+1)

    # Calculate min and max value for magnitutde range
    max_mag = 1 - 10**(-mag_range[1])
    min_mag = 1 - 10**(-mag_range[0])
    
    # Modify depth bound for shallow events ~5km(1km?)

    sbvals[:,0] = sbvals[:,0]*(long_range[1] - long_range[0])+long_range[0]
    sbvals[:,1] = sbvals[:,1]*(lat_range[1] - lat_range[0])+lat_range[0]
    sbvals[:,2] = sbvals[:,2]*(depth_range[1] - depth_range[0])+depth_range[0]
    sbvals[:,3] = sbvals[:,3]*(max_mag - min_mag) + min_mag
    sbvals[:, 3] = -np.log(1 - sbvals[:,3]) / np.log(10)
    
    # print(sbvals)
    # Only accept points inside bounds
    valid_test_pts_idx = check_valid(bounds, sbvals[:,:2])
    valid_test_pts = sbvals[valid_test_pts_idx]
    
    while valid_test_pts.shape[0] < nsamp:
        curr_len = nsamp - valid_test_pts.shape[0]
        addon_pts = np.full((curr_len, dim_num), np.nan)
        for j in range(curr_len):
            addon_pts[j, :], _ = sq.i4_sobol(dim_num, seed=count+1+skip+j)
        
        count += (j+1)

        # Calculate min and max value for magnitutde range
        max_mag = 1 - 10**(-mag_range[1])
        min_mag = 1 - 10**(-mag_range[0])

        addon_pts[:,0] = addon_pts[:,0]*(long_range[1] - long_range[0])+long_range[0]
        addon_pts[:,1] = addon_pts[:,1]*(lat_range[1] - lat_range[0])+lat_range[0]
        addon_pts[:,2] = addon_pts[:,2]*(depth_range[1] - depth_range[0])+depth_range[0]
        addon_pts[:,3] = addon_pts[:,3]*(max_mag - min_mag) + min_mag
        addon_pts[:, 3] = -np.log(1 - addon_pts[:,3]) / np.log(10)

        valid_addon_pts_idx = check_valid(bounds, addon_pts[:,:2])
        valid_test_pts = np.vstack((valid_test_pts, addon_pts[valid_addon_pts_idx]))
        
    return valid_test_pts[:,[1,0,2,3]].copy()
 
 
def eval_theta_prior(thetas, bounds, depth_range, mag_range):
    sample_bounds = compute_sample_bounds(bounds)
    long_range = sample_bounds[0]
    lat_range = sample_bounds[1]

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
    # lat_prob = 1/np.abs(lat_range[1]-lat_range[0])
    # long_prob = 1/np.abs(long_range[1] - long_range[0])
    # depth_prob = 1/np.abs(depth_range[1] - depth_range[0])
    # mag_prob = (np.log(10)/10**thetas[:,3]) / ((1 - 10**(-mag_range[1])) - (1 - 10**(-mag_range[0])))

    # return lat_prob*long_prob*depth_prob*mag_prob 



def eval_importance(thetas, bounds, depth_range, mag_range):
    if len(thetas.shape) == 1:
        thetas = thetas.reshape((1,-1))

    sample_bounds = compute_sample_bounds(bounds)
    lat_range = sample_bounds[0]
    long_range = sample_bounds[1]
    
    if len(thetas.shape) == 1:
        thetas = thetas.reshape((1,-1))
    
    if len(thetas.shape) == 1:
        thetas = thetas.reshape((1,-1))

    lat_prob = 1/np.abs(lat_range[1]-lat_range[0])
    long_prob = 1/np.abs(long_range[1] - long_range[0])
    depth_prob = 1/np.abs(depth_range[1] - depth_range[0])
    mag_prob = (np.log(10)/10**thetas[:,3]) / ((1 - 10**(-mag_range[1])) - (1 - 10**(-mag_range[0])))

    # area = calc_area(bounds)
    area = 1 # Only sampling a square domain for now
    
    return lat_prob*long_prob*depth_prob*mag_prob/area 


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
