import numpy as np
import sobol_seq as sq
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib.path as mpltPath

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
        
    if len(thetas.shape) == 1:
        thetas = thetas.reshape((1,-1))

    lat_prob = 1/np.abs(lat_range[1]-lat_range[0])
    long_prob = 1/np.abs(long_range[1] - long_range[0])
    depth_prob = 1/np.abs(depth_range[1] - depth_range[0])
    mag_prob = (np.log(10)/10**thetas[:,3]) / ((1 - 10**(-mag_range[1])) - (1 - 10**(-mag_range[0])))

    area = calc_area(bounds)
    
    return lat_prob*long_prob*depth_prob*mag_prob/area 



def eval_importance(thetas, bounds, depth_range, mag_range):
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

    area = calc_area(bounds)
    
    return lat_prob*long_prob*depth_prob*mag_prob/area 
