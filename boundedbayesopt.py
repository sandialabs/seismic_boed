import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import skopt 
from skopt import Optimizer, expected_minimum, dump
from skopt.learning.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy import stats
from scipy.optimize import minimize, OptimizeResult
import matplotlib.path as mpltPath
import copy
import inspect
from sklearn.neighbors import BallTree
from utils import read_bounds
# from GE_Beamforming_Utils import convert_coords_to_utm
import os

class BoundedBayesOpt():
    def __init__(self, kernel, bounds_file, init_X=None, init_Y=None, target_fun=None, noise=.1, acq_func=None, max_model_queue_size=30):
        self.specs = {"args": copy.copy(inspect.currentframe().f_locals),
              "function": "Optimizer"}
        # Initialize arrays for storing samples from target function
        self.X_sample = init_X
        self.Y_sample = init_Y
        
        # Initialize surrogate function
        self.gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise**2)
        
        # Use bounds file to get boundaries for sampling valid points
        # Get file extension of bounds file
        bounds_name, bounds_ext = os.path.splitext(bounds_file)
        # If JSON file provided, initialize boundaries
        if bounds_ext == '.json':
            self.__handle_bounds(bounds_file)
        # If KML file provided, intialize coordinates
        elif bounds_ext == '.kml':
            self.__get_kml_coords(bounds_file)
        # Only two filetypes suprported
        else:
            raise ValueError(f"bounds filename must have extension '.json' or '.kml', not '{bounds_ext}")

        
        # Initialize target function if provided
        self.target_fun = target_fun
        
        self.models = []
        self.max_model_queue_size = max_model_queue_size
            
    def __handle_bounds(self, bounds_file):
        bounds = read_bounds(bounds_file, sensor_bounds=True)

        if not isinstance(bounds, np.ndarray):
            bounds = np.array(bounds)
            
        if len(bounds.shape) == 2:
            bounds = bounds.reshape((1, *bounds.shape))
        elif (len(bounds.shape) >3 or len(bounds.shape) < 2):
            raise ValueError('Must pass array of arrays of 2d boundary coordinates')
        self.bounds = []

        for i in range(len(bounds)):
            self.bounds.append(bounds[i][:,[1,0]].copy())

        min_x = self.bounds[0][:,0].min()
        min_y = self.bounds[0][:,1].min()
        max_x = self.bounds[0][:,0].max()
        max_y = self.bounds[0][:,1].max()

        for i in range(1,len(bounds)):
            curr_minx = self.bounds[i][:,0].min()
            curr_miny = self.bounds[i][:,1].min()
            curr_maxx = self.bounds[i][:,0].max()
            curr_maxy = self.bounds[i][:,1].max()

            if curr_minx < min_x:
                min_x = curr_minx
            if curr_miny < min_y:
                min_y = curr_miny
            if curr_maxx > max_x:
                max_x = curr_maxx
            if curr_maxy > max_y:
                max_y = curr_maxy

        self.sample_bounds = np.array([[min_x, max_x], 
                                  [min_y, max_y]])
        
        self.json = True
        
    def __get_kml_coords(self, filename):
        """
        Pulls lat/lon coordinates of roads from KML file
        and initializes a KD tree that can be used for finding
        the shortest distance between a given point and a road.

        Inputs
        ------
        filename (str) : path to .kml file containing coordinates

        Returns
        -------
        None, stores the following object attributes:
        
        kml (np.ndarray) : Array of coordinates describing roads, with latitude 
                           in the first column and longitude in the second
        max_kml_dist (float) : maximum distance a point may be from a road to
                               be considered a valid position for sensor
                               placement (currently hardcoded)
        """
        # Import inside function to avoid requiring PyKML for those who won't use it
        from pykml import parser
        output_mat = []

        with open(filename) as f:
            contents = parser.parse(f)

        sup_fold_string  = contents.getroot().Document.Folder

        for e in sup_fold_string.Folder:
            for f in e.S_Roads_IISSSSSSSSD:

                if hasattr(f, "MultiGeometry") is False:

                    coord_string = f.LineString.coordinates

                    tmp_string = str(coord_string)
                    coord_string = tmp_string.split(" ")

                    for idx, mystr in enumerate(coord_string):
                        coord_string[idx] = mystr.strip()

                    # Last one appears to be junk
                    coord_string = coord_string[:-1]

                    output = np.zeros((len(coord_string), 3))

                    for idx, mystr in enumerate(coord_string):
                        output[idx, 0] = float(mystr.split(",")[0])
                        output[idx, 1] = float(mystr.split(",")[1])
                        output[idx, 2] = float(mystr.split(",")[2])

                    output_mat.append(output)

                if hasattr(f, "MultiGeometry") is True:

                    for g in f.MultiGeometry:

                        coord_string = g.LineString.coordinates

                        tmp_string = str(coord_string)
                        coord_string = tmp_string.split(" ")

                        for idx, mystr in enumerate(coord_string):
                            coord_string[idx] = mystr.strip()

                        # Last one appears to be junk
                        coord_string = coord_string[:-1]

                        output = np.zeros((len(coord_string), 3))

                        for idx, mystr in enumerate(coord_string):
                            output[idx, 0] = float(mystr.split(",")[0])
                            output[idx, 1] = float(mystr.split(",")[1])
                            output[idx, 2] = float(mystr.split(",")[2])

                        output_mat.append(output)

        latlon_out = np.asarray(output_mat, dtype='object')

        x_utm = []
        y_utm = []
        z_utm = []

        lats = []
        lons = []


        for latlon_coords in latlon_out:
            
            # temp = convert_coords_to_utm(latlon_coords, demean=False)
            
            # x_utm += temp[0].flatten().tolist()
            # y_utm += temp[1].flatten().tolist()
            
            lons += latlon_coords[:,0].tolist()
            lats += latlon_coords[:,1].tolist()
            
        z_utm = np.ones(len(x_utm)).tolist()

        self.kml_latloncoords = np.vstack((lats,lons)).T
        # self.kml_utmcoords = np.vstack((x_utm, y_utm, z_utm)).T

        self.kml_tree = BallTree(np.deg2rad(self.kml_latloncoords), metric='haversine')
        self.json = False
        self.max_kml_dist = .5
        
        min_x = np.min(lats)
        max_x = np.max(lats)
        min_y = np.min(lons)
        max_y = np.max(lons)

        self.sample_bounds = np.array([[min_x, max_x], 
                                       [min_y, max_y]])       

    def check_valid(self, points):
        if self.json:
            masks = []
            for polygon in self.bounds:
                valid_region = mpltPath.Path(polygon)
                valid_pts_idx = valid_region.contains_points(points)
                masks.append(valid_pts_idx)
            point_is_valid = np.any(masks, axis=0)

        else:
            # Get unit distances from points to nearest road
            dists = self.kml_tree.query(np.deg2rad(points))[0]
            # Convert distances to kilometers by multiplying by radius of earth
            dists = dists * 6371
            # Check where distances are smaller than max allowed
            point_is_valid = np.where(dists < self.max_kml_dist)[0]

        return point_is_valid
        

    def expected_improvement(self, X, xi=0.01):
        '''
        Computes the EI at points X based on existing samples X_sample
        and Y_sample using a Gaussian process surrogate model.

        Args:
            X: Points at which EI shall be computed (m x d).
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.
            xi: Exploitation-exploration trade-off parameter.

        Returns:
            Expected improvements at points X.
        '''
        # Elminate points outside the valid boundaries
        valid_X_idx = self.check_valid(X)
        valid_X = X[valid_X_idx]

        # If no points were valid, just return no EI
        if valid_X.shape[0] == 0:
            return np.array([0] * X.shape[0])

        # Get predictions out of surrogate function
        mu, sigma = self.gpr.predict(valid_X, return_std=True)
        mu_sample = self.gpr.predict(self.X_sample)

        # Shape matching
        sigma = sigma.reshape(-1, 1)

        # Get current optimum
        mu_sample_opt = np.max(mu_sample)

        # Calculate EI
        with np.errstate(divide='warn'):
            imp = (mu - mu_sample_opt - xi).reshape((-1,1))
            Z = imp / sigma
            ei = imp * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
            # Set EI where sigma was 0 to 0
            sigma_is_0 = sigma==0.
            ei[sigma_is_0.ravel()] = 0.0

        return ei.ravel()
        
    def ask(self, n_restarts=25):
        '''
        Proposes the next sampling point by optimizing the acquisition function.
    
        Args:
            acquisition: Acquisition function.
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.

        Returns:
            Location of the acquisition function maximum.
        '''
        # Check current sample size
        dim = self.X_sample.shape[1]
        # Initialize minimization vals
        min_val = 1
        min_x = None

        def min_obj(X):
            # Minimization objective is the negative acquisition function
            return -self.expected_improvement(X.reshape(-1, dim))

        # Sample n_restarts initial points for minimizer
        test_pts = np.random.uniform(self.sample_bounds[:,0], self.sample_bounds[:,1], size=(n_restarts, dim))

        # Only accept points inside bounds
        valid_test_pts_idx = self.check_valid(test_pts)
        valid_test_pts = test_pts[valid_test_pts_idx]
        # Resample until we have n_restarts points
        while valid_test_pts.shape[0] < n_restarts:
            addon_pts = np.random.uniform(self.sample_bounds[:,0], 
                                          self.sample_bounds[:,1], 
                                          size=(n_restarts - valid_test_pts.shape[0], dim))
            valid_addon_pts_idx = self.check_valid(addon_pts)
            valid_test_pts = np.vstack((valid_test_pts, addon_pts[valid_addon_pts_idx]))
            
        # Find the best optimum by starting from n_restart different random points.
        for x0 in valid_test_pts:

            res = minimize(min_obj, x0=x0, bounds=self.sample_bounds, method='L-BFGS-B')        
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x   
        return min_x.reshape(-1, 2)
    
    def tell(self, X_next, Y_next):
        if not isinstance(Y_next, (list, np.ndarray)):
            Y_next = np.array(Y_next)
        
        # Add sample to previous samples
        if self.X_sample is None:
            self.X_sample = X_next
        else:
            self.X_sample = np.vstack((self.X_sample, X_next))
            
        if self.Y_sample is None:
            self.Y_sample = Y_next
        else:
            try:
                self.Y_sample = np.vstack((self.Y_sample, Y_next))
            except:
                print(self.Y_sample)
                print(Y_next)
                raise ValueError('found error')

        
        # Update surrogate function
        self.gpr.fit(self.X_sample, self.Y_sample)
        if self.max_model_queue_size is None:
            self.models.append(self.gpr)
        elif len(self.models) < self.max_model_queue_size:
            self.models.append(self.gpr)
        else:
            # Maximum list size obtained, remove oldest model.
            self.models.pop(0)
            self.models.append(self.gpr)
    
    def run_opt(self, target_fun=None, n_iter=40, n_restarts=25):
        if target_fun is not None:
            self.target_fun = target_fun
        if self.target_fun is None:
            raise ValueError('Must provide a target function either on initialization or in the args to this function')
        for i in range(n_iter):
            X_next = self.ask(n_restarts)
            Y_next = self.target_fun(X_next)
            self.tell(X_next, Y_next)
        
        return (X_next, Y_next)
    
    def create_result(self, Xi, yi, space=None, rng=None, specs=None, models=None):
        """
        Initialize an `OptimizeResult` object.

        Parameters
        ----------
        Xi : list of lists, shape (n_iters, n_features)
            Location of the minimum at every iteration.

        yi : array-like, shape (n_iters,)
            Minimum value obtained at every iteration.

        space : Space instance, optional
            Search space.

        rng : RandomState instance, optional
            State of the random state.

        specs : dict, optional
            Call specifications.

        models : list, optional
            List of fit surrogate models.

        Returns
        -------
        res : `OptimizeResult`, scipy object
            OptimizeResult instance with the required information.
        """
        res = OptimizeResult()
        yi = np.asarray(yi)
        if np.ndim(yi) == 2:
            yi = np.ravel(yi[:, 0])
        best = np.argmin(yi)
        res.x = Xi[best]
        res.fun = yi[best]
        res.func_vals = yi
        res.x_iters = Xi
        res.models = models
        res.space = space
        res.random_state = rng
        res.specs = specs
        return res

    def get_result(self):
            """Returns the same result that would be returned by opt.tell()
            but without calling tell

            Returns
            -------
            res : `OptimizeResult`, scipy object
                OptimizeResult instance with the required information.

            """
            result = self.create_result(self.X_sample, self.Y_sample, models=self.models)
            result.specs = self.specs
            return result
        
    def expected_minimum(self, random_state=None):
        """Compute the minimum over the predictions of the last surrogate model.
        Uses `expected_minimum_random_sampling` with `n_random_starts` = 100000,
        when the space contains any categorical values.

        .. note::
            The returned minimum may not necessarily be an accurate
            prediction of the minimum of the true objective function.

        Parameters
        ----------
        res : `OptimizeResult`, scipy object
            The optimization result returned by a `skopt` minimizer.

        n_random_starts : int, default=20
            The number of random starts for the minimization of the surrogate
            model.

        random_state : int, RandomState instance, or None (default)
            Set random state to something other than None for reproducible
            results.

        Returns
        -------
        x : list
            location of the minimum.
        fun : float
            the surrogate function value at the minimum.
        """
        res = self.get_result()
        return res.x, res.fun
