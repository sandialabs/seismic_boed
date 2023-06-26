#!/usr/bin/env python
# coding: utf-8
import numpy as np
import time
import sys
import os

from subprocess import Popen, PIPE
import shlex

from utils import read_opt_file, write_input_file
from sample_gen import sample_sensors

from skopt import Optimizer, expected_minimum, dump
from skopt.learning.gaussian_process.kernels import RBF, WhiteKernel
from skopt.learning import GaussianProcessRegressor
from boundedbayesopt import BoundedBayesOpt as BBO

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    t0 = time.time()
    if len(sys.argv) == 5:
        #Load optimization parameters + parameters that will be needed to bulid the input file to the eig code
        #Needed for input files: nlpts, ndata, lat_range, long_range, depth_range, sensors
        #Assume we are doing sequential greedy sensor placement
        #Opt parameters: sensor_lat_range, sensor_long_range, #random initial trial points, #total number of trials
        #                sensor type and accuracy, optimization criteria (e.g. UCB, EI)
        #Also need info for how to run mpi and how many sensors to place

        nopt_random, nopt_total, sensor_lat_range, sensor_long_range, bounds_file, sensor_params, opt_type, nlpts_data, nlpts_space, ndata, lat_range, long_range, depth_range, mag_range, sensors, mpirunstring, sampling_file, nsensor_place = read_opt_file(sys.argv[1])

        save_file = sys.argv[2]
        save_path = sys.argv[3]
        verbose = int(sys.argv[4])
        location_bounds = np.load(bounds_file, allow_pickle=True)
        os.makedirs(save_path, exist_ok=True)

        if verbose == 1:
            t1 = time.time()-t0
            print("Configuring Optimizer: Nsensor " + str(nsensor_place) + " Ntrials "+ str(nopt_total)+ " " +str(t1), flush=True)
    else:
        #python3 network_opt.py inputs_opt.dat sensor_opt.npz 1
        #verbose options: 0 (only output is the final sensor network), 1 full output
        sys.exit('Usage: python3 network_opt.py loc_file save_file save_path verbose')
    
    
    for isamp in range(0,nsensor_place):
        if verbose == 1:
            t1 = time.time() - t0
            print("Starting optimization of sensor "+ str(sensors.shape[0]+1) + " " + str(t1), flush=True)
        

        #Initialize the optimizer not it minimizes
        #0 -> EI    
        if opt_type == 0:
            # Specify appropriate length scale
            kernel = 1.0 * RBF(length_scale=[1.0, 1.0,], length_scale_bounds=(0.2, 1)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-2, 5e-1))
            gp = GaussianProcessRegressor(kernel=kernel,alpha=0.0, normalize_y=True)

            opt = BBO(kernel, location_bounds)

        sensor_lat_range = opt.sample_bounds[0]
        sensor_long_range = opt.sample_bounds[1]

        #Randomly selection the random trial points (right now this is going to be the same every isamp stage)
        #becuase we are intializing the psuedo random seed the same each time.
        sensor_loc_random = sample_sensors(sensor_lat_range,sensor_long_range, nopt_random,nlpts_data+nlpts_space)
        
        # Make sure trial points are within bounds
        valid_loc_idx = opt.check_valid(sensor_loc_random)
        valid_trial_pts = sensor_loc_random[valid_loc_idx]
        it=0
        while len(valid_trial_pts) < nopt_random:
            addon_pts = sample_sensors(sensor_lat_range, sensor_long_range, nopt_random, nlpts_data+nlpts_space+it)
            valid_addons_idx = opt.check_valid(addon_pts)
            valid_trial_pts = np.vstack((valid_trial_pts, addon_pts[valid_addons_idx]))
            it += 1
        
        #For each trial point:
        #     Write the input ifile with the sensor info under consideration
        #     run the eig calculator
        #     read and store eig val + eig std
        eigdata = np.zeros([nopt_random,3])
        for inc in range(0,nopt_random):
            if verbose == 1:
                t1 = time.time() - t0
                print(str(sensors.shape[0]+1) + ' ' + str(inc) + ' ' + str(t1), flush=True)
            
            #write temp input file
            if verbose==1:
                fname = f'input_runner_{sensors.shape[0]+1}-{inc}.dat'
            else:
                fname = 'input_runner.dat'
            sloc_trial = valid_trial_pts[inc,:]

            write_input_file(os.path.join(save_path, fname), nlpts_data, nlpts_space, ndata, lat_range, long_range,
            depth_range, mag_range, sloc_trial, sensor_params, sensors, sampling_file, bounds_file)

            #run my MPI
            process = Popen(shlex.split(mpirunstring + " python3 eig_calc.py " + os.path.join(save_path, fname) + " outputs.npz 0"), stdout=PIPE, stderr=PIPE, shell=False)
            stdout, stderr = process.communicate()
            outputdata = np.array([float(item) for item in (stdout.decode("utf-8").rstrip("\n")).split()])

            eigdata[inc,:] = outputdata
        
        opt.tell(valid_trial_pts, -1.0*eigdata[:,0])
        
        eigdata_full = np.zeros([nopt_total,3])
        eigdata_full[0:nopt_random,:] = eigdata

        #Iterage
        for inc in range(nopt_random,nopt_total):
            if verbose == 1:
                t1 = time.time() - t0
                print(str(sensors.shape[0]+1) + ' '+ str(inc)+ " " + str(-1.0*opt.get_result().fun) + " " + str(t1), flush=True)
            
            #write temp input file
            if verbose==1:
                fname = f'input_runner_{sensors.shape[0]+1}-{inc}.dat'
            else:
                fname = 'input_runner.dat'

            #get the test pt
            sloc_trial = np.array(opt.ask())
            write_input_file(os.path.join(save_path, fname), nlpts_data, nlpts_space, ndata, lat_range, long_range,
            depth_range, mag_range, sloc_trial, sensor_params, sensors, sampling_file, bounds_file)

            #run my MPI
            process = Popen(shlex.split(mpirunstring + " python3 eig_calc.py " + os.path.join(save_path, fname) + " outputs.npz 0"), stdout=PIPE, stderr=PIPE, shell=False)
            stdout, stderr = process.communicate()
            outputdata = np.array([float(item) for item in (stdout.decode("utf-8").rstrip("\n")).split()])
            eigdata_full[inc,:] = outputdata

            #update the optimizer
            opt.tell(sloc_trial.tolist(),-1.0*outputdata[0])

        #now find optimial placement
        newsensor, neig = opt.expected_minimum()

        #append sensor to the list of sensors
        sensorvec = np.zeros(sensors.shape[1])
        sensorvec[0:2] = newsensor
        sensorvec[2:] = sensor_params
        sensors = np.vstack((sensors,sensorvec))
        #save the optimization results for fun
        if verbose == 1:
            # Filenames for outputs
            opt_obj_str = f'opt_obj{sensors.shape[0]+1}.pkl'
            opt_result_str = f'result{sensors.shape[0]+1}.pkl'
            eig_result_str = f'result_eigdata{sensors.shape[0]+1}.npz'

            # Paths for outputs
            opt_obj_path = os.path.join(save_path, opt_obj_str)
            opt_result_path = os.path.join(save_path, opt_result_str)
            eig_result_path = os.path.join(save_path, eig_result_str)

            dump(opt.get_result(), opt_result_path)
            dump(opt, opt_obj_path)
            np.savez(eig_result_path, sensors=sensors,eigdata_full=eigdata_full,Xs=np.array(opt.X_sample))
    
    # Path to save final output
    result_path = os.path.join(save_path, save_file)

    if verbose == 0:        
            np.savez(result_path, sensors=sensors)
        
    if verbose == 1:
            t1 = time.time() - t0
            print("Returning Results: " + str(t1), flush=True)
            print(sensors, flush=True)
            np.savez(result_path, sensors=sensors)
