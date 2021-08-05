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




if __name__ == '__main__':
    t0 = time.time()
    if len(sys.argv) == 5:
        #Load optimization parameters + parameters that will be needed to bulid the input file to the eig code
        #Needed for input files: nlpts, ndata, lat_range, long_range, depth_range, sensors
        #Assume we are doing sequential greedy sensor placement
        #Opt parameters: sensor_lat_range, sensor_long_range, #random initial trial points, #total number of trials
        #                sensor type and accuracy, optimization criteria (e.g. UCB, EI)
        #Also need info for how to run mpi and how many sensors to place

        nopt_random, nopt_total, sensor_lat_range, sensor_long_range, sensor_params, opt_type, nlpts_data, nlpts_space, ndata, lat_range, long_range, depth_range, mag_range, sensors, mpirunstring, nsensor_place = read_opt_file(sys.argv[1])

        save_file = sys.argv[2]
        save_path = sys.argv[3]
        verbose = int(sys.argv[4])
 
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
        
        #Randomly selection the random trial points (right now this is going to be the same every isamp stage)
        #becuase we are intializing the psuedo random seed the same each time.
        sensor_loc_random = sample_sensors(sensor_lat_range,sensor_long_range, nopt_random,nlpts_data+nlpts_space)

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
            fname = 'input_runner.dat'
            sloc_trial = sensor_loc_random[inc,:]
            print(sloc_trial)
            write_input_file(fname, nlpts_data, nlpts_space, ndata, lat_range, long_range, depth_range, mag_range, sloc_trial, sensor_params, sensors)

            #run my MPI
            process = Popen(shlex.split(mpirunstring + " python3 eig_calc.py input_runner.dat outputs.npz 0"), stdout=PIPE, stderr=PIPE, shell=False)
            stdout, stderr = process.communicate()
            print(stdout)
            outputdata = np.array([float(item) for item in (stdout.decode("utf-8").rstrip("\n")).split()])
            eigdata[inc,:] = outputdata

        #Initialize the optimizer not it minimizes
        #0 -> EI    
        if opt_type == 0:
            opt = Optimizer([(sensor_lat_range[0],sensor_lat_range[1]),(sensor_long_range[0],sensor_long_range[1])], "GP", n_initial_points=0, acq_optimizer="lbfgs", acq_func="EI")

        opt.tell(sensor_loc_random.tolist(),(-1.0*eigdata[:,0]).tolist())

        eigdata_full = np.zeros([nopt_total,3])
        eigdata_full[0:nopt_random,:] = eigdata

        #Iterage
        for inc in range(nopt_random,nopt_total):
            
            if verbose == 1:
                t1 = time.time() - t0
                print(str(sensors.shape[0]+1) + ' '+ str(inc)+ " " + str(-1.0*opt.get_result().fun) + " " + str(t1), flush=True)
            
            #write temp input file
            fname = 'input_runner.dat'

            #get the test pt
            sloc_trial = np.array(opt.ask())
            write_input_file(fname, nlpts_data, nlpts_space, ndata, lat_range, long_range, depth_range, mag_range, sloc_trial, sensor_params, sensors)

            #run my MPI
            process = Popen(shlex.split(mpirunstring + " python3 eig_calc.py input_runner.dat outputs.npz 0"), stdout=PIPE, stderr=PIPE, shell=False)
            stdout, stderr = process.communicate()
            outputdata = np.array([float(item) for item in (stdout.decode("utf-8").rstrip("\n")).split()])
            eigdata_full[inc,:] = outputdata

            print("OUTPUTDATA:", outputdata.shape)
            print(outputdata)
            print(sloc_trial.tolist())
            #update the optimizer
            opt.tell(sloc_trial.tolist(),-1.0*outputdata[0])

            #save the optimization results for fun
            if verbose == 1:
                # Filenames for outputs
                opt_result_str = f'result{sensors.shape[0]+1}.pkl'
                eig_result_str = f'result_eigdata{sensors.shape[0]+1}.npz'

                # Paths for outputs
                opt_result_path = os.path.join(save_path, opt_result_str)
                eig_result_path = os.path.join(save_path, eig_result_str)

                dump(opt.get_result, opt_result_path)
                np.savez(eig_result_path, sensors=sensors,eigdata_full=eigdata_full,Xs=np.array(opt.Xi))

        #now find optimial placement
        newsensor, neig = expected_minimum(opt.get_result())

        #append sensor to the list of sensors
        sensorvec = np.zeros(sensors.shape[1])
        sensorvec[0:2] = newsensor
        sensorvec[2:] = sensor_params
        sensors = np.vstack((sensors,sensorvec))
    
    # Path to save final output
    result_path = os.path.join(save_path, save_file)
    opt_res_path = os.path.join(save_path, 'opt_obj.pkl')

    if verbose == 0:        
            np.savez(result_path, sensors=sensors)
            dump(opt, opt_res_path)
        
    if verbose == 1:
            t1 = time.time() - t0
            print("Returning Results: " + str(t1), flush=True)
            print(sensors, flush=True)
            np.savez(result_path, sensors=sensors)
            dump(opt, opt_res_path)
