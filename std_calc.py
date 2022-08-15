#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from utils import read_input_file, plot_surface
# from sample_gen import generate_theta_data, sample_theta_space, eval_theta_prior, eval_importance
from meas_data_gen import generate_data
from like_models import compute_loglikes

import time
import sys
import os
import importlib

from mpi4py import MPI


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size() #Assume size, ndata, and nlpts all have divisiblity
    rank = comm.Get_rank()

    # make each rank of its own seed
    np.random.seed(int(time.time()) + rank)
    
    
    #Rank 0 intializes stuff
    if rank == 0:
        #useful to keep track of time
        t0 = time.time()
        
        if len(sys.argv) == 4:
            nlpts_data, nlpts_space, ndata, lat_range, long_range, depth_range, mag_range, sampling_fname, sensors = read_input_file(sys.argv[1])
            save_file = sys.argv[2]
            verbose = int(sys.argv[3])
            sampling_file = importlib.import_module(sampling_fname[:-4])
            generate_theta_data = sampling_file.generate_theta_data
            sample_theta_space = sampling_file.sample_theta_space
            eval_importance = sampling_file.eval_importance
            eval_theta_prior = sampling_file.eval_theta_prior


            if (verbose == 1 or verbose == 2):
                print("Configuring Run: " + str(t0), flush=True)
        
        else:
            #mpiexec --bind-to core --npernode 36 --n 576 python3 eig_calc.py inputs.dat outputs.npz 1
            #verbose options: 0 (only output is to the screen with EIG STD and MIN_ESS), 1 full output, 2 simpel output file
            sys.exit('Usage: python3 eig_calc.py loc_file save_path verbose')
            
        #Sample prior some how to generate events that we will use to generate data
        #This is not distributed
        #Set seed to 0 here
        theta_data = generate_theta_data(lat_range, long_range, depth_range, mag_range, nlpts_data, 0)
        nthetadim = theta_data.shape[1]
        
        #The data samples need to be importance corrected
        data_importance_evals = eval_importance(theta_data,lat_range,long_range,depth_range,mag_range)
        data_prior_evals = eval_theta_prior(theta_data,lat_range,long_range,depth_range,mag_range)
        data_importance_weight = data_prior_evals/data_importance_evals

        counts = nthetadim * nlpts_data // size * np.ones(size, dtype=int)
        dspls = range(0, nlpts_data * nthetadim, nthetadim * nlpts_data // size)
    else:
        #prepare to send variables to cores
        nlpts_data = None
        nlpts_space = None
        ndata = None
        lat_range = None
        long_range = None
        depth_range = None
        mag_range = None
        sensors = None
        nthetadim = None
        theta_data = None
        counts = None
        dspls = None
        sampling_fname = None
    
    #Distribute everythong to the cores
    nlpts_data = comm.bcast(nlpts_data, root=0)
    nlpts_space = comm.bcast(nlpts_space, root=0)
    ndata = comm.bcast(ndata, root=0)
    lat_range = comm.bcast(lat_range, root=0)
    long_range = comm.bcast(long_range, root=0)
    depth_range = comm.bcast(depth_range, root=0)
    mag_range = comm.bcast(mag_range, root=0)
    sensors = comm.bcast(sensors, root=0)
    nthetadim = comm.bcast(nthetadim, root=0)
    sampling_fname = comm.bcast(sampling_fname, root=0)

    if rank != 0:
        sampling_file = importlib.import_module(sampling_fname[:-4])
        eval_importance = sampling_file.eval_importance
        eval_theta_prior = sampling_file.eval_theta_prior 
      
    if rank == 0 and verbose == 1:
        t1 = time.time() - t0
        print("Generating Synthetic Data: " + str(t1), flush=True)
    
    local_nlpts_data = nlpts_data // size
    #prepaire to recieve the theta_data for each node
    # holder for everyone to recive there part of the theta vector
    recvtheta_data = np.zeros((local_nlpts_data, nthetadim))
    
    # get your theta values to use for computing the synthetics data
    comm.Scatterv([theta_data, counts, dspls, MPI.DOUBLE], recvtheta_data, root=0)
    
    
    #Every core generate hypohtetical dataset on their bit of theta
    #Generate Hypothetical Datasets

    dataveclen = np.int(sensors.shape[0]*sensors[0,3])
    localdataz = np.zeros([local_nlpts_data*ndata,dataveclen])
    localmeasnoise = np.zeros(local_nlpts_data)
    for ievent in range(0,local_nlpts_data):
        if rank == 0 and verbose == 1:
            t1 = time.time() - t0
            print(str(ievent) + " of " + str(local_nlpts_data) + " " + str(t1), flush=True)

        theta = recvtheta_data[ievent,:]
        data_returns =  generate_data(theta,sensors,ndata)
        #print(data_returns[1].shape)
        localdataz[(ievent*ndata):((ievent+1)*ndata),:] = data_returns[0]
        localmeasnoise[ievent] = data_returns[1]



    #Gather the synthetic data
    scounts = local_nlpts_data * ndata * dataveclen
    rcounts = local_nlpts_data * ndata * dataveclen * np.ones(size, dtype=int)
    rdspls = range(0, nlpts_data * ndata * dataveclen, local_nlpts_data * ndata * dataveclen)
    mscounts = local_nlpts_data
    mrcounts = mscounts * np.ones(size,dtype=int)
    mrdspls = range(0,mscounts*size,mscounts)
    dataz = None
    measnoise = None

    if rank == 0:
        dataz = np.zeros((nlpts_data*ndata,dataveclen))
        measnoise = np.zeros(nlpts_data)
    print('made it here')
    comm.Gatherv([localdataz, scounts, MPI.DOUBLE], [dataz, rcounts, rdspls, MPI.DOUBLE], root=0)
    comm.Gatherv([localmeasnoise,mscounts,MPI.DOUBLE], [measnoise, mrcounts, mrdspls, MPI.DOUBLE],root=0)

    
    #Now summarize and return results
    if rank == 0:
        print('Made it here 2')
        mean_mn = np.mean(measnoise)
        if verbose == 0:
            np.savez(save_file, std=mean_mn)
            
        if verbose == 1:
            print('saving results')
            t1 = time.time() - t0
            print("Returning Results: " + str(t1), flush=True)
        
            np.save(save_file, mean_mn)
            
        #Probs should retrun some uncertainty on this...
        print(str(np.mean(measnoise)))
