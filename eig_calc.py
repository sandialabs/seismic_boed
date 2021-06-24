#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from utils import read_input_file, plot_surface
from sample_gen import generate_theta_data, sample_theta_space, eval_theta_prior, eval_importance
from data_gen import generate_data
from like_models import compute_loglikes

import time
import sys
import os

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
            nlpts_data, nlpts_space, ndata, lat_range, long_range, depth_range, mag_range, sensors = read_input_file(sys.argv[1])
            save_file = sys.argv[2]
            verbose = int(sys.argv[3])

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
        sensors = None
        nthetadim = None
        theta_data = None
        counts = None
        dspls = None
    
    #Distribute everythong to the cores
    nlpts_data = comm.bcast(nlpts_data, root=0)
    nlpts_space = comm.bcast(nlpts_space, root=0)
    ndata = comm.bcast(ndata, root=0)
    lat_range = comm.bcast(lat_range, root=0)
    long_range = comm.bcast(long_range, root=0)
    depth_range = comm.bcast(depth_range, root=0)
    sensors = comm.bcast(sensors, root=0)
    nthetadim = comm.bcast(nthetadim, root=0)

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
    for ievent in range(0,local_nlpts_data):
        if rank == 0 and verbose == 1:
            t1 = time.time() - t0
            print(str(ievent) + " of " + str(local_nlpts_data) + " " + str(t1), flush=True)
            
        theta = recvtheta_data[ievent,:]
        localdataz[(ievent*ndata):((ievent+1)*ndata),:] = generate_data(theta,sensors,ndata)


    #Gather the synthetic data
    scounts = local_nlpts_data * ndata * dataveclen
    rcounts = local_nlpts_data * ndata * dataveclen * np.ones(size, dtype=int)
    rdspls = range(0, nlpts_data * ndata * dataveclen, local_nlpts_data * ndata * dataveclen)
    dataz = None
       
    if rank == 0:
        dataz = np.zeros((nlpts_data*ndata,dataveclen))
        
    comm.Gatherv([localdataz, scounts, MPI.DOUBLE], [dataz, rcounts, rdspls, MPI.DOUBLE], root=0)    
    
    #Now everyone needs a copy of the full dataset now
    dataz = comm.bcast(dataz, root=0)
         
    #Define the event space descritization
    #We assume that these are uniformly sampled from the prior so they all have equal aprior likelihood.
    #This may be something that we should change in the future to add a prior likleihood associated with each sample
    #so that we dont have to consider them to be uniform.
    
    if rank == 0:
        #seed with nlpts_data so that it starts sampling after that so we dont overlap pts.
        theta_space = sample_theta_space(lat_range,long_range, depth_range, nlpts_space,nlpts_data) # Could return theta space and sample weight    
        counts = nthetadim * nlpts_space // size * np.ones(size, dtype=int)
        dspls = range(0, nlpts_space * nthetadim, nthetadim * nlpts_space // size)
    else:
        #prepare to send variables to cores
        theta_space = None
        counts = None
        dspls = None
    
    if rank == 0 and verbose == 1:
        t1 = time.time() - t0
        print("Computing Likelihood: " + str(t1), flush=True)
            
    #prepaire to recieve the theta_space for each node
    # holder for everyone to recive there part of the theta vector
    local_nlpts_space = nlpts_space // size
    recvtheta_space = np.zeros((local_nlpts_space, nthetadim))
    
    # get your theta values to use for computing likelihoods
    comm.Scatterv([theta_space, counts, dspls, MPI.DOUBLE], recvtheta_space, root=0)    
    
    
    #Now everyone computes a bunch of likelihoods
    #Compute likelhood of each event given dataset
    local_loglikes = np.zeros([local_nlpts_space,nlpts_data*ndata])
    local_weight_loglikes = local_loglikes.copy()

    for ievent in range(0,local_nlpts_space):
        if rank == 0 and verbose == 1:
            t1 = time.time() - t0
            print(str(ievent) + " of " + str(local_nlpts_space) + " " + str(t1), flush=True)
            
        theta = recvtheta_space[ievent,:]
        prior_evals = eval_theta_prior(theta,lat_range,long_range,depth_range)
        importance_evals = eval_importance(theta,lat_range,long_range,depth_range)
        importance_weight = prior_evals/importance_evals
    
        #compute likelihoods
        local_loglikes[ievent,:] = compute_loglikes(theta,sensors,dataz)
        local_weight_loglikes[ievent,:] = local_loglikes[ievent,:] + np.log(importance_weight)

    
    #Gather the likelihoods
    scounts = local_nlpts_space * ndata * nlpts_data
    rcounts = local_nlpts_space * ndata * nlpts_data * np.ones(size, dtype=int)
    rdspls = range(0, nlpts_space * ndata * nlpts_data, local_nlpts_space * ndata * nlpts_data)
    loglikes = None
    weight_loglikes = None
       
    if rank == 0:
        loglikes = np.zeros((nlpts_space, nlpts_data * ndata))
        weight_loglikes = loglikes.copy()
        
    comm.Gatherv([local_loglikes, scounts, MPI.DOUBLE], [loglikes, rcounts, rdspls, MPI.DOUBLE], root=0)
    comm.Gatherv([local_weight_loglikes, scounts, MPI.DOUBLE], [weight_loglikes, rcounts, rdspls, MPI.DOUBLE], root=0)   

    #I really want transpose of loglike for everything
    if rank==0:
        loglikes = loglikes.transpose().copy()
        weight_loglikes = weight_loglikes.transpose().copy()

    #Now need to compute EIG/KL
    if rank == 0 and verbose == 1:
        t1 = time.time() - t0
        print("Computing EIG: " + str(t1), flush=True)
    
    
    #Distribute loglikes for each bit of data to the cores
    local_ndataz = ndata*nlpts_data // size
    if rank == 0:  
        counts = local_ndataz * nlpts_space * np.ones(size, dtype=int)
        dspls = range(0, nlpts_data * ndata * nlpts_space, local_ndataz * nlpts_space)
      
    else:
        #prepare to send variables to cores
        counts = None
        dspls = None
    
            
    #prepaire to recieve the loglikes for each separed data at the node
    # holder for everyone to recive there part of the loglikes we need
    recloglikes = np.zeros((local_ndataz, nlpts_space))
    recweight_loglikes = np.zeros((local_ndataz, nlpts_space))
    
    # get your theta values to use for computing likelihoods
    comm.Scatterv([loglikes, counts, dspls, MPI.DOUBLE], recloglikes, root=0)  
    comm.Scatterv([weight_loglikes, counts, dspls, MPI.DOUBLE], recweight_loglikes, root=0)     
    
    #Now compute the ig for each set of likelihoods    
    #Combine data to compute posterior and KL divergence
    local_ig = np.zeros(local_ndataz)
    local_ess = np.zeros(local_ndataz)
    for idata in range(0,local_ndataz):
        if rank == 0 and verbose == 1:
            t1 = time.time() - t0
            print(str(idata) + " of " + str(local_ndataz) + " " + str(t1), flush=True)
        loglike = recloglikes[idata,:]
        weight_loglike = recweight_loglikes[idata,:]

        probs = np.exp(weight_loglike - np.max(weight_loglike))/np.sum(np.exp(weight_loglike - np.max(weight_loglike)))
        logprobs = (loglike - np.max(weight_loglike)) - np.log(np.sum(np.exp(weight_loglike - np.max(weight_loglike))))
        local_ig[idata] = np.sum(probs*(likes)) - np.log(np.mean(np.exp(weight_loglikes - np.max(weight_loglikes)))) - np.max(weight_loglikes)
        
        #lets also compute ess of the weights so we can return that too...
        local_ess[idata] = 1.0 / np.sum(probs**2)
        
        
    #now gather all the igs and ess
    scounts = local_ndataz
    rcounts = local_ndataz * np.ones(size, dtype=int)
    rdspls = range(0, ndata*nlpts_data, local_ndataz)

    # send core information to the root
    ig = None
    ess = None
    if rank == 0:
        ig = np.zeros(ndata*nlpts_data)
        ess = np.zeros(ndata*nlpts_data)
    comm.Gatherv([local_ig, scounts, MPI.DOUBLE], [ig, rcounts, rdspls, MPI.DOUBLE], root=0)    
    comm.Gatherv([local_ess, scounts, MPI.DOUBLE], [ess, rcounts, rdspls, MPI.DOUBLE], root=0)  
 
    
    #Now summarize and return results
    if rank == 0:
        eig = np.mean(ig)
        seig = np.std(ig)
        miness = np.min(ess)
        if verbose == 0:
            np.savez(save_file, eig=eig, seig=seig, miness=miness)
            
        if verbose == 1:
            t1 = time.time() - t0
            print("Returning Results: " + str(t1), flush=True)
        
            np.savez(save_file, eig=eig, seig=seig, ig=ig, ess=ess, miness=miness, theta_data=theta_data,
                 theta_space=theta_space, sensors=sensors, lat_range=lat_range, long_range=long_range,
                     depth_range=depth_range, mag_range=mag_range, loglikes=loglikes, dataz=dataz)

            
        #Probs should retrun some uncertainty on this...
        print(str(eig) + " " + str(seig) + " " + str(miness), flush=True)
