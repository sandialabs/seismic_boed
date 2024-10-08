#!/usr/bin/env python
# coding: utf-8

import importlib
import os
import sys
import time
import warnings

import numpy as np
from mpi4py import MPI

# from sample_gen import generate_theta_data, sample_theta_space, eval_theta_prior, eval_importance
from data_gen import generate_data
from like_models import compute_loglikes
from utils import plot_surface, read_bounds, read_input_file

# warnings.filterwarnings('error')

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()  # Assume size, ndata, and nlpts all have divisiblity
    rank = comm.Get_rank()

    # make each rank of its own seed
    np.random.seed(int(time.time()) + rank)

    # Rank 0 intializes stuff
    if rank == 0:
        # useful to keep track of time
        t0 = time.time()

        if len(sys.argv) == 4:
            # Read input arguments from file
            nlpts_data, nlpts_space, ndata, bounds_fname, sampling_fname, sensors = (
                read_input_file(sys.argv[1])
            )
            # Path to save outputs
            save_file = sys.argv[2]
            # Verbosity level
            verbose = int(sys.argv[3])

            # Set up functions for sampling events
            samplingfile_name, samplingfile_ext = os.path.splitext(sampling_fname)
            if samplingfile_ext != ".py":
                raise ValueError(
                    f"Sampling file must have filetype '.py', not {samplingfile_ext}"
                )

            # Import sampling file as module and save functions
            sampling_module = importlib.import_module(samplingfile_name)
            generate_theta_data = sampling_module.generate_theta_data
            sample_theta_space = sampling_module.sample_theta_space
            eval_importance = sampling_module.eval_importance
            eval_theta_prior = sampling_module.eval_theta_prior

            # Set up bounds for each dimension of theta
            location_bounds, depth_range, mag_range = read_bounds(
                bounds_fname, sensor_bounds=False
            )

            if verbose == 1 or verbose == 2:
                print("Configuring Run: " + str(t0), flush=True)

        else:
            # mpiexec --bind-to core --npernode 36 --n 576 python3 eig_calc.py inputs.dat outputs.npz 1
            # verbose options: 0 (only output is to the screen with EIG STD and MIN_ESS), 1 full output, 2 simple output file
            sys.exit("Usage: python3 eig_calc.py loc_file save_path verbose")

        # Sample prior according to provided functions to generate events that we will use to generate data
        # This is not distributed
        theta_data = generate_theta_data(
            location_bounds, depth_range, mag_range, nlpts_data, 0
        )
        nthetadim = theta_data.shape[1]

        # The data samples need to be importance corrected
        data_importance_evals = eval_importance(
            theta_data, location_bounds, depth_range, mag_range
        )
        data_prior_evals = eval_theta_prior(
            theta_data, location_bounds, depth_range, mag_range
        )
        data_importance_weight = data_prior_evals / data_importance_evals

        # Parameters for broadcasting to cores
        counts = nthetadim * nlpts_data // size * np.ones(size, dtype=int)
        dspls = range(0, nlpts_data * nthetadim, nthetadim * nlpts_data // size)

    else:
        # prepare to send variables to cores
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
        samplingfile_name = None
        bounds_fname = None

    # Distribute everything to the cores
    nlpts_data = comm.bcast(nlpts_data, root=0)
    nlpts_space = comm.bcast(nlpts_space, root=0)
    ndata = comm.bcast(ndata, root=0)
    depth_range = comm.bcast(depth_range, root=0)
    mag_range = comm.bcast(mag_range, root=0)
    sensors = comm.bcast(sensors, root=0)
    nthetadim = comm.bcast(nthetadim, root=0)
    samplingfile_name = comm.bcast(samplingfile_name, root=0)
    bounds_fname = comm.bcast(bounds_fname, root=0)

    if rank != 0:
        sampling_module = importlib.import_module(samplingfile_name)
        eval_importance = sampling_module.eval_importance
        eval_theta_prior = sampling_module.eval_theta_prior
        location_bounds = read_bounds(bounds_fname, sensor_bounds=False)[0]

    if rank == 0 and verbose == 1:
        t1 = time.time() - t0
        print("Generating Synthetic Data: " + str(t1), flush=True)

    local_nlpts_data = nlpts_data // size
    # prepare to recieve the theta_data for each node
    # holder for everyone to recieve their part of the theta vector
    recvtheta_data = np.zeros((local_nlpts_data, nthetadim))

    # get theta values to use for computing the synthetics data
    comm.Scatterv([theta_data, counts, dspls, MPI.DOUBLE], recvtheta_data, root=0)

    # Every core generate hypohtetical dataset on their bit of theta
    # Generate Hypothetical Datasets

    dataveclen = int(sensors.shape[0] * 4)
    localdataz = np.zeros([local_nlpts_data * ndata, dataveclen])
    for ievent in range(0, local_nlpts_data):
        if rank == 0 and verbose == 1:
            t1 = time.time() - t0
            print(
                str(ievent + 1) + " of " + str(local_nlpts_data) + " " + str(t1),
                flush=True,
            )

        theta = recvtheta_data[ievent, :]
        localdataz[(ievent * ndata) : ((ievent + 1) * ndata), :] = generate_data(
            theta, sensors, ndata
        )

    # Gather the synthetic data
    scounts = local_nlpts_data * ndata * dataveclen
    rcounts = local_nlpts_data * ndata * dataveclen * np.ones(size, dtype=int)
    rdspls = range(
        0, nlpts_data * ndata * dataveclen, local_nlpts_data * ndata * dataveclen
    )
    dataz = None

    if rank == 0:
        dataz = np.zeros((nlpts_data * ndata, dataveclen))

    comm.Gatherv(
        [localdataz, scounts, MPI.DOUBLE], [dataz, rcounts, rdspls, MPI.DOUBLE], root=0
    )

    # Now everyone needs a copy of the full dataset
    dataz = comm.bcast(dataz, root=0)

    # Define the event space descritization

    if rank == 0:
        # seed with nlpts_data so that it starts sampling after that so we dont overlap pts.
        theta_space = sample_theta_space(
            location_bounds, depth_range, mag_range, nlpts_space, nlpts_data
        )
        counts = nthetadim * nlpts_space // size * np.ones(size, dtype=int)
        dspls = range(0, nlpts_space * nthetadim, nthetadim * nlpts_space // size)
    else:
        # prepare to send variables to cores
        theta_space = None
        counts = None
        dspls = None

    if rank == 0 and verbose == 1:
        t1 = time.time() - t0
        print(
            "Computing Likelihood: " + str(t1),
            "Rank",
            rank,
            "comm size",
            size,
            flush=True,
        )

    # prepare to recieve the theta_space for each node
    # holder for everyone to recieve their part of the theta vector
    local_nlpts_space = nlpts_space // size
    recvtheta_space = np.zeros((local_nlpts_space, nthetadim))

    # get theta values to use for computing likelihoods
    comm.Scatterv([theta_space, counts, dspls, MPI.DOUBLE], recvtheta_space, root=0)

    # Now everyone computes a bunch of likelihoods
    # Compute likelhood of each event given dataset
    local_loglikes = np.zeros([local_nlpts_space, nlpts_data * ndata])
    local_weight_loglikes = local_loglikes.copy()

    for ievent in range(0, local_nlpts_space):
        if rank == 0 and verbose == 1:
            t1 = time.time() - t0
            print(
                str(ievent + 1) + " of " + str(local_nlpts_space) + " " + str(t1),
                flush=True,
            )

        theta = recvtheta_space[ievent, :]

        importance_evals = eval_importance(
            theta, location_bounds, depth_range, mag_range
        )
        prior_evals = eval_theta_prior(theta, location_bounds, depth_range, mag_range)
        importance_weight = prior_evals / importance_evals

        # compute likelihoods
        local_loglikes[ievent, :] = compute_loglikes(theta, sensors, dataz)
        local_weight_loglikes[ievent, :] = local_loglikes[ievent, :] + np.log(
            importance_weight
        )

    # Gather the likelihoods
    scounts = local_nlpts_space * ndata * nlpts_data
    rcounts = local_nlpts_space * ndata * nlpts_data * np.ones(size, dtype=int)
    rdspls = range(
        0, nlpts_space * ndata * nlpts_data, local_nlpts_space * ndata * nlpts_data
    )
    loglikes = None
    weight_loglikes = None

    if rank == 0:
        loglikes = np.zeros((nlpts_space, nlpts_data * ndata))
        weight_loglikes = loglikes.copy()

    comm.Gatherv(
        [local_loglikes, scounts, MPI.DOUBLE],
        [loglikes, rcounts, rdspls, MPI.DOUBLE],
        root=0,
    )
    comm.Gatherv(
        [local_weight_loglikes, scounts, MPI.DOUBLE],
        [weight_loglikes, rcounts, rdspls, MPI.DOUBLE],
        root=0,
    )

    # We really want transpose of loglike for everything
    if rank == 0:
        loglikes = loglikes.transpose().copy()
        weight_loglikes = weight_loglikes.transpose().copy()

    # Now need to compute EIG/KL
    if rank == 0 and verbose == 1:
        t1 = time.time() - t0
        print("Computing EIG: " + str(t1), flush=True)

    # Distribute loglikes for each bit of data to the cores
    local_ndataz = ndata * nlpts_data // size
    if rank == 0:
        counts = local_ndataz * nlpts_space * np.ones(size, dtype=int)
        dspls = range(0, nlpts_data * ndata * nlpts_space, local_ndataz * nlpts_space)

    else:
        # prepare to send variables to cores
        counts = None
        dspls = None

    # prepare to recieve the loglikes for each separated data at the node
    # holder for everyone to recieve their part of the loglikes we need
    recloglikes = np.zeros((local_ndataz, nlpts_space))
    recweight_loglikes = np.zeros((local_ndataz, nlpts_space))

    # get theta values to use for computing likelihoods
    comm.Scatterv([loglikes, counts, dspls, MPI.DOUBLE], recloglikes, root=0)
    comm.Scatterv(
        [weight_loglikes, counts, dspls, MPI.DOUBLE], recweight_loglikes, root=0
    )

    # Now compute the ig for each set of likelihoods
    # Combine data to compute posterior and KL divergence
    local_ig = np.zeros(local_ndataz)
    local_ess = np.zeros(local_ndataz)
    for idata in range(0, local_ndataz):
        if rank == 0 and verbose == 1:
            t1 = time.time() - t0
            print(
                str(idata + 1) + " of " + str(local_ndataz) + " " + str(t1), flush=True
            )
        loglike = recloglikes[idata, :]
        weight_loglike = recweight_loglikes[idata, :]

        probs = np.exp(weight_loglike - np.max(weight_loglike)) / np.sum(
            np.exp(weight_loglike - np.max(weight_loglike))
        )

        local_ig[idata] = np.sum(
            probs
            * (
                loglike
                - np.log(np.mean(np.exp(weight_loglike - np.max(weight_loglike))))
            )
        ) - np.max(weight_loglike)

        # lets also compute ess of the weights so we can return that too...
        local_ess[idata] = 1.0 / np.sum(probs**2)

    # now gather all the igs and ess
    scounts = local_ndataz
    rcounts = local_ndataz * np.ones(size, dtype=int)
    rdspls = range(0, ndata * nlpts_data, local_ndataz)

    # send core information to the root
    ig = None
    ess = None
    if rank == 0:
        ig = np.zeros(ndata * nlpts_data)
        ess = np.zeros(ndata * nlpts_data)
    comm.Gatherv(
        [local_ig, scounts, MPI.DOUBLE], [ig, rcounts, rdspls, MPI.DOUBLE], root=0
    )
    comm.Gatherv(
        [local_ess, scounts, MPI.DOUBLE], [ess, rcounts, rdspls, MPI.DOUBLE], root=0
    )
    # Now summarize and return results
    if rank == 0:
        weights_arr = np.repeat(data_importance_weight, ndata)
        eig = np.mean(ig * weights_arr)
        veig = 1 / len(weights_arr) * (np.mean((ig * weights_arr) ** 2) - eig**2)
        seig = np.sqrt(veig)
        miness = np.min(ess)

        if verbose == 0:
            np.savez(save_file, eig=eig, seig=seig, miness=miness)

        if verbose == 1:
            t1 = time.time() - t0
            print("Returning Results: " + str(t1), flush=True)

            np.savez(
                save_file,
                eig=eig,
                seig=seig,
                ig=ig,
                ess=ess,
                miness=miness,
                theta_data=theta_data,
                theta_space=theta_space,
                sensors=sensors,
                location_bounds=location_bounds,
                depth_range=depth_range,
                mag_range=mag_range,
                loglikes=loglikes,
                weight_loglike=weight_loglike,
                dataz=dataz,
                data_importance_weight=data_importance_weight,
            )

        print(str(eig) + " " + str(seig) + " " + str(miness), flush=True)
