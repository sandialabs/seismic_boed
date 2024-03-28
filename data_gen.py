import numpy as np
import like_models as lm
import time
import warnings
warnings.filterwarnings('error')
import sys
import os
import signal
import joblib
import pickle
import geographiclib
import sobol_seq as sq

import numpy as np

from scipy import stats
from scipy import interpolate as itp

from obspy import geodetics
from obspy.taup import TauPyModel
from obspy.taup.taup_geo import calc_dist
from obspy.geodetics.base import gps2dist_azimuth
from obspy.geodetics.base import degrees2kilometers

from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import haversine_distances

import warnings
warnings.filterwarnings('ignore')
"""
Seismic sensors
------------------------------
------------------------------
"""
def seismic_gen_arrival_normal(theta, sensors, ndata):
    # Variance is combination of arrival time and general sensor variance
    #compute tt mean, model std, measruement std
    tt_data = lm.seismic_compute_tt(theta, sensors)

    mean_tt = tt_data[:,0]
    stdmodel = tt_data[:,1]
    measurenoise = tt_data[:,2]


    #compute corr matrix
    corr = lm.seismic_compute_corr(theta, sensors)
    cov = np.multiply(np.outer(stdmodel,stdmodel),corr) + np.diag(measurenoise**2.0)
#     min_eig = np.min(np.real(np.linalg.eigvals(cov)))

    return np.random.multivariate_normal(mean_tt, cov, ndata, tol=1e-5)

def generate_seismic_data(theta,sensors,ndata):
    #compute detection probablity
    probs = lm.seismic_detection_probability(theta,sensors)
    
    #make probs bigger
    fullprobs = np.outer(np.ones(ndata),probs)
    u_mat = np.random.uniform(size = fullprobs.shape)
    
    #sample arrival times
    atimes = seismic_gen_arrival_normal(theta, sensors, ndata)

    #create dummy data to match shapes with other sensor types
    dummy_data = np.empty((ndata, 2*sensors.shape[0]))
    dummy_data.fill(np.nan)

    #get data[probs arrivaltimes]
    data = np.concatenate((atimes,u_mat<fullprobs, dummy_data),axis=1)
    return data


"""
Instant arrival sensors
------------------------------
------------------------------
"""
def instant_gen_arrival_normal(theta, sensors, ndata):
    # Variance is combination of arrival time and general sensor variance
    #compute tt mean, model std, measruement std
    tt_data = lm.instant_compute_tt(theta, sensors)
    mean_tt = tt_data[:,0]
    stdmodel = tt_data[:,1]
    measurenoise = tt_data[:,2]


    #compute corr matrix
    corr = lm.seismic_compute_corr(theta, sensors)
    cov = np.multiply(np.outer(stdmodel,stdmodel),corr) + np.diag(measurenoise**2.0)
#     min_eig = np.min(np.real(np.linalg.eigvals(cov)))

    return np.random.multivariate_normal(mean_tt, cov, ndata, tol=1e-5)

def generate_instant_data(theta,sensors,ndata):
    #compute detection probablity
    probs = lm.instant_detection_probability(theta,sensors)

    #make probs bigger
    fullprobs = np.outer(np.ones(ndata),probs)
    u_mat = np.random.uniform(size = fullprobs.shape)

    #sample arrival times
    atimes = instant_gen_arrival_normal(theta, sensors, ndata)

    #create dummy data to match shapes with other sensor types
    dummy_data = np.empty((ndata, 2*sensors.shape[0]))
    dummy_data.fill(np.nan)

    #get data[probs arrivaltimes]
    data = np.concatenate((atimes,u_mat<fullprobs, dummy_data), axis=1)
    return data


"""
Infrasound sensors
------------------------------
------------------------------
"""
def infrasound_gen_arrival_normal(theta, sensors, ndata):
    # Variance is combination of arrival time and general sensor variance
    #compute tt mean, model std, measruement std
    tt_data = lm.infrasound_compute_tt(theta, sensors)
    mean_tt = tt_data[:,0]
    stdmodel = tt_data[:,1]
    measurenoise = tt_data[:,2]


    #compute corr matrix
    corr = np.eye(measurenoise.shape[0])
    cov = np.multiply(np.outer(stdmodel,stdmodel),corr) + np.diag(measurenoise**2.0)

    return np.random.multivariate_normal(mean_tt, cov, ndata)

def infrasound_gen_incident_vonmises(theta, sensors, ndata):
    angle_data = lm.infrasound_compute_incident(theta, sensors)
    mean_angle = angle_data[:,0]
    kappa_angle = angle_data[:,1]

    incidents = np.zeros((len(sensors),ndata))
    for i in range(len(sensors)):
        vmf = stats.vonmises_line(kappa_angle[i], loc=mean_angle[i])
        u = np.random.uniform(vmf.cdf(0), vmf.cdf(np.pi), size=ndata)
        val = vmf.ppf(u)

        if np.any(np.isnan(val)):
            print('NAN encountered incident')
            return u, vmf
        incidents[i] = val

    return incidents.T
        
def infrasound_gen_azimuth_vonmises(theta, sensors, ndata):
    azmth_data = lm.infrasound_compute_azimuth(theta, sensors)
    mean_azmth = azmth_data[:,0]
    kappa_azmth = azmth_data[:,1]
    
    azimuths = np.zeros((len(sensors),ndata))
    for i in range(len(sensors)): 
        vmf = stats.vonmises_line(kappa_azmth[i], loc=mean_azmth[i])
        val = vmf.rvs(size=ndata)
       
        if np.any(np.isnan(val)):
            print('NAN encountered azimuth')
            return u, vmf
        azimuths[i] = val
    
    return azimuths.T    
    
def generate_infrasound_data(theta,sensors,ndata):
    #compute detection probablity
    probs = lm.infrasound_detection_probability(theta,sensors)
    
    #make probs bigger
    fullprobs = np.outer(np.ones(ndata),probs)
    u_mat = np.random.uniform(size = fullprobs.shape)
    
    #sample arrival times
    atimes = infrasound_gen_arrival_normal(theta, sensors, ndata)
    
    #sample incident angles
    incidents = infrasound_gen_incident_vonmises(theta, sensors, ndata)
    
    azimuths = infrasound_gen_azimuth_vonmises(theta, sensors, ndata)
    
    #get data[probs arrivaltimes]
    data = np.concatenate((atimes, u_mat<fullprobs, azimuths, incidents),axis=1)
    return data


"""
Seismic arrays
------------------------------
------------------------------
"""
def array_gen_incident_vonmises(theta, sensors, ndata):
    angle_data = lm.array_compute_incident(theta, sensors)
    mean_angle = angle_data[:,0]
    kappa_angle = angle_data[:,1]
    
    incidents = np.zeros((len(sensors),ndata))
    for i in range(len(sensors)): 
        vmf = stats.vonmises_line(kappa_angle[i], loc=mean_angle[i])
        try:
            u = np.random.uniform(vmf.cdf(0), vmf.cdf(np.pi), size=ndata)
        except:
            print(f'failed vals: {vmf.cdf(0), vmf.cdf(np.pi)}')   
            print(f'data gen failed at kappa {kappa_angle[i]}, mean {mean_angle[i]}')
            u = np.random.uniform(vmf.cdf(0), vmf.cdf(np.pi), size=ndata)
            print('retook u')
        val = vmf.ppf(u)
       
        if np.any(np.isnan(val)):
            print('NAN encountered')
            return u, vmf
        incidents[i] = val
    
    return incidents.T
                   
def array_gen_azimuth_vonmises(theta, sensors, ndata):
    azmth_data = lm.array_compute_azimuth(theta, sensors)
    mean_azmth = azmth_data[:,0]
    kappa_azmth = azmth_data[:,1]
    
    azimuths = np.zeros((len(sensors),ndata))
    for i in range(len(sensors)): 
        vmf = stats.vonmises_line(kappa_azmth[i], loc=mean_azmth[i])
        val = vmf.rvs(size=ndata)
       
        if np.any(np.isnan(val)):
            print('NAN encountered')
            return u, vmf
        azimuths[i] = val
    
    return azimuths.T    
    
def generate_array_data(theta,sensors,ndata):
    #compute detection probablity
    probs = lm.seismic_detection_probability(theta,sensors)
    
    #make probs bigger
    fullprobs = np.outer(np.ones(ndata),probs)
    u_mat = np.random.uniform(size = fullprobs.shape)
    
    #sample arrival times
    atimes = seismic_gen_arrival_normal(theta, sensors, ndata)
    
    #sample incident angles
    incidents = array_gen_incident_vonmises(theta, sensors, ndata)
    
    azimuths = array_gen_azimuth_vonmises(theta, sensors, ndata)
    
    #get data[probs arrivaltimes]
    data = np.concatenate((atimes, u_mat<fullprobs, azimuths, incidents),axis=1)
    return data

"""
Generate data
------------------------------
------------------------------
"""
def generate_data(theta, sensors, ndata):
    def split_data(data):
        # splits data into arrival times, detections, azimuths, and incidents

        nsens = int(data.shape[1]/4)
        atimes = data[:,:nsens]
        detections = data[:,nsens:2*nsens]
        azmths = data[:,2*nsens:3*nsens]
        incdnt = data[:,3*nsens:]

        return atimes, detections, azmths, incdnt

    num_sensors = sensors.shape[0]

    # Get locations of each sensor type
    seismic_idx = np.where(sensors[:,4]==0)[0]
    instant_idx = np.where(sensors[:,4]==1)[0]
    infra_idx = np.where(sensors[:,4]==2)[0]
    array_idx = np.where(sensors[:,4]==3)[0]
    
    # Control for nonexistent sensors
    seismic_exists = len(seismic_idx)!=0
    instant_exists = len(instant_idx)!=0
    infra_exists = len(infra_idx)!=0
    array_exists = len(array_idx)!=0
    
    # Split sensors by type
    seismic_sensors = sensors[seismic_idx]
    instant_sensors = sensors[instant_idx]
    infrasound_sensors = sensors[infra_idx]
    array_sensors = sensors[array_idx]
    
    # Generate data from each sensor type
    if seismic_exists:
        seismic_data = generate_seismic_data(theta, seismic_sensors, ndata)
    if instant_exists:
        instant_data = generate_instant_data(theta, instant_sensors, ndata)
    if infra_exists:
        infra_data = generate_infrasound_data(theta, infrasound_sensors, ndata)
    if array_exists:
        array_data = generate_array_data(theta, array_sensors, ndata)

    # Split each sensor type's data into detections, arrivals, azimuths, incident angles
    # in order to recombine into a single dataset
    if seismic_exists:
        seis_atimes, seis_detects, seis_azmths, seis_incdnts = split_data(seismic_data)
    if instant_exists:
        inst_atimes, inst_detects, inst_azmths, inst_incdnts = split_data(instant_data)
    if infra_exists:
        infra_atimes, infra_detects, infra_azmths, infra_incdnts = split_data(infra_data)
    if array_exists:
        array_atimes, array_detects, array_azmths, array_incdnts = split_data(array_data)

    # Create matrix for storing all generated data
    total_data = np.zeros((ndata, 4*num_sensors))
    
    # Now add data from each sensor type into its column corresponding to sensor 
    # positions in original sensor array
    
    # Seismic sensor data
    if seismic_exists:
        total_data[:,seismic_idx] = seis_atimes
        total_data[:,num_sensors + seismic_idx] = seis_detects
        total_data[:,2*num_sensors + seismic_idx] = seis_azmths
        total_data[:,3*num_sensors + seismic_idx] = seis_incdnts
    
    # Instant origin data
    if instant_exists:
        total_data[:,instant_idx] = inst_atimes
        total_data[:,num_sensors + instant_idx] = inst_detects
        total_data[:,2*num_sensors + instant_idx] = inst_azmths
        total_data[:,3*num_sensors + instant_idx] = inst_incdnts
        
    # Infrasound data
    if infra_exists:
        total_data[:,infra_idx] = infra_atimes
        total_data[:,num_sensors + infra_idx] = infra_detects
        total_data[:,2*num_sensors + infra_idx] = infra_azmths
        total_data[:,3*num_sensors + infra_idx] = infra_incdnts
    
    # Seismic array data
    if array_exists:
        total_data[:,array_idx] = array_atimes
        total_data[:,num_sensors + array_idx] =  array_detects
        total_data[:,2*num_sensors + array_idx] = array_azmths
        total_data[:,3*num_sensors + array_idx] = array_incdnts
        
#     # Arrival times
#     total_data[:,seismic_idx] = seis_atimes
#     total_data[:,instant_idx] = inst_atimes
#     total_data[:,infra_idx] = infra_atimes
#     total_data[:,array_idx] = array_atimes

#     # Detections
#     total_data[:,num_sensors + seismic_idx] = seis_detects
#     total_data[:,num_sensors + instant_idx] = inst_detects
#     total_data[:,num_sensors + infra_idx] = infra_detects
#     total_data[:,num_sensors + array_idx] =  array_detects

#     # Azimuths
#     total_data[:,2*num_sensors + seismic_idx] = seis_azmths
#     total_data[:,2*num_sensors + instant_idx] = inst_azmths
#     total_data[:,2*num_sensors + infra_idx] = infra_azmths
#     total_data[:,2*num_sensors + array_idx] = array_azmths

#     # Incident angles
#     total_data[:,3*num_sensors + seismic_idx] = seis_incdnts
#     total_data[:,3*num_sensors + instant_idx] = inst_incdnts
#     total_data[:,3*num_sensors + infra_idx] = infra_incdnts
#     total_data[:,3*num_sensors + array_idx] = array_incdnts

    return total_data

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


