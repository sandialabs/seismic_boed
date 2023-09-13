import time
import math
import joblib
import pickle
import geographiclib

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
warnings.filterwarnings("ignore")

"""
Models for sesmic sensors
-------------------------------------------------
-------------------------------------------------
"""
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance (in km) between two points 
    using their longitude and latitude (in degrees).
    """
    # Radius of the Earth
    correction = 0.9996400
    r = 6372.8 * 1000 * correction

    # Convert degrees to radians 
    # First point
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    # Second Point
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) 
    return r * c

def seismic_detection_probability(theta,sensors):
    slat = theta[0]
    slong = theta[1]
    sdepth = theta[2]
    smag = theta[3]

    model = LogisticRegression()
    model.classes_ = np.asarray([0, 1])
    # Params for Utah model
#     model.coef_ = np.asarray([[-0.0297509 , -2.81876856,  1.14055126]])
#     model.intercept_ = np.asarray(1.95254635)

    # Params for Europe model
    model.coef_ = np.array([[ -0.0297509,   -0.606856,    1.14055126]])
    model.intercept_ = np.array(1.55254635)
    
    probs = np.ones(sensors.shape[0])
    
    for isens in range(0,sensors.shape[0]):  
        delta = geodetics.locations2degrees(slat, slong, sensors[isens][0], sensors[isens][1])
        x = np.asarray([sdepth, delta, smag]).reshape(1, -1)
        probs[isens] = model.predict_proba(x)[0][1]

    probs[np.where(probs == 1)[0]] = .99999
    return probs


#Improved TT STD Model
def seismic_tt_std_cal(depth, dist):
    params = np.array([ -2.56472101e+02,   1.74085841e+01,  -1.19406851e-03,
        -1.66597693e-04,   9.91187799e-08,   1.17056345e+01,
        -8.13371656e-01,   8.42272315e-04,   2.80802067e-06,
         3.60764706e-09,   1.21624463e-01,  -8.63214584e-03,
        -7.45214280e-06,   1.25905904e-07,  -1.21859595e-10,
        -1.19443389e-02,   8.67089566e-04,  -1.11453054e-06,
        -1.11651169e-09,  -7.07702285e-12,   1.39921004e-04,
        -1.03707528e-05,   1.97886651e-08,  -2.93026404e-11,
         1.57527073e-13,   1.23554461e-03,   1.84229583e-02,
        -2.18347658e-04,   9.53900469e-07,  -1.26729653e-09,
         3.64225546e-02,  -3.39939908e-03,   3.16659333e-05,
        -1.02362420e-07,   1.11272994e-10,  -3.75764388e-03,
         2.73743286e-04,  -2.10488918e-06,   5.69748506e-09,
        -5.17365299e-12,   1.49883472e-04,  -9.66468438e-06,
         6.99133567e-08,  -1.79611764e-10,   1.56300210e-13,
        -1.98606449e-06,   1.21461076e-07,  -8.71278276e-10,
         2.26330123e-12,  -2.03871471e-15,  -3.61047221e+01,
         1.75671838e+01])
    
    ndim = 5
    poly0 = np.reshape(params[0:(ndim**2)],(ndim,ndim))
    poly1 = np.reshape(params[(ndim**2):(2*ndim**2)],(ndim,ndim))
    g = -1.0*np.abs(params[2*ndim**2])
    h = np.abs(params[2*ndim**2+1])
    
    boolg = 1.0/(1.0+np.exp(-h*(depth+g)))
    std0 = np.polynomial.polynomial.polyval2d(depth,dist,poly0)
    std1 = np.polynomial.polynomial.polyval2d(depth,dist,poly1)
    std = std0*boolg + std1*(1.0-boolg)
    return std

def seismic_snr_cal(dist, mag, snroffset):
    #fit from TA Arrray Data nominally, snroffset = 0
    a = 0.90593911
    b = 0.92684044
    c = 5.54237317
    return a*mag + c - b*np.log(dist+10**-10) + snroffset

#magnitude dependeint measurment error
def seismic_meas_std_cal(dist, mag, snroffset):  
    #add nugget for numerical stablity
    logsnr = seismic_snr_cal(dist, mag, snroffset)
    
    #Uncertainty in Phase Arrival Time Picks for Regional Seismic Events: An Experimental Design
    # Velasco et al 2001 (SAND Report)
    #Simple IDC SNR only model
    
    #Nomial values from "Improving Regional Seismic Event Location in China"
    #Steck et al 2001
    
    sig0 = 10.0
    gamma = 0.01
    tl = 5
    tu = 50
    
    if logsnr < np.log(tl):
        sig = sig0
    else:
        if logsnr > np.log(tu):
                sig = gamma*sig0
        else:
                sig = sig0 - (1.0-gamma)*sig0/(np.log(tu)-np.log(tl)) * (logsnr - np.log(tl))
    return sig

def seismic_compute_corr(theta, sensors):
    rlats = sensors[:,0]
    rlongs = sensors[:,1]
    corr = np.eye(len(rlats))
    
    lscal = 147.5
    
    for isens in range(0,len(rlats)):
        for jsens in range(isens,len(rlats)):
            #147.5 was fit from the 1D profiles
#             if sensors[isens, 4] == 1:
#                 # Make instant arrival sensors uncorrelated with all others
#                 if isens == jsens:
#                     corr[isens, jsens] = 1
#                 else:
#                     corr[isens, jsens] = 0
#                     corr[jsens, isens] = 0

            azmthdata = geodetics.gps2dist_azimuth(rlats[isens], rlongs[isens], rlats[jsens], rlongs[jsens])
            corr[isens,jsens] = np.exp(-1.0/(lscal**2) * (azmthdata[0]/1000.0)**2)
            corr[jsens,isens] = np.exp(-1.0/(lscal**2) * (azmthdata[0]/1000.0)**2)
            
    return corr


def seismic_compute_tt(theta, sensors):
    model = TauPyModel(model="iasp91")
    # with open('cubic_reg_europe.pkl', 'rb') as inp:
    with open('cubic_reg.pkl', 'rb') as inp:
        reg = pickle.load(inp)
    with open('TTstd.pkl', 'rb') as inp:
        std_reg = pickle.load(inp)
    # event
    src_lat, src_long, zdepth, src_mag = theta
    # sensors
    rlats, rlongs, sensor_fidelity, *_ = sensors.T
    
    #mean and model std and measurment std
    ptime = np.zeros((len(rlats),3))
    #iangle = np.zeros(len(rlats))
    #azmth = np.zeros(len(rlats))

    for isens, (rlat, rlong, fidelity) in enumerate(zip(rlats, rlongs, sensor_fidelity)):
        arrivals = model.get_travel_times_geo(source_depth_in_km=zdepth, source_latitude_in_deg=src_lat, source_longitude_in_deg=src_long, receiver_latitude_in_deg=rlat, receiver_longitude_in_deg=rlong, phase_list=["P","p"])
        #azmthdata = geodetics.gps2dist_azimuth(src_lat, src_long, rlat, rlong)

        #record data from the first arrival. Assume always in the azimuthal plane.
        ptime[isens,0] = arrivals[0].time
        #iangle[isens] = arrivals[0].incident_angle
        #azmth[isens] = azmthdata[1]

        deltakm = geodetics.degrees2kilometers(geodetics.locations2degrees(src_lat, src_long, rlat, rlong))
        # model_std = tt_std_cal(zdepth, deltakm)
        # ptime[isens,1] = model_std

        #models

        dist = calc_dist(src_lat, src_long, rlat, rlong, 6371.0, 0)
        x = [zdepth, dist]

        # if math.isnan(reg(x)[0]):
        #     print("nan")
        #     print(f'SRC: {src_lat, src_long, src_mag}')
        #     print(f'REC: {rlat, rlong}')
        #     print(f'INPUTS: {zdepth, dist}')
        #     print('----------------------')
        # ptime[isens,0] = reg(x)[0]

        # model_std = std_reg(x)[0]

        ptime[isens,1] = std_reg(x)[0]

        measure_std = seismic_meas_std_cal(dist, src_mag, fidelity)
        
        ptime[isens,2] = measure_std

    return ptime



#Compute likelhood of each event given dataset


def seismic_arrival_likelihood_gaussian(theta, sensors, data):
    #compute mean
    tt_data = seismic_compute_tt(theta, sensors)

    mean_tt = tt_data[:,0]
    stdmodel = tt_data[:,1]
    measurenoise = tt_data[:,2]
    
    #compute corr matrix
    corr = seismic_compute_corr(theta, sensors)
    cov = np.multiply(np.outer(stdmodel,stdmodel),corr) + np.diag(measurenoise**2.0)
    
    [ndata, ndpt] = data.shape
    nsens = int(ndpt/4)
    
    loglike = np.zeros(ndata)
    
    for idata in range(0,ndata):
        mask = data[idata,nsens:2*nsens]
        maskidx = np.nonzero(mask)[0]
        
        #I could decompose this into an iterative operation where I sequentally condition on each sensor? That would maybe be computationally more tractable becuase I could apply to all as update the distributions?
        
        if len(maskidx) > 0:
            #only extracts detections
            means = mean_tt[maskidx]
            dataval = data[idata,maskidx]
            covmat = cov[maskidx[:, None], maskidx]
            
            res = dataval - means
            onevec = np.ones(means.shape)
            a = np.sum(np.linalg.solve(covmat,res))
            b = np.sum(np.linalg.solve(covmat,onevec))
        
        
            #THESE LINES ARE Maybe wrong BECAUSE THE DISTRIBUTION IS IMPROPER
            logdets = np.linalg.slogdet(covmat)[1]
            loglike[idata] = -0.5*np.sum(np.multiply(res, np.linalg.solve(covmat,res))) + (1.0 - len(maskidx))/2.0*np.log(2*np.pi) - 0.5*logdets - 0.5*np.log(b) + np.divide(a**2, 2.0*b)
    return loglike

def seismic_detection_likelihood(theta, sensors, data):
    probs = seismic_detection_probability(theta,sensors)
    
    [ndata, ndpt] = data.shape
    nsens = int(ndpt/4)
    
    loglike = np.zeros(ndata)
    
    for idata in range(0,ndata):
        mask = data[idata,nsens:2*nsens]
        loglike[idata] = np.sum(mask*np.log(probs) + (1.0 - mask) * np.log(1.0 - probs))
    return loglike

"""
Models for instant arrival arrays
-------------------------------------------------
-------------------------------------------------
"""

def instant_detection_probability(theta,sensors):
    # if sensor type is instant arrival, assume detection
    probs = np.ones(sensors.shape[0]) * 1-1e-15
    return probs


def instant_compute_tt(theta, sensors):
    # give instant arrival sensors negligible error
    ptime = np.zeros((sensors.shape[0],3))
    ptime[:,0] = 0
    ptime[:,1] = .001
    ptime[:,2] = .001
    
    return ptime


#Compute likelhood of each event given dataset
def instant_arrival_likelihood_gaussian(theta, sensors, data):
    #compute mean
    tt_data = instant_compute_tt(theta, sensors)

    mean_tt = tt_data[:,0]
    stdmodel = tt_data[:,1]
    measurenoise = tt_data[:,2]
    
    #compute corr matrix
    corr = seismic_compute_corr(theta, sensors) # FIX THIS (IT WORKS BUT ITS UGLY)
    corr = np.eye(corr.shape[0])
    cov = np.multiply(np.outer(stdmodel,stdmodel),corr) + np.diag(measurenoise**2.0)
    
    [ndata, ndpt] = data.shape
    nsens = int(ndpt/4)
    
    loglike = np.zeros(ndata)
    
    for idata in range(0,ndata):
        mask = data[idata,nsens:2*nsens]
        maskidx = np.nonzero(mask)[0]
        
        #I could decompose this into an iterative operation where I sequentally condition on each sensor? That would maybe be computationally more tractable becuase I could apply to all as update the distributions?
        
        if len(maskidx) > 0:
            #only extracts detections
            means = mean_tt[maskidx]
            dataval = data[idata,maskidx]
            covmat = cov[maskidx[:, None], maskidx]
            
            res = dataval - means
            onevec = np.ones(means.shape)
            a = np.sum(np.linalg.solve(covmat,res))
            b = np.sum(np.linalg.solve(covmat,onevec))
        
        
            #THESE LINES ARE Maybe wrong BECAUSE THE DISTRIBUTION IS IMPROPER
            logdets = np.linalg.slogdet(covmat)[1]
            loglike[idata] = -0.5*np.sum(np.multiply(res, np.linalg.solve(covmat,res))) + (1.0 - len(maskidx))/2.0*np.log(2*np.pi) - 0.5*logdets - 0.5*np.log(b) + np.divide(a**2, 2.0*b)
    return loglike


def instant_detection_likelihood(theta, sensors, data):
    probs = instant_detection_probability(theta,sensors)
    
    [ndata, ndpt] = data.shape
    nsens = int(ndpt/4)
    
    loglike = np.zeros(ndata)

    for idata in range(0,ndata):
        mask = data[idata,nsens:2*nsens]
        loglike[idata] = np.sum(mask*np.log(probs) + (1.0 - mask) * np.log(1.0 - probs))
    return loglike


"""
Models for Infrasound arrays
-------------------------------------------------
-------------------------------------------------
"""
def mag_to_kt(mag):
    # values fit for local mag
    d, f = [3.909197277671451, 2.998020624687433]
    energy_J = 10**(d*mag + f)
    energy_tJ = energy_J / 1e12
    energy_kT = energy_tJ / 4.184

    return energy_kT

def sigmoid_params(lower, upper):
    a = np.log((.95/.05)**2) / (upper - lower)
    b = np.log(.95/.05) / a + lower
    
    return a, b

def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a*(x-b)))
    
# LANL and AFTAC models for calculating peak pressure
def calc_P_lanl(yieldkt,deltakm):
    P=3.37 + 0.68*np.log10(yieldkt)-1.36*np.log10(deltakm)
    return P

def calc_P_aftac(yieldkt,deltadeg):
    P=0.92+0.5*np.log10(yieldkt)-1.47*np.log10(deltadeg)
    return P

def log_P_to_Pa(P):
    Pa=10**P
    return Pa

def infrasound_detection_probability(theta, sensors):
    # values from notebook
    lower_detection_threshold = 0.36432363
    upper_detection_threshold = 1.54355851
    
    # Set lower threshold val at %5 detection prob and upper threshold val at 95%
    a,b = sigmoid_params(lower_detection_threshold, 
                         upper_detection_threshold)
    
    slat = theta[0]
    slong = theta[1]
    sdepth = theta[2]
    smag = theta[3]
    syield = mag_to_kt(smag)
    
    probs = np.ones(sensors.shape[0])

    for isens in range(0,sensors.shape[0]):
        if sensors[isens, 4] == 1:
            # if sensor type is instant arrival, assume detection
            probs[isens] = 1 - 1e-15 
            
        else:
            delta_deg = geodetics.locations2degrees(slat, slong, sensors[isens, 0], sensors[isens, 1])
            delta_km = degrees2kilometers(delta_deg)

            lanl_peak = log_P_to_Pa(calc_P_lanl(syield, delta_km))
            aftac_peak = log_P_to_Pa(calc_P_aftac(syield, delta_deg))

            if lanl_peak > aftac_peak:
                pressure_high = lanl_peak
                pressure_low = aftac_peak

            else:
                pressure_high = aftac_peak
                pressure_low = lanl_peak

            prob = np.mean(sigmoid(np.linspace(pressure_low, pressure_high), a, b))
            
            probs[isens] = prob
    
    probs[np.where(probs >= 1)[0]] = .99999
    return probs


def infrasound_detection_likelihood(theta, sensors, data):
    probs = infrasound_detection_probability(theta,sensors)
    
    [ndata, ndpt] = data.shape
    nsens = int(ndpt/4)
    
    loglike = np.zeros(ndata)
    
    for idata in range(0,ndata):
        mask = data[idata,nsens:2*nsens]
        loglike[idata] = np.sum(mask*np.log(probs) + (1.0 - mask) * np.log(1.0 - probs))
    return loglike

#Infrasound TT STD Model
def infrasound_tt_std_cal():
    weights = np.array([.054, .090, .856])
    means = np.array([1.0/0.327, 1.0/0.293, 1.0/0.260])
    stds = np.array([0.066, 0.080, 0.330])
    
    gmm_var = np.sum(weights * stds**2) + np.sum(weights * means**2) - np.sum(weights*means)**2
    
    return gmm_var


def infrasound_snr_cal(dist, mag, snroffset, upper_detection_threshold=1.54355851):
    syield = mag_to_kt(mag)
    
    delta_deg = dist
    delta_km = degrees2kilometers(delta_deg)
    dist = delta_deg
    
    lanl_peakvel = calc_P_lanl(syield, delta_km)
    aftac_peakvel = calc_P_aftac(syield, delta_deg)

    lanl_peak = log_P_to_Pa(lanl_peakvel)
    aftac_peak = log_P_to_Pa(aftac_peakvel)
    
    if lanl_peak > aftac_peak:
        pressure_low = aftac_peak
    else:
        pressure_low = lanl_peak
        
    snr = pressure_low / upper_detection_threshold
    logsnr = np.log(snr) + snroffset
    
    return np.exp(logsnr)


def infrasound_meas_std_cal(dist, mag, snroffset):
    #fit from TA Arrray Data nominally, snroffset = 0
    a = 0.90593911
    b = 0.92684044
    c = 5.54237317
        
    # Take minimum signal vs maximum noise, ratio between the two is the SNR
    logsnr = np.log(infrasound_snr_cal(dist, mag, snroffset))

    # SNR model from velasco paper:
    # (add nugget for numerical stablity)
    # logsnr = a*mag + c - b*np.log(delta_deg+10**-10) + snroffset

    #Uncertainty in Phase Arrival Time Picks for Regional Seismic Events: An Experimental Design
    # Velasco et al 2001 (SAND Report)
    #Simple IDC SNR only model

    #Nomial values from "Improving Regional Seismic Event Location in China"
    #Steck et al 2001

    sig0 = 10.0
    gamma = 0.01
    tl = 5
    tu = 50

    if logsnr < np.log(tl):
        sig = sig0
    else:
        if logsnr > np.log(tu):
                sig = gamma*sig0
        else:
                sig = sig0 - (1.0-gamma)*sig0/(np.log(tu)-np.log(tl)) * (logsnr - np.log(tl))

    return sig


def infrasound_compute_tt(theta, sensors):
    # event
    src_lat, src_long, zdepth, src_mag = theta
    # sensors
    rlats, rlongs, sensor_fidelity, *_ = sensors.T
    
    # travel time model
    vel_model = GaussianMixture(n_components=3, covariance_type='spherical')
    weights = np.array([.054, .090, .856])
    means = np.array([1.0/0.327, 1.0/0.293, 1.0/0.260]).reshape((-1,1))
    stds = np.array([0.066, 0.080, 0.330])

    vel_model.weights_ = weights
    vel_model.means_ = means
    vel_model.covariances_ = stds
    vel_model.n_features = 1
    
    #mean and model std and measurment std
    ptime = np.zeros((len(rlats),3))

    for isens, (rlat, rlong, fidelity) in enumerate(zip(rlats, rlongs, sensor_fidelity)):
        delta_deg = geodetics.locations2degrees(src_lat, src_long, rlat, rlong)
        delta_km = degrees2kilometers(delta_deg)
        # Model is defined in s/km, take reciprocal for km/s
        vel = 1/vel_model.sample()[0]
        ptime[isens,0] = vel/delta_km

        model_std = infrasound_tt_std_cal()
        ptime[isens,1] = model_std

        measure_std = infrasound_meas_std_cal(delta_deg, src_mag, fidelity)
        
        ptime[isens,2] = measure_std

    return ptime


#Compute likelhood of each event given dataset


def infrasound_arrival_likelihood_gaussian(theta, sensors, data):
    #compute mean
    tt_data = infrasound_compute_tt(theta, sensors)

    mean_tt = tt_data[:,0]
    stdmodel = tt_data[:,1]
    measurenoise = tt_data[:,2]
    
    #compute corr matrix -- assume independence on infrasound for now
    corr = np.eye(measurenoise.shape[0])
    cov = np.multiply(np.outer(stdmodel,stdmodel),corr) + np.diag(measurenoise**2.0)
    
    [ndata, ndpt] = data.shape
    nsens = int(ndpt/4)
    
    loglike = np.zeros(ndata)
    
    for idata in range(0,ndata):
        mask = data[idata,nsens:2*nsens]
        maskidx = np.nonzero(mask)[0]
        
        #I could decompose this into an iterative operation where I sequentally condition on each sensor? That would maybe be computationally more tractable becuase I could apply to all as update the distributions?
        
        if len(maskidx) > 0:
            #only extracts detections
            means = mean_tt[maskidx]
            dataval = data[idata,maskidx]
            covmat = cov[maskidx[:, None], maskidx]
            
            res = dataval - means
            onevec = np.ones(means.shape)
            a = np.sum(np.linalg.solve(covmat,res))
            b = np.sum(np.linalg.solve(covmat,onevec))
        
        
            #THESE LINES ARE Maybe wrong BECAUSE THE DISTRIBUTION IS IMPROPER
            logdets = np.linalg.slogdet(covmat)[1]
            loglike[idata] = -0.5*np.sum(np.multiply(res, np.linalg.solve(covmat,res))) + (1.0 - len(maskidx))/2.0*np.log(2*np.pi) - 0.5*logdets - 0.5*np.log(b) + np.divide(a**2, 2.0*b)
    return loglike


def compute_kappa(N, snr, delta=2):
    return 1
    if N == 1:
        kappa_0  = 1 # Control for single sensors...update this to be actually principled
    print('compute kappa N:', N)
    print('compute kappa SNR:', snr)
    F = N * snr + 1
    kappa_0 = 2 * (N - 1)/N * (F - 1)
    
    cos_phihalf = 1 - np.log(2)/kappa_0
    if cos_phihalf < -1:
        cos_phihalf = 1
    if cos_phihalf > 1:
        cos_phihalf = 1
    print('cos phihalf:', cos_phihalf)
    sin_phihalf = np.sqrt(1 - cos_phihalf**2)
    
    kappa = np.log(2) / (1 - cos_phihalf*np.cos(delta) + sin_phihalf*np.sin(delta))
    print('compute kappa value:', kappa)
    return kappa


def infrasound_compute_incident(theta, sensors):
    src_lat, src_long, zdepth, src_mag = theta
    # sensors
    rlats, rlongs, sensor_fidelity, *_ = sensors.T
    model = TauPyModel(model="iasp91")
    angle = np.zeros((len(rlats) ,2))
    N = sensors.shape[0]

    for isens, (rlat, rlong, fidelity) in enumerate(zip(rlats, rlongs, sensor_fidelity)):
        arrivals = model.get_travel_times_geo(source_depth_in_km=zdepth, source_latitude_in_deg=src_lat, 
                                       source_longitude_in_deg=src_long, receiver_latitude_in_deg=rlat, 
                                       receiver_longitude_in_deg=rlong, phase_list=["P","p"])
        
        angle[isens,0] = arrivals[0].incident_angle *np.pi/180
        
        delta_deg = geodetics.locations2degrees(src_lat, src_long, rlat, rlong)
        snr = infrasound_snr_cal(delta_deg, src_mag, fidelity)
        
        angle[isens,1] = compute_kappa(N, snr)
    return angle


def infrasound_incident_likelihood(theta,sensors,data):
    angle_data = infrasound_compute_incident(theta, sensors)
    [ndata, ndpt] = data.shape
    nsens = int(ndpt/4)

    loglike = np.zeros(ndata)
    
    for idata, mask in enumerate(data[:,nsens:2*nsens]):
        maskidx = np.nonzero(mask)[0]
        
        if len(maskidx) == 0:
            continue
            
        detected_angledata = angle_data[maskidx]
        means = detected_angledata[:,0]
        kappas = detected_angledata[:,1]

        local_likes = []
        detected_data = data[idata, 3*nsens:][maskidx]

        for imask in range(len(detected_data)):
            dist = stats.vonmises(kappas[imask], loc=means[imask])
            const = dist.cdf(np.pi) - dist.cdf(0)
            tmp = dist.logpdf(detected_data[imask]) - np.log(const)
            if np.isnan(tmp):
                print(f'NAN in infrasound (inc) likelihood, kappa {kappas[imask]}, mean {means[imask]}, val {detected_data[imask]}')

            local_likes.append(tmp)
        
        loglike[idata] = sum(local_likes)
    
    return loglike


def infrasound_compute_azimuth(theta, sensors):
    src_lat, src_long, zdepth, src_mag = theta
    # sensors
    rlats, rlongs, sensor_fidelity, *_ = sensors.T
    azmth = np.zeros((len(rlats) ,2))

    N = sensors.shape[0]

    for isens, (rlat, rlong, fidelity) in enumerate(zip(rlats, rlongs, sensor_fidelity)):
        curr_azmth = gps2dist_azimuth(rlat, rlong, src_lat, src_long)[1]
        
        azmth[isens,0] = curr_azmth * np.pi/180
        
        delta_deg = geodetics.locations2degrees(src_lat, src_long, rlat, rlong)
        snr = infrasound_snr_cal(delta_deg, src_mag, fidelity)
        
        azmth[isens,1] = compute_kappa(N, snr)
        
    return azmth


def infrasound_azimuth_likelihood(theta,sensors,data):
    azmth_data = infrasound_compute_azimuth(theta, sensors)
    [ndata, ndpt] = data.shape
    nsens = int(ndpt/4)
    loglike = np.zeros(ndata)
    
    for idata, mask in enumerate(data[:,nsens:2*nsens]):
        maskidx = np.nonzero(mask)[0]
        
        if len(maskidx) == 0:
            continue
            
        detected_azmthdata = azmth_data[maskidx]
        means = detected_azmthdata[:,0]
        kappas = detected_azmthdata[:,1]

        local_likes = []
        detected_data = data[idata, 2*nsens:3*nsens][maskidx]

        for imask in range(len(detected_data)):
            dist = stats.vonmises(kappas[imask], loc=means[imask])
            tmp = dist.logpdf(detected_data[imask])
            local_likes.append(tmp)
            if np.isinf(tmp):
                print(f'INF in infrasound likelihood, kappa {kappas[imask]}, mean {means[imask]}, val {detected_data[imask]}')
    
        loglike[idata] = sum(local_likes)
        
    return loglike


"""
Models for Seismic Arrays
-------------------------------------------------
-------------------------------------------------
"""
def array_compute_incident(theta, sensors):
    src_lat, src_long, zdepth, src_mag = theta
    # sensors
    rlats, rlongs, sensor_fidelity, *_ = sensors.T
    model = TauPyModel(model="iasp91")
    angle = np.zeros((len(rlats) ,2))
    N = sensors.shape[0]

    for isens, (rlat, rlong, fidelity) in enumerate(zip(rlats, rlongs, sensor_fidelity)):
        arrivals = model.get_travel_times_geo(source_depth_in_km=zdepth, source_latitude_in_deg=src_lat, 
                                       source_longitude_in_deg=src_long, receiver_latitude_in_deg=rlat, 
                                       receiver_longitude_in_deg=rlong, phase_list=["P","p"])
        
        angle[isens,0] = arrivals[0].incident_angle *np.pi/180
        
        delta_deg = geodetics.locations2degrees(src_lat, src_long, rlat, rlong)
        snr = infrasound_snr_cal(delta_deg, src_mag, fidelity)
#         logsnr = seismic_snr_cal(delta_deg, src_mag, fidelity)
        
#         angle[isens,1] = compute_kappa(N, np.exp(logsnr))
        angle[isens,1] = compute_kappa(N, snr)
    return angle


def array_incident_likelihood(theta,sensors,data):
    angle_data = array_compute_incident(theta, sensors)
    [ndata, ndpt] = data.shape
    nsens = int(ndpt/4)

    loglike = np.zeros(ndata)
    
    for idata, mask in enumerate(data[:,nsens:2*nsens]):
        maskidx = np.nonzero(mask)[0]
        
        if len(maskidx) == 0:
            continue
            
        detected_angledata = angle_data[maskidx]
        means = detected_angledata[:,0]
        kappas = detected_angledata[:,1]

        local_likes = []
        detected_data = data[idata, 3*nsens:][maskidx]

        for imask in range(len(detected_data)):
            dist = stats.vonmises(kappas[imask], loc=means[imask])
            const = dist.cdf(np.pi) - dist.cdf(0)
            tmp = dist.logpdf(detected_data[imask]) - np.log(const)
            if np.isinf(tmp):
                print(f'INF in array (inc) likelihood, kappa {kappas[imask]}, mean {means[imask]}, val {detected_data[imask]}')
            local_likes.append(tmp)
        
        loglike[idata] = sum(local_likes)
    
    return loglike


def array_compute_azimuth(theta, sensors):
    src_lat, src_long, zdepth, src_mag = theta
    # sensors
    rlats, rlongs, sensor_fidelity, *_ = sensors.T
    azmth = np.zeros((len(rlats) ,2))

    N = sensors.shape[0]

    for isens, (rlat, rlong, fidelity) in enumerate(zip(rlats, rlongs, sensor_fidelity)):
        curr_azmth = gps2dist_azimuth(rlat, rlong, src_lat, src_long)[1]
        
        azmth[isens,0] = curr_azmth * np.pi/180
        
        delta_deg = geodetics.locations2degrees(src_lat, src_long, rlat, rlong)
#         logsnr = seismic_snr_cal(delta_deg, src_mag, fidelity)
        snr = infrasound_snr_cal(delta_deg, src_mag, fidelity)
#         azmth[isens,1] = compute_kappa(N, np.exp(logsnr))
        azmth[isens,1] = compute_kappa(N, snr)
        
    return azmth


def array_azimuth_likelihood(theta,sensors,data):
    azmth_data = array_compute_azimuth(theta, sensors)
    [ndata, ndpt] = data.shape
    nsens = int(ndpt/4)

    loglike = np.zeros(ndata)
    
    for idata, mask in enumerate(data[:,nsens:2*nsens]):
        maskidx = np.nonzero(mask)[0]
        
        if len(maskidx) == 0:
            continue
            
        detected_azmthdata = azmth_data[maskidx]
        means = detected_azmthdata[:,0]
        kappas = detected_azmthdata[:,1]

        local_likes = []
        detected_data = data[idata, 2*nsens:3*nsens][maskidx]
        for imask in range(len(detected_data)):
            dist = stats.vonmises(kappas[imask], loc=means[imask])
            tmp = dist.logpdf(detected_data[imask])
            local_likes.append(tmp)
            if np.isinf(tmp):
                (f'INF in array likelihood, kappa {kappas[imask]}, mean {means[imask]}, val {detected_data[imask]}')
        loglike[idata] = sum(local_likes)
        
    return loglike


"""
Detection likelihood models
-------------------------------------------------
-------------------------------------------------
"""
def extract_sensor_data(sensor_idx, total_data):
    """
    Function that extracts data corresponding to a specific set of sensors from
    from the total dataset and compiles it into a single dataset
    
    Inputs
    ------
    sensor_idx (list)    : list of sensor indices to extract data for
    total_data (ndarray) : array of shape (number of data realizations x 4 * total number of sensors) 
                           that contains all data generated
                           
    Returns
    -------
    sensor_data (ndarray) : array of shape (number of data realizations x 4 * number of specific sensors)
                            that contains only data generated by those specific sensors
    """
    nsens = len(sensor_idx)
    ntotal_sens = int(total_data.shape[1]/4)
    ndata = total_data.shape[0]
    sensor_data = np.zeros((ndata, nsens*4))
    
    # Extract arrivals
    sensor_data[:, :nsens] = total_data[:,sensor_idx]
    # Extract detections
    sensor_data[:, nsens:2*nsens] = total_data[:,ntotal_sens + sensor_idx]
    # Extract azimuths
    sensor_data[:, 2*nsens:3*nsens] = total_data[:, 2*ntotal_sens + sensor_idx]
    # Extract incident angles
    sensor_data[:, 3*nsens:] = total_data[:, 3*ntotal_sens + sensor_idx]
    
    return sensor_data
    

def compute_seismic_loglikes(theta,sensors,data):
    detect_loglikes = seismic_detection_likelihood(theta,sensors,data)
    arrival_loglikes = seismic_arrival_likelihood_gaussian(theta, sensors, data)

    loglikes = detect_loglikes + arrival_loglikes
    
#     return loglikes
    return loglikes


def compute_instant_loglikes(theta,sensors,data):
    detect_loglikes = instant_detection_likelihood(theta,sensors,data)
    arrival_loglikes = instant_arrival_likelihood_gaussian(theta, sensors, data)
    
    loglikes = detect_loglikes + arrival_loglikes
    
#     return loglikes
    return loglikes


def compute_infrasound_loglikes(theta,sensors,data):
    detect_loglikes = infrasound_detection_likelihood(theta,sensors,data)
    arrival_loglikes = infrasound_arrival_likelihood_gaussian(theta, sensors, data)
    incident_loglikes = infrasound_incident_likelihood(theta, sensors, data)
    azimuth_loglikes = infrasound_azimuth_likelihood(theta, sensors, data)
    
    loglikes = detect_loglikes + arrival_loglikes + incident_loglikes + azimuth_loglikes
    
#     return loglikes
    return loglikes


def compute_array_loglikes(theta,sensors,data):
    detect_loglikes = seismic_detection_likelihood(theta,sensors,data)
    arrival_loglikes = seismic_arrival_likelihood_gaussian(theta, sensors, data)
    incident_loglikes = array_incident_likelihood(theta, sensors, data)
    azimuth_loglikes = array_azimuth_likelihood(theta, sensors, data)

    loglikes = detect_loglikes + arrival_loglikes + incident_loglikes + azimuth_loglikes
    
#     return loglikes
    return loglikes

def compute_loglikes(theta,sensors,data):
    # split out sensor types into their own arrays and call relevant functions
    # This assumes independent sensor types
    seismic_idx = np.where(sensors[:,4]==0)[0]
    instant_idx = np.where(sensors[:,4]==1)[0]
    infrasound_idx = np.where(sensors[:,4]==2)[0]
    array_idx = np.where(sensors[:,4]==3)[0]
    
    seismic_sensors = sensors[seismic_idx] # Seismic arrival sensors and data
    seismic_data = extract_sensor_data(seismic_idx, data)
    
    instant_sensors = sensors[instant_idx] # Instant arrival sensors and data
    instant_data = extract_sensor_data(instant_idx, data)
    
    infrasound_sensors = sensors[infrasound_idx] # Infrasound sensors and data
    infrasound_data = extract_sensor_data(infrasound_idx, data)
    
    array_sensors = sensors[array_idx] # Seismic arrays and data
    array_data = extract_sensor_data(array_idx, data)
    
    seismic_loglikes = compute_seismic_loglikes(theta, seismic_sensors, seismic_data)
    instant_loglikes = compute_instant_loglikes(theta, instant_sensors, instant_data)
    infrasound_loglikes = compute_infrasound_loglikes(theta, infrasound_sensors, infrasound_data)
    array_loglikes = compute_array_loglikes(theta, array_sensors, array_data)
#     print('array:', array_loglikes)
#     print('seismic:', seismic_loglikes)
#     print('instant:', instant_loglikes)
#     print('infrasound:', infrasound_loglikes)
    
    loglikes = seismic_loglikes + instant_loglikes + infrasound_loglikes + array_loglikes
    
    return loglikes
