import numpy as np

from obspy import geodetics
from obspy.taup import TauPyModel

from sklearn.linear_model import LogisticRegression


def detection_probability(theta,sensors):
    slat = theta[0]
    slong = theta[1]
    sdepth = theta[2]
    smag = theta[3]

    model = LogisticRegression()
    model.classes_ = np.asarray([0, 1])
    model.coef_ = np.asarray([[-0.0297509 , -2.81876856,  1.14055126]])
    model.intercept_ = np.asarray(1.95254635)
    
    
    probs = np.ones(sensors.shape[0])
    
    for isens in range(0,sensors.shape[0]):
            delta = geodetics.locations2degrees(slat, slong, sensors[isens][0], sensors[isens][1])
            x = np.asarray([sdepth, delta, smag]).reshape(1, -1)
            probs[isens] = model.predict_proba(x)[0][1]
    return probs

def compute_mean_tt(theta, sensors):
    model = TauPyModel(model="iasp91")
    src_lat = theta[0]
    src_long = theta[1]
    zdepth = theta[2]

    rlats = sensors[:,0]
    rlongs = sensors[:,1]
    
    ptime = np.zeros(len(rlats))
    #iangle = np.zeros(len(rlats))
    #azmth = np.zeros(len(rlats))

    for isens in range(0,len(rlats)):
        rlat = rlats[isens]
        rlong = rlongs[isens]

        #models
        arrivals = model.get_travel_times_geo(source_depth_in_km=zdepth, source_latitude_in_deg=src_lat, source_longitude_in_deg=src_long, receiver_latitude_in_deg=rlat, receiver_longitude_in_deg=rlong, phase_list=["P","p"])
        #azmthdata = geodetics.gps2dist_azimuth(src_lat, src_long, rlat, rlong)

        #record data from the first arrival. Assume always in the azimuthal plane.
        ptime[isens] = arrivals[0].time
        #iangle[isens] = arrivals[0].incident_angle
        #azmth[isens] = azmthdata[1]
        
    return ptime

def compute_corr(theta, sensors):
    rlats = sensors[:,0]
    rlongs = sensors[:,1]
    corr = np.eye(len(rlats))
    
    lscal = 147.5
    
    for isens in range(0,len(rlats)):
        for jsens in range(isens,len(rlats)):
            #147.5 was fit from the 1D profiles
            azmthdata = geodetics.gps2dist_azimuth(rlats[isens], rlongs[isens], rlats[jsens], rlongs[jsens])
            corr[isens,jsens] = np.exp(-1.0/(lscal**2) * (azmthdata[0]/1000.0)**2)
            corr[jsens,isens] = np.exp(-1.0/(lscal**2) * (azmthdata[0]/1000.0)**2)
            
    return corr



#Compute likelhood of each event given dataset


def arrival_likelihood_gaussian(theta, sensors, data):
    #compute mean
    mean_tt = compute_mean_tt(theta, sensors)
    
    #compute corr matrix
    corr = compute_corr(theta, sensors)
    
    measurenoise = sensors[:,2]
    stdmodel = (2.75758229e-02)*mean_tt + (-5.57985096e-04)*(mean_tt**2.0) + (1.63610033e-05)*(mean_tt**3.0)
    
    cov = np.multiply(np.outer(stdmodel,stdmodel),corr) + np.diag(measurenoise**2.0)
    
    [ndata, ndpt] = data.shape
    nsens = np.int(ndpt/2)
    
    loglike = np.zeros(ndata)
    
    for idata in range(0,ndata):
        mask = data[idata,nsens:]
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

def detection_likelihood(theta, sensors, data):
    probs = detection_probability(theta,sensors)
    print(probs)
    
    [ndata, ndpt] = data.shape
    nsens = np.int(ndpt/2)
    
    loglike = np.zeros(ndata)
    
    for idata in range(0,ndata):
        mask = data[idata,nsens:]
        loglike[idata] = np.sum(mask*np.log(probs) + (1.0 - mask) * np.log(1.0 - probs))
        return loglike


def compute_loglikes(theta,sensors,data):
    dloglikes = detection_likelihood(theta,sensors,data)
    aloglikes = arrival_likelihood_gaussian(theta, sensors, data)
    loglikes = dloglikes + aloglikes
    
    return loglikes