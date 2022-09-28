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
        if sensors[isens, 4] == 1:
            # if sensor type is instant arrival, assume detection
            probs[isens] = 1 - 1e-15    
        else:
            delta = geodetics.locations2degrees(slat, slong, sensors[isens][0], sensors[isens][1])
            x = np.asarray([sdepth, delta, smag]).reshape(1, -1)
            probs[isens] = model.predict_proba(x)[0][1]
    return probs


#Improved TT STD Model
def tt_std_cal(depth, dist):
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

#magnitude dependeint measurment error
def meas_std_cal(dist, mag, snroffset):
    #fit from TA Arrray Data nominally, snroffset = 0
    a = 0.90593911
    b = 0.92684044
    c = 5.54237317
    
    #add nugget for numerical stablity
    logsnr = a*mag + c - b*np.log(dist+10**-10) + snroffset
    
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

def compute_corr(theta, sensors):
    rlats = sensors[:,0]
    rlongs = sensors[:,1]
    corr = np.eye(len(rlats))
    
    lscal = 147.5
    
    for isens in range(0,len(rlats)):
        for jsens in range(isens,len(rlats)):
            #147.5 was fit from the 1D profiles
            if sensors[isens, 4] == 1:
                # Make instant arrival sensors uncorrelated with all others
                if isens == jsens:
                    corr[isens, jsens] = 1
                else:
                    corr[isens, jsens] = 0
                    corr[jsens, isens] = 0

            azmthdata = geodetics.gps2dist_azimuth(rlats[isens], rlongs[isens], rlats[jsens], rlongs[jsens])
            corr[isens,jsens] = np.exp(-1.0/(lscal**2) * (azmthdata[0]/1000.0)**2)
            corr[jsens,isens] = np.exp(-1.0/(lscal**2) * (azmthdata[0]/1000.0)**2)
            
    return corr


def compute_tt(theta, sensors):
    model = TauPyModel(model="iasp91")
    src_lat = theta[0]
    src_long = theta[1]
    zdepth = theta[2]
    src_mag = theta[3]

    rlats = sensors[:,0]
    rlongs = sensors[:,1]
    sensor_fidelity = sensors[:,2]
    
    #mean and model std and measurment std
    ptime = np.zeros((len(rlats),3))
    #iangle = np.zeros(len(rlats))
    #azmth = np.zeros(len(rlats))

    for isens in range(0,len(rlats)):
        rlat = rlats[isens]
        rlong = rlongs[isens]

        #models
        arrivals = model.get_travel_times_geo(source_depth_in_km=zdepth, source_latitude_in_deg=src_lat, source_longitude_in_deg=src_long, receiver_latitude_in_deg=rlat, receiver_longitude_in_deg=rlong, phase_list=["P","p"])
        #azmthdata = geodetics.gps2dist_azimuth(src_lat, src_long, rlat, rlong)
        #record data from the first arrival. Assume always in the azimuthal plane.
        if sensors[isens, 4] == 1:
            # instant arrival time for sensors of that type
            ptime[isens, 0] = 0
        else:
            ptime[isens,0] = arrivals[0].time
        #iangle[isens] = arrivals[0].incident_angle
        #azmth[isens] = azmthdata[1]

        deltakm = geodetics.degrees2kilometers(geodetics.locations2degrees(src_lat, src_long, rlat, rlong))
        if sensors[isens,4] ==1:
            model_std = 0.001
        else:
            model_std = tt_std_cal(zdepth, deltakm)
        
        ptime[isens,1] = model_std

        if sensors[isens, 4] == 1:
            # give instant arrival sensors negligible measurement error
            measure_std = .001
        else:
            measure_std = meas_std_cal(deltakm, src_mag, sensor_fidelity[isens])
        
        ptime[isens,2] = measure_std

    return ptime



#Compute likelhood of each event given dataset


def arrival_likelihood_gaussian(theta, sensors, data):
    #compute mean
    tt_data = compute_tt(theta, sensors)

    mean_tt = tt_data[:,0]
    stdmodel = tt_data[:,1]
    measurenoise = tt_data[:,2]
    
    #compute corr matrix
    corr = compute_corr(theta, sensors)
    cov = np.multiply(np.outer(stdmodel,stdmodel),corr) + np.diag(measurenoise**2.0)
    
    [ndata, ndpt] = data.shape
    nsens = int(ndpt/2)
    
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
    
    [ndata, ndpt] = data.shape
    nsens = int(ndpt/2)
    
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
