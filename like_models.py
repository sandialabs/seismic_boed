import os
import pickle
import warnings
from pathlib import Path

import ml_utils
import numpy as np
from obspy import geodetics
from obspy.geodetics.base import (
    degrees2kilometers,
    gps2dist_azimuth,
    kilometers2degrees,
)
from obspy.taup import TauPyModel
from obspy.taup.taup_geo import calc_dist
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")


def _resolve_repo_root():
    env_root = os.environ.get("SEISMIC_OED_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parent


BASE_DIR = _resolve_repo_root()
CUBIC_PATH = BASE_DIR / "cubic_reg.pkl"
TTSTD_PATH = BASE_DIR / "TTstd.pkl"

_POWER_GATE_DIAGNOSTICS = {
    "total_checked": 0,
    "failed_lat_local": 0,
    "failed_lon_local": 0,
    "failed_depth_m": 0,
    "failed_gaussian_variance": 0,
    "failed_mt_norm": 0,
    "power_disabled_count": 0,
}


def reset_power_gate_diagnostics():
    for key in _POWER_GATE_DIAGNOSTICS:
        _POWER_GATE_DIAGNOSTICS[key] = 0


def get_power_gate_diagnostics():
    return dict(_POWER_GATE_DIAGNOSTICS)



# --------------------------------------------------------------------------
# Power Model Helpers (The Heteroscedastic Noise Logic)
# --------------------------------------------------------------------------
def observation_variance_power(
    log_power, sigma_min=0.1, sigma_max=2.0, log_power_ref=-15.5, decay_rate=0.5
):
    """
    Computes additional observation variance that depends on signal strength.
    Low signal (<-15.5) -> High variance (sigma_max)
    High signal (>-15.5) -> Low variance (sigma_min)
    """
    sigma = sigma_min + (sigma_max - sigma_min) / (
        1.0 + np.exp(decay_rate * (log_power - log_power_ref))
    )
    return sigma**2


def compute_power(theta, sensors, stype):
    """
    Computes predicted power using the MLP and returns error components.

    Returns
    -------
    result : np.array (n_sensors x 4)
        [predicted_log_power, model_sigma, measurement_sigma, enabled_mask]
    """
    n_sensors = sensors.shape[0]
    if stype not in ["seismic", "array"] or n_sensors == 0:
        return np.column_stack(
            [
                np.full(n_sensors, np.nan),
                np.full(n_sensors, np.nan),
                np.full(n_sensors, np.nan),
                np.zeros(n_sensors, dtype=bool),
            ]
        )

    map_strategy = os.environ.get("SEISMIC_OED_FIDELITY_MAP", "clamped_linear")
    gaussian_variance = ml_utils.map_sensor_fidelity_to_gaussian_variance(
        sensors[:, 2], strategy=map_strategy
    )

    enabled_mask, gate_masks, _ = ml_utils.evaluate_power_domain_gates(
        theta,
        sensors,
        gaussian_variance=gaussian_variance,
    )

    _POWER_GATE_DIAGNOSTICS["total_checked"] += int(n_sensors)
    _POWER_GATE_DIAGNOSTICS["failed_lat_local"] += int(np.count_nonzero(~gate_masks["lat_local"]))
    _POWER_GATE_DIAGNOSTICS["failed_lon_local"] += int(np.count_nonzero(~gate_masks["lon_local"]))
    _POWER_GATE_DIAGNOSTICS["failed_depth_m"] += int(np.count_nonzero(~gate_masks["depth_m"]))
    _POWER_GATE_DIAGNOSTICS["failed_gaussian_variance"] += int(
        np.count_nonzero(~gate_masks["gaussian_variance"])
    )
    _POWER_GATE_DIAGNOSTICS["failed_mt_norm"] += int(np.count_nonzero(~gate_masks["mt_norm"]))
    _POWER_GATE_DIAGNOSTICS["power_disabled_count"] += int(np.count_nonzero(~enabled_mask))

    pred_log_power = np.full(n_sensors, np.nan, dtype=float)
    model_sigma = np.full(n_sensors, np.nan, dtype=float)
    measure_sigma = np.full(n_sensors, np.nan, dtype=float)

    if np.any(enabled_mask):
        if ml_utils.TORCH_AVAILABLE:
            power_model = ml_utils.get_power_model()
            pred_log_power[enabled_mask] = power_model.predict_log_power(
                theta,
                sensors[enabled_mask],
                gaussian_variance=gaussian_variance[enabled_mask],
            )
            model_sigma[enabled_mask] = 0.3
            measure_sigma[enabled_mask] = np.sqrt(gaussian_variance[enabled_mask])

    return np.column_stack([pred_log_power, model_sigma, measure_sigma, enabled_mask])


def power_likelihood(theta, sensors, data, stype):
    """
    Computes Log-Likelihood for Power data (Column Block 5).
    Uses a Heteroscedastic Gaussian on Log-Power.
    """
    # Get predictions and base errors
    # result shape: (N_sensors, 3) -> [pred, sigma_model, sigma_measure]
    power_components = compute_power(theta, sensors, stype)

    pred_log_power = power_components[:, 0]
    sigma_model = power_components[:, 1]
    sigma_measure = power_components[:, 2]
    enabled_mask = power_components[:, 3].astype(bool)

    # Calculate the Signal-Dependent Noise (The "Noise Floor" Effect)
    # This adds variance when the signal is weak
    var_hetero = observation_variance_power(pred_log_power)

    # Total Variance = Model^2 + Measure^2 + SignalDep^2
    sigma_sq_total = sigma_model**2 + sigma_measure**2 + var_hetero

    # Prepare Data Extraction
    # Data shape is (ndata, Total_Cols)
    # We need to find where the power data lives.
    # Architecture: [Arrivals(N) | Detect(N) | Az(N) | Inc(N) | POWER(N)]
    [ndata, ndpt] = data.shape
    nsens = int(ndpt / 5)  # Updated stride to 5

    loglike = np.zeros(ndata)

    # Loop over data realizations (vectorizing this is possible but complex due to masking)
    for idata in range(ndata):
        # Power data is in the 5th block (indices 4*nsens to 5*nsens)
        # However, we only evaluate power if the sensor DETECTED the event.
        # Check Detections (Block 2: indices nsens to 2*nsens)
        detection_mask = data[idata, nsens : 2 * nsens].astype(bool)
        finite_mask = np.isfinite(pred_log_power) & np.isfinite(sigma_sq_total)
        maskidx = np.nonzero(detection_mask & enabled_mask & finite_mask)[0]

        if len(maskidx) == 0:
            continue

        # Extract observed power for detecting sensors
        obs_power = data[idata, 4 * nsens : 5 * nsens][maskidx]

        # Extract predictions for detecting sensors
        curr_pred = pred_log_power[maskidx]
        curr_var = sigma_sq_total[maskidx]

        # Gaussian Log Likelihood
        # -0.5 * log(2pi * sigma^2) - 0.5 * (obs - pred)^2 / sigma^2
        ll_terms = (
            -0.5 * np.log(2 * np.pi * curr_var)
            - 0.5 * (obs_power - curr_pred) ** 2 / curr_var
        )

        loglike[idata] = np.sum(ll_terms)

    return loglike


# --------------------------------------------------------------------------
# Existing Helper Functions (Unchanged but included for context)
# --------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    correction = 0.9996400
    r = 6372.8 * 1000 * correction
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def seismic_snr_cal(dist, mag, snroffset):
    """Compute SNR from model fit to TA Array Data"""
    a = 0.90593911
    b = 0.92684044
    c = 5.54237317
    return a * mag + c - b * np.log(dist + 10**-10) + snroffset


# Helper functions for infrasound models
# --------------------------------------
def mag_to_kt(mag):
    """Convert local magnitude to kilotons"""
    # values fit for local mag
    d, f = [3.909197277671451, 2.998020624687433]
    energy_J = 10 ** (d * mag + f)
    energy_tJ = energy_J / 1e12
    energy_kT = energy_tJ / 4.184

    return energy_kT


def sigmoid_params(lower, upper):
    """
    Compute parameters that define sigmoid function with 'lower' value evaluating
    to .05 and 'uppper' value evaluating to .95.
    """
    a = np.log((0.95 / 0.05) ** 2) / (upper - lower)
    b = np.log(0.95 / 0.05) / a + lower

    return a, b


def sigmoid(x, a, b):
    """
    Sigmoid function for evaluating detection likelihoods
    """
    return 1.0 / (1.0 + np.exp(-a * (x - b)))


# LANL and alt models for calculating peak pressure
def calc_P_lanl(yieldkt, delta_km):
    """
    Computes peak pressure according to LANL model, see
    'Infrasonic Monitoring' by Whitaker, 1995, in Proceedings of the 17th Annual
    Seismic Research Symposium
    """
    P = 3.37 + 0.68 * np.log10(yieldkt) - 1.36 * np.log10(delta_km)
    return P


def calc_P_alt(yieldkt, deltadeg):
    """
    Computes peak pressure according to alternative model, see
    'Capability Estimation of Infrasound Networks"' by Clauter and Brandford,
    1995, AFTAC Report
    """
    P = 0.92 + 0.5 * np.log10(yieldkt) - 1.47 * np.log10(deltadeg)
    return P


def log_P_to_Pa(P):
    "Convert log peak pressure to peak pressure"
    Pa = 10**P
    return Pa


def infrasound_snr_cal(delta_km, mag, snroffset, upper_detection_threshold=1.54355851):
    """Compute SNR for infrasound models as ratio between minimum of LANL signal
    model signal and alt model signal and maximum noise"""
    syield = mag_to_kt(mag)

    delta_deg = kilometers2degrees(delta_km)

    lanl_peakvel = calc_P_lanl(syield, delta_km)
    alt_peakvel = calc_P_alt(syield, delta_deg)

    lanl_peak = log_P_to_Pa(lanl_peakvel)
    alt_peak = log_P_to_Pa(alt_peakvel)

    if lanl_peak > alt_peak:
        pressure_low = alt_peak
    else:
        pressure_low = lanl_peak

    snr = pressure_low / upper_detection_threshold
    logsnr = np.log(snr) + snroffset

    return logsnr


def meas_std_cal(delta_km, mag, snroffset, stype):
    """
    Compute uncertainty in phase arrival picks, model from
    Uncertainty in Phase Arrival Time Picks for Regional Seismic Events:
    An Experimental Design by Velasco et al 2001 (Sand Report)

    Nominal values from "Improving Regional Seismic Event Location in China by
    Steck et al 2001
    """
    if stype == "infrasound":
        logsnr = infrasound_snr_cal(delta_km, mag, snroffset)
    elif stype == "seismic":
        logsnr = seismic_snr_cal(delta_km, mag, snroffset)

    sig0 = 10.0
    gamma = 0.01
    tl = 5
    tu = 50

    if logsnr < np.log(tl):
        sig = sig0
    else:
        if logsnr > np.log(tu):
            sig = gamma * sig0
        else:
            sig = sig0 - (1.0 - gamma) * sig0 / (np.log(tu) - np.log(tl)) * (
                logsnr - np.log(tl)
            )

    return sig


def compute_kappa(N, snr, delta=2):
    """Compute concentration kappa for von Mises distributions based on model
    from 'Improved Bayesian Infrasonic Source Localization for regional
    infrasound' by Blom et al., 2015"""
    if N == 1:
        kappa_0 = (
            1  # Control for single sensors...update this to be actually principled
        )

    F = N * snr + 1
    kappa_0 = 2 * (N - 1) / N * (F - 1)

    cos_phihalf = 1 - np.log(2) / kappa_0
    if cos_phihalf < -1:
        cos_phihalf = 1
    if cos_phihalf > 1:
        cos_phihalf = 1
    sin_phihalf = np.sqrt(1 - cos_phihalf**2)

    kappa = np.log(2) / (1 - cos_phihalf * np.cos(delta) + sin_phihalf * np.sin(delta))
    return kappa


# Detection likelihoods
# -------------------------
def detection_probability(theta, sensors, stype="seismic"):
    """
    Compute probability of detections at each sensor given theta

    Inputs
    ------
    theta (np.array) : (m x 4) array containing parameters for each seismic event
    sensors (np.array) : (n x 5) array containing parameters for each seismic sensor
    stype (str) : type of sensor model to use. One of 'seismic', 'instant',
                  'infrasound', or 'array'.

    Returns
    -------
    probs (np.array) : (n x m) array containing the probability of a sensor (row)
                       detecting a given seismic event (column).
    """
    if stype in ["seismic", "array"]:
        slat = theta[0]
        slong = theta[1]
        sdepth = theta[2]
        smag = theta[3]

        model = LogisticRegression()
        model.classes_ = np.asarray([0, 1])

        # Params for model trained on US TA data (in Utah)
        model.coef_ = np.asarray([[-0.0297509, -2.81876856, 1.14055126]])
        model.intercept_ = np.asarray(1.95254635)

        probs = np.ones(sensors.shape[0])

        for isens in range(0, sensors.shape[0]):
            delta = geodetics.locations2degrees(
                slat, slong, sensors[isens][0], sensors[isens][1]
            )
            x = np.asarray([sdepth, delta, smag]).reshape(1, -1)
            probs[isens] = model.predict_proba(x)[0][1]

    elif stype == "instant":
        # if sensor type is instant arrival, assume detection
        probs = np.ones(sensors.shape[0]) * 1 - 1e-15

    elif stype == "infrasound":
        # values from notebook
        lower_detection_threshold = 0.36432363
        upper_detection_threshold = 1.54355851

        # Set lower threshold val at %5 detection prob and upper threshold val at 95%
        a, b = sigmoid_params(lower_detection_threshold, upper_detection_threshold)

        slat = theta[0]
        slong = theta[1]
        sdepth = theta[2]
        smag = theta[3]
        syield = mag_to_kt(smag)

        probs = np.ones(sensors.shape[0])

        for isens in range(0, sensors.shape[0]):
            if sensors[isens, 4] == 1:
                # if sensor type is instant arrival, assume detection
                probs[isens] = 1 - 1e-15

            else:
                delta_deg = geodetics.locations2degrees(
                    slat, slong, sensors[isens, 0], sensors[isens, 1]
                )
                delta_km = degrees2kilometers(delta_deg)

                lanl_peak = log_P_to_Pa(calc_P_lanl(syield, delta_km))
                alt_peak = log_P_to_Pa(calc_P_alt(syield, delta_deg))

                if lanl_peak > alt_peak:
                    pressure_high = lanl_peak
                    pressure_low = alt_peak

                else:
                    pressure_high = alt_peak
                    pressure_low = lanl_peak

                prob = np.mean(sigmoid(np.linspace(pressure_low, pressure_high), a, b))

                probs[isens] = prob

    else:
        raise ValueError(
            f"Cannot compute detection probabilities for sensor type {stype}"
        )

    probs[np.where(probs >= 1)[0]] = 0.99999
    return probs


def detection_likelihood(theta, sensors, data, stype):
    """
    Compute the loglikelihood of the given data being generated by the sensors
    under the given seismic event

    Inputs
    ------
    theta (np.array) : (m x 4) array containing parameters for each seismic event
    sensors (np.array) : (n x 5) array containing parameters for each seismic sensor
    data (np.array) : (m*ndata x n) array containing the data generated by each sensor
    stype (str) : type of sensor model to use. One of 'seismic', 'instant',
                  'infrasound', or 'array'.

    Returns
    -------
    loglike (np.array) : (m*ndata x n) array containing the loglikelihood of the
                         data being generated by the sensor under the given
                         seismic event
    """
    # probs = detection_probability(theta, sensors, stype)
    probs = detection_probability(theta, sensors, "seismic")

    [ndata, ndpt] = data.shape
    nsens = int(ndpt / 5)

    loglike = np.zeros(ndata)

    for idata in range(0, ndata):
        mask = data[idata, nsens : 2 * nsens]
        loglike[idata] = np.sum(
            mask * np.log(probs) + (1.0 - mask) * np.log(1.0 - probs)
        )
    return loglike


# Arrival likelihoods
# --------------------
def compute_corr(sensors, stype):
    """
    Compute correlations between sensors
    """
    rlats = sensors[:, 0]
    rlongs = sensors[:, 1]
    corr = np.eye(len(rlats))

    if stype in ["seismic", "array"]:
        lscal = 147.5

        for isens in range(0, len(rlats)):
            for jsens in range(isens, len(rlats)):
                # 147.5 was fit from the 1D profiles
                azmthdata = geodetics.gps2dist_azimuth(
                    rlats[isens], rlongs[isens], rlats[jsens], rlongs[jsens]
                )
                corr[isens, jsens] = np.exp(
                    -1.0 / (lscal**2) * (azmthdata[0] / 1000.0) ** 2
                )
                corr[jsens, isens] = np.exp(
                    -1.0 / (lscal**2) * (azmthdata[0] / 1000.0) ** 2
                )

    return corr


def tt_std_cal(depth, dist, stype):
    """
    Compute standard deviation of travel times using 5th degree polynomial
    model fit to US Transportable Array data for seismic/instant/array sensors
    or GMM model from 'Evaluation of a pair-based, joint-likelihood association
    approach for regional infrasound event identification' by Blom et al.
    for infrasound sensors

    Inputs
    ------
    depth (float) : event depth
    dist (float) : source-receiver distance
    stype (str) : sensor model to use. Options are 'sesmic', 'instant',
                  'infrasound', 'array'.

    Returns
    -------
    std (float) : standard deviation of travel times for given depth/distance
                  pair
    """
    if stype in ["seismic", "instant", "array"]:
        params = np.array(
            [
                -2.56472101e02,
                1.74085841e01,
                -1.19406851e-03,
                -1.66597693e-04,
                9.91187799e-08,
                1.17056345e01,
                -8.13371656e-01,
                8.42272315e-04,
                2.80802067e-06,
                3.60764706e-09,
                1.21624463e-01,
                -8.63214584e-03,
                -7.45214280e-06,
                1.25905904e-07,
                -1.21859595e-10,
                -1.19443389e-02,
                8.67089566e-04,
                -1.11453054e-06,
                -1.11651169e-09,
                -7.07702285e-12,
                1.39921004e-04,
                -1.03707528e-05,
                1.97886651e-08,
                -2.93026404e-11,
                1.57527073e-13,
                1.23554461e-03,
                1.84229583e-02,
                -2.18347658e-04,
                9.53900469e-07,
                -1.26729653e-09,
                3.64225546e-02,
                -3.39939908e-03,
                3.16659333e-05,
                -1.02362420e-07,
                1.11272994e-10,
                -3.75764388e-03,
                2.73743286e-04,
                -2.10488918e-06,
                5.69748506e-09,
                -5.17365299e-12,
                1.49883472e-04,
                -9.66468438e-06,
                6.99133567e-08,
                -1.79611764e-10,
                1.56300210e-13,
                -1.98606449e-06,
                1.21461076e-07,
                -8.71278276e-10,
                2.26330123e-12,
                -2.03871471e-15,
                -3.61047221e01,
                1.75671838e01,
            ]
        )

        ndim = 5
        poly0 = np.reshape(params[0 : (ndim**2)], (ndim, ndim))
        poly1 = np.reshape(params[(ndim**2) : (2 * ndim**2)], (ndim, ndim))
        g = -1.0 * np.abs(params[2 * ndim**2])
        h = np.abs(params[2 * ndim**2 + 1])

        boolg = 1.0 / (1.0 + np.exp(-h * (depth + g)))
        std0 = np.polynomial.polynomial.polyval2d(depth, dist, poly0)
        std1 = np.polynomial.polynomial.polyval2d(depth, dist, poly1)
        std = std0 * boolg + std1 * (1.0 - boolg)

    elif stype in ["infrasound"]:
        weights = np.array([0.054, 0.090, 0.856])
        means = np.array([1.0 / 0.327, 1.0 / 0.293, 1.0 / 0.260])
        stds = np.array([0.066, 0.080, 0.330])

        std = (
            np.sum(weights * stds**2)
            + np.sum(weights * means**2)
            - np.sum(weights * means) ** 2
        )

    return std


def compute_tt(theta, sensors, stype):
    """
    Compute travel time between given theta and all sensors in network using
    obspy's TauPy model for seismic/instant/arrival sensors and using the GMM
    model from 'Evaluation of a pair-based, joint-likelihood association
    approach for regional infrasound event identification' by Blom et al.
    for infrasound sensors.

    Inputs
    ------
    theta (np.array) : (1 x 4) array containing parameters for each seismic event
    sensors (np.array) : (n x 5) array containing parameters for each seismic sensor
    stype (str) : type of sensor model to use. One of 'seismic', 'instant',
                  'infrasound', or 'array'.

    Returns
    -------
    ptime (np.array) : (n x 3) array whose columns are predicted travel time,
                       prediected model error (computed in tt_std_cal),
                       and predicted measurement error (computed in meas_std_cal).
    """
    # event
    src_lat, src_long, zdepth, src_mag = theta
    # sensors
    rlats, rlongs, sensor_fidelity, *_ = sensors.T
    
    if stype in ["seismic", "instant", "array"]:
        model = TauPyModel(model="iasp91")
        
        # Updated to use absolute paths
        with open(CUBIC_PATH, "rb") as inp:
            reg = pickle.load(inp)
        with open(TTSTD_PATH, "rb") as inp:
            std_reg = pickle.load(inp)

        # ... rest of the function remains identical ...
        # mean and model std and measurment std
        ptime = np.zeros((len(rlats), 3))

        for isens, (rlat, rlong, fidelity) in enumerate(
            zip(rlats, rlongs, sensor_fidelity)
        ):
            arrivals = model.get_travel_times_geo(
                source_depth_in_km=zdepth,
                source_latitude_in_deg=src_lat,
                source_longitude_in_deg=src_long,
                receiver_latitude_in_deg=rlat,
                receiver_longitude_in_deg=rlong,
                phase_list=["P", "p"],
            )

            # record data from the first arrival. Assume always in the azimuthal plane.
            ptime[isens, 0] = arrivals[0].time

            delta_km = geodetics.degrees2kilometers(
                geodetics.locations2degrees(src_lat, src_long, rlat, rlong)
            )
            model_std = tt_std_cal(zdepth, delta_km, stype)
            ptime[isens, 1] = model_std

            # models
            dist = calc_dist(src_lat, src_long, rlat, rlong, 6371.0, 0)
            x = [zdepth, dist]

            measure_std = meas_std_cal(delta_km, src_mag, fidelity, stype)

            ptime[isens, 2] = measure_std

    else:
        # travel time model
        vel_model = GaussianMixture(n_components=3, covariance_type="spherical")
        weights = np.array([0.054, 0.090, 0.856])
        means = np.array([1.0 / 0.327, 1.0 / 0.293, 1.0 / 0.260]).reshape((-1, 1))
        stds = np.array([0.066, 0.080, 0.330])

        vel_model.weights_ = weights
        vel_model.means_ = means
        vel_model.covariances_ = stds
        vel_model.n_features = 1

        # mean and model std and measurment std
        ptime = np.zeros((len(rlats), 3))

        for isens, (rlat, rlong, fidelity) in enumerate(
            zip(rlats, rlongs, sensor_fidelity)
        ):
            delta_deg = geodetics.locations2degrees(src_lat, src_long, rlat, rlong)
            delta_km = degrees2kilometers(delta_deg)
            # Model is defined in s/km, take reciprocal for km/s
            vel = 1 / vel_model.sample()[0]
            # ptime[isens, 0] = vel / delta_km
            ptime[isens, 0] = delta_km / vel

            model_std = tt_std_cal(0, 0, stype)
            ptime[isens, 1] = model_std

            measure_std = meas_std_cal(delta_deg, src_mag, fidelity, stype)

            ptime[isens, 2] = measure_std

    return ptime


def arrival_likelihood_gaussian(theta, sensors, data, stype):
    """
    Generate samples from normal distribution with mean defined by predicted
    arrival time and model + measurement error (all of which are computed in
    compute_tt function)

    Inputs
    ------
    theta (np.array) : (m x 4) array containing parameters for each seismic event
    sensors (np.array) : (n x 5) array containing parameters for each seismic sensor
    data (np.array) : (m*ndata x n) array containing generated realizations of data
    stype (str) : type of sensor model to use. One of 'seismic', 'instant',
                  'infrasound', or 'array'.

    Returns
    -------
    loglike (np.array) : (ndata x 1) array containing loglikelihood of each
                         data realization
    """
    # compute mean
    tt_data = compute_tt(theta, sensors, stype)
    # HARD CODING FOR TESTING FIX THIS
    # tt_data = compute_tt(theta, sensors, "seismic")

    mean_tt = tt_data[:, 0]
    stdmodel = tt_data[:, 1]
    measurenoise = tt_data[:, 2]

    corr = compute_corr(sensors, stype)
    cov = np.multiply(np.outer(stdmodel, stdmodel), corr) + np.diag(measurenoise**2.0)

    [ndata, ndpt] = data.shape
    nsens = int(ndpt / 5)

    loglike = np.zeros(ndata)

    for idata in range(0, ndata):
        mask = data[idata, nsens : 2 * nsens]
        maskidx = np.nonzero(mask)[0]

        # I could decompose this into an iterative operation where I sequentally condition on each sensor? That would maybe be computationally more tractable becuase I could apply to all as update the distributions?

        if len(maskidx) > 0:
            # only extracts detections
            means = mean_tt[maskidx]
            dataval = data[idata, maskidx]
            covmat = cov[maskidx[:, None], maskidx]

            res = dataval - means
            onevec = np.ones(means.shape)
            a = np.sum(np.linalg.solve(covmat, res))
            b = np.sum(np.linalg.solve(covmat, onevec))

            # THESE LINES ARE Maybe wrong BECAUSE THE DISTRIBUTION IS IMPROPER
            logdets = np.linalg.slogdet(covmat)[1]
            loglike[idata] = (
                -0.5 * np.sum(np.multiply(res, np.linalg.solve(covmat, res)))
                + (1.0 - len(maskidx)) / 2.0 * np.log(2 * np.pi)
                - 0.5 * logdets
                - 0.5 * np.log(b)
                + np.divide(a**2, 2.0 * b)
            )

    return loglike


# Incident angle likelihoods
# ---------------------------
def compute_incident(theta, sensors, stype):
    """
    Compute incident angle between given theta and all sensors in network using
    obspy's TauPy model.

    Inputs
    ------
    theta (np.array) : (1 x 4) array containing parameters for each seismic event
    sensors (np.array) : (n x 5) array containing parameters for each seismic sensor
    stype (str) : type of sensor model to use. One of 'seismic', 'instant',
                  'infrasound', or 'array'.

    Returns
    -------
    angle (np.array) : (n x 2) array whose columns are predicted incident angle
                       and prediected concentration (computed in compute_kappa).
    """
    if stype in ["infrasound", "array"]:
        src_lat, src_long, zdepth, src_mag = theta
        # sensors
        rlats, rlongs, sensor_fidelity, *_ = sensors.T
        model = TauPyModel(model="iasp91")
        angle = np.zeros((len(rlats), 2))
        N = sensors.shape[0]

        for isens, (rlat, rlong, fidelity) in enumerate(
            zip(rlats, rlongs, sensor_fidelity)
        ):
            arrivals = model.get_travel_times_geo(
                source_depth_in_km=zdepth,
                source_latitude_in_deg=src_lat,
                source_longitude_in_deg=src_long,
                receiver_latitude_in_deg=rlat,
                receiver_longitude_in_deg=rlong,
                phase_list=["P", "p"],
            )

            angle[isens, 0] = arrivals[0].incident_angle * np.pi / 180

            delta_deg = geodetics.locations2degrees(src_lat, src_long, rlat, rlong)
            snr = infrasound_snr_cal(delta_deg, src_mag, fidelity)

            angle[isens, 1] = compute_kappa(N, snr)

    else:
        raise ValueError(
            f"Incident angle could not be computed for sensor type: {stype}"
        )

    return angle


def incident_likelihood(theta, sensors, data, stype):
    """
    Generate samples from von Mises distribution with mean defined by predicted
    incident angle and concentration (both of which are computed in
    compute_incident function)

    Inputs
    ------
    theta (np.array) : (m x 4) array containing parameters for each seismic event
    sensors (np.array) : (n x 5) array containing parameters for each seismic sensor
    data (np.array) : (m*ndata x n) array containing generated realizations of data
    stype (str) : type of sensor model to use. One of 'seismic', 'instant',
                  'infrasound', or 'array'.

    Returns
    -------
    loglike (np.array) : (ndata x 1) array containing loglikelihood of each
                         data realization
    """
    if stype in ["infrasound", "array"]:
        angle_data = compute_incident(theta, sensors, stype)
        [ndata, ndpt] = data.shape
        nsens = int(ndpt / 5)

        loglike = np.zeros(ndata)

        for idata, mask in enumerate(data[:, nsens : 2 * nsens]):
            maskidx = np.nonzero(mask)[0]

            if len(maskidx) == 0:
                continue

            detected_angledata = angle_data[maskidx]
            means = detected_angledata[:, 0]
            kappas = detected_angledata[:, 1]

            local_likes = []
            detected_data = data[idata, 3 * nsens :][maskidx]

            for imask in range(len(detected_data)):
                dist = stats.vonmises(kappas[imask], loc=means[imask])
                const = dist.cdf(np.pi) - dist.cdf(0)
                tmp = dist.logpdf(detected_data[imask]) - np.log(const)
                local_likes.append(tmp)

            loglike[idata] = sum(local_likes)

    else:
        loglike = 0

    return loglike


# Azimuth angle likelihoods
# --------------------------
def compute_azimuth(theta, sensors, stype):
    """
    Compute incident angle between given theta and all sensors in network.

    Inputs
    ------
    theta (np.array) : (1 x 4) array containing parameters for each seismic event
    sensors (np.array) : (n x 5) array containing parameters for each seismic sensor
    stype (str) : type of sensor model to use. One of 'seismic', 'instant',
                  'infrasound', or 'array'.

    Returns
    -------
    azmth (np.array) : (n x 2) array whose columns are predicted incident angle
                       and prediected concentration (computed in compute_kappa).
    """
    if stype in ["infrasound", "array"]:
        src_lat, src_long, zdepth, src_mag = theta
        # sensors
        rlats, rlongs, sensor_fidelity, *_ = sensors.T
        azmth = np.zeros((len(rlats), 2))

        N = sensors.shape[0]

        for isens, (rlat, rlong, fidelity) in enumerate(
            zip(rlats, rlongs, sensor_fidelity)
        ):
            curr_azmth = gps2dist_azimuth(rlat, rlong, src_lat, src_long)[1]

            azmth[isens, 0] = curr_azmth * np.pi / 180

            delta_deg = geodetics.locations2degrees(src_lat, src_long, rlat, rlong)
            #         logsnr = seismic_snr_cal(delta_deg, src_mag, fidelity)
            snr = infrasound_snr_cal(delta_deg, src_mag, fidelity)
            #         azmth[isens,1] = compute_kappa(N, np.exp(logsnr))
            azmth[isens, 1] = compute_kappa(N, snr)

    else:
        raise ValueError(f"Azimuth cannot be computed for sensor type: {stype}")

    return azmth


def azimuth_likelihood(theta, sensors, data, stype):
    """
    Generate samples from von Mises distribution with mean defined by predicted
    azimuth angle and concentration (both of which are computed in
    compute_incident function)

    Inputs
    ------
    theta (np.array) : (m x 4) array containing parameters for each seismic event
    sensors (np.array) : (n x 5) array containing parameters for each seismic sensor
    data (np.array) : (m*ndata x n) array containing generated realizations of data
    stype (str) : type of sensor model to use. One of 'seismic', 'instant',
                  'infrasound', or 'array'.

    Returns
    -------
    loglike (np.array) : (ndata x 1) array containing loglikelihood of each
                         data realization
    """
    if stype in ["seismic", "instant"]:
        loglike = 0

    if stype in [
        "infrasound",
    ]:
        azmth_data = compute_azimuth(theta, sensors, stype)
        [ndata, ndpt] = data.shape
        nsens = int(ndpt / 5)

        loglike = np.zeros(ndata)

        for idata, mask in enumerate(data[:, nsens : 2 * nsens]):
            maskidx = np.nonzero(mask)[0]

            if len(maskidx) == 0:
                continue

            detected_azmthdata = azmth_data[maskidx]
            means = detected_azmthdata[:, 0]
            kappas = detected_azmthdata[:, 1]

            local_likes = []
            detected_data = data[idata, 2 * nsens : 3 * nsens][maskidx]
            for imask in range(len(detected_data)):
                dist = stats.vonmises(kappas[imask], loc=means[imask])
                tmp = dist.logpdf(detected_data[imask])
                local_likes.append(tmp)
            loglike[idata] = sum(local_likes)

    else:
        loglike = 0

    return loglike


"""
Detection likelihood models
-------------------------------------------------
-------------------------------------------------
"""


def extract_sensor_data(sensor_idx, total_data):
    """
    Extracts data for specific sensors.
    UPDATED: Now handles 5 blocks of data (Arrival, Detect, Az, Inc, Power)
    """
    nsens = len(sensor_idx)
    ntotal_sens = int(total_data.shape[1] / 5)  # DIVIDE BY 5 NOW
    ndata = total_data.shape[0]
    sensor_data = np.zeros((ndata, nsens * 5))

    # Extract Arrivals
    sensor_data[:, :nsens] = total_data[:, sensor_idx]
    # Extract Detections
    sensor_data[:, nsens : 2 * nsens] = total_data[:, ntotal_sens + sensor_idx]
    # Extract Azimuths
    sensor_data[:, 2 * nsens : 3 * nsens] = total_data[:, 2 * ntotal_sens + sensor_idx]
    # Extract Incident Angles
    sensor_data[:, 3 * nsens : 4 * nsens] = total_data[:, 3 * ntotal_sens + sensor_idx]
    # Extract Power (NEW)
    sensor_data[:, 4 * nsens :] = total_data[:, 4 * ntotal_sens + sensor_idx]

    return sensor_data


def compute_sensor_loglikes(theta, sensors, data, stype="seismic"):
    if sensors.shape[0] == 0:
        return np.zeros(data.shape[0])

    # Existing likelihoods
    detect_loglikes = detection_likelihood(theta, sensors, data, stype=stype)
    arrival_loglikes = arrival_likelihood_gaussian(theta, sensors, data, stype=stype)
    incident_loglikes = incident_likelihood(theta, sensors, data, stype=stype)
    azimuth_loglikes = azimuth_likelihood(theta, sensors, data, stype=stype)

    # NEW: Power likelihood
    # Only calculate for seismic/array sensors, others get 0
    if stype in ["seismic", "array"]:
        power_loglikes = power_likelihood(theta, sensors, data, stype=stype)
    else:
        power_loglikes = 0.0

    loglikes = (
        detect_loglikes
        + arrival_loglikes
        + power_loglikes
        # + incident_loglikes + azimuth_loglikes (Uncomment if you use them)
    )
    return loglikes


def compute_loglikes(theta, sensors, data):
    """
    Computes the log likelihoods of data given paramaters theta for each sensor.

    Inputs
    ------
    theta (array_like) : array containing seismic parameters
    sensors (array_like) : array containing sensors
    data (array_like) : array containing observed data at each sensor

    Returns
    -------
    loglikes (array_like) : array of shape ( containing log likelihoods
    """
    # split out sensor types into their own arrays and call relevant functions
    # This assumes independent sensor types
    seismic_idx = np.where(sensors[:, 4] == 0)[0]
    instant_idx = np.where(sensors[:, 4] == 1)[0]
    infrasound_idx = np.where(sensors[:, 4] == 2)[0]
    array_idx = np.where(sensors[:, 4] == 3)[0]

    seismic_sensors = sensors[seismic_idx]  # Seismic arrival sensors and data
    seismic_data = extract_sensor_data(seismic_idx, data)

    instant_sensors = sensors[instant_idx]  # Instant arrival sensors and data
    instant_data = extract_sensor_data(instant_idx, data)

    infrasound_sensors = sensors[infrasound_idx]  # Infrasound sensors and data
    infrasound_data = extract_sensor_data(infrasound_idx, data)

    array_sensors = sensors[array_idx]  # Seismic arrays and data
    array_data = extract_sensor_data(array_idx, data)

    seismic_loglikes = compute_sensor_loglikes(
        theta, seismic_sensors, seismic_data, stype="seismic"
    )
    instant_loglikes = compute_sensor_loglikes(
        theta, instant_sensors, instant_data, stype="instant"
    )
    infrasound_loglikes = compute_sensor_loglikes(
        theta, infrasound_sensors, infrasound_data, stype="infrasound"
    )
    array_loglikes = compute_sensor_loglikes(
        theta, array_sensors, array_data, stype="array"
    )

    loglikes = (
        seismic_loglikes + instant_loglikes + infrasound_loglikes + array_loglikes
    )

    return loglikes
