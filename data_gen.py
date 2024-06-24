import warnings

import numpy as np
from scipy import stats

import like_models as lm

warnings.filterwarnings("ignore")


def gen_arrival_normal(theta, sensors, ndata, stype):
    """
    Simulates arrival data by drawing from a normal distribution with mean and
    variance defined by sensor models

    Inputs
    ------
    theta (np.array) : (m x 4) array containing parameters for each seismic event
    sensors (np.array) : (n x 5) array containing parameters for each seismic sensor
    ndata (int) : number of realizations of data to generate
    stype (str) : type of sensor model to use. One of 'seismic', 'instant',
                  'infrasound', or 'array'.

    Returns
    -------
    arrivals (np.array) : (ndata*m x n) array containing sampled arrival data.
    """
    # Variance is combination of arrival time and general sensor variance
    # compute tt mean, model std, measruement std
    tt_data = lm.compute_tt(theta, sensors, stype)

    mean_tt = tt_data[:, 0]
    stdmodel = tt_data[:, 1]
    measurenoise = tt_data[:, 2]

    # compute corr matrix
    corr = lm.compute_corr(sensors, stype)
    cov = np.multiply(np.outer(stdmodel, stdmodel), corr) + np.diag(measurenoise**2.0)

    arrivals = np.random.multivariate_normal(mean_tt, cov, ndata, tol=1e-5)
    return arrivals


def gen_incident_vonmises(theta, sensors, ndata, stype):
    """
    Simulates incident angle data by drawing from a von Mises distribution with
    mean and concentration (kappa) defined by sensor models. Data only simulated
    for sensors of type 'infrasound' and 'array'.

    Inputs
    ------
    theta (np.array) : (m x 4) array containing parameters for each seismic event
    sensors (np.array) : (n x 5) array containing parameters for each seismic sensor
    ndata (int) : number of realizations of data to generate
    stype (str) : type of sensor model to use. One of 'seismic', 'instant',
                  'infrasound', or 'array'.

    Returns
    -------
    incidents (np.array) : (ndata*m x n) array containing sampled incident angle data.
    """
    if stype in ["infrasound", "array"]:
        angle_data = lm.compute_incident(theta, sensors, stype)
        mean_angle = angle_data[:, 0]
        kappa_angle = angle_data[:, 1]

        incidents = np.zeros((len(sensors), ndata))
        for i in range(len(sensors)):
            vmf = stats.vonmises_line(kappa_angle[i], loc=mean_angle[i])
            u = np.random.uniform(vmf.cdf(0), vmf.cdf(np.pi), size=ndata)
            val = vmf.ppf(u)

            if np.any(np.isnan(val)):
                print("NAN encountered incident")
                return u, vmf
            incidents[i] = val

        incidents = incidents.T

    elif stype in ["seismic", "instant"]:
        incidents = np.empty((ndata, sensors.shape[0]))
        incidents.fill(np.nan)

    return incidents


def gen_azimuth_vonmises(theta, sensors, ndata, stype):
    """
    Simulates azimuth angle data by drawing from a von Mises distribution with
    mean and concentration (kappa) defined by sensor models. Data only simulated
    for sensors of type 'infrasound' and 'array'.

    Inputs
    ------
    theta (np.array) : (m x 4) array containing parameters for each seismic event
    sensors (np.array) : (n x 5) array containing parameters for each seismic sensor
    ndata (int) : number of realizations of data to generate
    stype (str) : type of sensor model to use. One of 'seismic', 'instant',
                  'infrasound', or 'array'.

    Returns
    -------
    azimuths (np.array) : (ndata*m x n) array containing sampled arrival data.
    """
    if stype in ["infrasound", "array"]:
        azmth_data = lm.compute_azimuth(theta, sensors, stype)
        mean_azmth = azmth_data[:, 0]
        kappa_azmth = azmth_data[:, 1]

        azimuths = np.zeros((len(sensors), ndata))
        for i in range(len(sensors)):
            vmf = stats.vonmises_line(kappa_azmth[i], loc=mean_azmth[i])
            val = vmf.rvs(size=ndata)

            if np.any(np.isnan(val)):
                print("NAN encountered azimuth")
                return vmf
            azimuths[i] = val

        azimuths = azimuths.T

    else:
        azimuths = np.empty((ndata, sensors.shape[0]))
        azimuths.fill(np.nan)

    return azimuths


def generate_sensor_data(theta, sensors, ndata, stype):
    """
    Generates all types of data for a given set of sensors and corresponding
    sensor type.

    Inputs
    ------
    theta (np.array) : (m x 4) array containing parameters for each seismic event
    sensors (np.array) : (n x 5) array containing parameters for each seismic sensor
    ndata (int) : number of realizations of data to generate
    stype (str) : type of sensor model to use. One of 'seismic', 'instant',
                  'infrasound', or 'array'.

    Returns
    -------
    data (np.array) : (ndata*m x n*4) array containing sampled data.
    """
    # compute detection probablity
    probs = lm.detection_probability(theta, sensors, stype)

    # make probs bigger
    fullprobs = np.outer(np.ones(ndata), probs)
    u_mat = np.random.uniform(size=fullprobs.shape)

    # sample arrival times
    atimes = gen_arrival_normal(theta, sensors, ndata, stype)

    # sample incident angles
    incidents = gen_incident_vonmises(theta, sensors, ndata, stype)

    azimuths = gen_azimuth_vonmises(theta, sensors, ndata, stype)

    # get data[probs arrivaltimes]
    data = np.concatenate((atimes, u_mat < fullprobs, azimuths, incidents), axis=1)
    return data


def generate_data(theta, sensors, ndata):
    """
    Generates all types of data for all sensors.

    Inputs
    ------
    theta (np.array) : (m x 4) array containing parameters for each seismic event
    sensors (np.array) : (n x 5) array containing parameters for each seismic sensor
    ndata (int) : number of realizations of data to generate

    Returns
    -------
    data (np.array) : (ndata*m x n*4) array containing sampled data.
    """

    def split_data(data):
        # splits data into arrival times, detections, azimuths, and incidents

        nsens = int(data.shape[1] / 4)
        atimes = data[:, :nsens]
        detections = data[:, nsens : 2 * nsens]
        azmths = data[:, 2 * nsens : 3 * nsens]
        incdnt = data[:, 3 * nsens :]

        return atimes, detections, azmths, incdnt

    num_sensors = sensors.shape[0]

    # Get locations of each sensor type
    seismic_idx = np.where(sensors[:, 4] == 0)[0]
    instant_idx = np.where(sensors[:, 4] == 1)[0]
    infra_idx = np.where(sensors[:, 4] == 2)[0]
    array_idx = np.where(sensors[:, 4] == 3)[0]

    # Control for nonexistent sensors
    seismic_exists = len(seismic_idx) != 0
    instant_exists = len(instant_idx) != 0
    infra_exists = len(infra_idx) != 0
    array_exists = len(array_idx) != 0

    # Split sensors by type
    seismic_sensors = sensors[seismic_idx]
    instant_sensors = sensors[instant_idx]
    infrasound_sensors = sensors[infra_idx]
    array_sensors = sensors[array_idx]

    # Generate data from each sensor type
    if seismic_exists:
        seismic_data = generate_sensor_data(
            theta, seismic_sensors, ndata, stype="seismic"
        )
    if instant_exists:
        instant_data = generate_sensor_data(
            theta, instant_sensors, ndata, stype="instant"
        )
    if infra_exists:
        infra_data = generate_sensor_data(
            theta, infrasound_sensors, ndata, stype="infrasound"
        )
    if array_exists:
        array_data = generate_sensor_data(theta, array_sensors, ndata, stype="array")

    # Split each sensor type's data into detections, arrivals, azimuths, incident angles
    # in order to recombine into a single dataset
    if seismic_exists:
        seis_atimes, seis_detects, seis_azmths, seis_incdnts = split_data(seismic_data)
    if instant_exists:
        inst_atimes, inst_detects, inst_azmths, inst_incdnts = split_data(instant_data)
    if infra_exists:
        infra_atimes, infra_detects, infra_azmths, infra_incdnts = split_data(
            infra_data
        )
    if array_exists:
        array_atimes, array_detects, array_azmths, array_incdnts = split_data(
            array_data
        )

    # Create matrix for storing all generated data
    total_data = np.zeros((ndata, 4 * num_sensors))

    # Now add data from each sensor type into its column corresponding to sensor
    # positions in original sensor array

    # Seismic sensor data
    if seismic_exists:
        total_data[:, seismic_idx] = seis_atimes
        total_data[:, num_sensors + seismic_idx] = seis_detects
        total_data[:, 2 * num_sensors + seismic_idx] = seis_azmths
        total_data[:, 3 * num_sensors + seismic_idx] = seis_incdnts

    # Instant origin data
    if instant_exists:
        total_data[:, instant_idx] = inst_atimes
        total_data[:, num_sensors + instant_idx] = inst_detects
        total_data[:, 2 * num_sensors + instant_idx] = inst_azmths
        total_data[:, 3 * num_sensors + instant_idx] = inst_incdnts

    # Infrasound data
    if infra_exists:
        total_data[:, infra_idx] = infra_atimes
        total_data[:, num_sensors + infra_idx] = infra_detects
        total_data[:, 2 * num_sensors + infra_idx] = infra_azmths
        total_data[:, 3 * num_sensors + infra_idx] = infra_incdnts

    # Seismic array data
    if array_exists:
        total_data[:, array_idx] = array_atimes
        total_data[:, num_sensors + array_idx] = array_detects
        total_data[:, 2 * num_sensors + array_idx] = array_azmths
        total_data[:, 3 * num_sensors + array_idx] = array_incdnts

    return total_data


def sample_sensors(lat_range, long_range, nsamp, skip):
    # Generate psuedo random sensor distribution for initial OED

    dim_num = 2
    sbvals = np.full((nsamp, dim_num), np.nan)
    for j in range(nsamp):
        sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1 + skip + j)

    sbvals[:, 0] = sbvals[:, 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
    sbvals[:, 1] = sbvals[:, 1] * (long_range[1] - long_range[0]) + long_range[0]

    return sbvals
