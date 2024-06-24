import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def read_bounds(bounds_file, sensor_bounds=False):
    """
    Loads boundary configuration options from specified file.

    Parameters
    ----------
    bounds_file (str)    : filepath to file containing domain information.
    sensor_bounds (bool) : declares whether function is returning bounds for sensor placement (True)
                           or event sampling (False).

    Returns
    -------
    latlon_bounds (array_like) : Array containing points that define a polygon inside which valid
                                 latitude and longitude coordinates may be chosen.

    depth_bounds (array_like)  : Array containing two points defining the range from which valid
                                 depths may be chosen. Returned only if sensor_bounds is False.
    mag_bounds (array_like)    : Array containing two points defining the range from which valid
                                 magnitudes may be chosen. Returned only if sensor_bounds is False.
    """
    # Load file
    with open(bounds_file, "r") as f:
        bounds = json.load(f)

    # Check which types of keys the dict contains
    has_lat = "lat_range" in bounds.keys()
    has_lon = "lon_range" in bounds.keys()
    has_ranges = has_lat and has_lon

    coords_keys = [f"coordinates_{i}" for i in range(1, len(bounds.keys()) - 1)]
    has_coords = any([key in coords_keys for key in bounds.keys()])

    allowable_keys = [
        "lat_range",
        "lon_range",
        "depth_range",
        "mag_range",
    ] + coords_keys

    # Ensure only correct combination of keys exists
    for key in bounds.keys():
        if key not in allowable_keys:
            raise ValueError(
                f"Key {key} in bounds file is not supported. Please remove."
            )
    if has_lat and not has_lon:
        raise ValueError(
            "Bounds file contains 'lat_range' but not 'lon_range'. Both must be specified."
        )
    elif has_lon and not has_lat:
        raise ValueError(
            "Bounds file contains 'lon_range' but not 'lat_range'. Both must be specified."
        )
    elif has_ranges and has_coords:
        raise ValueError(
            "Bounds file contains keys 'lat_range', 'lon_range', and 'coordinates_*'. Only one type of bound specification (ranges or coordinates) may be used."
        )

    if has_ranges:
        # Use range values to create a square boundary defined by corner coordinates

        # Extract ranges
        lat_range = bounds["lat_range"]
        lon_range = bounds["lon_range"]

        # Define square (polygons use longitude as x coordinate)
        latlon_bounds = np.array(
            [
                [lon_range[0], lat_range[0]],
                [lon_range[0], lat_range[1]],
                [lon_range[1], lat_range[1]],
                [lon_range[1], lat_range[0]],
            ]
        )

    elif has_coords:
        # Compile all polygons into one list
        latlon_bounds = []
        for key in coords_keys:
            # Store each polygon as numpy array
            latlon_bounds.append(np.array(bounds[key]))

    if sensor_bounds:
        # Only return lat/lon boundary when dealing with sensors
        return latlon_bounds

    # When dealing with events return depth and mag ranges as well
    depth_range = bounds["depth_range"]
    mag_range = bounds["mag_range"]

    return latlon_bounds, depth_range, mag_range


# Load Configuration
def read_input_file(file):
    with open(file, "r") as readdata:
        nlpts_data = int(readdata.readline())
        nlpts_space = int(readdata.readline())
        ndata = int(readdata.readline())
        event_boundary_file = readdata.readline().strip("\n")

        sampling_file = readdata.readline().strip("\n")

        # rest of lines are sensors
        sensorlines = readdata.readlines()
        numsen = len(sensorlines)

        # lat, long, measurement noise std, length of data vector for sensor, sensor type
        nsensordata = len(np.fromstring(sensorlines[0], dtype=float, sep=","))

        sensors = np.zeros([numsen, nsensordata])
        for inc in range(0, numsen):
            sensorline = np.fromstring(sensorlines[inc], dtype=float, sep=",")
            sensors[inc, :] = sensorline

    return nlpts_data, nlpts_space, ndata, event_boundary_file, sampling_file, sensors


def old_read_input_file(file):
    with open(file, "r") as readdata:
        nlpts_data = int(readdata.readline())
        nlpts_space = int(readdata.readline())
        ndata = int(readdata.readline())
        lat_range = np.fromstring(readdata.readline(), dtype=float, sep=",")
        long_range = np.fromstring(readdata.readline(), dtype=float, sep=",")
        depth_range = np.fromstring(readdata.readline(), dtype=float, sep=",")
        mag_range = np.fromstring(readdata.readline(), dtype=float, sep=",")

        sampling_file = readdata.readline().strip("\n")

        # rest of lines are sensors
        sensorlines = readdata.readlines()
        numsen = len(sensorlines)

        # lat, long, measurement noise std, length of data vector for sensor, sensor type
        nsensordata = len(np.fromstring(sensorlines[0], dtype=float, sep=","))

        sensors = np.zeros([numsen, nsensordata])
        for inc in range(0, numsen):
            sensorline = np.fromstring(sensorlines[inc], dtype=float, sep=",")
            sensors[inc, :] = sensorline

    return (
        nlpts_data,
        nlpts_space,
        ndata,
        lat_range,
        long_range,
        depth_range,
        mag_range,
        sampling_file,
        sensors,
    )


# Write Configuration
def write_input_file(
    file,
    nlpts_data,
    nlpts_space,
    ndata,
    event_boundary_file,
    sampling_file,
    sloc_trial,
    sensor_params,
    sensors,
):

    writedata = open(file, "w+")
    writedata.write(str(int(nlpts_data)) + "\n")
    writedata.write(str(int(nlpts_space)) + "\n")
    writedata.write(str(int(ndata)) + "\n")

    # Boundary file for sampling events
    writedata.write(event_boundary_file + "\n")

    # write sampling filename
    writedata.write(sampling_file + "\n")

    # rest of lines are sensors
    writedata.write(
        (np.array2string(sensors, separator=",", max_line_width=1000))
        .replace("[", "")
        .replace("],", "")
        .replace("]", "")
        .replace(" ", "")
        + "\n"
    )

    # lat, long, measurement noise std, length of data vector for sensor, sensor type
    # write the new sensor
    writedata.write(
        (np.array2string(sloc_trial, separator=",", max_line_width=1000))
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "")
        + ","
        + (np.array2string(sensor_params, separator=",", max_line_width=1000))
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "")
        + "\n"
    )

    writedata.close()
    return


# Load Configuration
def read_opt_file(file):
    with open(file, "r") as readdata:
        nopt_random = int(readdata.readline())
        nopt_total = int(readdata.readline())
        opt_bounds_file = readdata.readline().strip("\n")
        sensor_params = np.fromstring(readdata.readline(), dtype=float, sep=",")
        opt_type = int(readdata.readline())

        nlpts_data = int(readdata.readline())
        nlpts_space = int(readdata.readline())
        ndata = int(readdata.readline())
        event_bounds_file = readdata.readline().strip("\n")

        mpirunstring = readdata.readline().strip("\n")
        sampling_file = readdata.readline().strip("\n")
        nsensor_place = int(readdata.readline())

        # rest of lines are sensors
        sensorlines = readdata.readlines()
        numsen = len(sensorlines)

        # lat, long, measurement noise std, length of data vector for sensor, sensor type
        nsensordata = len(np.fromstring(sensorlines[0], dtype=float, sep=","))

        sensors = np.zeros([numsen, nsensordata])
        for inc in range(0, numsen):
            sensorline = np.fromstring(sensorlines[inc], dtype=float, sep=",")
            sensors[inc, :] = sensorline
    return (
        nopt_random,
        nopt_total,
        opt_bounds_file,
        sensor_params,
        opt_type,
        nlpts_data,
        nlpts_space,
        ndata,
        event_bounds_file,
        sensors,
        mpirunstring,
        sampling_file,
        nsensor_place,
    )


def plot_surface(data, output_path="eig_plots", depth_step=1, mag_step=1, stepsize=100):

    t0 = time.time()
    print(f"Configuring data for plots: {time.time() - t0}")
    # Specify training data for Gaussian Processs
    target = data["ig"]
    inputs = data["theta_data"]  # (lat, long, depth, magnitude)
    # Specify graphing domain
    lat_range = data["lat_range"]
    long_range = data["long_range"]

    depth_range = data["depth_range"]
    mag_range = data["mag_range"]

    # Format target variable
    target = target.reshape(len(inputs), -1).mean(axis=1)

    # Create domain to make predictions on

    # Lat/long features
    x = np.linspace(lat_range[0], lat_range[1], stepsize)
    y = np.linspace(long_range[0], long_range[1], stepsize)
    xv, yv = np.meshgrid(x, y)
    xy = np.vstack([xv.ravel(), yv.ravel()]).T

    # Create full featureset
    domain = np.zeros((stepsize**2, 4))
    domain[:, :2] = xy

    print(f"Training GP model: {time.time() -t0}")
    model = GPR()
    model.fit(inputs, target)

    depth_slices = np.arange(depth_range[0], depth_range[1] + depth_step, depth_step)
    mag_slices = np.arange(mag_range[0], mag_range[1] + mag_step, mag_step)

    now = datetime.now()
    timestamp = f"{now.year}-{now.month}-{now.day}_{now.hour}:{now.minute}:{now.second}"
    save_dir = os.path.join(output_path, timestamp)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    total_plots = len(depth_slices) * len(mag_slices)
    curr_plot = 1

    for depth_slice in depth_slices:
        for mag_slice in mag_slices:
            print(f"Generating plot {curr_plot} of {total_plots}: {time.time()-t0}")
            # Specify 2d slice (depth and magnitude features)
            domain[:, 2] = depth_slice
            domain[:, 3] = mag_slice

            # Make predictions to generate map
            preds = model.predict(domain)

            # Plot IG map
            plt.pcolormesh(
                xv,
                yv,
                preds.reshape((stepsize, stepsize)),
                shading="auto",
                cmap="viridis",
            )
            plt.colorbar()

            # Plot sensor locations
            plt.scatter(
                data["sensors"][:, 0],
                data["sensors"][:, 1],
                marker="o",
                facecolors="none",
                edgecolors="red",
                label="Sensor location",
            )

            # Label and plot
            plt.xlabel("Latitude")
            plt.ylabel("Longitude")
            plt.title(
                f"Expected Information Gain for events with depth = {depth_slice}, mag = {mag_slice}"
            )

            plt.legend()

            plotname = (
                f"depth-{np.round(depth_slice,3)}_mag-{np.round(mag_slice,3)}.pdf"
            )
            plt.savefig(os.path.join(save_dir, plotname))

            plt.close()
            plt.clf()
            curr_plot += 1
