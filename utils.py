import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

#Load Configuration
def read_input_file(file):
    with open(file, 'r') as readdata:
        nlpts_data  = int(readdata.readline())
        nlpts_space  = int(readdata.readline())
        ndata = int(readdata.readline())
        lat_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        long_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        depth_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        mag_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        
        sampling_file = readdata.readline()
        
        #rest of lines are sensors
        sensorlines=readdata.readlines()
        numsen = len(sensorlines)
        
        #lat, long, measurement noise std, length of data vector for sensor, sensor type
        nsensordata = len(np.fromstring(sensorlines[0], dtype=float, sep=','))
        
        sensors = np.zeros([numsen,nsensordata])
        for inc in range(0,numsen):
            sensorline = np.fromstring(sensorlines[inc], dtype=float, sep=',')
            sensors[inc,:] = sensorline
    return nlpts_data, nlpts_space, ndata, lat_range, long_range, depth_range, mag_range, sampling_file, sensors


#Write Configuration
def write_input_file(file, nlpts_data, nlpts_space, ndata, lat_range, long_range, depth_range, mag_range, sloc_trial, sensor_params, sensors, sampling_file):

    writedata=open(file,"w+")
    writedata.write(str(int(nlpts_data)) + "\n")
    writedata.write(str(int(nlpts_space)) + "\n")
    writedata.write(str(int(ndata)) + "\n")
    
    #,max_line_width=1000 to keep numppy from splitting the sensor description up onto multiple lines.
    writedata.write((np.array2string(lat_range,separator=',',max_line_width=1000)).replace('[','').replace(']','').replace(' ', '') + "\n")
    writedata.write((np.array2string(long_range,separator=',',max_line_width=1000)).replace('[','').replace(']','').replace(' ', '') + "\n")
    writedata.write((np.array2string(depth_range,separator=',',max_line_width=1000)).replace('[','').replace(']','').replace(' ', '') + "\n")
    writedata.write((np.array2string(mag_range,separator=',',max_line_width=1000)).replace('[','').replace(']','').replace(' ', '') + "\n")

    #Read sampling filename
    sampling_file = writedata.write(sampling_file)

    #rest of lines are sensors
    writedata.write((np.array2string(sensors,separator=',',max_line_width=1000)).replace('[','').replace('],','').replace(']','').replace(' ', '') + "\n")

    #lat, long, measurement noise std, length of data vector for sensor, sensor type
    #write teh new sensor
    writedata.write((np.array2string(sloc_trial,separator=',',max_line_width=1000)).replace('[','').replace(']','').replace(' ', '') + "," + (np.array2string(sensor_params,separator=',',max_line_width=1000)).replace('[','').replace(']','').replace(' ', '') + "\n")

    writedata.close()
    return

#Load Configuration
def read_opt_file(file):
    with open(file, 'r') as readdata:
        nopt_random = int(readdata.readline())
        nopt_total = int(readdata.readline())
        sensor_lat_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        sensor_long_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        bounds_file = readdata.readline().strip('\n')
        sensor_params = np.fromstring(readdata.readline(), dtype=float, sep=',')
        opt_type = int(readdata.readline())
        
        nlpts_data  = int(readdata.readline())
        nlpts_space  = int(readdata.readline())
        ndata = np.int(readdata.readline())
        lat_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        long_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        depth_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        mag_range = np.fromstring(readdata.readline(), dtype=float, sep=',')

        mpirunstring = readdata.readline()
        sampling_file = readdata.readline()
        nsensor_place = int(readdata.readline())
        
        #rest of lines are sensors
        sensorlines=readdata.readlines()
        numsen = len(sensorlines)
        
        #lat, long, measurement noise std, length of data vector for sensor, sensor type
        nsensordata = len(np.fromstring(sensorlines[0], dtype=float, sep=','))
        
        sensors = np.zeros([numsen,nsensordata])
        for inc in range(0,numsen):
            sensorline = np.fromstring(sensorlines[inc], dtype=float, sep=',')
            sensors[inc,:] = sensorline
    return nopt_random, nopt_total, sensor_lat_range, sensor_long_range, bounds_file, sensor_params, opt_type, nlpts_data, nlpts_space, ndata, lat_range, long_range, depth_range, mag_range, sensors, mpirunstring, sampling_file, nsensor_place


def plot_surface(data,
                 output_path='eig_plots', 
                 depth_step=1, mag_step=1, 
                 stepsize=100):

    t0 = time.time()
    print(f'Configuring data for plots: {time.time() - t0}')
    # Specify training data for Gaussian Processs
    target = data['ig']
    inputs = data['theta_data'] # (lat, long, depth, magnitude)
    # Specify graphing domain
    lat_range = data['lat_range']
    long_range = data['long_range']
    
    depth_range = data['depth_range']
    mag_range = data['mag_range']

    # Format target variable
    target = target.reshape(len(inputs),-1).mean(axis=1)
    
    # Create domain to make predictions on

    # Lat/long features
    x = np.linspace(lat_range[0], lat_range[1], stepsize)
    y = np.linspace(long_range[0], long_range[1], stepsize)
    xv, yv = np.meshgrid(x, y)
    xy = np.vstack([xv.ravel(), yv.ravel()]).T

    # Create full featureset
    domain = np.zeros((stepsize**2, 4))
    domain[:,:2] = xy

    print(f'Training GP model: {time.time() -t0}')
    model = GPR()
    model.fit(inputs,target)
    
    depth_slices = np.arange(depth_range[0], depth_range[1]+depth_step, depth_step)
    mag_slices = np.arange(mag_range[0], mag_range[1]+mag_step, mag_step)

    now = datetime.now()
    timestamp = f'{now.year}-{now.month}-{now.day}_{now.hour}:{now.minute}:{now.second}'
    save_dir = os.path.join(output_path, timestamp)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    total_plots = len(depth_slices)* len(mag_slices)
    curr_plot = 1

    for depth_slice in depth_slices:
        for mag_slice in mag_slices:
            print(f'Generating plot {curr_plot} of {total_plots}: {time.time()-t0}')
            # Specify 2d slice (depth and magnitude features)
            domain[:,2] = depth_slice
            domain[:,3] = mag_slice
            
            # Make predictions to generate map
            preds = model.predict(domain)
            
            # Plot IG map
            plt.pcolormesh(xv, yv, preds.reshape((stepsize, stepsize)),
                        shading='auto', cmap='viridis')
            plt.colorbar()

            # Plot sensor locations
            plt.scatter(data['sensors'][:,0],data['sensors'][:,1], 
                        marker='o',facecolors='none', edgecolors='red', 
                        label='Sensor location')
            
            # Label and plot
            plt.xlabel('Latitude')
            plt.ylabel('Longitude')
            plt.title(f'Expected Information Gain for events with depth = {depth_slice}, mag = {mag_slice}')

            plt.legend()

            plotname = f'depth-{np.round(depth_slice,3)}_mag-{np.round(mag_slice,3)}.pdf'
            plt.savefig(os.path.join(save_dir, plotname))

            plt.close()
            plt.clf()
            curr_plot += 1
            
