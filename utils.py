import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

#Load Configuration
def read_input_file(file):
    with open(file, 'r') as readdata:
        nlpts_data  = np.int(readdata.readline())
        nlpts_space  = np.int(readdata.readline())
        ndata = np.int(readdata.readline())
        lat_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        long_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        depth_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        mag_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        vis_slices = np.fromstring(readdata.readline(), dtype=float, sep=',')
        
        #rest of lines are sensors
        sensorlines=readdata.readlines()
        numsen = len(sensorlines)
        
        #lat, long, measurement noise std, length of data vector for sensor, sensor type
        nsensordata = len(np.fromstring(sensorlines[0], dtype=float, sep=','))
        
        sensors = np.zeros([numsen,nsensordata])
        for inc in range(0,numsen):
            sensorline = np.fromstring(sensorlines[inc], dtype=float, sep=',')
            sensors[inc,:] = sensorline
    return nlpts_data, nlpts_space, ndata, lat_range, long_range, depth_range, mag_range, vis_slices, sensors


#Write Configuration
def write_input_file(file, nlpts_data, nlpts_space, ndata, lat_range, long_range, depth_range, sloc_trial, sensor_params, sensors):

    writedata=open(file,"w+")
    writedata.write(str(np.int(nlpts_data)) + "\n")
    writedata.write(str(np.int(nlpts_space)) + "\n")
    writedata.write(str(np.int(ndata)) + "\n")
    
    #,max_line_width=1000 to keep numppy from splitting the sensor description up onto multiple lines.
    writedata.write((np.array2string(lat_range,separator=',',max_line_width=1000)).replace('[','').replace(']','').replace(' ', '') + "\n")
    writedata.write((np.array2string(long_range,separator=',',max_line_width=1000)).replace('[','').replace(']','').replace(' ', '') + "\n")
    writedata.write((np.array2string(depth_range,separator=',',max_line_width=1000)).replace('[','').replace(']','').replace(' ', '') + "\n")

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
        nopt_random = np.int(readdata.readline())
        nopt_total = np.int(readdata.readline())
        sensor_lat_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        sensor_long_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        sensor_params = np.fromstring(readdata.readline(), dtype=float, sep=',')
        opt_type = np.int(readdata.readline())
        
        nlpts_data  = np.int(readdata.readline())
        nlpts_space  = np.int(readdata.readline())
        ndata = np.int(readdata.readline())
        lat_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        long_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        depth_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        
        mpirunstring = readdata.readline()
        nsensor_place = np.int(readdata.readline())
        
        #rest of lines are sensors
        sensorlines=readdata.readlines()
        numsen = len(sensorlines)
        
        #lat, long, measurement noise std, length of data vector for sensor, sensor type
        nsensordata = len(np.fromstring(sensorlines[0], dtype=float, sep=','))
        
        sensors = np.zeros([numsen,nsensordata])
        for inc in range(0,numsen):
            sensorline = np.fromstring(sensorlines[inc], dtype=float, sep=',')
            sensors[inc,:] = sensorline
    return nopt_random, nopt_total, sensor_lat_range, sensor_long_range, sensor_params, opt_type, nlpts_data, nlpts_space, ndata, lat_range, long_range, depth_range, sensors, mpirunstring, nsensor_place


def plot_surface(data,
                 output_file='eiggraph_depth0mag0_test.pdf', 
                 depth_slice=0, mag_slice=1, stepsize=100):

    # Specify training data for Gaussian Processs
    target = data['ig']
    print(f'Target shape:{target.shape}')
    print(target)
    inputs = data['theta_data'] # (lat, long, depth, magnitude)
    print(f'Input shape: {inputs.shape}')
    print(inputs)
    # Specify graphing domain
    lat_range = data['lat_range']
    long_range = data['long_range']

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
    # Specify 2d slice (depth and magnitude features)
    domain[:,2] = depth_slice
    domain[:,3] = mag_slice
    
    # Create and fit model, make predictions to generate map
    model = GPR()
    model.fit(inputs,target)
    preds = model.predict(domain)

    print(f'Max pred: {preds.max()}')
    print(f'Min pred: {preds.min()}')
    
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
    plt.title(f'Expected Information Gain for events with depth = {depth_slice}')

    plt.legend()
    plt.savefig(output_file)
    plt.show()
            