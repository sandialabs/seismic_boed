import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

import sys
import time
from datetime import datetime
import os

def select_training_samples(samples, targets, depth_slice, mag_slice, depth_tol, mag_tol):
    depth_low = depth_slice - depth_tol
    depth_high = depth_slice + depth_tol

    mag_low = mag_slice - mag_tol
    mag_high = mag_slice + mag_tol

    # Mask that selects samples whose magnitude and depth are in the desired range
    mask = [((samples[:,2]<=depth_high) & (samples[:,2]>=depth_low)) & 
            ((samples[:,3]<=mag_high) & (samples[:,3]>=mag_low))]

    training_inputs = samples[tuple(mask)]
    training_targets = targets[tuple(mask)]

    return training_inputs, training_targets


def plot_surface(data,
                 t0,
                 output_path='eig_plots', 
                 depth_step=1, mag_step=1,
                 depth_tol=10, mag_tol=.5, 
                 stepsize=100):

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

    # print(f'Training GP model: {time.time() -t0}')
    # model = GPR()
    # model.fit(inputs,target)
    
    # Enumerate slices to visualize
    depth_slices = np.arange(depth_range[0], depth_range[1]+depth_step, depth_step)
    mag_slices = np.arange(mag_range[0], mag_range[1]+mag_step, mag_step)

    now = datetime.now()
    timestamp = f'{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}'
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
            
            training_inputs, training_targets = select_training_samples(inputs, target, 
                                                                        depth_slice, mag_slice, 
                                                                        depth_tol, mag_tol)

            # Train GP to generate surface
            if training_inputs.shape[0]==0:
                print(f'No samples found within {depth_tol}m of depth slice and within {mag_tol} of mag slice, skipping this plot')
                curr_plot += 1
                continue

            print(f'Training GP model: {time.time() -t0}')
            model = GPR()
            model.fit(training_inputs, training_targets)
    #             
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


if __name__ == '__main__':
    data_file = sys.argv[1]
    control_file = sys.argv[2]

    t0 = time.time()

    print('Loading data: {t0}')
    with open(control_file, 'r') as f:
        depth_step, mag_step = np.fromstring(f.readline(), dtype=float, sep=',')
        depth_tol, mag_tol = np.fromstring(f.readline(), dtype=float, sep=',')
    
    data = np.load(data_file)

    plot_surface(data, t0, depth_step, mag_step, depth_tol, mag_tol)





