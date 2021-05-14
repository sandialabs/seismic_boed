import numpy as np
import os

#Load Configuration
def read_input_file(file):
    with open(file, 'r') as readdata:
        nlpts_data  = np.int(readdata.readline())
        nlpts_space  = np.int(readdata.readline())
        ndata = np.int(readdata.readline())
        lat_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        long_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        depth_range = np.fromstring(readdata.readline(), dtype=float, sep=',')
        
        #rest of lines are sensors
        sensorlines=readdata.readlines()
        numsen = len(sensorlines)
        
        #lat, long, measurement noise std, length of data vector for sensor, sensor type
        nsensordata = len(np.fromstring(sensorlines[0], dtype=float, sep=','))
        
        sensors = np.zeros([numsen,nsensordata])
        for inc in range(0,numsen):
            sensorline = np.fromstring(sensorlines[inc], dtype=float, sep=',')
            sensors[inc,:] = sensorline
    return nlpts_data, nlpts_space, ndata, lat_range, long_range, depth_range, sensors


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