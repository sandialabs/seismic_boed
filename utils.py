import numpy as np

#Load Configuration
def read_input_file(file):
    with open(file, 'r') as readdata:
        nlpts  = np.int(readdata.readline())
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
    return nlpts, ndata, lat_range, long_range, depth_range, sensors