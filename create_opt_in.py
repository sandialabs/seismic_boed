import numpy as np

def write_input_file(file, nlpts_data, nlpts_space, ndata, lat_range, long_range, depth_range, mag_range, sloc_trial, sensor_params, sensors):

    writedata=open(file,"w+")
    writedata.write(str(np.int(nlpts_data)) + "\n")
    writedata.write(str(np.int(nlpts_space)) + "\n")
    writedata.write(str(np.int(ndata)) + "\n")
    
    #,max_line_width=1000 to keep numppy from splitting the sensor description up onto multiple lines.
    writedata.write((np.array2string(lat_range,separator=',',max_line_width=1000)).replace('[','').replace(']','').replace(' ', '') + "\n")
    writedata.write((np.array2string(long_range,separator=',',max_line_width=1000)).replace('[','').replace(']','').replace(' ', '') + "\n")
    writedata.write((np.array2string(depth_range,separator=',',max_line_width=1000)).replace('[','').replace(']','').replace(' ', '') + "\n")
    writedata.write((np.array2string(mag_range,separator=',',max_line_width=1000)).replace('[','').replace(']','').replace(' ', '') + "\n")

    #rest of lines are sensors
    writedata.write((np.array2string(sensors,separator=',',max_line_width=1000)).replace('[','').replace('],','').replace(']','').replace(' ', '') + "\n")

    #lat, long, measurement noise std, length of data vector for sensor, sensor type
    #write teh new sensor
    writedata.write((np.array2string(sloc_trial,separator=',',max_line_width=1000)).replace('[','').replace(']','').replace(' ', '') + "," + (np.array2string(sensor_params,separator=',',max_line_width=1000)).replace('[','').replace(']','').replace(' ', '') + "\n")

    writedata.close()
    return