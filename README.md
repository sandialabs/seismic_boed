# Seismic Optimal Experimental Design (OED)

## Overview
This code provides the functionality to 

## Contents
<ol>
<ol>
# Cloning this repo
To clone this repo, first you need to have/generate an ssh key on the system you wish to clone it to. Then add your ssh key into your cee-gitlab settings. See [https://cee-gitlab.sandia.gov/-/profile/keys](https://cee-gitlab.sandia.gov/-/profile/keys) for details. After the key has been added you can clone normally via:<br>
git clone git@cee-gitlab.sandia.gov:tacatan/seismic_oed.git

# Required Python Packages
## List of Packages
<ol>
<li>Numpy [https://numpy.org/](https://numpy.org/)</li>
<li>Obspy [https://docs.obspy.org/](https://docs.obspy.org/)</li>
<li>Scikit-optimize [https://scikit-optimize.github.io/stable/](https://scikit-optimize.github.io/stable/)</li>
<li>Sobol-seq [https://pypi.org/project/sobol-seq/](https://pypi.org/project/sobol-seq/)</li> 
<li>mpi4py [https://mpi4py.readthedocs.io/en/stable/](https://mpi4py.readthedocs.io/en/stable/)</li>
<li>Scikit-learn [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)</li>
</ol>

## Installing Python Packages on Sandia HPC with pip (e.g. numpy)
[tacatan@skybridge-login7 seismic_oed]$ export http_proxy="http://user:pass@proxy.sandia.gov:80"<br>
[tacatan@skybridge-login7 seismic_oed]$ export https_proxy="http://user:pass@proxy.sandia.gov:80"<br>
[tacatan@skybridge-login7 seismic_oed]$ pip install --cert=/etc/pki/ca-trust/extracted/openssl/ca-bundle.trust.crt --user numpy<br>

# Network Analysis

## Define an input file e.g. inputs.dat
|Line Number| Description        | Example      |
| -------------| ------------- |:-------------:| 
|Line 1| # Synthetic events to test | 32 |
|Line 2| # Possible Events in the event space | 8192 |
|Line 3| # Realizations of Data per event | 2 |
|Line 4| Latitude Range | 40.0, 42.0 |
|Line 5| Longitude Range | -112.0, -108.38 |
|Line 6| Depth Range | 0.0, 40.0 |
|Line 7| Sensor 1: Lat, Long, Noise Std, Length of sensor output vec, Sensor type| 40.0, -111.5,0.1,2,0 |
|Line 8| Sensor 2: Lat, Long, Noise Std, Length of sensor output vec, Sensor type| 41.0, -111.9,0.1,2,0 |
|... | ... | ... |
|Line 6+N| Sensor N: Lat, Long, Noise Std, Length of sensor output vec, Sensor type| 40.0, -110.0,0.1,2,0 |

Note that for HPC simulations, # Synthetic events to test and # Possible Events in the event space must be divisible by the number of cores. For example this configuration is valid if ncores = 1,2,4,8,16, or 32.

## Running the Analysis code (eig_calc.py)
This code assumes you are running it on skybridge where the number of cores per node is 16. For ghost the number of cores is 36. If you are running locally on your machine, you do not have to worry about the number of cores per node. For HPC systems, before running the code you have to request access to a certain number of cores/nodes. This can be done either interactively or with a batch script.

### Running on Sandia HPC Interactively

To submit an Interactive Job: salloc -N2 --time=2:00:00 --account=FY210056 --partition=short,batch
This code request 2 node (-N) for 2 hours. The account number for seismic monitoring research is FY210056. This is requesting the nodes from either the short or batch partition. See the Sandia HPC documentation for more details [https://computing.sandia.gov/](https://computing.sandia.gov/).

Once you have an allocation, you can now submit the job. This is done with the following syntax:<br>
mpiexec --bind-to core --npernode [#cores per node] --n [#Total cores] python3 eig_calc.py [Input File] [Ouptut Numpy File] [Verbose]

For example on skybridge:<br>
mpiexec --bind-to core --npernode 16 --n 32 python3 eig_calc.py inputs.dat outputs.npz 1

Verbose has three levels. 0 has no output other than the final EIG estimate, EIG Standard Deviation, and Statistic of EIG effective sample size. 1 is the most verbose with printed statements throughout the code describing what is going on and all the different calculated quantieis are stored in the output file. 2 is a more limited version of 1 with less IO and quantities stored in the output file.


### Running on Sandia HPC with script

First make a script for example eig_batch.bash:

```bash
#!/bin/bash
## Do not put any commands or blank lines before the #SBATCH lines
#SBATCH --nodes=16                   # Number of nodes - all cores per node are allocated to the job
#SBATCH --time=04:00:00              # Wall clock time (HH:MM:SS) - once the job exceeds this time, the job will be terminated (default is 5 minutes)
#SBATCH --account=FY210056           # WC ID
#SBATCH --job-name=seis_eig          # Name of job
#SBATCH --partition=short            # partition/queue name: short or batch
                                      #            short: 4hrs wallclock limit
                                      #            batch: nodes reserved for > 4hrs (default)
#SBATCH --qos=normal                  # Quality of Service: long, large, priority or normal
                                      #           normal: request up to 48hrs wallclock (default)
                                      #           long:   request up to 96hrs wallclock and no larger than 64nodes
                                      #           large:  greater than 50% of cluster (special request)
                                      #           priority: High priority jobs (special request)
#SBATCH --export=ALL                 # export environment variables form the submission env (modules, bashrc etc)

nodes=$SLURM_JOB_NUM_NODES           # Number of nodes - the number of nodes you have requested (for a list of SLURM environment variables see "man sbatch")
cores=16                             # Number MPI processes to run on each node (a.k.a. PPN)
                                     # CTS1 has 36 cores per node, skybridge 16

mpiexec --bind-to core --npernode $cores --n $(($cores*$nodes)) python3 eig_calc.py inputs.dat outputs.npz 0
```

This can then be submitted by:<br>
sbatch eig_batch.bash


### Running locally with mpi4py
Running eig_calc locally is much the same other than you do not need to deal with requesting cores and defining the cores per node. For example:<br>
mpiexec -n 4 python3 eig_calc.py inputs.dat outputs.npz 0

This will run locally on 4 cores.

## Analysis outputs
At the end of running eig_calc.py the code will print to screen the following results: EIG, standard deviation of the EIG, and the minimum effective sample size (ESS) for all realizations of synthetic data of the weighted samples that make up the posterior distribution estimate. In some sense, the std and ess numbers relate to the variance and bias of the EIG estimator.

Additionally, an output numpy file (e.g. outputs.npz) is created. The variables stored in this file depend on the verbose flag. They are given in the following table. 

|Variable Name| Description        | Verbose Levels |
| -------------| ------------- |:-------------:| 
|eig| Estimated Expected Information Gain (EIG) overall experiments | 0,1,2 |
|seig| Standard Deviation of Estimated EIG overall experiments | 0,1,2 |
|ig|  Information Gain assessed for each experiment| 1,2  |
|ess| Effective Sample Size (ESS) of weighted posterior samples for each experiment| 1,2 |
|miness| Min ESS overall experiments  | 0,1,2 |
|theta_data| Paramters (lat,long,depth,magnitude) of synthetic events | 1 |
|theta_space|Parameters (lat,long,depth,magnitude) that define the posterior event space | 1,2 |
|sensors| Sensor network configuration | 1,2 |
|lat_range| Latitude Range of Events | 1,2 |
|long_range| Longitude Range of Events | 1,2 |
|depth_range| Depth Range of Events | 1,2 |
|loglikes| Loglikelihood of event candidate event for every synthetic experiment | 1 |
|dataz| Data used of each synthetic experiment | 1 |

# Network Optimization

## Define an input file e.g. inputs_opt.dat
|Line Number| Description        | Example      |
| -------------| ------------- |:-------------:| 
|Line 1| # Random initial sensors to build GP model | 8 |
|Line 2| # Sensors to test during each optimization level | 16 |
|Line 3| Sensor Latitude Range | 40.0, 42.0 |
|Line 4| Sensor Longitude Range | -112.0, -108.38 |
|Line 5| Fixed Sensor Parameters:  Noise Std, Length of sensor output vec, Sensor type | 0.1,2,0 |
|Line 6| Optimization Objective (Currenlty only 0 i.e. EI is supported) | 0 |
|Line 7| # Synthetic events to test | 512 |
|Line 8| # Possible Events in the event space | 8192 |
|Line 9| # Realizations of Data per event | 8 |
|Line 10| Event Latitude Range | 40.0, 42.0 |
|Line 11| Event Longitude Range | -112.0, -108.38 |
|Line 12| Event Depth Range | 0.0, 40.0 |
|Line 13| MPI string to run code | mpiexec --bind-to core --npernode 16 --n 256 |
|Line 14| # sensors to add e.g. number of optimization levels| 5 |
|Line 15| Sensor 1: Lat, Long, Noise Std, Length of sensor output vec, Sensor type| 40.0, -111.5,0.1,2,0 |
|Line 16| Sensor 2: Lat, Long, Noise Std, Length of sensor output vec, Sensor type| 41.0, -111.9,0.1,2,0 |
|... | ... | ... |
|Line 14+N| Sensor N: Lat, Long, Noise Std, Length of sensor output vec, Sensor type| 40.0, -110.0,0.1,2,0 |

This code will take initial set of N sensors defined by this list and then iteratively add the number of sensors defined by line 14 e.g. 5 in this example to the existing network.

Note that for HPC simulations, # Synthetic events to test and # Possible Events in the event space must be divisible by the number of cores. For example this configuration is valid if ncores that is a power of 2 up to 512.

## Running the Optimization code (network_opt.py)
There are similar assumptions about running the optimization code as running the analysis code. The optimization code in effect just calls the analysis code as part of the optimization loop. Therefore, the optimization code does not use MPI since it is just a wrapper around the analysis code that uses MPI to compute the EIG.

### Running on Sandia HPC Interactively
First request an allocation like before. Here the number of nodes and cores needs to line up with what is defined in teh optimization input file (e.g. Line 13 of inputs_opt.dat) e.g.:<br>
To submit an Interactive Job: salloc -N16 --time=2:00:00 --account=FY210056 --partition=short,batch

Then run the code. This is done with the following syntax:<br>
python3 network_opt.py [Input File] [Ouptut Numpy File] [Verbose]

For example on skybridge:<br>
python3 network_opt.py inputs_opt.dat opt_network.npz 1

There are two levels of verbose. 0 corresponds to no output except the final list of sensors in the output file e.g. opt_network.npz. 1 gives print statements to describe how the optimzation process is proceeding. At every step the results of the optimization model and the EIG statistics used to fit that model are saved in the files result*.pkl and result_eigdata*.npz respectively, where * is the sensor number being placed.

### Running on Sandia HPC with script

First make a script for example opt_batch.bash:

```bash
#!/bin/bash
## Do not put any commands or blank lines before the #SBATCH lines
#SBATCH --nodes=16                   # Number of nodes - all cores per node are allocated to the job
#SBATCH --time=04:00:00              # Wall clock time (HH:MM:SS) - once the job exceeds this time, the job will be terminated (default is 5 minutes)
#SBATCH --account=FY210056           # WC ID
#SBATCH --job-name=seis_opt          # Name of job
#SBATCH --partition=short            # partition/queue name: short or batch
                                      #            short: 4hrs wallclock limit
                                      #            batch: nodes reserved for > 4hrs (default)
#SBATCH --qos=normal                  # Quality of Service: long, large, priority or normal
                                      #           normal: request up to 48hrs wallclock (default)
                                      #           long:   request up to 96hrs wallclock and no larger than 64nodes
                                      #           large:  greater than 50% of cluster (special request)
                                      #           priority: High priority jobs (special request)
#SBATCH --export=ALL                 # export environment variables form the submission env (modules, bashrc etc)

nodes=$SLURM_JOB_NUM_NODES           # Number of nodes - the number of nodes you have requested (for a list of SLURM environment variables see "man sbatch")
cores=16                             # Number MPI processes to run on each node (a.k.a. PPN)
                                     # CTS1 has 36 cores per node, skybridge 16

python3 network_opt.py inputs_opt.dat opt_network.npz 1
```

This can then be submitted by:<br>
sbatch opt_batch.bash

### Running locally
Running network_opt.py locally is much the same others. The input file e.g. inputs_opt.dat will need to be modified so that the mpi script corresponds to how you run MPI code locally. For example, line 13 for my machine would be "mpiexec -n 8" to run on 8 cores.

Then to run the code simply run it like in the other examples:<br>
python3 network_opt.py inputs_opt.dat opt_network.npz 1

## Optimization output
When network_opt.py finishes, it will display the optimized sensor nework configuration e.g. for each sensor its lat, long, noise level, number of output variables, and sensor type. This network configuration will then be saved in the output numpy file (e.g. opt_network.npz). 

Additionally, if the verbose flag is set to 1, two files are created per optimization level where a new sensor number is being places. The first file is result*.pkl, where * is the sensor number being placed. This file contains an optization result object. This object contains information about the GP surrogate used to find the optimization objective and the data used to fit it. More information can be found in the documentation [https://scikit-optimize.github.io/stable/auto_examples/store-and-load-results.html](https://scikit-optimize.github.io/stable/auto_examples/store-and-load-results.html). The second file, result_eigdata*.npz is a numpy file that contains three variables: sensors, eigdata_full, and Xs. sensors lists the current network configuration before optimization. eigdata_full contains the [EIG, std EIG, minESS] for each trial new sensor location to augment the current network. Xs includes the trial sensor locations.

# Configuring the Bayesian OED models
There are several basic compoents to the Bayesian model for OED that need to be specified. The first three componets are for both optimization and analysis. The first is how we sample the event space. This means how we are going to generate our synthetic events for generating data and our set of candidate events that we use during inference. Second is sampling the synthetic data whcih means how we take a synthetic event and a sensor configuration and then generate a synthetic dataset for that event. Third, are the likelihood models that we use in Bayesian inference to compute the posterior probablity of our candidate events given the synthetic data. Finally, if we are also doing optimization, we need to define how we generate random trial sensor configuratin to seed the optimization algorithm.

## Sampling the event space
The module sample_gen.py contains two functions sample_prior and descritize_space that are used to sample the event space. Modifying these functions can be used to change the model. The two functions serve very similar perposes, e.g. returning a set of events, so for many applications they can be the same. An event corresponds to the theta vector that contains the full distribution we are considering about an event like an earthquake or explosion. Both of these functions take as input the variables lat_range, long_range, depth_range, nsamp, and skip. The variables lat_range, long_range, and depth_range correspond to the limits of our spatial domain for generating events. The variable nsamp defines how many events we want to generate. And skip is a seed variable that tells us how to start our quasi-random number generator so that we dont duplicate events. Both these codes return the vairable sbvals which contain the description of the event. This event description can have any dimension that are set by modifying these codes. In the examples provided, it is 4D corresponding to lat, long, depth, and event magnitude. In principle, other characteristics could be added such as a seismic moment tensor or a source time function parameter. The provided codes uses the quasi monte carlo sobol sequence to generate a uniform set of events, but any sampling method could be used in these codes.

The sample_prior function returns a set of events generated from the prior distribution over data generating events. These events will be used to generate the synthetic data and in the code are called theta_data. For computing EIG, the prior distribution over data generating events should be the prior distribution over event hypothesies. However, for some applications it may make sense to bias this distribution meaning that you care more about EIG about a certain type of events. For example, you may only care about EIG for events less than magnitude 2 or events within 1km of the surface. This information could be used to bias the distribution.

```python
def sample_prior(lat_range,long_range, depth_range, nsamp, skip):
    dim_num = 4
    sbvals = np.full((nsamp, dim_num), np.nan)
    for j in range(nsamp):
        sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1+skip+j)
    
    sbvals[:,0] = sbvals[:,0]*(lat_range[1] - lat_range[0])+lat_range[0]
    sbvals[:,1] = sbvals[:,1]*(long_range[1] - long_range[0])+long_range[0]
    sbvals[:,2] = sbvals[:,2]*(depth_range[1] - depth_range[0])+depth_range[0]
    sbvals[:, 3] = -np.log(1 - sbvals[:,3]) / np.log(10)
    sbvals[:, 3] += 0.5
    
    return sbvals
```
The descritize_space function returns a set of events distributed according to the prior over event hypotheses e.g. our prior knowledge in Bayesian inference. These events will be used to define the space of candidate events whose likelhood we will infer from the synthetic data. In the code this is the variable theta_space. These finite number of events from the prior will in effect be used to discretize the posterior distribution so that solving the Bayesian inference problem is easier. In the current version of the code, these events must be distributed according to the prior over candiate events.
```python
def descritize_space(lat_range,long_range, depth_range, nsamp,skip):
    dim_num = 4
    sbvals = np.full((nsamp, dim_num), np.nan)
    for j in range(nsamp):
        sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1+skip+j)    
    
    sbvals[:,0] = sbvals[:,0]*(lat_range[1] - lat_range[0])+lat_range[0]
    sbvals[:,1] = sbvals[:,1]*(long_range[1] - long_range[0])+long_range[0]
    sbvals[:,2] = sbvals[:,2]*(depth_range[1] - depth_range[0])+depth_range[0]
    sbvals[:, 3] = -np.log(1 - sbvals[:,3]) / np.log(10)
    sbvals[:, 3] += 0.5
    
    return sbvals
```

## Sampling synthetic data
The module data_gen.py contains the functions needed to generate synthetic data. The core function is generate_data. This function takes as input three variables: theta, sensors, and ndata. Theta is an event descrption (e.g. lat, long, depth, mag). Sensors is the network configuration (e.g. lat, long, noise std, num variables, and sensor type for each sensor). And ndata is the number of synthetic data realizations to generate for each data generating event. This function returns the synthetic data for each sensor for each of the data realizations with this set of event characteristics.

Inside the generate_data function, the data generating functions are very flexible and can be modified to be anything. However, it is important that these data generating functions correspond to the likelihood functions e.g. that the data D is in fact distributed according to the likelihood D ~ p(D|theta). Therefore, when constructing these functions it is often helpful to call functions from the like_models.py module imporated as lm.

In this example, the data generate for the sensors has 2 parts. First is just an indicator function that registers 1 if the sensor detects an event and 0 otherwise. The second is the time at which the station registers the event. The functions lm.detection_probablity and gen_arrival_normal are used to generate this data.

```python
def generate_data(theta,sensors,ndata):
    probs = lm.detection_probability(theta,sensors)
    fullprobs = np.outer(np.ones(ndata),probs)
    u_mat = np.random.uniform(size = fullprobs.shape)
    
    atimes = gen_arrival_normal(theta, sensors, ndata)    
    data = np.concatenate((atimes,u_mat<fullprobs),axis=1)

    return data
```

## Constructing the likelihood
In module like_models.py we have our likleihood funciton models. Therefore, this is very similar to the data_gen.py module except instead of generating the synthetic data given an event, it computes the log liklihood of synthetic data given an event. The core function for this is compute_loglikes. This function takes as input the variables theta, sensors, and data. Theta coresponds to the event hypothesis whose likelihood we want to assess. Sensors corresonds to the sensor network configuration. And data, is the synthetic data that we want to compute the likelihood of for each sensor. In this code, data is the full dataset for all experiments since for each experiment we need to compute the likelihood for each event hypothesis so it is most efficent to do it in this vectorized form. So data has dimensions [nlpts_data * ndata, # of sensors * Length of sensor output vec] and corresponds to the variables dataz in the eig_calc.py code. The output, loglikes, of this code is the log likelihood of the data given the event hypothesis theta and has dimensions [nlpts_data * ndata].

Within the compute_loglike funciton, any sensor type model can be implemented as long as it agrees with the models used in the data_gen.py module. In this example, the likelihood, given the event hypothesis theta, is computed based on the probablity of detecting an arrival at each station and if an arrival is detected then the probability of detecting an arrival with that arrival time. Other likleihood models could be easily added to the module and put into this funciton.
```python
def compute_loglikes(theta,sensors,data):
    dloglikes = detection_likelihood(theta,sensors,data)
    aloglikes = arrival_likelihood_gaussian(theta, sensors, data)
    loglikes = dloglikes + aloglikes
    
    return loglikes
```

## Sampling sensor locations
The module sample_gen.py contains a functions sample_sensors that is used to generate random sensor locations. This is useful for seeding the optimization problem. These initial stations are used to build an initial GP model of the optimization surface that can then be refined through Bayesian optimzation. It takes as input the variables: lat_range, long_range, nsamp, and skip. Lat_range and Long_range define the domain on which we can place sensors. Nsamp defines the number of quasi-random sensors we want to generate. Skip defines the seed of the quasi random generator. The output of the code, sbvals, is a set of sensor locations. In principle any method could be used to generate the random stations but in this code we use a quasi random uniform distribution through the Sobol sequence. 

```python
def sample_sensors(lat_range,long_range, nsamp,skip):
    dim_num = 2
    sbvals = np.full((nsamp, dim_num), np.nan)
    for j in range(nsamp):
        sbvals[j, :], _ = sq.i4_sobol(dim_num, seed=1+skip+j)    
    
    sbvals[:,0] = sbvals[:,0]*(lat_range[1] - lat_range[0])+lat_range[0]
    sbvals[:,1] = sbvals[:,1]*(long_range[1] - long_range[0])+long_range[0]
    
    return sbvals
```
