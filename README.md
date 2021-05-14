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

# Running Network Optimization or Analysis Code

## Network Analysis

### Define an input file e.g. inputs.dat
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

## Network Optimization

### Define an input file e.g. inputs_opt.dat
|Line Number| Description        | Example      |
| -------------| ------------- |:-------------:| 
|Line 1| # Random initial sensors to build GP model | 8 |
|Line 2| # Sensors to test during each optimization level | 16 |
|Line 3| Sensor Latitude Range | 40.0, 42.0 |
|Line 4| Sensor Longitude Range | -112.0, -108.38 |
|Line 5| Fixed Sensor Parameters:  Noise Std, Length of sensor output vec, Sensor type | 0.1,2,0 |
|Line 6| Optimization Objective (Currenlty on 0 -> EI is supported) | 0 |
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
