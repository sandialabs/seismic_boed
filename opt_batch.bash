#!/bin/bash
## Do not put any commands or blank lines before the #SBATCH lines
#SBATCH --nodes=14                   # Number of nodes - all cores per node are allocated to the job
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
#SBATCH --export=ALL		     # export environment variables form the submission env (modules, bashrc etc)

nodes=$SLURM_JOB_NUM_NODES           # Number of nodes - the number of nodes you have requested (for a list of SLURM environment variables see "man sbatch")
cores=16                             # Number MPI processes to run on each node (a.k.a. PPN)
                                     # CTS1 has 36 cores per node, skybridge 16

#mpiexec --bind-to core --npernode $cores --n $(($cores*$nodes)) python ~/st-mcmc-cme/examples/repressilator/repressilator_stmcmc_full.py
python3 -u network_opt.py inputs_opt.dat opt_network.npz 1
