#!/usr/bin/env python
# coding: utf-8

import numpy as np

from utils import read_input_file
from sample_gen import sample_prior,descritize_space
from data_gen import generate_data
from like_models import compute_loglikes

import time
import sys
import os


from mpi4py import MPI


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size() #Assume size, ndata, and nlpts all have divisiblity
    rank = comm.Get_rank()

    # make each rank of its own seed
    np.random.seed(int(time.time()) + rank)
    print(rank)