# Seismic Bayesian Optimal Experiment Design in Python

This code is a Python implementation of Bayesian optimal experiment design (BOED)
for seismic monitoring networks.
This code provides the tools necessary to analyze and optimize seismic
monitoring networks.
Currently we target the location problem in which
we want to study how well the network will identify the location of an
event and then optimize the network to provide better locations. The
user can specify models for generating synthetic data and assessing the
likelihood of that synthetic data for different sensors and events in
the domain of candidate events. The code is designed to use MPI so that
it can run on HPC resources because OED is computationally expensive.

See [**Installation**](./sboed/installation.md) and the first tutorial,
[**Getting Started**](./tutorials/basics.ipynb) to begin using
this code. More details on the theory and application can be found in [the
accompanying paper](nonexistent_arxiv_link.com).

```{tableofcontents}

```
