# Package Installation

To clone this repo, first you need to have/generate an ssh key on the system you wish to clone it to.
Then add your ssh key into your GitHub settings.
See the [GitHub documentation](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) for details.
After the key has been added you can clone normally via:

```shell
git clone git@github.com:sandialabs/seismic_boed.git
```

## Required packages

The following Python packages are required in order to run the `seismic-boed` code:

- [Numpy](https://numpy.org)
- [Matplotlib](https://matplotlib.org)
- [Obspy](https://docs.obspy.org)
- [Scikit-optimize](https://scikit-optimize.github.io/stable)
- [Sobol-seq](https://pypi.org/project/sobol-seq/)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
- [Scikit-learn](https://scikit-learn.org/stable/)
