# Installing Python Packages on HPC with pip
###[tacatan@skybridge-login7 seismic_oed]$ export http_proxy="http://user:pass@proxy.sandia.gov:80"
###[tacatan@skybridge-login7 seismic_oed]$ export https_proxy="http://user:pass@proxy.sandia.gov:80"
###[tacatan@skybridge-login7 seismic_oed]$ pip install --cert=/etc/pki/ca-trust/extracted/openssl/ca-bundle.trust.crt --user scikit-optimize

# Basic OED
## You can run via mpiexec -n 5 python3 eig_calc.py inputs.dat outputs.npz 0

