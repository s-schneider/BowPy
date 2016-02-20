#Example script for Radon transform

import numpy as np
import scipy.io as sio
import datetime
import math
from radon import nextpow2
from scipy import sparse

import radon


data = sio.loadmat("mtz_radon/Radon/data.mat")

# t        - time axis.
# Delta    - distance (offset) axis.
# M        - Amplitudes of phase arrivals.
# indicies - list of indicies relevent to the S670S phase.

# Define some variables for RT.

t = data['t']
IDelta = data['Delta']
M = data['M']
indicies = data['indicies']

mu=[5e-2]
P_axis=np.arange(-1,1.01,0.01)
Idelta = np.mean(IDelta)

# Invert to Radon domain using unweighted L2 inversion, linear path
# functions and an average distance parameter.

delta=IDelta
p = P_axis
weights = np.ones(IDelta.size)
ref_dist =Idelta
hyperparameters = mu
line_model="Linear"
inversion_model="L2"

tic = datetime.datetime.now()
R=radon.radon_inverse(t, IDelta, M, P_axis, np.ones(IDelta.size), Idelta, "Linear", "L2", mu)
#radon_inverse(t,delta,M,p,weights,ref_dist,line_model,inversion_model,hyperparameters)



toc = datetime.datetime.now()
time = toc-tic
print( "Elapsed time is %s seconds." % str(time))
