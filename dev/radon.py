import obspy
from obspy.core.util.geodetics import gps2DistAzimuth, kilometer2degrees, locations2degrees
from obspy import read as read_st
from obspy import read_inventory as read_inv
from obspy import readEvents as read_cat
from obspy.taup import TauPyModel

import numpy
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
import scipy.signal as signal
from numpy import genfromtxt

import Muenster_Array_Seismology_Vespagram as MAS
from Muenster_Array_Seismology import get_coords

import os
import datetime

import fk_work
import fk_work as fkw
from fk_work import fk_filter

def radon_inverse(t,delta,M,p,weights,ref_dist,line_model,inversion_model,hyperparameters):
#This function inverts move-out data to the Radon domain given the inputs:
# -t        -- vector of time axis.
# -delta    -- vector of distance axis.
# -M        -- matrix of move-out data, ordered size(M)==[length(delta),length(t)].
# -p        -- vector of slowness axis you would like to invert to.
# -weights  -- weighting vector that determines importance of each trace.
#              set vector to ones for no preference.
# -ref_dist -- reference distance the path-function will shift about.
#
# -line_model, select one of the following options for path integration:
#     'linear'     - linear paths in the spatial domain (default)
#     'parabolic'  - parabolic paths in the spatial domain.
#
# -inversion model, select one of the following options for regularization schema:
#     'L2'       - Regularized on the L2 norm of the Radon domain (default)
#     'L1'       - Non-linear regularization based on L1 norm and iterative
#                  reweighted least sqaures (IRLS) see Sacchi 1997.
#     'Cauchy'   - Non-linear regularization see Sacchi & Ulrych 1995
#
# -hyperparameters, trades-off between fitting the data and chosen damping.
#
#Output radon domain is ordered size(R)==[length(p),length(t)].
#
#Known limitations:
# - Assumes evenly sampled time axis.
# - Assumes move-out data isn't complex.
#
#
# References: Schultz, R., Gu, Y. J., 2012. Flexible Matlab implementation
#             of the Radon Transform.  Computers and Geosciences [In Preparation]
#
#             An, Y., Gu, Y. J., Sacchi, M., 2007. Imaging mantle 
#             discontinuities using least-squares Radon transform. 
#             Journal of Geophysical Research 112, B10303.
#
# Author: R. Schultz, 2012
# Adapted by: S. Schneider, 2016

# Define some array/matrices lengths.
	it=len(t)
	iF=math.pow(2,nextpow2(it)+1) # Double length
	iDelta=len(delta)
	ip=len(p)
	iw=len(weights)

#Exit if inconsistent data is input.
"""
MATLAB ORIGINAL:
  if(min([iDelta,it]~=size(M)))
      fprintf('Dimensions inconsistent!\nsize(M)~=[length(delta),length(t)]\n');
      R=0;
      return;
  end;
  if(iw~=iDelta)
      fprintf('Dimensions inconsistent!\nlength(delta)~=length(weights)\n');
      R=0;
      return;
  end;
"""
	if 

	return()

def nextpow2(i):
	n = 1
	count = 0
	while n < abs(i):
		n *= 2
		count+=1
	return count
