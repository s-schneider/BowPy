from __future__ import absolute_import, print_function

import numpy
import numpy as np
from numpy import genfromtxt
import math

import matplotlib

# If using a Mac Machine, otherwitse comment the next line out:
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
import scipy.signal as signal
import scipy.io as sio

import os
import datetime

import obspy
from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees, locations2degrees
from obspy.taup import TauPyModel
from obspy import read as read_st
from obspy import read_inventory as read_inv
from obspy import read_events as read_cat

import sipy
import sipy.misc.Muenster_Array_Seismology_Vespagram as MAS
import sipy.filter.fk as fk
import sipy.filter.radon as radon
import sipy.util.fkutil as fku
import sipy.util.base as base
import sipy.util.array_util as au

from sipy.util.data_request import data_request
from sipy.filter.fk import fk_filter, fk_reconstruct, pocs_recon
from sipy.util.fkutil import  nextpow2, find_subsets, slope_distribution, makeMask, create_iFFT2mtx, fktrafo
from sipy.util.array_util import get_coords, attach_network_to_traces, attach_coordinates_to_traces,\
stream2array, array2stream, attach_network_to_traces, attach_coordinates_to_traces, attach_epidist2coords, epidist2nparray, epidist2list, \
alignon, partial_stack, gaps_fill_zeros, vespagram
from sipy.util.picker import get_polygon

import os
import sys

def qtest_pocs(st_rec, st_orginal, alpharange, alphadecrease, irange):
	"""
	Runs the selected method in a certain range of parameters (iterations and alpha), returns a table of Q values ,defined as:

	Q = 10 * log( || d_org || ^2 _2  / ||  d_org - d_rec || ^2 _2 )

	The highest Q value is the one to be chosen.
	"""

	method  = 'reconstruct'
	dmethod	= 'linear'

	st_org  = st_orginal.copy()
	data_org= stream2array(st_org)

	for alpha in alpharange:
		
		print("##################### CURRENT ALPHA %f  #####################\n" % alpha )
		for i in irange:

			print('POCS RECON WITH %i ITERATIONS' % maxiter, end="\r")
			sys.stdout.flush()
			srs = st_rec.copy()
			st_pocsrec = pocs_recon(srs, maxiter=i, method=method, dmethod=dmethod, alpha=alpha)
			drec = stream2array(st_pocsrec, normalize=True)
			Q_tmp = np.linalg.norm(data_org,2)**2. / np.linalg.norm(data_org - drec,2)**2.
			Q = 10.*np.log(Q_tmp)	
			Qall.append([alpha, maxiter, Qlin])


	return Qall

def qtest_plot():

	return



