"""
Testscript for all cases
"""
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
from sipy.filter.fk import fk_filter, fktrafo, fk_reconstruct
from sipy.util.fkutil import  nextpow2, find_subsets, slope_distribution, makeMask, create_iFFT2mtx
from sipy.util.array_util import get_coords, attach_network_to_traces, attach_coordinates_to_traces,\
stream2array, array2stream, attach_network_to_traces, attach_coordinates_to_traces, attach_epidist2coords, epidist2nparray, epidist2list, \
alignon, partial_stack, gaps_fill_zeros, vespagram
from sipy.util.picker import get_polygon

import os

noise = np.fromfile('../data/test_datasets/randnumbers/noisearr.txt')
noise = noise.reshape(20,300)

with open("../data/test_datasets/ricker/rickerlist.dat", 'r') as fh:
	rickerlist = np.array(fh.read().split()).astype('str')

noisefoldlist = ["no_noise","10pct_noise", "20pct_noise", "50pct_noise", "60pct_noise", "80pct_noise"]
noiselevellist = np.array([0., 0.1, 0.2, 0.5, 0.6, 0.8]) 

peaks = np.array([[-13.95      ,   6.06      ,  20.07      ],[  8.46648822,   8.42680793,   8.23354933]])
errors = []
#stream = read_st(PATH)

for i, noisefolder in enumerate(noisefoldlist):
	print("##################### NOISELVL %i %% #####################\n" % int(noiselevellist[i] * 100.) )
	for filein in rickerlist:
		print("##################### CURRENT FILE %s  #####################\n" % filein )
		PICPATH = filein[:filein.rfind("/"):] + "/" + noisefolder + "/"
		if not os.path.isdir(PICPATH):
			os.mkdir(PICPATH)
		PATH = filein
		stream = read_st(PATH)
		#if i != 0:
		data = stream2array(stream.copy(), normalize=True) + noiselevellist[i] * noise
		srs = array2stream(data)

		
		prename = 	'boxcar_auto_noise_' + str(noiselevellist[i]) +  'orig' + '.png'
		prepath = PICPATH + prename
		fku.plot(srs, savefig=prepath)
		name = 'boxcar_auto_noise_' + str(noiselevellist[i]) +  '.png'
		picpath = PICPATH + name
		st_rec = fk_reconstruct(srs, slopes=[-2,2], deltaslope=0.001, maskshape=['boxcar', None], solver='ilsmr',method='interpolate', mu=42, tol=1e-12, peakinput=peaks)
		st_rec.normalize()
		fku.plot(st_rec, savefig=picpath)

		prename ='boxcar_size1_noise_' + str(noiselevellist[i]) +  'orig' + '.png'
		prepath = PICPATH + prename
		fku.plot(srs, savefig=prepath)
		name = 'boxcar_size1_noise_' + str(noiselevellist[i]) +  '.png'
		picpath = PICPATH + name
		st_rec = fk_reconstruct(srs, slopes=[-2,2], deltaslope=0.001, maskshape=['boxcar', 1], solver='ilsmr',method='interpolate', mu=42, tol=1e-12, peakinput=peaks)
		st_rec.normalize()
		fku.plot(st_rec, savefig=picpath)


		taperrange = [0.5, 1, 1.5]
		for ts in taperrange:
			print("##################### %s, NOISE: %f, :CURRENTLY TAPERING WITH %f  #####################\n" % (filein, int(noiselevellist[i] * 100.), ts) )
			try:
				prename = 'taper_' + str(ts) + "_" + str(noiselevellist[i]) +  'orig' + '.png'
				prepath = PICPATH + prename
				fku.plot(srs, savefig=prepath)
				st_rec = fk_reconstruct(srs, slopes=[-2,2], deltaslope=0.001, maskshape=['taper', ts], solver='ilsmr',method='interpolate', mu=42, tol=1e-12, peakinput=peaks)
				name = 'taper_' + str(ts) + "_" + str(noiselevellist[i]) +  '.png'
				picpath = PICPATH + name
				st_rec.normalize()
				fku.plot(st_rec, savefig=picpath)
			except:
				error.append(picpath)
				continue

		bwrange = [1,2,4,8]
		for bw in bwrange:
			print("##################### %s, NOISE: %f, :CURRENTLY BUTTERWORTHING WITH %f  #####################\n" % (filein, int(noiselevellist[i] * 100.), bw) )
			try:
				prename = 'butterworth_' + str(bw) + "_" + str(noiselevellist[i]) +  'orig' + '.png'
				prepath = PICPATH + prename
				fku.plot(srs, savefig=prepath)
				st_rec = fk_reconstruct(srs, slopes=[-2,2], deltaslope=0.001, maskshape=['butterworth', bw], solver='ilsmr',method='interpolate', mu=42, tol=1e-12, peakinput=peaks)
				name = 'butterworth_' + str(bw) + "_" + str(noiselevellist[i]) +  '.png'
				picpath = PICPATH + name
				st_rec.normalize()
				fku.plot(st_rec, savefig=picpath)
			except:
				error.append(picpath)
				continue
