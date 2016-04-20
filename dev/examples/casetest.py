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
from sipy.filter.fk import fk_filter, fktrafo, fk_reconstruct, pocs_recon
from sipy.util.fkutil import  nextpow2, find_subsets, slope_distribution, makeMask, create_iFFT2mtx
from sipy.util.array_util import get_coords, attach_network_to_traces, attach_coordinates_to_traces,\
stream2array, array2stream, attach_network_to_traces, attach_coordinates_to_traces, attach_epidist2coords, epidist2nparray, epidist2list, \
alignon, partial_stack, gaps_fill_zeros, vespagram
from sipy.util.picker import get_polygon

import os
import sys

noise = np.fromfile('../data/test_datasets/randnumbers/noisearr.txt')
noise = noise.reshape(20,300)

with open("../data/test_datasets/ricker/rickerlist.dat", 'r') as fh:
	rickerlist = np.array(fh.read().split()).astype('str')

noisefoldlist = ["no_noise","10pct_noise", "20pct_noise", "50pct_noise", "60pct_noise", "80pct_noise"]
noiselevellist = np.array([0., 0.1, 0.2, 0.5, 0.6, 0.8]) 
alphalist = np.linspace(0.01, 0.9, 50)
maxiterlist = np.arange(51)[1:]
bwlist = [1,2,4]
taperlist = [2,4,5,8,200]



peaks = np.array([[-13.95      ,   6.06      ,  20.07      ],[  8.46648822,   8.42680793,   8.23354933]])
errors = []
FPATH = '/home/s_schn42/dev/FK-filter/data/test_datasets/ricker/'
for i, noisefolder in enumerate(noisefoldlist):
	print("##################### NOISELVL %i %% #####################\n" % int(noiselevellist[i] * 100.) )
	for filein in rickerlist:
		print("##################### CURRENT FILE %s  #####################\n" % filein )
		PICPATH = filein[:filein.rfind("/"):] + "/" + noisefolder + "/"
		FNAME 	= filein[filein.rfind("/"):].split('/')[1].split('.')[0]
		plotname = FPATH + FNAME + 'pocs_' + 'nlvl' + str(noiselevellist[i]) + '_linear''.png'
		plotname2 = FPATH + FNAME + 'pocs_' + 'nlvl' + str(noiselevellist[i]) + '_mask' '.png'
		if not os.path.isdir(PICPATH):
			os.mkdir(PICPATH)
		PATH = filein
		stream = read_st(PATH)
		d0 = stream2array(stream.copy(), normalize=True)
		data = stream2array(stream.copy(), normalize=True) + noiselevellist[i] * noise
		srs = array2stream(data)
		Qlinall = []
		Qbwmaskall = []
		Qtapermaskall = []
		
		if 'original' in PICPATH:
			DOMETHOD = 'denoise'
		else:
			DOMETHOD = 'recon'

		for alpha in alphalist:

			name1 = 'pocs_' + str(noiselevellist[i]) + '-noise_' + '{:01.2}'.format(alpha) + '-alpha_' + 'linear' + '.png'
			name2 = 'pocs_' + str(noiselevellist[i]) + '-noise_' + '{:01.2}'.format(alpha) + '-alpha_' + 'exp' + '.png'		
			picpath1 = PICPATH + name1
			picpath2 = PICPATH + name2
			plotnameQall = PICPATH + 'pocs_' + str(noiselevellist[i]) + '{:01.2}'.format(alpha) + '-alpha' + DOMETHOD + 'lin'
			plotnameQmbwall = PICPATH + 'pocs_' + str(noiselevellist[i]) + '{:01.2}'.format(alpha) + '-alpha' + DOMETHOD + 'mask_bw'
			plotnameQmtaperall = PICPATH + 'pocs_' + str(noiselevellist[i]) + '{:01.2}'.format(alpha) + '-alpha' + DOMETHOD + 'mask_taper'
			plotnameQmvaryall = PICPATH + 'pocs_' + str(noiselevellist[i]) + '{:01.2}'.format(alpha) + '-alpha' + 'norm' + 'mask_vary'

			
			print("##################### CURRENT ALPHA %f  #####################\n" % alpha )
			for maxiter in maxiterlist:

				print('POCS RECON WITH %i ITERATIONS' % maxiter, end="\r")
				sys.stdout.flush()
				print('LINEAR\n')
				data_org = d0.copy()
				srs = array2stream(data.copy())
				st_rec = pocs_recon(srs, maxiter, method=DOMETHOD, dmethod='linear', alpha=alpha)
				drec = stream2array(st_rec, normalize=True)
				Q = np.linalg.norm(data_org,2)**2. / np.linalg.norm(data_org - drec,2)**2.
				Qlin = 10.*np.log(Q)	
				Qlinall.append([alpha, maxiter, Qlin])

				for bws in bwlist:
					print('USING BW, %i, MASK' % int(bws), end="\r")
					sys.stdout.flush()
					data_org = d0.copy()
					srs = array2stream(data.copy())
					st_rec = pocs_recon(srs, maxiter, method=DOMETHOD, dmethod='mask', alpha=alpha, beta=None, peaks=peaks, maskshape=['butterworth', bws])
					drecmask = stream2array(st_rec, normalize=True)
					Q = np.linalg.norm(data_org,2)**2. / np.linalg.norm(data_org - drecmask,2)**2.
					
					Qbwmaskall.append([alpha, maxiter, 10.*np.log(Q), bws])

				for taper in taperlist:
					print('USING TAPER, %i, MASK' % int(taper), end="\r")
					sys.stdout.flush()
					data_org = d0.copy()
					srs = array2stream(data.copy())
					st_rec = pocs_recon(srs, maxiter, method=DOMETHOD, dmethod='mask', alpha=alpha, beta=None, peaks=peaks, maskshape=['taper', taper])
					drecmask = stream2array(st_rec, normalize=True)
					Q = np.linalg.norm(data_org,2)**2. / np.linalg.norm(data_org - drecmask,2)**2.
					
					Qtapermaskall.append([alpha, maxiter, 10.*np.log(Q), taper])


		savepath = plotnameQall + '.dat'
		np.savetxt(savepath, Qlinall)
	
		savepath = plotnameQmbwall + '.dat'
		np.savetxt(savepath, Qbwmaskall)
	
		savepath = plotnameQmtaperall + '.dat'
		np.savetxt(savepath, Qtapermaskall)	

		#savepath = plotnameQmvaryall + '.dat'
		#np.savetxt(savepath, Qmaskvaryall)			


#############################################

with open("/home/s_schn42/dev/FK-filter/data/test_datasets/ricker/Qvalues/masklist.dat", 'r') as fh:
	masklist = np.array(fh.read().split()).astype('str')

with open("/home/s_schn42/dev/FK-filter/data/test_datasets/ricker/Qvalues/linearlist.dat", 'r') as fh:
	linearlist = np.array(fh.read().split()).astype('str')

FPATH = "/home/s_schn42/dev/FK-filter/data/test_datasets/ricker/"

for name in masklist:
	ifile = "/home/s_schn42/dev/FK-filter/data/test_datasets/ricker/Qvalues/" + name
	Qraw = np.loadtxt(ifile)

	Q=[]
	for p in Qraw:
		if not p[2] == np.float64('inf'):
			Q.append(p)
	point = np.array(Q).transpose()

	fname = name.split('png')[0]
	plotname = FPATH + fname + 'png'

	fig, ax = plt.subplots()	
	cmap = 'seismic'
	scat = ax.scatter(point[0], point[1], c=point[2], s=60, cmap=cmap)#, s=50)
	ax.set_xlim(0.,0.91)
	ax.set_ylim(0,50)
	ax.autoscale(False)
	ax.set_xlabel('Alpha')
	ax.set_ylabel('No of iterations')
	scat.set_clim(-10,10)
	cbar = plt.colorbar(scat)
	cbar.ax.set_xlabel('Q')
	plt.savefig(plotname)
	plt.close("all")

for name in linearlist:
	ifile = "/home/s_schn42/dev/FK-filter/data/test_datasets/ricker/Qvalues/" + name
	Qraw = np.loadtxt(ifile)

	Q=[]
	for p in Qraw:
		if not p[2] == np.float64('inf'):
			if not p[0] == 0:
				if not p[1] ==0:
					Q.append(p)
	point = np.array(Q).transpose()

	fname = name.split('png')[0]
	plotname = FPATH + fname + 'png'

	fig, ax = plt.subplots()	
	cmap = 'seismic'
	scat = ax.scatter(point[0], point[1], c=point[3], s=60, cmap=cmap)#, s=50)
	ax.set_xlim(0.,0.91)
	ax.set_ylim(0,50)
	ax.autoscale(False)
	ax.set_xlabel('Alpha')
	ax.set_ylabel('No of iterations')
	scat.set_clim(-10,10)
	cbar = plt.colorbar(scat)
	cbar.ax.set_xlabel('Q')
	plt.savefig(plotname)
	plt.close("all")


####################################


