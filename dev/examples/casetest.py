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

folderlist=[]

with open("/home/s_schn42/dev/FK-Toolbox/data/test_datasets/ricker/Qvalues/denoisemask_bw_list.dat", 'r') as fh:
	dmask_bwlist = np.array(fh.read().split()).astype('str')
	folderlist.append(dmask_bwlist)

with open("/home/s_schn42/dev/FK-Toolbox/data/test_datasets/ricker/Qvalues/denoisemask_taper_list.dat", 'r') as fh:
	dmask_taperlist = np.array(fh.read().split()).astype('str')
	folderlist.append(dmask_taperlist)

with open("/home/s_schn42/dev/FK-Toolbox/data/test_datasets/ricker/Qvalues/reconmask_bw_list.dat", 'r') as fh:
	rmask_bwlist = np.array(fh.read().split()).astype('str')
	folderlist.append(rmask_bwlist)

with open("/home/s_schn42/dev/FK-Toolbox/data/test_datasets/ricker/Qvalues/reconmask_taper_list.dat", 'r') as fh:
	rmask_taperlist = np.array(fh.read().split()).astype('str')
	folderlist.append(rmask_taperlist)

with open("/home/s_schn42/dev/FK-Toolbox/data/test_datasets/ricker/Qvalues/SSAlist.dat", 'r') as fh:
	ssalist = np.array(fh.read().split()).astype('str')
	folderlist.append(ssalist)

FPATH = "/home/s_schn42/dev/FK-Toolbox/data/test_datasets/ricker/"

bwlist = [1,2,4]
taperlist = [2,4,5,8,200]


for sublist in folderlist:
	for name in sublist:

		ifile = "/home/s_schn42/dev/FK-Toolbox/data/test_datasets/ricker/" + name
		Qraw = np.loadtxt(ifile)
		
		if 'bw' in name:
			for i, bw in enumerate(bwlist):
				Q=[]
				for p in Qraw[i::3]:
					if not p[2] == np.float64('inf'):
						Q.append(p)
				point = np.array(Q).transpose()

				
				fname = name.split('dat')[0]
				plotname = FPATH + fname + 'bw' + str(bw) + '.png'
		
				fig, ax = plt.subplots()	
				cmap = 'seismic'
				scat = ax.scatter(point[0], point[1], c=point[2], s=100, cmap=cmap)#, s=50)
				ax.set_xlim(0.,0.91)
				ax.set_ylim(0,12)
				ax.autoscale(False)
				ax.set_xlabel('Alpha')
				ax.set_ylabel('No of iterations')
				scat.set_clim(-10,10)
				cbar = plt.colorbar(scat)
				cbar.ax.set_xlabel('Q')
				plt.savefig(plotname)
				plt.close("all")

		if 'taper' in name:
			for i, taper in enumerate(bwlist):
				Q=[]
				for p in Qraw[i::5]:
					if not p[2] == np.float64('inf'):
						Q.append(p)
				point = np.array(Q).transpose()

				fname = name.split('dat')[0]
				plotname = FPATH + fname + 'taper' + str(taper) + '.png'
		
				fig, ax = plt.subplots()	
				cmap = 'seismic'
				scat = ax.scatter(point[0], point[1], c=point[2], s=100, cmap=cmap)#, s=50)
				ax.set_xlim(0.,0.91)
				ax.set_ylim(0,12)
				ax.autoscale(False)
				ax.set_xlabel('Alpha')
				ax.set_ylabel('No of iterations')
				scat.set_clim(-10,10)
				cbar = plt.colorbar(scat)
				cbar.ax.set_xlabel('Q')
				plt.savefig(plotname)
				plt.close("all")

		if 'SSA' in name:

			Q=[]
			for p in Qraw:
				if not p[2] == np.float64('inf'):
					Q.append(p)
			point = np.array(Q).transpose()

			fname = name.split('dat')[0]
			plotname = FPATH + fname + 'SSA' + '.png'
	
			fig, ax = plt.subplots()	
			cmap = 'seismic'
			scat = ax.scatter(point[0], point[1], c=point[2], s=100, cmap=cmap)#, s=50)
			ax.set_xlim(0.,0.91)
			ax.set_ylim(0,12)
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


#################################### IMSHOW PLOT #####################################
import os
yrange = np.linspace(0.01, 0.9, 10)
yextent = np.zeros(yrange.size)
xrange = np.arange(11)[1:]

for sublist in folderlist:
	for name in sublist:

		ifile = "/home/s_schn42/dev/FK-Toolbox/data/test_datasets/ricker/" + name
		Qraw = np.loadtxt(ifile)
		Qmat = np.zeros((10,10))

		if 'bw' in name:
			for i, bw in enumerate(bwlist):
				for xi, x in enumerate(xrange):
					for yi, y in enumerate(yrange):
						for j, item in enumerate(Qraw[i::3]):
							if item[1] == x and item[0]==y:
								Qmat[xi,yi] = item[2]

				
				fname = name.split('dat')[0]
				plotname = FPATH + fname + 'bw' + str(bw) + '.png'
				rmname =FPATH + fname + 'bw_imshow' + str(bw) + '.png'
				os.remove(rmname)

				fig, ax = plt.subplots(frameon=False)
				Qplot = Qmat.copy()
				maxindex = Qmat.argmax()
				Qmax=np.zeros(Qplot.shape)
				#Qmax[:,:]= np.float64('nan')
				Qmax[np.unravel_index(maxindex, Qmat.shape)] = 10000

				extent =(0.01, 1, 1,11)
				im1 = ax.imshow(Qplot, aspect='auto', origin='lower', interpolation='none',cmap='Blues', extent=extent)
				ax.set_ylabel('No of iterations', fontsize=fs)
				ax.set_xlabel(r'$\alpha$', fontsize=fs)
				ax.tick_params(axis='both', which='both', labelsize=fs)
				cbar = fig.colorbar(im)
				cbar.ax.set_ylabel('Q', fontsize=fs)
				cbar.ax.tick_params(labelsize=fs)

				# Customize major tick labels

				ax.yaxis.set_major_locator(ticker.FixedLocator(np.linspace(1.5,10.5,10)))
				ax.yaxis.set_major_formatter(ticker.FixedFormatter(np.linspace(1,10,10).astype('int')))

				for vi, value in enumerate(yrange):
					yextent[vi] = "{:.2f}".format(value)
				ax.xaxis.set_major_locator(ticker.FixedLocator(np.linspace(.05,.95,10)))#np.linspace(1.5,99.5,9)))
				ax.xaxis.set_major_formatter(ticker.FixedFormatter(yextent))
				fig.set_size_inches(12,10)
				fig.savefig(plotname, dpi=300)
				plt.close("all")

		if 'taper' in name:
			for i, taper in enumerate(bwlist):
				for xi, x in enumerate(xrange):
					for yi, y in enumerate(yrange):
						for j, item in enumerate(Qraw[i::]):
							if item[1] == x and item[0]==y:
								Qmat[xi,yi] = item[2]

				fname = name.split('dat')[0]
				plotname = FPATH + fname + 'taper' + str(bw) + '.png'
				rmname = FPATH + fname + 'taper_imshow' + str(bw) + '.png'
				os.remove(rmname)

				fig, ax = plt.subplots(frameon=False)
				Qplot = Qmat.copy()
				maxindex = Qmat.argmax()
				Qmax=np.zeros(Qplot.shape)
				#Qmax[:,:]= np.float64('nan')
				Qmax[np.unravel_index(maxindex, Qmat.shape)] = 10000

				extent =(0.01, 1, 1,11)
				im1 = ax.imshow(Qplot, aspect='auto', origin='lower', interpolation='none',cmap='Blues', extent=extent)
				ax.set_ylabel('No of iterations', fontsize=fs)
				ax.set_xlabel(r'$\alpha$', fontsize=fs)
				ax.tick_params(axis='both', which='both', labelsize=fs)
				cbar = fig.colorbar(im)
				cbar.ax.set_ylabel('Q', fontsize=fs)
				cbar.ax.tick_params(labelsize=fs)

				# Customize major tick labels

				ax.yaxis.set_major_locator(ticker.FixedLocator(np.linspace(1.5,10.5,10)))
				ax.yaxis.set_major_formatter(ticker.FixedFormatter(np.linspace(1,10,10).astype('int')))

				for vi, value in enumerate(yrange):
					yextent[vi] = "{:.2f}".format(value)
				ax.xaxis.set_major_locator(ticker.FixedLocator(np.linspace(.05,.95,10)))#np.linspace(1.5,99.5,9)))
				ax.xaxis.set_major_formatter(ticker.FixedFormatter(yextent))
				fig.set_size_inches(12,10)
				fig.savefig(plotname, dpi=300)
				plt.close("all")

		if 'SSA' in name:

			for xi, x in enumerate(xrange):
				for yi, y in enumerate(yrange):
					for j, item in enumerate(Qraw[i::]):
						if item[1] == x and item[0]==y:
							Qmat[xi,yi] = item[2]

			fname = name.split('dat')[0]
			plotname = FPATH + fname + 'SSA' + '.png'
			rmname = FPATH + fname + 'SSA_imshow' + str(bw) + '.png'	
			os.remove(rmname)

			fig, ax = plt.subplots(frameon=False)
			Qplot = Qmat.copy()
			maxindex = Qmat.argmax()
			Qmax=np.zeros(Qplot.shape)
			#Qmax[:,:]= np.float64('nan')
			Qmax[np.unravel_index(maxindex, Qmat.shape)] = 10000

			extent =(0.01, 1, 1,11)
			im1 = ax.imshow(Qplot, aspect='auto', origin='lower', interpolation='none',cmap='Blues', extent=extent)
			ax.set_ylabel('No of iterations', fontsize=fs)
			ax.set_xlabel(r'$\alpha$', fontsize=fs)
			ax.tick_params(axis='both', which='both', labelsize=fs)
			cbar = fig.colorbar(im)
			cbar.ax.set_ylabel('Q', fontsize=fs)
			cbar.ax.tick_params(labelsize=fs)

			# Customize major tick labels

			ax.yaxis.set_major_locator(ticker.FixedLocator(np.linspace(1.5,10.5,10)))
			ax.yaxis.set_major_formatter(ticker.FixedFormatter(np.linspace(1,10,10).astype('int')))

			for vi, value in enumerate(yrange):
				yextent[vi] = "{:.2f}".format(value)
			ax.xaxis.set_major_locator(ticker.FixedLocator(np.linspace(.05,.95,10)))#np.linspace(1.5,99.5,9)))
			ax.xaxis.set_major_formatter(ticker.FixedFormatter(yextent))
			fig.set_size_inches(12,10)
			fig.savefig(plotname, dpi=300)
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

