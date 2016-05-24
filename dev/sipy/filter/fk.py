from __future__ import absolute_import, print_function
import obspy
from obspy import Stream
from obspy.core.event.event import Event
from obspy.core.inventory.inventory import Inventory

import sys

import numpy
import numpy as np
import math
import matplotlib.pyplot as plt
from sipy.util.array_util import get_coords
import datetime
import scipy as sp
import scipy.signal as signal
from scipy import sparse
from scipy.optimize import fmin_cg

from sipy.util.array_util import epidist2nparray, attach_epidist2coords, alignon, stack
from sipy.util.fkutil import ls2ifft_prep, line_cut, line_set_zero, shift_array,\
							find_peaks, slope_distribution, makeMask, create_iFFT2mtx, cg_solver, lstsqs, pocs
from sipy.util.base import nextpow2, array2stream, array2trace, stream2array
from sipy.util.picker import get_polygon

def fk_filter(st, inv=None, event=None, ftype='extract', fshape=['spike'], phase=None, polygon=12, normalize=True, stack=False,
					slopes=[-3,3], deltaslope=0.05, slopepicking=False, smoothpicks=False, dist=0.5, maskshape=['boxcar',None], 
					order=4., peakinput=False):
	"""
	Import stream, the function applies an 2D FFT, removes a certain window around the
	desired phase to surpress a slownessvalue corresponding to a wavenumber and applies an 2d iFFT.
	To fill the gap between uneven distributed stations use array_util.gaps_fill_zeros(). A method to interpolate the signals in the
	fk-domain is beeing build, also a method using a norm minimization method.
	Alternative is an nonequidistant 2D Lombard-Scargle transformation.

	param st: Stream
	type st: obspy.core.stream.Stream

	param inv: inventory
	type inv: obspy.station.inventory.Inventory

	param event: Event
	type event: obspy.core.event.Event

	param ftype: type of method, default is 'eliminate-polygon', possible inputs are:
				 -eliminate
				 -extract
				 -eliminate-polygon
				 -extract-polygon
				 -mask
				 -fk

	type ftype: string

	param fshape: fshape[0] describes the shape of the fk-filter in case of ftype is 'eliminate' or 'extract'. Possible inputs are:
				 -spike (default)
				 -boxcar
				 -taper
				 -butterworth

				  fshape[1] is an additional attribute to the shape of taper and butterworth, for:
				 -taper: fshape[1] = slope of sides
				 -butterworth: fshape[1] = number of poles

				  fshape[3] describes the length of the filter shape, respectivly wavenumber corner points around k=0,
				
				 e.g.: fshape['taper', 2, 4] produces a symmetric taper with slope of side = 2, where the signal is reduced about 50% at k=+-2


	type  fshape: list

	param phase: name of the phase to be investigated
	type  phase: string

	param polygon: number of vertices of polygon for fk filter, only needed 
				   if ftype is set to eliminate-polygon or extract-polygon.
				   Default is 12.
	type  polygon: int
	
	param normalize: normalize data to 1
	type normalize: bool

	param SSA: Force SSA algorithm or let it check, default:False
	type SSA: bool

	returns:	stream_filtered, the filtered stream.
			


	References: Yilmaz, Thomas

	Author: S. Schneider 2016

	 This program is free software: you can redistribute it and/or modify
	 it under the terms of the GNU General Public License as published
	 by the Free Software Foundation, either version 3 of the License, or
	 any later version.

	 This program is distributed in the hope that it will be useful,
	 but WITHOUT ANY WARRANTY; without even the implied warranty of
	 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	 GNU General Public License for more details: http://www.gnu.org/licenses/
	"""

	# Convert format and prepare Variables.

	# Check for Data type of variables.
	if not type(st ) == Stream:
		print( "Wrong input type of stream, must be obspy.core.stream.Stream" )
		raise TypeError

	if len(fshape) ==  1:
		fshape = [fshape[0], None, None]
	
	st_tmp = st.copy()
	ArrayData = stream2array(st_tmp, normalize)
	
	ix = ArrayData.shape[0]
	iK = int(math.pow(2,nextpow2(ix)))
	
	try:
		yinfo = epidist2nparray(attach_epidist2coords(inv, event, st_tmp))
		dx = (yinfo.max() - yinfo.min() + 1) / yinfo.size
		k_axis = np.fft.fftfreq(iK, dx)	
	except:
		print("\nNo inventory or event-information found. \nContinue without specific distance and wavenumber information.")
		yinfo=None
		dx=None
		k_axis=None

	it = ArrayData.shape[1]
	iF = int(math.pow(2,nextpow2(it)))
	dt = st_tmp[0].stats.delta
	f_axis = np.fft.fftfreq(iF,dt)


	# Calc mean diff of each epidist entry if it is reasonable
	# do a partial stack and apply filter.


	"""
	2D Frequency-Space / Wavenumber-Frequency Filter #########################################################
	"""

	# 2D f-k Transformation 
	# Note array_fk has f on the x-axis and k on the y-axis!!!
	# For interaction the conj.-transposed Array is shown!!!


	# Decide when to use SSA to fill the gaps, calc mean distance of each epidist entry
	# if it differs too much --> SSA


	if ftype in ("eliminate"):
		if phase:
			if not isinstance(event, Event) and not isinstance(inv, Inventory):
				msg='For alignment on phase calculation inventory and event information is needed, not found.'
				raise IOError(msg)

			st_al = alignon(st_tmp, inv, event, phase)
			ArrayData = stream2array(st_al, normalize)
			array_fk = np.fft.fft2(ArrayData, s=(iK,iF))
			array_filtered_fk = line_set_zero(array_fk, shape=fshape)

		else:
			array_fk = np.fft.fft2(ArrayData, s=(iK,iF))
			array_filtered_fk = line_set_zero(array_fk, shape=fshape)

	elif ftype in ("extract"):
		if phase:
			if not isinstance(event, Event) and not isinstance(inv, Inventory):
				msg='For alignment on phase calculation inventory and event information is needed, not found.'
				raise IOError(msg)

			st_al = alignon(st_tmp, inv, event, phase)
			ArrayData = stream2array(st_al, normalize)
			array_fk = np.fft.fft2(ArrayData, s=(iK,iF))
			array_filtered_fk = line_cut(array_fk, shape=fshape)

		else:
			array_fk = np.fft.fft2(ArrayData, s=(iK,iF))
			array_filtered_fk = line_cut(array_fk, shape=fshape)

		array_filtered = np.fft.ifft2(array_filtered_fk, s=(iK,iF)).real


		# Convert to Trace or Stream object.
		array_filtered = array_filtered[0:ix, 0:it]
		if stack:
			stacked_array = stack(array_filtered, order)
			stream_filtered = array2trace(stacked_array, st_original=st.copy())
		else:
			stream_filtered = array2stream(array_filtered,st_original=st.copy())

		return stream_filtered, array_filtered_fk
	
	elif ftype in ("eliminate-polygon"):
		array_fk = np.fft.fft2(ArrayData, s=(iK,iF))
		if phase:
			if not isinstance(event, Event) and not isinstance(inv, Inventory):
				msg='For alignment on phase calculation inventory and event information is needed, not found.'
				raise IOError(msg)
			st_al = alignon(st_tmp, inv, event, phase)
			ArrayData = stream2array(st_al, normalize)
			array_fk = np.fft.fft2(ArrayData, s=(iK,iF))
			array_filtered_fk = _fk_eliminate_polygon(array_fk, polygon, ylabel=r'frequency-domain f in $\frac{1}{Hz}$', \
													  yticks=f_axis, xlabel=r'wavenumber-domain k in $\frac{1}{^{\circ}}$', xticks=k_axis)

		else:
			array_filtered_fk = _fk_eliminate_polygon(array_fk, polygon, ylabel=r'frequency-domain f in $\frac{1}{Hz}$', \
													  yticks=f_axis, xlabel=r'wavenumber-domain k in $\frac{1}{^{\circ}}$', xticks=k_axis)


	elif ftype in ("extract-polygon"):
		array_fk = np.fft.fft2(ArrayData, s=(iK,iF))
		if phase:
			if not isinstance(event, Event) and not isinstance(inv, Inventory):
				msg='For alignment on phase calculation inventory and event information is needed, not found.'
				raise IOError(msg)

				st_al = alignon(st_tmp, inv, event, phase)
				ArrayData = stream2array(st_al, normalize)
				array_fk = np.fft.fft2(ArrayData, s=(iK,iF))
				array_filtered_fk = _fk_extract_polygon(array_fk, polygon, ylabel=r'frequency-domain f in $\frac{1}{Hz}$', \
													yticks=f_axis, xlabel=r'wavenumber-domain k in $\frac{1}{^{\circ}}$', xticks=k_axis)
		else:
			array_filtered_fk = _fk_extract_polygon(array_fk, polygon, ylabel=r'frequency-domain f in $\frac{1}{Hz}$', \
												yticks=f_axis, xlabel=r'wavenumber-domain k in $\frac{1}{^{\circ}}$', xticks=k_axis)

		array_filtered = np.fft.ifft2(array_filtered_fk, s=(iK,iF)).real


		# Convert to Trace or Stream object.
		array_filtered = array_filtered[0:ix, 0:it]
		if stack:
			stacked_array = stack(array_filtered, order)
			stream_filtered = array2trace(stacked_array, st_original=st.copy())
		else:
			stream_filtered = array2stream(array_filtered,st_original=st.copy())

		stream_filtered = array2trace(stacked_array, st_original=st.copy())

		return stream_filtered

	elif ftype in ("mask"):
		array_fk = np.fft.fft2(ArrayData)
		M, prange, peaks = slope_distribution(array_fk, slopes, deltaslope, peakpick=None, mindist=dist, smoothing=smoothpicks, interactive=slopepicking)
		W = makeMask(array_fk, peaks[0], maskshape)
		array_filtered_fk =  array_fk * W
		array_filtered = np.fft.ifft2(array_filtered_fk)
		stream_filtered = array2stream(array_filtered, st_original=st.copy())
		return stream_filtered


	elif ftype in ("fk"):
		if phase:
			if not isinstance(event, Event) and not isinstance(inv, Inventory):
				msg='For alignment on phase calculation inventory and event information is needed, not found.'
				raise IOError(msg)

			st_al = alignon(st_tmp, inv, event, phase)
			ArrayData = stream2array(st_al, normalize)
			array_fk = np.fft.fft2(ArrayData, s=(iK,iF))
			### BUILD DOUBLE TAPER ###
			#array_filtered_fk = 

		else:
			array_fk = np.fft.fft2(ArrayData, s=(iK,iF))
			### BUILD DOUBLE TAPER ###
			#array_filtered_fk = 


	else:
		print("No type of filter specified")
		raise TypeError

	array_filtered = np.fft.ifft2(array_filtered_fk, s=(iK,iF)).real


	# Convert to Stream object.
	array_filtered = array_filtered[0:ix, 0:it]
	stream_filtered = array2stream(array_filtered, st_original=st.copy())

	return stream_filtered



"""
FFT FUNCTIONS 
"""
def fktrafo(stream, inv, event, normalize=True):
	"""
	Calculates the f,k - transformation of the data in stream. Returns the trafo as an array.

	:param st: Stream
	:type st: obspy.core.stream.Stream

	:param inv: inventory
	:type inv: obspy.station.inventory.Inventory

	:param event: Event
	:type event: obspy.core.event.Event

	returns
	:param fkdata: f,k - transformation of data in stream
	:type fkdata: numpyndarray
	"""
	st_tmp = stream.copy()
	ArrayData = stream2array(st_tmp, normalize)
	
	ix = ArrayData.shape[0]
	iK = int(math.pow(2,nextpow2(ix)))
	it = ArrayData.shape[1]
	iF = int(math.pow(2,nextpow2(it)))

	fkdata = np.fft.fft2(ArrayData, s=(iK,iF))
	
	return fkdata

def fk_reconstruct(st, slopes=[-3,3], deltaslope=0.05, slopepicking=False, smoothpicks=False, dist=0.5, maskshape=['boxcar',None], 
					method='denoise', solver="iterative",  mu=5e-2, tol=1e-12, fulloutput=False, peakinput=False, alpha=0.9):
	"""
	This functions reconstructs missing signals in the f-k domain, using the original data,
	including gaps, filled with zeros, and its Mask-array (see makeMask, and slope_distribution.
	If all traces are avaiable it is a useful method to de-noise the data.
	Uses the following cost function to minimize:

			J = ||dv - T FHmtx2D Yw Dv ||^{2}_{2} + mu^2 ||Dv||^{2}_{2}
			
			J := Cost function
			dv:= Column-wise-ordered long vector of the 2D signal d (columns: t-domain, rows: x-domain)
			DV:= Column-wise-ordered long vector of the	f-k-spectrum D ( columns: f-domain, rows: k-domain)
			Yw := Diagonal matrix built from the column-wise-ordered long vector of Mask
			T := Sampling matrix which maps the fully sampled desired seismic data to the available samples.
				 For de-noising problems T = I (identity matrix)
			mu := Trade-off parameter between misfit and model norm


	Minimizing is done via a method of the LSMR solver, de-noising (1-2 iterations), reconstruction(8-10) iterations.
	T FHmtx2D Yw Dv will be formed to one matrix A, so at the end the equation system that will be solved has the form:
			
							|   A    |		  | dv |
							|    	 | * Dv = |    |
							| mu * I |		  | 0  |


	:param st: Stream with missing traces, to be reconstructed or complete stream to be de-noised
	:type  st: obspy.core.stream.Stream

	:param slopes: Range of slopes to investigate for mask-function
	:type  slopes: list

	:param deltaslope: stepsize inbetween slopes.
	:type  deltaslope: float

	:param slopepicking: If True peaks of slopedistribution can be picked by hand.
	:type  slopepicking: bool

	:param smoothpicks: Determines the smoothing of the Slopedistribution, default off. If enabled the distribution ist smoothened by
						convoluting it with a boxcar of size smoothpicks.
	:type  smoothpicks: int

	:param dist: Minimum distance inbetween maximum picks.
	:type  dist: float

	:param maskshape: maskshape[0] describes the shape of the lobes of the mask. Possible inputs are:
				 -boxcar (default)
				 -taper
				 -butterworth

				  maskshape[1] is an additional attribute to the shape of taper and butterworth, for:
				 -taper: maskshape[1] = slope of sides
				 -butterworth: maskshape[1] = number of poles
				
				 e.g.: maskshape['taper', 2] produces a symmetric taper with slope of side = 2.


	:type  maskshape: list

	:param method: Desired fk-method, options are 'denoise' and 'interpolate'
	:type  method: string

	:param solver: Solver used for method. Options are 'lsqr' and 'iterative'.
				   If method is 'denoise' only the iterative solver is used.
	:type  solver: string
	
	:param mu:	Damping parameter for the solver
	:type  mu:	float

	:param tol: Tolerance for solver to abort iteration.
	:type  tol: float

	:param fulloutput: If True, the function additionally outputs FH, dv, Dv, Ts and Yw
	:type  fulloutput: bool

	:param peakinput: Chosen peaks of the distribution, insert here if the peaks are not to be meant to recalculated
	:type  peakinput: np.ndarray

	######  returns:

	:param st_rec: Stream with reconstructed signals on the missing traces
	:type  st_rec: obspy.core.stream.Stream
	
	## if fulloutput=True

	:param st_rec: Stream with reconstructed signals on the missing traces
	:type  st_rec: obspy.core.stream.Stream

	:param FH: 2DiFFT-matrix for column-wise ordered longvector of the f-k spectrum
	:type  FH: scipy.sparse.csc.csc_matrix

	:param dv: Column-wise ordered longvector of the t-x data
	:type  dv: numpy.ndarray
	
	:param Dv: Column-wise ordered longvector of the f-k spectrum of dv
	:type  Dv: numpy.ndarray

	:param Ts: Sampling-matrix, which maps desired to available data
	:type  Ts: scipy.sparse.dia.dia_matrix

	:param Yw: Diagonal matrix constructed of the column-wise ordered longvector of the mask-matrix
	:type  Yw: scipy.sparse.dia.dia_matrix

	Example:
				from obspy import read as read_st
				import sipy
				
				stream = read_st("../data/synthetics_uniform/SUNEW.QHD")

				#Example around PP.
				stream_org = st.copy()
				d = sipy.util.array_util.stream2array(stream_org)
				ArrayData = np.zeros((d.shape[0], 300))
				for i, trace in enumerate(d):
					ArrayData[i,:]=trace[400:700]
				stream = sipy.util.array_util.array2stream(ArrayData, stream_org)
	
				dssa = sipy.filter.fk.fk_reconstruct(stream, mu=5e-2, method='interpolate')
				
				stream_ssa = sipy.util.array_util.array2stream(dssa, stream)

				sipy.util.fkutil.plot(stream_ssa)

	Author: S. Schneider, 2016
	Reference:	Mostafa Naghizadeh, Seismic data interpolation and de-noising in the frequency-wavenumber
				domain, 2012, GEOPHYSICS
	"""

	# Prepare data.
	st_tmp 		= st.copy()
	ArrayData	= stream2array(st_tmp, normalize=False)
	ADT 		= ArrayData.copy().transpose()

	fkData 		= np.fft.fft2(ArrayData)
	fkDT 		= np.fft.fft2(ADT)

	# Look for missing Traces
	recon_list 	= []

	for i, trace in enumerate(st_tmp):
		try:
			if trace.stats.zerotrace == 'True':
				recon_list.append(i)
		except AttributeError:
			if sum(trace.data) == 0. :
				recon_list.append(i)
		except:
			continue
	print(recon_list)

	# Calculate mask-function W.
	try:	
		if peakinput.any():
			peaks = peakinput
	except:
		print("Calculating slope distribution...\n")
		M, prange, peaks = slope_distribution(fkData, slopes, deltaslope, peakpick=None, mindist=dist, smoothing=smoothpicks, interactive=slopepicking)
		if fulloutput:
			kin = 'n'
			while kin in ('n', 'N'):
				plt.figure()
				plt.title('Magnitude-Distribution')
				plt.xlabel('Slope in fk-domain')
				plt.ylabel('Magnitude of slope')
				plt.plot(prange, M)
				plt.plot(peaks[0], peaks[1]/peaks[1].max()*M.max(), 'ro')
				plt.show()
				kin = raw_input("Use picks? (y/n) \n")
				if kin in ['y' , 'Y']:
					print("Using picks, continue \n")
				elif kin in ['n', 'N']:
					print("Don't use picks, please re-pick \n")
					M, prange, peaks = slope_distribution(fkData, slopes, deltaslope, peakpick=None, mindist=dist, smoothing=smoothpicks, interactive=True)
	
	print("Creating mask function with %i significant linear events \n" % len(peaks[0]) )
	W = makeMask(fkData, peaks[0], maskshape)

	# If fulloutput is desired, a bunch of messages and user interaction appears.
	if fulloutput:
		plt.figure()
		plt.subplot(3,1,1)
		plt.gca().yaxis.set_major_locator(plt.NullLocator())
		plt.gca().xaxis.set_major_locator(plt.NullLocator())
		plt.title("fk-spectrum")
		plt.imshow(abs(np.fft.fftshift(fkData)), aspect='auto', interpolation='none')
		plt.subplot(3,1,2)
		plt.gca().yaxis.set_major_locator(plt.NullLocator())
		plt.gca().xaxis.set_major_locator(plt.NullLocator())
		plt.title("Mask-function")
		plt.imshow(np.fft.fftshift(W), aspect='auto', interpolation='none')
		plt.subplot(3,1,3)
		plt.gca().yaxis.set_major_locator(plt.NullLocator())
		plt.gca().xaxis.set_major_locator(plt.NullLocator())
		plt.title("Applied mask-function")
		plt.imshow(abs(np.fft.fftshift(W*fkData)), aspect='auto', interpolation='none')
		plt.show()
		kin = raw_input("Use Mask? (y/n) \n")
		if kin in ['y' , 'Y']:
			print("Using Mask, continue \n")
		elif kin in ['n', 'N']:
			msg="Don't use Mask, exit"
			raise IOError(msg)

	# Checking for number of iteration and reconstruction behavior.
	maxiter=None
	interpol = False
	if isinstance(method, str):
		if method in ("denoise"):
				maxiter = 2
				recon_list = []
		elif method in ("interpolate"):
				maxiter = 10
				interpol = True

	elif isinstance(method, int):
		maxiter=method

	print("maximum %i" %maxiter)
	if solver in ("lsqr", "leastsquares", "ilsmr", "iterative", "cg", "fmin"):
		pocs = False
		# To keep the order it would be better to transpose W to WT
		# but for creation of Y, WT has to be transposed again,
		# so this step can be skipped.
		Y 	= W.reshape(1,W.size)[0]
		Yw 	= sparse.diags(Y)

		# Initialize arrays for cost-function.
		dv 	= ADT.transpose().reshape(1, ADT.size)[0]
		Dv	= fkDT.transpose().reshape(1, fkDT.size)[0]
	
		T = np.ones((ArrayData.shape[0], ArrayData.shape[1]))
		T[recon_list] = 0.
		T = T.reshape(1, T.size)[0]

		Ts = sparse.diags(T)
	

		# Create sparse-matrix with iFFT operations.	
		print("Creating iFFT2 operator as a %ix%i matrix ...\n" %(fkDT.shape[0]*fkDT.shape[1], fkDT.shape[0]*fkDT.shape[1]))	

		FH = create_iFFT2mtx(fkDT.shape[0], fkDT.shape[1])
		print("... finished\n")

		# Create model matrix A.
		print("Creating sparse %ix%i matrix A ...\n" %(FH.shape[0], FH.shape[1]))	
		A =  Ts.dot(FH.dot(Yw))
		print("Starting reconstruction...\n")

		if solver in ("lsqr", "leastsquares"):
			print(" ...using iterative least-squares solver...\n")
			x = sparse.linalg.lsqr(A, dv, mu, atol=tol, btol=tol, conlim=tol, iter_lim=maxiter)
			print("istop = %i \n" % x[1])
			print("Used iterations = %i \n" % x[2])
			print("residual Norm ||x||_2 = %f \n " % x[8])
			print("Misfit ||Ax - b||_2= %f \n" % x[4]) 
			print("Condition number = %f \n" % x[6])
			
			Dv_rec = x[0]

		elif solver in ("ilsmr", "iterative"):
			print(" ...using iterative LSMR solver...\n")
			x = sparse.linalg.lsmr(A,dv,mu, atol=tol, btol=tol, conlim=tol, maxiter=maxiter)
			print("istop = %i \n" % x[1])
			print("Used iterations = %i \n" % x[2])
			print("Misfit = %f \n " % x[3])
			print("Modelnorm = %f \n" % x[4])
			print("Condition number = %f \n" % x[5])
			print("Norm of Dv = %f \n" % x[6]) 
			Dv_rec = x[0]

		elif solver in ("cg"):
			A 		= Ts.dot(FH.dot(Yw))
			Ah 		= A.conjugate().transpose()
			madj 	= Ah.dot(dv)
			E 		= mu * sparse.eye(A.shape[0])
			B 		= A + E
			Binv 	= sparse.linalg.inv(B)
			x 		= sparse.linalg.cg(Binv, madj, maxiter=maxiter)
			Dv_rec 	= x[0]

		elif solver in ('fmin'):
			A 		= Ts.dot(FH.dot(Yw))
			global arg1
			global arg2
			global arg3
			arg1 = dv
			arg2 = A
			arg3 = mu

			def J(x):
				COST = np.linalg.norm(arg1 - arg2.dot(x), 2)**2. + arg3*np.linalg.norm(x,2)**2.
				return COST 

			Dv_rec = sp.optimize.fmin_cg(J, x0=Dv, maxiter=10)			

		data_rec = np.fft.ifft2(Dv_rec.reshape(fkData.shape)).real

	elif solver in ("pocs"):
		pocs=True
		threshold = abs( (fkData*W.astype('complex').max()) ) 

		for i in range(maxiter):
			data_tmp 								= ArrayData.copy()
			fkdata 									= np.fft.fft2(data_tmp) * W.astype('complex')
			fkdata[ np.where(abs(fkdata) < threshold)] 	= 0. + 0j
			threshold = threshold * alpha
			#if i % 10 == 0.:
			#	plt.imshow(abs(fkdata), aspect='auto', interpolation='none')
			#	plt.savefig("%s.png" % i)
			data_tmp 								= np.fft.ifft2(fkdata).real.copy()
			ArrayData[recon_list] 					= data_tmp[recon_list]
	
		data_rec = ArrayData.copy()
	else:
		print("No solver or method specified.")
		return

	

	if interpol:
		st_rec = st.copy()
		for i in recon_list:
			st_rec[i].data = data_rec[i,:]
			st_rec[i].stats.zerotrace = 'reconstructed'


	else:
		st_rec = array2stream(data_rec, st)

	if fulloutput and not pocs:
		return st_rec, FH, dv, Dv, Dv_rec, Ts, Yw
	else:
		return st_rec #, x[8], x[4]

def pocs_recon(st, maxiter, dmethod='denoise', method='linear', alpha=0.9, beta=None, peaks=None, maskshape=None, 
			   dt=None, p=None, flow=None, fhigh=None, slidingwindow=False):
	"""
	This functions reconstructs missing signals in the f-k domain, using the original data,
	including gaps, filled with zeros. It applies the projection onto convex sets (pocs) algorithm in
	2D.

	Reference: 3D interpolation of irregular data with a POCS algorithm, Abma & Kabir, 2006

	:param st:
	:type  st:

	:param maxiter:
	:type  maxiter:

	:param nol: Number of loops
	:type  nol:

	returns:

	:param st_rec:
	:type  st_rec:
	"""

	st_tmp 		= st.copy()
	ArrayData 	= stream2array(st_tmp, normalize=True)
	recon_list 	= []

	if dmethod in ('reconstruct'):
		for i, trace in enumerate(st_tmp):
			try:
				if trace.stats.zerotrace in ['True']:
					recon_list.append(i)

			except AttributeError:
				if sum(trace.data) == 0. :
					recon_list.append(i)

			except:
				continue
		
		noft = recon_list
		
	elif dmethod in ('denoise', 'de-noise'):
		noft = range(ArrayData.shape[0])
	
	print(noft)
	ADfinal = pocs(ArrayData, maxiter, noft, alpha, beta, method, dmethod, peaks, maskshape, dt, p, flow, fhigh, slidingwindow)

	#datap = ADfinal.copy()

	st_rec 	= array2stream(ADfinal, st)

	return st_rec

def _fk_extract_polygon(data, polygon, xlabel=None, xticks=None, ylabel=None, yticks=None):
	"""
	Only use with the function fk_filter!
	Function to test the fk workflow with synthetic data
	param data:	data of the array
	type data:	numpy.ndarray
	"""
	# Shift 0|0 f-k to center, for easier handling
	dsfk = np.fft.fftshift(data.conj().transpose())

	# Define polygon by user-input.
	#indicies = get_polygon(np.log(abs(dsfk)), polygon, xlabel, xticks, ylabel, yticks)
	indicies = get_polygon(abs(dsfk), polygon, xlabel, xticks, ylabel, yticks)
	# Create new array, only contains extractet energy, pointed to with indicies
	dsfk_extract = np.zeros(dsfk.shape)
	dsfk_extract.conj().transpose().flat[ indicies ]=1.
	data_fk = dsfk * dsfk_extract
	
	data_fk = np.fft.ifftshift(data_fk.conj().transpose())

	return data_fk


def _fk_eliminate_polygon(data, polygon, xlabel=None, xticks=None, ylabel=None, yticks=None):
	"""
	Only use with the function fk_filter!
	Function to test the fk workflow with synthetic data
	param data:	data of the array
	type data:	numpy.ndarray
	"""
	# Shift 0|0 f-k to center, for easier handling
	dsfk = np.fft.fftshift(data.conj().transpose())
	
	# Define polygon by user-input.
	#indicies = get_polygon(np.log(abs(dsfk)), polygon, xlabel, xticks, ylabel, yticks)
	indicies = get_polygon(abs(dsfk), polygon, xlabel, xticks, ylabel, yticks)
	# Create new array, contains all the energy, except the eliminated, pointed to with indicies
	dsfk_elim = dsfk.conj().transpose()
	dsfk_elim.flat[ indicies ]=0.
	data_fk = dsfk_elim.conj().transpose()

	data_fk = np.fft.ifftshift(data_fk.conj().transpose())

	return data_fk

"""
LS FUNCTIONS
"""
def _fk_ls_filter_extract_phase_sp(ArrayData, y_dist=False, radius=None, maxk=False):
	"""
	Only use with the function fk_filter!
	FK-filter using the Lomb-Scargle Periodogram with the scipy library
	param data:	data of the array
	type data:	numpy.ndarray

	param snes:	slownessvalue of the desired extracted phase
	type snes:	int
	"""
	return 

def _fk_ls_filter_eliminate_phase_sp(ArrayData, y_dist=False, radius=None, maxk=False):
	"""
	Only use with the function fk_filter!
	Function to test the fk workflow with synthetic data
	param data:	data of the array
	type data:	numpy.ndarray

	param snes:	slownessvalue of the desired extracted phase
	type snes:	int
	"""
	# Define freq Array 
	freq = np.zeros((len(ArrayData), len(ArrayData[0]) / 2  + 1)) + 1j

	for i in range(len(ArrayData)):
		freq_new = np.fft.rfftn(ArrayData[i])
		freq[i] = freq_new

	# Define k Array
	freqT = freq.conj().transpose()
	knum = np.zeros( ( len(freqT), len(freqT[0])  /2 +1 ))
		     
	#calc best range
	N = len(freqT[0])
	dN = ( max(freqT[0]) - min(freqT[0]) / N )
	
	f_temp = np.fft.rfftfreq(len(freqT[0]), dN) * 2.* np.pi

	#1. try: 
	#period_range = np.linspace(min_wavelength, max_bound, len(freqT[0]))
	#2. try: 
	#period_range = np.linspace(f_temp[1], max(f_temp), N)
	#3. try:
	period_range = f_temp
	#period_range = period_range.astype('float')
	period_range[0] = 1.
	#after this change the first outputparameter of the periodogram to a 
	#correlation between the signal and a e^0 function ;)
	period_range = period_range.astype('float')
	
	for j in range(len(freqT)):
		k_new = signal.lombscargle(y_dist, abs(freqT[j]), period_range)
		k_new = ls2ifft_prep(k_new, abs(freqT[j]))
		knum[j] = k_new

			
	#change dtype to integer, for further processing
	period_range = period_range.astype('int')
	fkspectra = knum
	dsfft = line_set_zero(fkspectra, 0, radius)
	
	return fkspectra.conj().transpose(), period_range
