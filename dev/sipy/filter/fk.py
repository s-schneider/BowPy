from __future__ import absolute_import
import obspy
from obspy.core.event.event import Event
from obspy.core.inventory.inventory import Inventory

import numpy
import numpy as np
import math
import matplotlib.pyplot as plt
from sipy.util.array_util import get_coords
import datetime
import scipy as sp
import scipy.signal as signal
from scipy.optimize import fmin_cg

from sipy.util.array_util import array2stream, stream2array, epidist2nparray, epidist
from sipy.util.fkutil import ls2ifft_prep, line_cut, line_set_zero, shift_array, get_polygon,\
							find_peaks, slope_distribution, makeMask
from sipy.util.base import nextpow2

def fk_filter(st, inv=None, event=None, trafo='FK', ftype='eliminate-polygon', phase=None, polygon=12, normalize=True, SSA=False):
	"""
	At this point prework with programs like seismic handler or SAC is needed to perform correctly


	Import stream, the function applies an 2D FFT, removes a certain window around the
	desired phase to surpress a slownessvalue corresponding to a wavenumber and applies an 2d iFFT.
	To fill the gap between uneven distributed stations use the SSA algorithm.
	Alternative is an nonequidistant 2D Lombard-Scargle transformation.

	param st: Stream
	type st: obspy.core.stream.Stream

	param inv: inventory
	type inv: obspy.station.inventory.Inventory

	param event: Event
	type event: obspy.core.event.Event

	param trafo: Type of transformation, default is 'FK', possible inputs are:
				 FK: for f-k transformation via numpy.fft.fft2
				 FX: for f-x transformation via numpy.fft.fft
				 LS: for a combination of 1D FFT in time-domain and and Lomb-Scargle
				     in the space-domain, via numpy.fft.fft and scipy.signal.lombscargle
	type trafo: string

	param ftype: type of method, default is 'eliminate-polygon', possible inputs are:
				 eliminate
				 extract
				
				 if trafo is set to FK, also:
				 eliminate-polygon
				 extract-polygon

	type ftype: string

	param phase: name of the phase to be investigated
	type phase: string

	param polygon: number of vertices of polygon for fk filter, only needed 
				   if ftype is set to eliminate-polygon or extract-polygon.
				   Default is 12.
	type polygon: int
	
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
	st_tmp = st.copy()
	ArrayData = stream2array(st_tmp, normalize)
	
	ix = ArrayData.shape[0]
	iK = int(math.pow(2,nextpow2(ix)+1))
	
	try:
		yinfo = epidist2nparray(epidist(inv, event, st_tmp))
		dx = (yinfo.max() - yinfo.min() + 1) / yinfo.size
		k_axis = np.fft.fftfreq(iK, dx)	
	except:
		print("\nNo inventory or event-information found. \nContinue without specific distance and wavenumber information.")
		yinfo=None
		dx=None
		k_axis=None

	it = ArrayData.shape[1]
	iF = int(math.pow(2,nextpow2(it)+1))
	dt = st_tmp[0].stats.delta
	f_axis = np.fft.fftfreq(iF,dt)


	# Calc mean diff of each epidist entry if it is reasonable
	# do a partial stack and apply filter.


	"""
	2D Frequency-Space / Wavenumber-Frequency Filter #########################################################
	"""

	# 2D f-k Transformation 
	# Decide when to use SSA to fill the gaps, calc mean distance of each epidist entry
	# if it differs too much --> SSA
	if trafo == "FK":
		
		# Note array_fk has f on the x-axis and k on the y-axis!!!
		# For interaction the conj.-transposed Array is shown!!! 
		array_fk = np.fft.fft2(ArrayData, s=(iK,iF))

		if ftype in ("eliminate"):
			array_filtered_fk = line_set_zero(array_fk)

		elif ftype in ("extract"):
			array_filtered_fk = line_cut(array_fk)				
		
		elif ftype in ("eliminate-polygon"):
			if isinstance(event, Event) and isinstance(inv, Inventory):
				array_filtered_fk = _fk_eliminate_polygon(array_fk, polygon, ylabel=r'frequency-domain f in $\frac{1}{Hz}$', \
														  yticks=f_axis, xlabel=r'wavenumber-domain k in $\frac{1}{^{\circ}}$', xticks=k_axis)
			else:
				msg='For wavenumber calculation inventory and event information is needed, not found.'
				raise IOError(msg)

		elif ftype in ("extract-polygon"):
			if isinstance(event, Event) and isinstance(inv, Inventory):
				array_filtered_fk = _fk_extract_polygon(array_fk, polygon, ylabel=r'frequency-domain f in $\frac{1}{Hz}$', \
														yticks=f_axis, xlabel=r'wavenumber-domain k in $\frac{1}{^{\circ}}$', xticks=k_axis)
			else:
				msg='For wavenumber calculation inventory and event information is needed, not found.'
				raise IOError(msg)

		else:
			print("No type of filter specified")
			raise TypeError

		array_filtered = np.fft.ifft2(array_filtered_fk, s=(iK,iF)).real

	# 2D f-x Transformation 
	elif trafo in ("FX"):
		array_fx = np.fft.fft(ArrayData, iF)
		if ftype in ("eliminate"):
			array_filtered = line_set_zero(array_fx)

		elif ftype in ("extract"):
			array_filtered = line_cut(array_fx)
			array_filtered = np.fft.ifft(ArrayData, iF).real

		else:
			msg = "No type of filter specified"
			raise TypeError(msg)

	# 2D FFT-LS 
	# elif trafo in ("LS"):

	# 	try:
	# 		yinfo = epidist2nparray(epidist(inv, event, st_tmp))
	# 	else:
	# 		msg='For wavenumber calculation inventory and event information is needed, not found.'
	# 		raise IOError(msg)	

	# 	# Apply filter.
	# 	if ftype in ("eliminate"):
	# 		array_filtered, periods = _fk_ls_filter_eliminate_phase_sp(ArrayData, y_dist=yinfo)

	# 	elif ftype in ("extract"):
	# 		array_filtered, periods = _fk_ls_filter_extract_phase_sp(ArrayData, y_dist=yinfo)

	# 	else:
	# 		print("No type of fk-filter specified")
	# 		raise TypeError		


	else:
		print("No valid input for type of Transformationtype")
		raise TypeError

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
	iK = int(math.pow(2,nextpow2(ix)+1))
	it = ArrayData.shape[1]
	iF = int(math.pow(2,nextpow2(it)+1))

	fkdata = np.fft.fft2(ArrayData, s=(iK,iF))
	
	return fkdata

def fk_reconstruct(st, inv, event, mu=5e-2):
	"""
	This functions reconstructs missing signals in the f-k domain, using the original data,
	including gaps, filled with zeros, and its Mask-array (see makeMask, and slope_distribution.
	Uses the following cost function to minimize:

			J = ||dv - T F^(-1) Wv Dv ||^{2}_{2} + mu^2 ||Dv||^{2}_{2}
			
			J := Cost function
			dv:= Column-wise-ordered long vector of the 2D signal d
			DV:= Column-wise-ordered long vector of the	f-k-spectrum
			W := Diagnoal matrix built from the column-wise-ordered long vector of Mask
			T := Sampling matrix which maps the fully sampled desired seismic data to the available samples.
				 For de-noising problems T = I (identity matrix)
			mu := Trade-off parameter between misfit and model norm

	Minimizing is done via a method of conjugate gradients, de-noising (1-2 iterations), reconstruction(8-10) iterations.
								scipy.optimize
								scipy.optimize.fmin_cg

	:param:

	returns:

	:param: 

	Reference:	Mostafa Naghizadeh, Seismic data interpolation and de-noising in the frequency-wavenumber
				domain, 2012, GEOPHYSICS
	"""
	peakpick = None
	deltaslope = 0.05
	slopes = [-10,10]

	# Prepare data.
	st_tmp = st.copy()
	ArrayData = stream2array(st_tmp, normalize=True)
	
	ix = ArrayData.shape[0]
	iK = int(math.pow(2,nextpow2(ix)+1))
	it = ArrayData.shape[1]
	iF = int(math.pow(2,nextpow2(it)+1))

	fkData = np.fft.fft2(ArrayData, s=(iK,iF))

	# Calculate mask-function W.
	M, prange, peaks = slope_distribution(fkData, slopes, deltaslope, peakpick)
	W = makeMask(fkData, peaks[0])

	# Prepare arrays for cost-function.
	dv = ArrayData.reshape(1, ArrayData.size)
	
	# First.
	#Dv = fkData.reshape(1, fkData.size)

	#Y = np.diag( W.reshape(1, W.size)[0]) 

	# Second.

	Dv = fkData 
	Y = W

	T = np.zeros((ArrayData.shape[0], ArrayData.shape[0]))
	for i,trace in enumerate(ArrayData):
		if sum(trace) == 0.:
			T[i] = 1.

	YDfft = Y * Dv
	YDifft = np.fft.ifft2( YDfft, s=(iK,iF) )

	dnew = np.dot(T, YDifft)

	# Create callable cost-function
	args = (dv, T, Y, mu, iK, iF)
	def _cost_function_denoise_interpolation(x, *args):
		"""
		Only use with the function fk_reconstruct!
		"""
		d, T, Y, mu, iK, iF = args

		YDfft = np.dot(Y, x)
		YDifft = np.fft.ifft2( YDfft, s=(iK,iF) )

		return np.linalg.norm(d - np.dot(T, YDifft) , 2) + mu**2. * np.linalg.norm(D, 2)

	# Initial conditions.
	x0 = Dv
	res1 = optimize.fmin_cg( _cost_function_denoise_interpolation , x0)


	return


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
	indicies = get_polygon(np.log(abs(dsfk)), polygon, xlabel, xticks, ylabel, yticks)

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
	indicies = get_polygon(np.log(abs(dsfk)), polygon, xlabel, xticks, ylabel, yticks)

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



