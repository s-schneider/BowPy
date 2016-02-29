from __future__ import absolute_import
import obspy
import numpy
import numpy as np
import math
import matplotlib.pyplot as plt
from sipy.util.array_util import get_coords
import datetime
import scipy as sp
import scipy.signal as signal

from sipy.util.array_util import array2stream, stream2array, epidist2nparray
from sipy.util.fkutil import ls2ifft_prep, line_cut, line_set_zero, shift_array, get_polygon, nextpow2

def fk_filter(st, inv=None, cat=None, trafo=None, ftype=None, phase=None, epi_dist=None, polygon=4, normalize=True, SSA=False):
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

	param cat: catalog
	type cat: obspy.core.event.Catalog

	param trafo: Type of transformation, possible inputs are:
				 FK: for f-k transformation via numpy.fft.fft2
				 FX: for f-x transformation via numpy.fft.fft
				 LS: for a combination of 1D FFT in time-domain and and Lomb-Scargle
				     in the space-domain, via numpy.fft.fft and scipy.signal.lombscargle
	type trafo: string

	param ftype: type of method, possible inputs are:
				 eliminate
				 extract
				
				 if trafo is set to FK, also:
				 eliminate-polygon
				 extract-polygon

	type ftype: string

	param phase: name of the phase to be investigated
	type phase: string
	
	param epidist: list of epidistances, corresponding to st
	type epidist: list

	param polygon: number of vertices of polygon for fk filter, only needed 
				   if ftype is set to eliminate-polygon or extract-polygon.
				   Default is 4.
	type polygon: int
	
	param normalize: normalize data to 1
	type normalize: bool

	param SSA: Force SSA algorithm or let it check, default:False
	type SSA: bool


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

	# Convert format.
	ArrayData = stream2array(st, normalize)


	if st and inv and cat:
		epidist = epidist2nparray(epidist_stream(st, inv, cat))
	
	if not epi_dist == None:
		epidist = epi_dist

	# Calc mean diff of each epidist entry if it is reasonable
	# do a partial stack and apply filter.
	it = ArrayData.shape[1]
	ix = ArrayData.shape[0]
	iF = int(math.pow(2,nextpow2(it)+1))
	iK = int(math.pow(2,nextpow2(ix)+1))
	"""
	2D Frequency-Space / Wavenumber-Frequency Filter #########################################################
	"""

	# 2D f-k Transformation 
	# Decide when to use SSA to fill the gaps, calc mean distance of each epidist entry
	# if it differs too much --> SSA
	if trafo == "FK":
		
		array_fk = np.fft.fft2(ArrayData, s=(iK,iF))

		if ftype == "eliminate":
			array_filtered_fk = line_set_zero(array_fk)
		elif ftype == "extract":
			array_filtered_fk = line_cut(array_fk)				

		elif ftype == "eliminate-polygon":
			array_filtered_fk = _fk_eliminate_polygon(array_fk,polygon)

		elif ftype == "extract-polygon":
			array_filtered_fk = _fk_extract_polygon(array_fk,polygon)

		else:
			print("No type of filter specified")
			raise TypeError

		array_filtered = np.fft.ifft2(array_filtered_fk, s=(iK,iF))

	# 2D f-x Transformation 
	elif trafo == "FX":
		array_fx = np.fft.fft(ArrayData, iF)
		if ftype == "eliminate":
			array_filtered = line_set_zero(array_fx)

		elif ftype == "extract":
			array_filtered = line_cut(array_fx)
		array_filtered = np.fft.ifft(ArrayData, iF)

	# 2D FFT-LS 
	# UNDER CONSTRUCTION
	elif trafo == "LS":

		# Apply filter.
		if ftype == "eliminate":
			array_filtered, periods = _fk_ls_filter_eliminate_phase_sp(ArrayData, y_dist=epidist)
		elif ftype == "extract":
			array_filtered, periods = _fk_ls_filter_extract_phase_sp(ArrayData, y_dist=epidist)
		else:
			print("No type of fk-filter specified")
			raise TypeError		


	else:
		print("No valid input for type of Transformationtype")
		raise TypeError

	# Convert to Stream object.
	array_filtered = array_filtered[0:ix, 0:it]
	stream_filtered = array2stream(array_filtered, st_original=st)

	return(stream_filtered)



"""
FFT FUNCTIONS 
"""

def _fk_extract_polygon(data, polygon):
	"""
	Only use with the function fk_filter!
	Function to test the fk workflow with synthetic data
	param data:	data of the array
	type data:	numpy.ndarray
	"""
	# Shift 0|0 f-k to center, for easier handling
	dsfk = np.fft.fftshift(data)

	# Define polygon by user-input.
	indicies = get_polygon(np.log(abs(dsfk)), no_of_vert=polygon)

	# Create new array, only contains extractet energy, pointed to with indicies
	dsfk_extract = np.zeros(dsfk.shape)
	dsfk_extract.conj().transpose().flat[ indicies ]=1.
	data_fk = dsfk * dsfk_extract
	
	data_fk = np.fft.ifftshift(data_fk)

	return(data_fk)


def _fk_eliminate_polygon(data,polygon):
	"""
	Only use with the function fk_filter!
	Function to test the fk workflow with synthetic data
	param data:	data of the array
	type data:	numpy.ndarray
	"""
	# Shift 0|0 f-k to center, for easier handling
	dsfk = np.fft.fftshift(data)
	
	# Define polygon by user-input.
	indicies = get_polygon(np.log(abs(dsfk)), no_of_vert=polygon)

	# Create new array, contains all the energy, except the eliminated, pointed to with indicies
	dsfk_elim = dsfk.conj().transpose()
	dsfk_elim.flat[ indicies ]=0.
	data_fk = dsfk_elim.conj().transpose()

	data_fk = np.fft.ifftshift(data_fk)

	return(data_fk)

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
	return()

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
	epidist=y_dist
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
		k_new = signal.lombscargle(epidist, abs(freqT[j]), period_range)
		k_new = ls2ifft_prep(k_new, abs(freqT[j]))
		knum[j] = k_new

			
	#change dtype to integer, for further processing
	period_range = period_range.astype('int')
	fkspectra = knum
	if maxk:
		max_k = maxrow(fkspectra)
		dsfft = line_set_zero(fkspectra, max_k)
	else:
		dsfft = line_set_zero(fkspectra, 0, radius)
	
	return(fkspectra.conj().transpose(), period_range)


