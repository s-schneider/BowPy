from __future__ import absolute_import
import obspy
import numpy
import numpy as np
import matplotlib.pyplot as plt
from sipy.utilities.array_util import get_coords
import datetime
import scipy as sp
import scipy.signal as signal

from sipy.utilities.array_util import array2stream, stream2array, epidist2nparray
from sipy.utilities.fkutil import ls2ifft_prep, line_cut, line_set_zero, shift_array

def fk_filter(st, ftype=None, inv=None, cat=None, phase=None, epi_dist=None, fktype=None, normalize=False, SSA=False):
	"""
	At this point prework with programs like seismic handler or SAC is needed to perform correctly


	Import stream, the function applies an 2D FFT, removes a certain window around the
	desired phase to surpress a slownessvalue corresponding to a wavenumber and applies an 2d iFFT.
	To fill the gap between uneven distributed stations use the SSA algorithm.
	Alternative is an nonequidistant 2D Lombard-Scargle transformation.

	param st: Stream
	type st: obspy.core.stream.Stream
	
	param ftype: Type of filter, FFT or LS (Lombard-Scargle)
	type ftype: string

	param inv: inventory
	type inv: obspy.station.inventory.Inventory

	param cat: catalog
	type cat: obspy.core.event.Catalog

	param phase: name of the phase to be investigated
	type phase: string
	
	param epidist: list of epidistances, corresponding to st
	type epidist: list

	param fktype: type of fk-filter, extraction or elimination
	type fktype: string
	
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



	"""
	2D Wavenumber-Frequency Filter #########################################################
	"""
	# 2D FFT-LS #####################################################################
	# UNDER CONSTRUCTION
	if ftype == "LS":
		#Convert format.
		ArrayData = stream2array(st)
		if st and inv and cat:
			epidist = epidist2nparray(epidist_stream(st, inv, cat))
		
		if not epi_dist == None:
			epidist = epi_dist

		# Apply filter.
		if fktype == "eliminate":
			array_filtered, periods = _fk_ls_filter_eliminate_phase_sp(ArrayData, y_dist=epidist)
		elif fktype == "extract":
			array_filtered, periods = _fk_ls_filter_extract_phase_sp(ArrayData, y_dist=epidist)
		else:
			print("No type of fk-filter specified")
			raise TypeError		
			
		fkspectra=array_filtered

		return(fkspectra, periods)

	#2D FFT #########################################################################
	# Decide when to use SSA to fill the gaps, calc mean distance of each epidist entry
	# if it differs too much --> SSA
	elif ftype == "FFT":
		#Convert format.
		ArrayData = stream2array(st, normalize)

		if st and inv and cat:
			epidist = epidist2nparray(epidist_stream(st, inv, cat))
		
		if not epi_dist == None:
			epidist = epi_dist

		# Calc mean diff of each epidist entry if it is reasonable
		# do a partial stack and apply filter.


		# Apply filter.
		if fktype == "fx-eliminate":
			array_filtered = _fx_fft_filter_eliminate_phase(ArrayData, radius=None)

		elif fktype == "fx-extract":
			array_filtered = _fx_fft_filter_extract_phase(ArrayData, radius=None)

		elif fktype == "fk":
			array_filtered = _fk_filter(ArrayData, indicies)

		else:
			print("No type of fk-filter specified")
			raise TypeError

		#Convert to Stream object
		stream_filtered = array2stream(array_filtered)

		for i in range(len(stream_filtered)):
			stream_filtered[i].meta = st[i].meta

		return(stream_filtered)

	else:
		print("No valid input for type of filter")
		raise TypeError

	"""
	return stream with filtered data ####################################
	"""



# FFT FUNCTIONS ############################################################################

def _fx_fft_filter_extract_phase(data, snes=False, y_dist=False, radius=None):
	"""
	Only use with the function fk_filter!
	Function to extract a desired phase in the f-x-domain.
	param data:	data of the array
	type data:	numpy.ndarray

	param snes:	slownessvalue of the desired extracted phase
				default is False, function expects corrected data
	type snes:	int
	"""
	# Shift array to desired phase.
	if snes:
		ds = shift_array(data, snes, y_dist)
	else:
		ds = data
	
	# Apply FFT.
	dsfft = np.fft.fftn(ds)
	# Extract zero-wavenumber
	dsfft = line_cut(dsfft, 0,radius)
	
	ds = np.fft.ifftn(dsfft)

	data_fk = ds

	return(data_fk.real)

def _fx_fft_filter_eliminate_phase(data, snes=False, y_dist=False, radius=None):
	"""
	Only use with the function fk_filter!
	Function to eliminate a given phase in the f-x-domain.
	param data:	data of the array
	type data:	numpy.ndarray

	param snes:	slownessvalue of the desired extracted phase
	type snes:	int
	"""
	if snes:
		ds = shift_array(data, snes, y_dist)
	else:
		ds = data
	
	# Apply FFT.
	dsfft = np.fft.fftn(ds)
	# Cut zero-wavenumber
	dsfft = line_set_zero(dsfft, 0, radius)
	
	ds = np.fft.ifftn(dsfft)

	data_fk = ds

	return(data_fk.real)

def _fk_fft_filter(data, indicies):
	"""
	Only use with the function fk_filter!
	Function to test the fk workflow with synthetic data
	param data:	data of the array
	type data:	numpy.ndarray

	param snes:	slownessvalue of the desired extracted phase
	type snes:	int
	"""
	ds = data
	
	# Apply FFT t-x  --> f-x.
	dsfx = np.fft.fftn(ds)
	# Apply second FFT f-x --> f-k.
	dsfk_tmp = np.fft.fftn(dsfx.conj().transpose()).conj().transpose()

	# Create new array, only contains extractet energy, pointed to with indicies
	dsfk = np.zeros(dsfk.shape)
	dsfk.conj().transpose().flat[ indicies ]=1
	dsfk = dsfk_tmp * dsfk

	# Apply inverse FFT from f-k --> t-x 	
	
	dsfx = np.fft.ifftn(dsfk.conj().transpose()).conj().transpose()
	ds = np.fft.ifftn(dsfx)

	data_fk = ds

	return(data_fk.real)





# LS FUNCTIONS ############################################################################
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


