"""
perform 2d fft
ordne k richtig zu, Formel finden!
Synthetics mit sinus wavelet und multiplen dessen
damit erstmal testen
"""



import obspy
from obspy import read
import numpy as np
import matplotlib as plt


def fk_filter(stream, inventory, catalog, phase):
	"""
	Import stream, inventory, catalog and phase you want to investigate.
	The function bins the data, applies an 2D FFT, removes a certain window around the
	desired phase to surpress mutliples and applies an 2d iFFT

	param stream: with ending like MSEED
	type stream: string

	param inventory: with ending xml
	type inventory: string

	param catalog: with ending xml
	type catalog: string

	param phase: name of the phase to be investigated
	type phase: string
	"""

	"""
	Read data ######################################################################
	"""
	st = obspy.read(stream)
	inv = obspy.read_inventory(inventory)
	cat = obspy.readEvents(catalog)

	tr0 = st[0]
	tr1 = st[1]

	x = tr0.data
	y = tr1.data


	"""
	Binning of the data ############################################################
	"""
	#pushing the trace data in an array

	#calc min and max epidist between source and receivers

	#create equidistant (delta x) x-mesh with ->  N artificial receiver / ghost receiver

	#assign stations to ghost receiver

	#calc local slowness of the phase and align all the traces assigned to the respective ghost with that slowness

	#beamform them

	#return N binned traces with equidistant delta x 


	"""
	Correction of global(array) slowness of phase ##################################
	"""

	#align all binned traces with the slowness of the imported phase

	"""
	2D FFT #########################################################################
	"""
	#apply 2D FFT

	np.fft.fft2(a, s=None, axes=(-2, -1))
		"""
		Parameters
		----------
		a : array_like
		    Input array, can be complex
		s : sequence of ints, optional
		    Shape (length of each transformed axis) of the output
		    (`s[0]` refers to axis 0, `s[1]` to axis 1, etc.).
		    This corresponds to `n` for `fft(x, n)`.
		    Along each axis, if the given shape is smaller than that of the input,
		    the input is cropped.  If it is larger, the input is padded with zeros.
		    if `s` is not given, the shape of the input along the axes specified
		    by `axes` is used.
		axes : sequence of ints, optional
		    Axes over which to compute the FFT.  If not given, the last two
		    axes are used.  A repeated index in `axes` means the transform over
		    that axis is performed multiple times.  A one-element sequence means
		    that a one-dimensional FFT is performed.

		Returns
		-------
		out : complex ndarray
		    The truncated or zero-padded input, transformed along the axes
		    indicated by `axes`, or the last two axes if `axes` is not given.
		"""

	#mute area around |f| > eps, choose eps dependent on your phase/data/i dont know yet

	#apply 2D iFFT

	"""
	Undo global-correction of the phase ############################################
	""" 

	"""
	return stream with filtered and binned data ####################################
	"""








