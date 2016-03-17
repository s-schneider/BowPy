from __future__ import absolute_import, print_function
import numpy
import numpy as np
import math

import sys
import matplotlib
# If using a Mac Machine, otherwitse comment the next line out:
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.path as mplPath

import obspy
import obspy.signal.filter as obsfilter
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees, locations2degrees
from obspy.taup import TauPyModel
from obspy.core.event.event import Event
from obspy import Stream, Trace, Inventory

from sipy.util.base import nextpow2
from sipy.util.array_util import get_coords, attach_coordinates_to_traces, attach_network_to_traces, stream2array
from sipy.util.picker import pick_data, FollowDotCursor
import datetime
import scipy as sp
import scipy.signal as signal
from scipy import sparse





"""
A collection of useful functions for handling the fk_filter and seismic data.

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

def plot(st, inv=None, event=None, zoom=1, yinfo=False, markphase=None, norm=None, clr='black', newfigure=True):
	"""
	Alpha Version!
	
	Needs inventory and event of catalog for yinfo using

	param st: 	stream
	type  st:	obspy.core.stream.Stream

	param inv: inventory
	type  inv:

	param event: event of the seismogram
	type  event:

	param zoom: zoom factor of the traces
	type  zoom:	float

	param yinfo:	Plotting with y info as distance of traces
	type  yinfo:		bool

	param markphase: Phases, that should be marked in the plot, default is "None"
	type  markphase: list

	param norm: Depiction of traces; unprocessed or normalized. Normalization options are:
				all - normalized on biggest value of all traces
				trace - each trace is normalized on its biggest value
	type  norm: string or bool

	param clr: Color of plot
	type  clr: string

	"""

	#check for Data input
	if not isinstance(st, Stream):
		if not isinstance(st, Trace):
			msg = "Wrong data input, must be Stream or Trace"
			raise TypeError(msg)

	if newfigure:
		plt.figure()
	
	if isinstance(st, Stream):
		t_axis = np.linspace(0,st[0].stats.delta * st[0].stats.npts, st[0].stats.npts)
		data = stream2array(st)
		
		spacing=2.

		# Set axis information and bools.
		plt.xlabel("Time in s")
		isinv = False
		isevent = False

		if isinstance(inv, Inventory) and isinstance(event,Event):
			# Calculates y-axis info using epidistance information of the stream.
			# Check if there is a network entry
			attach_network_to_traces(st,inv[0])
			attach_coordinates_to_traces(st, inv, event)
			depth = event.origins[0]['depth']/1000.
			isinv = True
			isevent = True

		yold=0
		
		# Normalize Data, if set to 'all'
		if norm in ['all']:
			data = data/data.max()
		for j, trace in enumerate(data):

			# Normalize trace, if set to 'trace'
			if norm in ['trace']:
				trace = trace/trace.max()

			try:
				y_dist = st[j].stats.distance
			except:
				y_dist = yold + 1
			if markphase and isinv and isevent:
				origin = event.origins[0]['time']
				m = TauPyModel('ak135')
				arrivals = m.get_travel_times(depth, y_dist, phase_list=markphase)
				timetable = [ [], [] ]
				for k, phase in enumerate(arrivals):
					phase_name = phase.name
					t = phase.time
					phase_time = origin + t - st[j].stats.starttime
					Phase_npt = int(phase_time/st[j].stats.delta)
					Phase = Phase_npt * st[j].stats.delta

					if Phase < t_axis.min() or Phase > t_axis.max():
						continue	
					else:	
						timetable[0].append(phase_name)
						timetable[1].append(Phase)
					

				if yinfo:
					plt.ylabel("Distance in deg")
					plt.xlabel("Time in s")
					plt.annotate('%s' % st[j].stats.station, xy=(1,y_dist+0.1))
					plt.plot(t_axis,zoom*trace+ y_dist, color=clr)
					plt.plot( (timetable[1],timetable[1]),(-1+y_dist,1+y_dist), color='red' )
					for time, key in enumerate(timetable[0]):
						plt.annotate('%s' % key, xy=(timetable[1][time],y_dist))
				else:
					plt.ylabel("No. of trace")
					plt.xlabel("Time in s")
					plt.gca().yaxis.set_major_locator(plt.NullLocator())
					plt.annotate('%s' % st[j].stats.station, xy=(1,spacing*j+0.1))
					plt.plot(t_axis,zoom*trace+ spacing*j, color=clr)
					plt.plot( (timetable[1],timetable[1]),(-1+spacing*j,1+spacing*j), color='red' )
					for time, key in enumerate(timetable[0]):
						plt.annotate('%s' % key, xy=(timetable[1][time],spacing*j))

			elif markphase and not isinv and not isevent:
				msg='Markphase needs Inventory and Event Information, not found.'
				raise IOError(msg)		
			
			else:

				if yinfo:
					try:
						plt.ylabel("Distance in deg")
						plt.xlabel("Time in s")
						plt.annotate('%s' % st[j].stats.station, xy=(1,y_dist+0.1))
						plt.plot(t_axis,zoom*trace+ y_dist, color=clr)
				
					except:
						msg='Oops, something not found.'
						raise IOError(msg)
				else:
					plt.ylabel("No. of trace")
					plt.xlabel("Time in s")
					plt.gca().yaxis.set_major_locator(plt.NullLocator())
					plt.annotate('%s' % st[j].stats.station, xy=(1,spacing*j+0.1))
					plt.plot(t_axis,zoom*trace+ spacing*j, color=clr)			
			
			yold = y_dist
		plt.ion()
		plt.draw()
		plt.show()
		plt.ioff()

	elif isinstance(st, Trace):

		t_axis = np.linspace(0,st.stats.delta * st.stats.npts, st.stats.npts)
		data = st.data.copy()

		if norm in ['all', 'All', 'trace', 'Trace']:
			data = data/data.max()
		try:
			y_dist = st.stats.distance
		except:
			print("No distance information attached to trace, no phases are calculated!")
			markphases=False

		if markphase:
			origin = event.origins[0]['time']
			depth = event.origins[0]['depth']/1000.
			m = TauPyModel('ak135')
			arrivals = m.get_travel_times(depth, y_dist, phase_list=markphase)
			timetable = [ [], [] ]
			for k, phase in enumerate(arrivals):
				phase_name = phase.name
				t = phase.time
				phase_time = origin + t - st.stats.starttime
				Phase_npt = int(phase_time/st.stats.delta)
				Phase = Phase_npt * st.stats.delta

				if Phase < t_axis.min() or Phase > t_axis.max():
					continue	
				else:	
					timetable[0].append(phase_name)
					timetable[1].append(Phase)

			plt.ylabel("Amplitude")
			plt.xlabel("Time in s")
			title = st.stats.network+'.'+st.stats.station+'.'+st.stats.location+'.'+st.stats.channel
			plt.title(title)
			#plt.gca().yaxis.set_major_locator(plt.NullLocator())
			plt.plot(t_axis,zoom*data, color=clr)
			plt.plot( (timetable[1],timetable[1]),(-0.5,0.5), color='red' )
			for time, key in enumerate(timetable[0]):
				plt.annotate('%s' % key, xy=(timetable[1][time]+5,0.55))
		else:
			plt.ylabel("Amplitude")
			plt.xlabel("Time in s")
			title = st.stats.network+'.'+st.stats.station+'.'+st.stats.location+'.'+st.stats.channel
			plt.title(title)
			plt.plot(t_axis, zoom*data, color=clr)

		plt.ion()
		plt.draw()
		plt.show()
		plt.ioff()

def plot_data(data, zoom=1, y_dist=1, label=None, clr='black'):
	"""
	Alpha Version!
	Time axis has no time-ticks --> Working on right now
	
	Needs inventory and catalog for yinfo using
	param st: 	array of data
	type st:	np.array 
	param zoom: zoom factor of the traces
	type zoom:	float
	param y_dist:	separating distance between traces, for example equidistant with "1" 
					or import epidist-list via epidist
	type y_dist:	int or list
	"""
	plt.figure()
	for i, trace in enumerate(data):
		if isinstance(y_dist,int):
			if i == 0:
				plt.plot(zoom*trace+ y_dist*i, color=clr, label=label)
			else:
				plt.plot(zoom*trace+ y_dist*i, color=clr)

	plt.ion()
	plt.draw()
	plt.show()
	plt.legend()
	plt.ioff()

def plot_fft(x, logscale=False, fftshift=False):
	"""
	Doing an fk-trafo and plotting it.

	param x:	Data of the array
	type x:		np.array

	param logscale:	Sets scaling of the plot to logarithmic
	type logscale:	boolean

	param fftshift: If True shifts zero values of f and k into the center of the plot
	type fftshift:	boolean

	param scaling:	Sets the scaling of the plot in terms of aspectratio y/x
	type scaling:	int
	"""
	
#	if not type(x) == np.array or numpy.ndarray:
#		fftx = stream2array(x)	
#	else:
	fftx = x
	y,x = fftx.shape
	
	if fftshift:
		plt.imshow((np.abs(x)), origin='lower',cmap=None, aspect=scaling)
		if logscale:
			plt.imshow(np.log(np.abs(np.fft.fftshift(fftx))), origin='lower',cmap=None, aspect=scaling)
		else:
			plt.imshow((np.abs(np.fft.fftshift(fftx))), origin='lower',cmap=None, aspect=scaling)
	if not fftshift:
		if logscale:
			plt.imshow(np.log(np.abs(fftx)), origin='lower',cmap=None, aspect=scaling)
		else:
			plt.imshow((np.abs(fftx)), origin='lower',cmap=None, aspect=scaling)

	plt.colorbar()
	plt.show()
	
def plot_fft_subplot(x, logscale=False, fftshift=False, scaling=1):
	"""
	Doing an fk-trafo and plotting it.

	param x:	Data of the array
	type x:		np.array

	param logscale:	Sets scaling of the plot to logarithmic
	type logscale:	boolean

	param fftshift: If True shifts zero values of f and k into the center of the plot
	type fftshift:	boolean

	param scaling:	Sets the scaling of the plot in terms of aspectratio y/x
	type scaling:	int
	"""
	
#	if not type(x) == np.array or numpy.ndarray:
#		fftx = stream2array(x)	
#	else:
	fftx = x
	y,x = fftx.shape
	scaling = float(x) / (float(y) * 2.)
	if fftshift:
		plt.imshow((np.abs(x)), origin='lower',cmap=None, aspect=scaling)
		if logscale:
			plt.imshow(np.log(np.abs(np.fft.fftshift(fftx))), origin='lower',cmap=None, aspect=scaling)
		else:
			plt.imshow((np.abs(np.fft.fftshift(fftx))), origin='lower',cmap=None, aspect=scaling)
	if not fftshift:
		if logscale:
			plt.imshow(np.log(np.abs(fftx)), origin='lower',cmap=None, aspect=scaling)
		else:
			plt.imshow((np.abs(fftx)), origin='lower',cmap=None, aspect=scaling)

	plt.colorbar()


def plot_data_im(x, color='Greys'):
	y,x = x.shape
	scaling = float(x) / (float(y) * 2.)	
	plt.imshow(x, origin='lower', cmap=color, interpolation='nearest', aspect=scaling)
	plt.show()

	
def multplot(data1, data2, Log):
	y,x = data1.shape
	scale1 = float(x) / (float(y) * 2.)
	y,x = data2.shape
	scale2 = float(x) / (float(y) * 2.)
	plt.subplot(2,1,1)
	plot_fft_subplot(data1, logscale=Log, scaling=scale1)
	plt.subplot(2,1,2)
	plot_fft_subplot(data2, logscale=Log, scaling=scale2)
	plt.show()

def kill(data, stat):
	"""
	Deletes the trace of a selected station from the array

	param data:	array data
	type data:	np.array

	param stat:	station(s)/trace(s) to be killed
	type stat: int or list
	"""

	data = np.delete(data, stat, 0)
	return(data)

def line_cut(array, stat=0):
	"""
	Sets the array to zero, except for the stat line and the radius values around it.
	"Cuts" one line out. 
	"""
	new_array = np.zeros(array.shape).astype('complex')
	new_array[stat] = array[stat]

	return(new_array)

def line_set_zero(array, stat=0):
	"""
	Sets lines zero in array
	"""
	new_array = array
	new_array[stat] = 0
	
	return(new_array)

def extract_nonzero(array):
	newarray = array[~np.all(array == 0, axis=1)]
	newindex = np.unique(array.nonzero()[0])
	return(newarray, newindex)

def convert_lsindex(ls_range, samplespacing):
	n = len(ls_range)
	fft_range = ls_range * n * samplespacing
	return(fft_range)

def ls2ifft_prep(ls_periodogram, data):
	"""
	Converts a periodogram of the lombscargle function into an array, that can be used
	to perform an IRFFT
	"""
	fft_prep = np.roll(ls_periodogram, 1)
	N = data.size
	a = 0
	for i in range(N):
		a = a + data[i]
	a = a/N
	fft_prep[0] = a
	return(fft_prep)
	
def shift_array(array, shift_value=0, y_dist=False):
	array_shift = array
	try:
		for i in range(len(array)):
			array_shift[i] = np.roll(array[i], -shift_value*y_dist[i])
	except (AttributeError, TypeError):
		for i in range(len(array)):
			array_shift[i] = np.roll(array[i], -shift_value*i)
	return(array_shift)

def makeMask(fkdata,slope):
	"""
	This function creates a Mask-array in shape of the original fkdata,
	with straight lines (value = 1.) along the angles, given in slope and 0 everywhere else.
	slope shows the position of L linear dominants in the f-k domain.

	:param fkdata:

	:param slope:


	Returns 

	:param W: Mask function W
	"""
	M = fkdata.copy()
	
	pnorm = 1/2. * ( float(M.shape[0])/float(M.shape[1]) )
	
	prange = slope * pnorm

	Mask = np.zeros(M.shape)
	W = np.zeros(M.shape)

	for m in prange:
		
		if m == 0.:
			Mask[0,:] = 1.
			Mask[1,:] = 1.
			Mask[Mask.shape[0]-1,:] = 1.
		for f in range(Mask.shape[1]):
			Mask[:,f] = np.roll(Mask[:,f], int(f*m))
		Mask[0,:] = 1.
		for f in range(Mask.shape[1]):
			Mask[:,f] = np.roll(Mask[:,f], -int(f*m))

	# Convolving each frequency slice of the mask with a boxcar
	# of size L. Widens the the maskfunction along k-axis.
	b = sp.signal.boxcar(slope.size)
	
	for i, fslice in enumerate(Mask.conj().transpose()):
		W[:,i] = sp.signal.convolve(fslice, b, mode=1)

	W[np.where(W!=0)]=1

	return W

def slope_distribution(fkdata, prange, pdelta, peakpick=None, delta_threshold=0, mindist=0, smoothing=1, interactive=False):
	"""
	Generates a distribution of slopes in a range given in prange.
	Needs fkdata as input. 

	k on the y-axis fkdata[0]
	f on the x-axis fkdata[1]

	:param fkdata: array-like dataset transformed to f-k domain.

	:param prange: range of slopes, with minimum and maximum value
				   Slopes are defined as nondimensional, by the formula

							m = 1/2 * (ymax - ymin)/(xmax - xmin),

				   respectively

							p = 1/2 * (kmax - kmin)/(fmax - fmin).

				   The factor 1/2 is due too the periodicity in the f-k domain.
				
	:type prange: array-like, tuple or list

	:param pdelta: stepsize of slope-interval
	:type pdelta: int
	
	:param peakpick: method to pick the peaks of distribution possible is:
						mod - minimum mean of distribution (default)
						mop  - minimum mean of peaks
						float - minimum float value
						None - pick all peaks
					 
	:type peakpick: int or float
	
	:param delta_threshold: Value to lower manually 
	:type delta_threshold: int

	:param mindist: minimum distance between two peaks
	:type mindist: float or int

	:param smoothing: Parameter to smooth distribution
	:type smoothing: float
	
	:param interactive: If True, picking by hand is enabled.
	:type interactive: boolean

	returns:

	:param MD: Magnitude distribution of the slopes p
	:type MD: 1D array, numpy.ndarray

	:param prange: Range of slopes
	:type prange: 1D array, numpy.ndarray


	:param peaks: position ( peaks[0] ) and value ( peaks[1] ) of the peaks.
	:type peaks: numpy.ndarray
	"""

	M = fkdata.copy()
	Mt = M.conj().transpose()
	fk_shift =	np.zeros(M.shape).astype('complex')
	
	pmin = prange[0]
	pmax = prange[1]
	N = abs(pmax - pmin) / pdelta + 1
	srange = np.linspace(pmin,pmax,N)
	MD = np.zeros(N)
	pnorm = 1/2. * ( float(M.shape[0])/float(M.shape[1]) )

	rend = float( len(srange) )
	for i, delta in enumerate(srange):

		p = delta*pnorm
		for j, trace in enumerate(Mt):
			shift = int(math.floor(p*j))		
			fk_shift[:,j] = np.roll(trace, shift)
		MD[i] = sum(abs(fk_shift[0,:])) / len(fk_shift[0,:])

		prcnt = 100*(i+1) / rend
		print("%i %% done" % prcnt, end="\r")
		sys.stdout.flush()	

	if interactive:
		
		peaks = pick_data(srange, MD, 'Slope in fk-domain', 'Magnitude of slope', 'Magnitude-Distribution')
		for j,pairs in enumerate(peaks):
			if len(pairs) > 1:
				maxm = 0
				for i, item in enumerate(pairs):
					if item[1] > maxm:
						maxm = item[1]
						idx = i
				peaks[j] = [(pairs[idx][0], maxm)]
		peaks = np.array(peaks).reshape(len(peaks), 2)

	else:
		MDconv = sp.signal.convolve(MD, sp.signal.boxcar(int(abs(pmin-pmax))*smoothing),mode=1)
		peaks_first = find_peaks(MDconv, srange, peakpick='All', mindist=0.3)
		peaks_first[1] = peaks_first[1]/peaks_first.max()*MD.max()

		# Calculate envelope of the picked peaks, and pick the 
		# peaks of the envelope.
		peak_env = obsfilter.envelope( peaks_first[1] )
		peaks = find_peaks( peaks_first[1], peaks_first[0], peak_env.mean() + delta_threshold)		
	
	return MD, srange, peaks

def find_peaks(data, drange=None, peakpick='mod', mindist=0.2):
	"""
	Finds peaks in given 1D array, by search for values in data
	that are higher than the two neighbours. Except first and last.
	:param data: 1D array-like

	:param drange: optional, range of the data distribution.

	returns:
	:param peaks: array with position on 0 axis and value on 1-axis.
	"""

	pos = []
	peak = []
	
	# Loop through all values except first and last.
	pick = None
	for p, value in enumerate(data[1:len(data)-1]):
		if peakpick in ['mod', 'MoD', 'Mod', 'MoP', 'Mop', 'mop']:
			if value > data[p] and value > data[p+2] and value > data.mean():
				pick = data[p+1]

		elif isinstance(peakpick, float) or isinstance(peakpick, int):
			if value > data[p] and value > data[p+2] and value > peakpick:
				pick = data[p+1]

		elif peakpick in ['all', 'All', 'AlL', 'ALl', 'ALL']:
			if value > data[p] and value > data[p+2]:
				pick = data[p+1]
			elif value < data[p] and value < data[p+2]:
				pick = data[p+1]

		elif not peakpick:
			if value > data[p] and value > data[p+2]:
				pick = data[p+1]

		if pick:		
			if len(pos)>0 and abs(pos[len(pos)-1] - drange[p+1]) <= mindist:
				pick = None
				continue
			else:		
				pos.append(drange[p+1])			
				peak.append(pick)

			pick = None

	peak = np.array(peak)

	# If mean of picks is choosen.
	if peakpick in ['MoP', 'Mop', 'mop']:
		newpeak = []
		newpos = []
		for i, value in enumerate(peak):
			if value > peak.mean():
				newpeak.append(value)
				newpos.append(pos[i])


		peaks = np.append([newpos], [newpeak], axis=0)

	else:
		peaks = np.append([pos], [peak], axis=0)

	return peaks

def find_subsets(numbers, target, bottom, top, minlen, partial=[], sets=[]):
	"""
	Generator to create all possible combinations of entrys in numbers, that sum up to target.
	example:	for value in subset_sum([0,1,2,3,4], 5, 0, 0, minlen=2):
					print value

	output:		[0, 1, 4]
				[0, 2, 3]
				[1, 4]
				[2, 3]
	""" 

	s = sum(partial)
	if s >= target *( 1. - bottom) and s <= target * ( 1. + top):
		if len(partial) >= minlen:
			yield sets
	if s > target:
		return
 
	for i, n in enumerate(numbers): #np.diff(numbers)):
		remaining = numbers[i+1:] # np.diff(numbers)[i+1:]
		for item in find_subsets(remaining, target, bottom, top, minlen, partial + [n], sets + [numbers[i]]):
			print(item)

def create_iFFT2mtx(nx, ny):
	"""
	Take advantage of the use of scipy.sparse library.
	Creates the Matrixoperator for an array x.
	
	:param x: Array to calculate the Operator for
	:type x: array-like

	returns
	:param sparse_iFFT2mtx: 2D iFFT operator for matrix of shape of x.
	:type sparse_iFFT2mtx: scipy.sparse.csr.csr_matrix

	"""
	N = nx * ny

	iDFT1 = np.fft.fft(sparse.eye(nx).toarray().transpose()).conj().transpose()
	iDFT2 = np.fft.fft(sparse.eye(ny).toarray().transpose()).conj().transpose()

	# Create Sparse matrix, with iDFT1 ny-times repeatet on the diagonal.

	# Initialze lil_matrix, to write diagonals in correct way.
	tmp = sparse.lil_matrix((N,N), dtype='complex')
	row = 0
	for i in range(ny):
		for j in range(nx):
			tmp[row, (i)*nx:(i+1)*nx] = iDFT1[j,:]
			row += 1

		#Screen feedback.
		prcnt = 50*(i+1) / float(ny) -1
		if prcnt >=0 : print("%i %% done" % prcnt, end="\r")
		sys.stdout.flush()	

	# Export tmp to a diagonal sparse matrix.
	sparse_iDFT1 = tmp.tocsc()

	# Initialze lil_matrix for iDFT2 and export it to sparse.
	tmp = sparse.lil_matrix((N,N), dtype='complex')
	row = 0	
	for i in range(ny):
		for j in range(nx):
			indx = np.arange(j,N,nx)
			tmp[row,indx] = iDFT2[i,:]
			row += 1

		prcnt = (50*(i+1) / float(ny)) + 49
		print("%i %% done" % prcnt, end="\r")
		sys.stdout.flush()	
	
	sparse_iDFT2 = tmp.tocsc()
	print("\n")
	print("100 %% done \n")

	# Calculate matrix dot-product iDFT2 * iDFT1 and divide it 
	# by the number of all samples (nx * ny)
	sparse_iFFT2mtx = sparse_iDFT2.dot(sparse_iDFT1)/float(N)
	
	return sparse_iFFT2mtx

def cg_solver(A,b,niter,x0=None):
	"""
	Conjugate gradient solver for Ax = b lstsqs problems, as shown in Tomographic 
	inversion via the conjugate gradient method, Scales, J. 1987
	Expect a mxn Matrix A, a rhs b and an optional startvalue x0
	
	:param A:
	:type A:
	
	:param b:
	:type b:

	:param x0:
	:type x0:

	:param niter:
	:type niter:

	returns

	:param:
	:type:
	"""
	print("--- Using CG-method --- \n \nInitiating matrices... \n \n")
	
	if not x0.any():
		x = np.zeros(A.shape[1])
	else:
		x = x0.copy()

	s = np.zeros(A.shape[0])
	r = np.zeros(A.shape[1])
	p = np.zeros(A.shape[1])
	q = np.zeros(A.shape[1])

	s = b - np.dot(A,x).real
	r = np.dot( A.conj().transpose(), b - np.dot(A, x).real )
	p = r.copy()
	q = np.dot(A, p).real
	
	print("Starting iterations. \n \n")
	for k in range(niter):
		#print("Currently in iteration %i" % int(k+1), end="\r")
		sys.stdout.flush()

		alpha = np.dot(r,r) / np.dot(q,q)
		x = x + alpha * p
		s = s - alpha * q
	
		r_old = r.copy()
		r = np.dot(A.conj().transpose(),s).real
		
		beta = np.dot(r,r) / np.dot(r_old, r_old)
		p = r + beta * p
		q = np.dot(A, p).real
		misfit = np.linalg.norm( np.dot(A, x).real - b, 2)
		print("Misfit after %i iterations is : %f \n" % (int(k+1), misfit) )
	
	mnorm = np.linalg.norm(x, 2)
	
	return x, misfit, mnorm

def lstsqs(A,b,mu=0):

	print("Calculating AhA")
	Ah = A.conjugate().transpose()
	AhA= Ah.dot(A)
	
	print("Calculating I")
	I = sparse.identity(A.shape[0])
	
	tmp = AhA + mu*I
	tmpI = sparse.linalg.inv(tmp)
	
	print("Calculating x")
	x = tmpI.dot(Ah.dot(b))
	print("..finished")
	return x

	
