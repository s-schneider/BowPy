from __future__ import absolute_import, print_function
import numpy
import numpy as np
import math

import sys
import matplotlib as mpl
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
from obspy.core.inventory.network import Network
from sipy.util.base import nextpow2, stream2array
from sipy.util.array_util import get_coords, attach_coordinates_to_traces, attach_network_to_traces
from sipy.util.picker import pick_data, FollowDotCursor
from sipy.filter.ssa import fx_ssa
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

def plot(st, inv=None, event=None, zoom=1, yinfo=False, epidistances=None, markphases=None, phaselabel=True, phaselabelclr='red', 
		norm='all', clr='black', clrtrace=None, newfigure=True, savefig=False, dpi=400, xlabel=None, ylabel=None, t_axis=None, 
		fs=15, tw=None, verbose=False):
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

	param markphases: Phases, that should be marked in the plot, default is "None"
	type  markphases: list

	param norm: Depiction of traces; unprocessed or normalized. Normalization options are:
				all - normalized on biggest value of all traces
				trace - each trace is normalized on its biggest value
	type  norm: string or bool

	param clr: Color of plot
	type  clr: string
	
	param clrtrace: dict containing tracenumber and color, e.g.:
						>>>		clrtrace = {1: 'red', 7: 'green'}
						>>>		trace 1 in red
						>>>		trace 7 in green
							
	type  clrtrace: list 

	"""

	#check for Data input
	if not isinstance(st, Stream):
		if not isinstance(st, Trace):
			try:
				if isinstance(yinfo,bool):
					yinfo = 1
				plot_data(st, zoom=zoom, y_dist=yinfo, clr=clr, newfigure=newfigure, savefig=savefig, dpi=dpi, xlabel=xlabel, ylabel=ylabel, t_axis=t_axis, fs=fs)
				return
			except:
				msg = "Wrong data input, must be Stream or Trace"
				raise TypeError(msg)
	if newfigure:
		# Set axis information and bools.
		fig, ax = plt.subplots()
		if xlabel:
			ax.set_xlabel(xlabel, fontsize=fs)
		else:
			ax.set_xlabel("Time(s)", fontsize=fs)
		if ylabel:
			ax.set_ylabel(ylabel, fontsize=fs)

		ax.tick_params(axis='both', which='major', labelsize=fs)	
	else:
		ax 	= plt.gca()
		fig = plt.gcf()


	if isinstance(st, Stream):

		# Check if just specific timewindow should be plotted.
		try:
			tw = np.array(tw)
			twdelta = tw.max() - tw.min()
			t_axis = np.linspace(tw.min(),tw.max(), twdelta/float(st[0].stats.delta))
			npts_min = int(tw.min()/float(st[0].stats.delta))
			npts_max = int(tw.max()/float(st[0].stats.delta))
		except:
			t_axis_max = st[0].stats.delta * st[0].stats.npts
			t_axis = np.linspace(0,t_axis_max, st[0].stats.npts)
			tw = np.array([0, t_axis_max])
			npts_min = 0
			npts_max = st[0].stats.npts

		data = stream2array(st)
	
		spacing=2.
		ax.set_xlim(tw.min(), tw.max())
		isinv = False
		isevent = False
		cclr = clr

		# Check if inventory and catalog is input, then calculate distances etc. 
		if isinstance(inv, Inventory): isinv   = True
		if isinstance(event,Event):    isevent = True

		if isinv:
			# Calculates y-axis info using epidistance information of the stream.
			# Check if there is a network entry
			attach_network_to_traces(st,inv)
			attach_coordinates_to_traces(st, inv, event)
		else:
			for trace in st:
				try:
					if not trace.stats.distance:
						isinv = False
						break
					else:
						isinv = True
				except:
						isinv = False
						break					
		
		try:
			depth = event.origins[0]['depth']/1000.
			isevent = True
		except AttributeError:
			try:
				depth = st[0].stats.depth	
				isevent = True		
			except AttributeError:
				isevent = False
			
		yold=0
		
		# Normalize Data, if set to 'all'
		if norm in ['all']:
			data = data/data.max()
		
		if yinfo:
			ymin = st[0].stats.distance
			ymax = st[0].stats.distance

		for j, trace in enumerate(data):

			# Normalize trace, if set to 'trace'
			if norm in ['trace']:
				trace = trace/trace.max()

			try:
				y_dist = st[j].stats.distance
			except:
				y_dist = yold + 1

			if markphases and isinv and isevent:
				try:
					origin = st[0].stats.origin
				except AttributeError:
					origin = event.origins[0]['time']
				except:
					msg=('No origin-time found in stream or event-file')
					raise IOError(msg)

				m = TauPyModel('ak135')
				arrivals = m.get_travel_times(depth, y_dist, phase_list=markphases)
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

				if not timetable[0] or not timetable[1]:
					print('Phases not in Seismogram')
					plt.close('all')
					return

				if yinfo:
					if not ylabel: ax.set_ylabel("Distance(deg)", fontsize=fs)

					if st[j].stats.distance < ymin: ymin = st[j].stats.distance
					if st[j].stats.distance > ymax: ymax = st[j].stats.distance

					try:
						if j in clrtrace: 
							cclr = clrtrace[j]
						else:
							cclr = clr
					except:
						cclr = clr

					ax.annotate('%s' % st[j].stats.station, xy=(1 + tw.min(),y_dist+0.1))
					ax.plot(t_axis,zoom*trace[npts_min: npts_max]+ y_dist, color=cclr)
					ax.plot( (timetable[1],timetable[1]),(-1+y_dist,1+y_dist), color=phaselabelclr )
					if verbose:
						print(timetable[1] + st[j].stats.shifttime)
						ax.plot( (timetable[1] + st[j].stats.shifttime, timetable[1] + st[j].stats.shifttime), \
								(-1+y_dist,1+y_dist), color=phaselabelclr )

					if phaselabel:
						for time, key in enumerate(timetable[0]):
							ax.annotate('%s' % key, xy=(timetable[1][time],y_dist))
							if verbose:
								ax.annotate('%s' % key, xy=(timetable[1][time] + st[j].stats.shifttime, y_dist))
					else:
						continue

				else:
					if not ylabel: ax.set_ylabel("No. of trace", fontsize=fs)

					try:
						if j in clrtrace: 
							cclr = clrtrace[j]
						else:
							cclr = clr
					except:
						cclr = clr

					fig.gca().yaxis.set_major_locator(plt.NullLocator())
					ax.annotate('%s' % st[j].stats.station, xy=(1 + tw.min(),spacing*j+0.1))
					ax.plot(t_axis,zoom*trace[npts_min: npts_max]+ spacing*j, color=cclr)
					ax.plot( (timetable[1],timetable[1]),(-1+spacing*j,1+spacing*j), color=phaselabelclr )
					if verbose:
						print(st[j].stats.shifttime)
						print(timetable[1] + st[j].stats.shifttime)

						ax.plot( (timetable[1] + st[j].stats.shifttime, timetable[1] + st[j].stats.shifttime), \
								(-1+spacing*j,1+spacing*j), color=phaselabelclr )

					if phaselabel:
						for time, key in enumerate(timetable[0]):
							ax.annotate('%s' % key, xy=(timetable[1][time],spacing*j))
							if verbose:
								ax.annotate('%s' % key, xy=(timetable[1][time] + st[j].stats.shifttime, spacing*j))
					else:
						continue

			elif markphases and not isinv:
				msg='markphases needs Inventory Information, not found.'
				raise IOError(msg)	
	
			elif markphases and not isevent:
				msg='markphases needs Event Information, not found.'
				raise IOError(msg)		
				
			elif type(epidistances) == numpy.ndarray or type(epidistances)==list:
				y_dist = epidistances
				if not ylabel: ax.set_ylabel("Distance(deg)", fontsize=fs)
				try:
					if j in clrtrace: 
						cclr = clrtrace[j]
					else:
						cclr = clr
				except:
					cclr = clr
				ax.annotate('%s' % st[j].stats.station, xy=(1 + tw.min(),y_dist[j]+0.1))
				ax.plot(t_axis, zoom*trace[npts_min: npts_max] + y_dist[j], color=cclr)

			else:
				if yinfo:

					try:
						if not ylabel: ax.set_ylabel("Distance(deg)", fontsize=fs)

						try:
							if j in clrtrace: 
								cclr = clrtrace[j]
							else:
								cclr = clr
						except:
							cclr = clr

					except:
						msg='Oops, something not found.'
						raise IOError(msg)
						
					if st[j].stats.distance < ymin: ymin = st[j].stats.distance
					if st[j].stats.distance > ymax: ymax = st[j].stats.distance
					ax.annotate('%s' % st[j].stats.station, xy=(1 + tw.min(),y_dist+0.1))
					ax.plot(t_axis,zoom*trace[npts_min: npts_max]+ y_dist, color=cclr)

				else:
					if not ylabel: ax.set_ylabel("No. of trace", fontsize=fs)
					try:
						if j in clrtrace: 
							cclr = clrtrace[j]
						else:
							cclr = clr
					except:
						cclr = clr

					fig.gca().yaxis.set_major_locator(plt.NullLocator())
					ax.annotate('%s' % st[j].stats.station, xy=(1 + tw.min(),spacing*j+0.1))
					ax.plot(t_axis,zoom*trace[npts_min: npts_max]+ spacing*j, color=cclr)			

			yold = y_dist


		if yinfo:
			ylim = (ymin-1, ymax+1)
			ax.set_ylim(ylim)

		if savefig:
			fig.set_size_inches(8,7)
			fig.savefig(savefig, dpi=dpi)
			plt.close("all")
		else:
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

		if markphases:
			origin = event.origins[0]['time']
			depth = event.origins[0]['depth']/1000.
			m = TauPyModel('ak135')
			arrivals = m.get_travel_times(depth, y_dist, phase_list=markphases)
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

			if not ylabel: ax.set_ylabel("Amplitude", fontsize=fs)
			title = st.stats.network+'.'+st.stats.station+'.'+st.stats.location+'.'+st.stats.channel
			ax.set_title(title, fontsize=fs)
			#plt.gca().yaxis.set_major_locator(plt.NullLocator())
			ax.plot(t_axis,zoom*data, color=clr)
			ax.plot( (timetable[1],timetable[1]),(-0.5,0.5), color=phaselabelclr )
			if phaselabel:
				for time, key in enumerate(timetable[0]):
					ax.annotate('%s' % key, xy=(timetable[1][time]+5,0.55))

		else:
			if not ylabel: ax.set_ylabel("Amplitude", fontsize=fs)
			title = st.stats.network+'.'+st.stats.station+'.'+st.stats.location+'.'+st.stats.channel
			ax.set_title(title, fontsize=fs)
			ax.plot(t_axis, zoom*data, color=clr)

		if savefig:
			plt.savefig(savefig)
			plt.close("all")
		else:
			plt.ion()
			plt.draw()
			plt.show()
			plt.ioff()

def plot_data(data, zoom=1, y_dist=1, label=None, clr='black', newfigure=True, savefig=False, dpi=400, xlabel=None, ylabel=None, t_axis=None, fs=15):
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
	if newfigure: 
		fig, ax = plt.subplots()
		ax.set_xlabel(xlabel, fontsize=fs)
		ax.set_ylabel(ylabel, fontsize=fs)
		ax.tick_params(axis='both', which='major', labelsize=fs)	
		ticks = mpl.ticker.FuncFormatter(lambda r, pos: '{0:g}'.format(r/y_dist))
		ax.yaxis.set_major_formatter(ticks)

	else:
		ax 	= plt.gca()
		fig = plt.gcf()

	for i, trace in enumerate(data):
		if isinstance(y_dist,int):
			try:
				if i == 0:
					ax.plot(t_axis, zoom*trace+ y_dist*i, color=clr, label=label)
				else:
					ax.plot(t_axis, zoom*trace+ y_dist*i, color=clr)
			except:
				if i == 0:
					ax.plot(zoom*trace+ y_dist*i, color=clr, label=label)
				else:
					ax.plot(zoom*trace+ y_dist*i, color=clr)					

	if savefig:
		fig.set_size_inches(12,10)
		fig.savefig(savefig, dpi=dpi)
		plt.close('all')
	else:
		plt.ion()
		plt.draw()
		ax.legend()
		plt.show()
		plt.ioff()

def plotfk(data, fs=15, savefig=False, dpi=400, logscale=False, hold=False):
	fig, ax = plt.subplots()
	ax.set_xlabel('Normalized Wavenumber', fontsize=fs)
	ax.set_ylabel('Normalized Frequency', fontsize=fs)
	#ax.xaxis.tick_top()
	ax.xaxis.set_ticks((-0.5, 0.0, 0.5))
	#ax.xaxis.set_label_position('top')
	ax.tick_params(axis='both', which='major', labelsize=fs)

	if logscale:
		im = ax.imshow(np.flipud(np.log(abs(np.fft.fftshift(data[:,:data.shape[1]/2].transpose(), axes=1)))), aspect='auto', extent=(-0.5, 0.5, 0, 0.5), interpolation='none')	
	else:
		im = ax.imshow(np.flipud(abs(np.fft.fftshift(data[:,:data.shape[1]/2].transpose(), axes=1))), aspect='auto', extent=(-0.5, 0.5, 0, 0.5), interpolation='none')
	cbar = fig.colorbar(im)
	cbar.ax.tick_params(labelsize=fs)
	cbar.ax.set_ylabel('R', fontsize=fs)
	#plt.gca().invert_yaxis()

	if savefig:
		fig.set_size_inches(7,8)
		fig.savefig(savefig, dpi=dpi)
		plt.close('all')
	else:
		if not hold:
			plt.ion()
			ax.legend()
			plt.show()
			plt.ioff()
		else:
			ax.legend()
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

def line_cut(array, shape):
	"""
	Sets the array to zero, except for the 0 line and given features given in shape.
	"Cuts" one line out + given shape. For detailed information look in sipy.filter.fk.fk_filter

	:param array: array-like
	:type  array: numpy.ndarray

	:param shape: shape and filter Information
	:type  shape: list
	"""

	fil=None
	name = shape[0]
	kwarg = shape[1] 
	length = shape[2]
	new_array = np.zeros(array.shape).astype('complex')
	if name in ['spike', 'Spike']:
		new_array[0] = array[0]
		return new_array

	elif name in ['boxcar', 'Boxcar'] and isinstance(length, int):
		new_array[0] = array[0]
		newrange = np.linspace(1, length, length).astype('int')
		for i in newrange:
			new_array[i] = array[i]
			new_array[new_array.shape[0]-i] = array[new_array.shape[0]-i]			
		return new_array

	elif name in ['butterworth', 'Butterworth', 'taper', 'Taper'] and isinstance(length, int):
		fil_lh = create_filter(name, array.shape[0]/2, length, kwarg)

	elif name in ['taper', 'Taper'] and isinstance(length, int):
		fil_lh = create_filter(name, array.shape[0]/2, array.shape[0]/2-length, kwarg)
		fil_lh = -1. * fil_lh + 1.

	fil_rh = np.flipud(fil_lh)[::-1][0:][::-1]
	fil = np.zeros(2*fil_lh.size)
	fil[:fil.size/2] = fil_lh
	fil[fil.size/2:] = fil_rh

	new_array = array.transpose() * fil
	new_array = new_array.transpose()

	return(new_array)

def line_set_zero(array, shape):
	"""
	Sets 0 line zero in array + features given in shape.
	For detailed information look in sipy.filter.fk.fk_filter

	:param array: array-like
	:type  array: numpy.ndarray

	:param shape: shape and filter Information
	:type  shape: list
	"""


	fil=None
	name = shape[0]
	kwarg = shape[1]
	length = shape[2]
	new_array = array

	if name in ['spike', 'Spike']:
		new_array[0] = np.zeros(array[0].size)
		return new_array

	elif name in ['boxcar', 'Boxcar'] and isinstance(length, int):
		new_array[0] = np.zeros(array[0].size)
		newrange = np.linspace(1, length, length).astype('int')
		for i in newrange:
			new_array[i] = np.zeros(array[new_array.shape[0]-i].size)
			new_array[new_array.shape[0]-i] = np.zeros(array[new_array.shape[0]-i].size)
		return new_array

	elif name in ['butterworth', 'Butterworth', 'taper', 'Taper'] and isinstance(length, int):
		fil_lh = create_filter(name, array.shape[0]/2, length, kwarg)

	elif name in ['taper', 'Taper'] and isinstance(length, int):
		fil_lh = create_filter(name, array.shape[0]/2, length, kwarg)
		fil_lh = -1. * fil_lh + 1.

	fil_rh = np.flipud(fil_lh)[::-1][1:][::-1]
	fil = np.zeros(2*fil_lh.size)
	fil[:fil.size/2] = fil_lh
	fil[fil.size/2+1:] = fil_rh
	newfil = np.ones(fil.shape)
	newfil = newfil - fil
	
	new_array = array.transpose() * newfil
	new_array = new_array.transpose()
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

def makeMask(fkdata, slope, shape, rth=0.4, expl_cutoff=False):
	"""
	This function creates a Mask-array in shape of the original fkdata,
	with straight lines (value = 1.) along the angles, given in slope and 0 everywhere else.
	slope shows the position of L linear dominants in the f-k domain.

	:param fkdata:

	:param slope:

	:param shape: shape[0] describes the shape of the lobes of the mask. Possible inputs are:
				 -boxcar (default)
				 -taper
				 -butterworth

				  shape[1] is an additional attribute to the shape of taper and butterworth, for:
				 -taper: maskshape[1] = slope of sides
				 -butterworth: maskshape[1] = number of poles
				
				 e.g.: maskshape['taper', 2] produces a symmetric taper with slope of side = 2.


	:type  maskshape: list
	
	:param rth =  Resamplethreshhold, marks the border between 0 and 1 for the resampling
	Returns 

	:param W: Mask function W
	"""
	M 			= fkdata.copy()
	
	# Because of resolution issues, upsampling to double size
	pnorm 		= 1/2. * ( float(M.shape[0]+1)/float(2. * M.shape[1]) )
	
	prange 		= slope * pnorm
	Mask 		= np.zeros((M.shape[0], 2.*M.shape[1]))
	maskshape 	= np.zeros((M.shape[0], 2.*M.shape[1]))
	W 			= np.zeros((M.shape[0], 2.*M.shape[1]))
	name 		= shape[0]
	arg 		= shape[1]

	if name in ['butterworth', 'Butterworth', 'taper', 'Taper']:
		if not expl_cutoff:
			cutoff 	= slope.size/2
		else:
			cutoff 	= expl_cutoff
 
		if cutoff < 1: cutoff = 1
		maskshape_tmp 	= create_filter(name, Mask.shape[0]/2, cutoff, arg)
		maskshape_lh 	= np.tile(maskshape_tmp, Mask.shape[1]).reshape(Mask.shape[1], maskshape_tmp.size).transpose()
		maskshape_rh 	= np.flipud(maskshape_lh)

		maskshape[:maskshape.shape[0]/2,:] = maskshape_lh
		maskshape[maskshape.shape[0]/2:,:] = maskshape_rh


	for m in prange:
		if m == 0.:
			Mask[0,:] = 1.
			Mask[1,:] = 1.
			Mask[Mask.shape[0]-1,:] = 1.

		for f in range(Mask.shape[1]):
			Mask[:,f] = np.roll(Mask[:,f], -int(math.floor(f*m)))

		if name in ['boxcar']:
			Mask[0,:] = 1.
		else:
			Mask = maskshape.copy()

		for f in range(Mask.shape[1]):
			Mask[:,f] = np.roll(Mask[:,f], int(f*m))
			
		W += Mask		
		Mask =  np.zeros((M.shape[0], 2.*M.shape[1]))

	# Convolving each frequency slice of the mask with a boxcar
	# of size L. Widens the the maskfunction along k-axis.
	if name in ['boxcar']:
		if arg:
			b = sp.signal.boxcar(arg)
		else:		
			b = sp.signal.boxcar(slope.size)
		for i, fslice in enumerate(W.conj().transpose()):
			W[:,i] = sp.signal.convolve(fslice, b, mode=1)

		W[np.where(W!=0)]=1.

		W[np.where(W > 1.)]=1.

	# Resample it to original size
	Wr = np.flipud(sp.signal.resample(W, M.shape[1], axis=1))
	Wr[ np.where(Wr > 1 ) ] = 1.
	Wr[ np.where(Wr < 0 ) ] = 0.
	if name in ['boxcar']:
		Wr[ np.where(Wr > rth ) ] = 1.
		Wr[ np.where(Wr < rth ) ] = 0.


	Wlhs	= np.roll(Wr[:,0:Wr.shape[1]/2-1], shift=1, axis=0)
	Wrhs	= Wr[:,1:Wr.shape[1]/2+1]
	Wrhs 	= np.roll(np.flipud(np.fliplr(Wrhs)), shift=0, axis=0)

	Wr[:,0:Wr.shape[1]/2-1] = Wlhs
	Wr[:,Wr.shape[1]/2:] = Wrhs
	return Wr

def slope_distribution(fkdata, prange, pdelta, peakpick=None, delta_threshold=0, mindist=0, smoothing=False, interactive=False):
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
	
	pnorm = 1/2. * ( float(M.shape[0])/float(M.shape[1]) )

	pmin = prange[0]
	pmax = prange[1]
	N = abs(pmax - pmin) / pdelta + 1
	MD = np.zeros(N)
	srange = np.linspace(pmin,pmax,N)

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
		peaks = np.array(peaks).reshape(len(peaks), 2).transpose()

	else:
		if smoothing:
			blen = int(abs(pmin-pmax))*smoothing
			if blen < 1 : blen=1
			MDconv = sp.signal.convolve(MD, sp.signal.boxcar(blen),mode=1)
		else:
			MDconv=MD
		peaks_first = find_peaks(MDconv, srange, peakpick='All', mindist=0.3)
		peaks_first[1] = peaks_first[1]/peaks_first.max()*MD.max()

		# Calculate envelope of the picked peaks, and pick the 
		# peaks of the envelope.
		peak_env = obsfilter.envelope( peaks_first[1] )
		peaks_tmp = find_peaks( peaks_first[1], peaks_first[0], peak_env.mean() + delta_threshold)		
	
		if peaks_tmp[0].size > 4:
			peaks = find_peaks( peaks_tmp[1], peaks_tmp[0], 0.5 + delta_threshold)
		else:
			peaks = peaks_tmp
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

def pocs(data, maxiter, noft, alpha=0.9, beta=None, method='linear', dmethod='denoise', peaks=None, maskshape=None, dt=None, p=None, flow=None, fhigh=None, slidingwindow=False, overlap=0.5):
	"""
	This functions reconstructs missing signals in the f-k domain, using the original data,
	including gaps, filled with zeros. It applies the projection onto convex sets (pocs) algorithm in
	2D.

	Reference: 3D interpolation of irregular data with a POCS algorithm, Abma & Kabir, 2006

	:param data:
	:type  data:

	:param maxiter:
	:type  maxiter:

	:param nol: Number of loops
	:type  nol:
	
	:param alpha: Factor of threshold decrease after each iteration
	:type  alpha: float

	:param method: Method to be used for the pocs algorithm
					-'linear', 'exp', 'mask' or 'ssa'

	:param peaks: Slope values for the mask

	:param maskshape: Shape of the corners of mask, see makemask

	returns:

	:param datap:
	:type  datap:
	"""
	#if not decrease in ('linear', 'exp', None):
	#	msg='No decrease method chosen'
	#	raise IOError(msg)

	ArrayData 	= data.copy()
	ix = ArrayData.shape[0]
	iK = int(math.pow(2,nextpow2(ix)))
	it = ArrayData.shape[1]
	iF = int(math.pow(2,nextpow2(it)))
	fkdata = np.fft.fft2(ArrayData, s=(iK,iF))
	threshold = abs(fkdata.max())
		
	ADold = ArrayData.copy()
	ADnew = ArrayData.copy()
	ADfinal = np.zeros(ArrayData.shape).astype('complex')

	if method in ('linear', 'exp'):
		if slidingwindow:
			if dmethod in ('reconstruct'):
				w_length = int(data.shape[1] / 3.)
				swh = np.hanning(w_length)

				loc = 0.
				inside = True
				while inside:
					curr_win = int(loc)
					ADtemp = ArrayData[:,curr_win:curr_win+w_length].copy()
						
					for i in range(maxiter):
						data_tmp 	= ADtemp.copy()
						fkdata 		= np.fft.fft2(data_tmp, s=(iK,iF))
						fkdata[ np.where(abs(fkdata) < threshold)] 	= 0. + 0j

						if method in ('linear'):
							threshold 	= threshold * alpha
						elif method in ('exp'):
							threshold 	= threshold * sp.exp(-(i+1) * alpha)

						data_tmp 	= np.fft.ifft2(fkdata, s=(iK,iF)).real[0:ix, 0:it].copy()
						ADtemp[noft] 	= data_tmp[noft][:,curr_win:curr_win+w_length].copy()

					

					if loc == 0.:
						ADfinal[:,curr_win:curr_win+w_length] = ADtemp.copy()
					else:
						ADfinal[:,curr_win:curr_win+w_length] = ADfinal[:,curr_win:curr_win+w_length] + ADtemp
						ADfinal[:,curr_win-int(overlap*w_length):int(curr_win)] = ( ADold[:,int((1-overlap)*w_length):] + ADtemp[:,:int(overlap*w_length)] ) / 2.

					ADold = ADtemp.copy()
					threshold = abs(np.fft.fft2(ADold, s=(iK,iF)).max())

					loc += overlap * w_length
					print(loc)
					if loc >= data.shape[1]: inside=False
	
		else:
			if dmethod in ('reconstruct'):
				ADtemp = ArrayData.copy()
				for i in range(maxiter):
					data_tmp 	= ADtemp.copy()
					fkdata 		= np.fft.fft2(data_tmp, s=(iK,iF))
					fkdata[ np.where(abs(fkdata) < threshold)] 	= 0. + 0j

					if method in ('linear'):
						threshold 	= threshold * alpha
					elif method in ('exp'):
						threshold 	= threshold * sp.exp(-(i+1) * alpha)

					data_tmp 	= np.fft.ifft2(fkdata, s=(iK,iF)).real[0:ix, 0:it].copy()
					ADtemp[noft] 	= data_tmp[noft]
					name = str(i) + '.png'
					# plt.ion()
					# plotfk(fkdata)
					# fig = plt.gcf()
					# fig.set_size_inches(7,8)
					# fig.savefig(name)
					# plt.ioff()
				ADfinal = ADtemp.copy()

				threshold = abs(np.fft.fft2(ADfinal, s=(iK,iF)).max())
			
			elif dmethod in ('denoise', 'de-noise'):
				for n in noft:
					ADtemp = ArrayData.copy()
					for i in range(maxiter):
						data_tmp 	= ADtemp.copy()
						fkdata 		= np.fft.fft2(data_tmp, s=(iK,iF))
						fkdata[ np.where(abs(fkdata) < threshold)] 	= 0. + 0j

						if method in ('linear'):
							threshold 	= threshold * alpha
						elif method in ('exp'):
							threshold 	= threshold * sp.exp(-(i+1) * alpha)

						data_tmp 	= np.fft.ifft2(fkdata, s=(iK,iF)).real[0:ix, 0:it].copy()
						ADtemp[n] 	= data_tmp[n]
						#save = 'pocsdata' + str(i) + '.png'
						#plot(data_tmp, ylabel='Distance(m)', xlabel='Time(s)', fs=22, yinfo=2, savefig=save)
					ADfinal = ADtemp.copy()

				threshold = abs(np.fft.fft2(ADfinal, s=(iK,iF)).max())

	elif method in ('mask'):
		W 		= makeMask(fkdata, peaks[0], maskshape)
		ADfinal = ArrayData.copy()
		for n in noft:
			ADtemp 	= ArrayData.copy()
			threshold = abs(W*np.fft.fft2(ADfinal, s=(iK,iF))).max()
			for i in range(maxiter):
				data_tmp 	=ADtemp.copy()
				fkdata 		= W * np.fft.fft2(data_tmp, s=(iK,iF))
				fkdata[ np.where(abs(fkdata) < threshold)] 	= 0. + 0j
				threshold 	= threshold * alpha
				data_tmp 	= np.fft.ifft2(fkdata, s=(iK,iF)).real[0:ix, 0:it].copy()
				ADtemp[n] 	= data_tmp[n]

			ADfinal[n] = ADtemp[n].copy()

	elif method in ('ssa'):

		for n in noft:
			for i in range(maxiter):
				data_tmp 		= ArrayData.copy()
				data_ssa 		= fx_ssa(data_tmp,dt,p,flow,fhigh)
				ArrayData 		= alpha * ArrayData			
				ArrayData[n] 	= (1. - alpha) * data_ssa[n]

			ADfinal[n] = ArrayData[n].copy()

	elif method in ('average'):
		threshold = beta * abs(np.fft.fft2(ArrayData, s=(iK,iF)).max())
		ADtemp = ArrayData.copy()
		for n in noft:
			for i in range(maxiter):
				data_tmp 	= ADtemp.copy()
				fkdata 		= np.fft.fft2(data_tmp, s=(iK,iF))
				fkdata[ np.where(abs(fkdata) < threshold)] 	= 0. + 0j

				ADtemp 		= alpha*data_tmp + (1. - alpha) * np.fft.ifft2(fkdata, s=(iK,iF)).real[0:ix, 0:it]
				ADtemp[n] 	= (1. - alpha) * np.fft.ifft2(fkdata, s=(iK,iF)).real[0:ix, 0:it][n]

			ADfinal = ADtemp.copy()


	elif method == 'maskvary':
				
		ADfinal = ArrayData.copy()
		ADtemp 	= ArrayData.copy()
		for n in noft:
			for i in range(maxiter):
				W 			= makeMask(fkdata, peaks, shape=maskshape, expl_cutoff=i)
				data_tmp 	= ADtemp.copy()
				fkdata 		= W * np.fft.fft2(data_tmp, s=(iK,iF))
				data_tmp 	= np.fft.ifft2(fkdata, s=(iK,iF)).real[0:ix, 0:it].copy()
				ADtemp[n]	= alpha * ArrayData[n].copy()			
				ADtemp[n]  += (1. - alpha) * data_tmp[n]

			ADfinal[n] = ArrayData[n].copy()

	else:
		print('no method specified')
		return

	datap = ADfinal.copy()

	return datap

def cg_solver(A,b,x0=None,niter=10):
	"""
	Conjugate gradient solver for Ax = b lstsqs problems, as shown in Tomographic 
	inversion via the conjugate gradient method, Scales, J. 1987
	Expect a mxn Matrix A, a rhs b and an optional startvalue x0
	
	:param A:
	:type A:
	
	:param dv:
	:type dv:

	:param x0:
	:type x0:

	:param niter:
	:type niter:

	returns

	:param:
	:type:
	"""
	
	if A.shape[0] != A.shape[1]:
		msg='Dimension missmatch, A should be NxN'
		raise IOError(msg)
	print("--- Using CG-method --- \n \nInitiating matrices... \n \n")

	
	
	if x0.any(): x = x0
	else: x = np.zeros(A.shape[1])

	r = b - A.dot(x).real
	p = r.copy()

	print("Starting iterations. \n \n")

	cont = True
	k = 1
	resnorm = 0.
	while cont:	

		alpha = np.dot(r,r) / np.dot(p,A.dot(p))
		x_new = x + alpha * p
		r_new = r - alpha * A.dot(p)
		
		beta = np.dot(r_new,r_new) / np.dot(r,r)
		p_new = r_new + beta * p
		
		x = x_new.copy()
		p = p_new.copy()


		#mcalc = A.dot(x).real - b
		#resnorm_old = resnorm
		#resnorm = np.linalg.norm(mcalc.transpose().toarray(), 2)

		#print("Misfit after %i iterations is : %f \n" % (int(k+1), resnorm) )
		if k == niter: cont=False
		#if resnorm < 1e-8: cont=False
		#elif abs(resnorm - resnorm_old) < 1e-8: cont=False
		k +=1

	#solnorm = np.linalg.norm(x.toarray(), 2)
	
	return x_new #, resnorm, solnorm

def dcg_solver(A, b, mu,niter,x0=None):
	"""
	Damped conjugate gradient solver for Ax = b lstsqs problems, as shown in Tomographic 
	inversion via the conjugate gradient method, Scales, J. 1987
	Expect a mxn Matrix A, a rhs b and an optional startvalue x0

	minimizes the following problem by applying CGLS to:

							|| (   A  )  	  ( b )	||^{2}	
						min || (mu * I) *x 	- ( 0 )	||_{2}  
				
				==>		min || G * m - d || ^{2}_{2} 
	
	"""
	print("--- Using dCG-method --- \n \nInitiating matrices... \n \n")
	
	# Create G matrix and d Vector.

	I = mu**2. * sparse.identity(A.shape[0])
	
	G = sparse.lil_matrix(np.vstack((A.toarray(), I.toarray())))
	G = G.tocsc()

	d = np.hstack( (b, np.zeros(I.shape[0])) )


	# Initialize startvalues.
	try:
		if not x0:
			m = np.zeros(A.shape[1])
			s = -d
	except ValueError:
		if x0.any(): 
			m = x0
			s = G.dot(m) - d
	except:
		msg = 'No valid input'
		raise IOError(msg)

	
	beta_old 	= 0.
	r 			= G.transpose().conjugate().dot(s)
	p_old 		= np.zeros(r.shape)

	print("Starting iterations. \n \n")

	cont = True
	k = 1
	while cont:	

		p 		= -r + beta_old * p_old
		alpha 	= np.linalg.norm(r,2)**2. /  p.dot(G.transpose().conjugate().toarray()).dot(G.dot(p))
		m_new 	= m + alpha * p
		s_new 	= s + alpha * G.dot(p)
		r_new	= G.transpose().conjugate().dot(s_new)
		beta	= np.linalg.norm(r_new,2)**2. / np.linalg.norm(r,2)**2.
	
		m			= m_new.copy()
		s			= s_new.copy()
		r			= r_new.copy()
		beta_old	= beta.copy()
		p_old		= p.copy()
		
		misfit 	= np.linalg.norm(G.dot(m) - d, 2)
		rnorm 	= np.linalg.norm(m,2)**2.

		plt.ion()
		plt.figure()		
		plt.imshow(abs(m.reshape(20,300)),aspect='auto', interpolation='none')
		plt.show()

		if k == niter: cont=False
		k +=1
		
		
	x = m.copy()
	return x


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

def create_filter(name, length, cutoff=None, ncorner=None):
	
	cut = float(cutoff)/float(length)
	m 	= float(ncorner)

	if name in ['butterworth', 'Butterworth']:
		x = np.linspace(0, 1, length)
		y = 1. / (1. + (x/float(cut))**(2.*ncorner))
	
	elif name in ['taper', 'Taper']:
		shift	= 0.
		fit = True
		while fit:
			cut += shift
			x 		= np.linspace(0, 1, length)
			y 		= (cut-x)*m + 0.5
			y[y>1.] = 1.
			y[y<0.] = 0.
			if y.max() >= 1: fit=False
			shift = 0.1
		
	else:
		msg='No valid name for filter found.'
		raise IOError(msg)

	return y

def fktrafo(stream, normalize=True):
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

def ifktrafo(fkdata, stream, normalize=True):
	"""
	Calculates the inverse f,k - transformation of the data in fkdata. Returns the trafo as an array.

	"""
	StreamData= stream2array(stream)
	ix   = StreamData.shape[0]
	iK   = int(math.pow(2,nextpow2(ix)))
	it   = StreamData.shape[1]
	iF   = int(math.pow(2,nextpow2(it)))

	fk_tmp = fkdata.copy()

	ArrayData = np.fft.ifft2(fkdata, s=(iK,iF))
	ArrayData = ArrayData[0:ix, 0:it]

	return ArrayData

def eval_fkarea(fkdata, no_of_phases, polygon, xlabel, xticks, ylabel, yticks):
	"""
	Calculates the mean of an area in the fk-domain, depending of the number of linear events.

	Author: Simon Schneider, 2016
	"""

	indicies = get_polygon(abs(dsfk), polygon, xlabel, xticks, ylabel, yticks)


	return fkdata_eval
	
