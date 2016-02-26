from __future__ import absolute_import
import numpy
import numpy as np
import matplotlib.pyplot as plt

import obspy
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees, locations2degrees
from obspy.taup import TauPyModel


from sipy.utilities.array_util import get_coords, attach_coordinates_to_traces, attach_network_to_traces,stream2array
import datetime
import scipy as sp
import scipy.signal as signal

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

def plot(st, inv=None, event=None, zoom=1, y_dist=1, yinfo=False, markphase=None):
	"""
	Alpha Version!
	
	Needs inventory and event of catalog for yinfo using

	param st: 	stream
	type st:	obspy.core.stream.Stream

	param inv:	inventory
	type inv:

	param event:	event of the seismogram
	type event:

	param zoom: zoom factor of the traces
	type zoom:	float

	param y_dist:	separating distance between traces, for example equidistant with "1" 
					or import epidist-list via epidist
	type y_dist:	int or list

	param yinfo:	Plotting with y info as distance of traces
	type yinfo:		bool

	param markphase: Phase, that should be marked in the plot, default is "None"
	type markphase: string
	"""

	#check for Data input
	if not type(st) == obspy.core.stream.Stream:
		msg = "Wrong data input, must be obspy.core.stream.Stream"
		raise TypeError(msg)

	t_axis = np.linspace(0,st[0].stats.delta * st[0].stats.npts, st[0].stats.npts)
	data = stream2array(st, normalize=True)
	
	spacing=2.

	# Set axis information.
	plt.ylabel("Distance in deg")
	plt.xlabel("Time in s")
	if inv and event:
		# Calculates y-axis info using epidistance information of the stream.
		# Check if there is a network entry
		if not st[0].meta.network:
			msg="No network entry found in stream"
			raise ValueError(msg)

		attach_coordinates_to_traces(st, inv, event)
		depth = event.origins[0]['depth']/1000.
		no_x,no_t = data.shape

		for j in range(no_x):
			y_dist = st[j].meta.distance

			if markphase:
				origin = event.origins[0]['time']
				m = TauPyModel('ak135')
				time = m.get_travel_times(depth, y_dist)
				for k in range(len(time)):
					if time[k].name != markphase:
						continue
					t = time[k].time
				phase_time = origin + t - st[j].stats.starttime
				Phase_npt = int(phase_time/st[j].stats.delta)
				Phase = Phase_npt * st[j].stats.delta
			
				if yinfo:
					plt.plot(t_axis,zoom*data[j]+ y_dist, color='black')
					plt.plot( (Phase,Phase),(-1+y_dist,1+y_dist), color='red' )			
				else:
					plt.plot(t_axis,zoom*data[j]+ spacing*j, color='black')
					plt.plot( (Phase,Phase),(-1+spacing*j,1+spacing*j), color='red' )

			else:
				if yinfo:
					plt.plot(t_axis,zoom*data[j]+ y_dist, color='black')
				else:
					plt.plot(t_axis,zoom*data[j]+ spacing*j, color='black')			

	else:
		print("no inventory and event given")
		raise ValueError

	plt.show()

def plot_data(data, zoom=1, y_dist=1, bins=None):
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

	for i in range(len(data)):
		if type(y_dist) == int:
			plt.plot(zoom*data[i]+ y_dist*i, color='black')
		if type(y_dist) == list or type(y_dist) == numpy.ndarray:
			plt.plot(zoom*data[i]+ y_dist[i], color='black')
	if type(bins) == numpy.ndarray:
		print('plotting bins')
		for j in range(bins.size):
			plt.plot( (0, data[0].size), (bins[j],bins[j]), color='red' )

	plt.show()


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

def line_cut_old(array, stat, radius=None):
	"""
	Old Version, if the new works fine --> delete
	Sets the array to zero, except for the stat line and the radius values around it.
	"Cuts" one line out. 
	"""
	x = len(array[0])
	y = len(array)
	
	new_array = np.array([ np.array(np.zeros(x), dtype=np.complex) ])
	new_line = new_array
	for i in range(y)[1:]:
		new_array = np.append(new_array, new_line, axis=0)
	
	if radius:
		for i in range(stat, stat + radius  + 1):
			new_array[i] = array[i]
			if i == 0:
				new_array[y] = array[y]
			else:
				new_array[y-i] = array[y-i]
	
	else:
		new_array[stat] = array[stat]

	
	return(new_array)

def line_cut(array, stat, radius=None):
	"""
	Sets the array to zero, except for the stat line and the radius values around it.
	"Cuts" one line out. 
	"""
	new_array = np.zeros( (len(array[0]),len(array)) )

	if radius:
		for i in range(stat, stat + radius  + 1):
			new_array[i] = array[i]
			if i == 0:
				new_array[y] = array[y]
			else:
				new_array[y-i] = array[y-i]
	else:
		new_array[stat] = array[stat]

	
	return(new_array)

def line_set_zero(array, stat, radius=None):
	"""
	Sets lines zero in array
	"""
	new_array = array
	end = len(array)-1
	if radius:
		for i in range(stat, stat + radius + 1):
			new_array[i] = 0
			if i == 0:
				new_array[end] = 0
			else:
				new_array[end-i] = 0

	else:
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

def nextpow2(i):
	#See Matlab documentary
	n = 1
	count = 0
	while n < abs(i):
		n *= 2
		count+=1
	return count

	
def shift_array(array, shift_value=0, y_dist=False):
	array_shift = array
	try:
		for i in range(len(array)):
			array_shift[i] = np.roll(array[i], -shift_value*y_dist[i])
	except (AttributeError, TypeError):
		for i in range(len(array)):
			array_shift[i] = np.roll(array[i], -shift_value*i)
	return(array_shift)

def get_polygon(data, no_of_vert=4):
	"""
	Interactive function to pick a polygon out of a figure and receive the vertices of it.
	:param data:
	:type:
	
	:param no_of_vert: number of vertices, default 4, 
	:type no_of_vert: int
	"""
	from sipy.utilities.polygon_interactor import PolygonInteractor
	from matplotlib.patches import Polygon
	
	# Define shape of polygon.
	y,x = data.shape
	aspectratio = float(x) / (float(y) * 2.)
	xmin= x/3.
	xmax= x*2./3.
	ymin= y/3.
	ymax= y*2./3.

	xs = []
	for i in range(no_of_vert):
		if i >= no_of_vert/2:
			xs.append(xmax)
		else:
			xs.append(xmin)

	ys = np.linspace(ymin, ymax, no_of_vert/2)
	ys = np.append(ys,ys[::-1]).tolist()

	poly = Polygon(list(zip(xs, ys)), animated=True, closed=False, fill=False)
	
	# Add polygon to figure.
	fig, ax = plt.subplots()
	ax.add_patch(poly)
	p = PolygonInteractor(ax, poly)
	plt.title("Pick polygon, close figure to save vertices")
	plt.imshow(abs(data), aspect=aspectratio)
	plt.show()		
	
	vertices = (poly.get_path().vertices).astype('int')
	
	if no_of_vert == 4:
		indicies = convert_polygon_to_flat_index(data, vertices)

	return(indicies)

def convert_polygon_to_flat_index(data, vertices):
	"""
	Converts vertices of a polygon taken of an imshow plot of data to 
	flat-indicies.
	
	:param data: speaks for itself
	:type data: numpy.ndarray

	:param vertices: also...
	:type vertices: numpy.ndarray
	
	"""

	xlen = vertices[:,0].max() - vertices[:,0].min()
	ylen = data.shape[0]
	#ylen = vertices[:,1].max() - vertices[:,1].min()
	indicies=np.zeros(xlen*ylen)		
	
	arr=np.zeros((xlen*ylen,2))
	i=0		
	for x in np.arange(vertices[:,0].min(), vertices[:,0].max()).astype('int'):
		for y in range(ylen):
			arr[i]=[x,y]
			i+=1	

	arr = arr.transpose().astype('int').tolist()
	flat_index= np.ravel_multi_index(arr, data.conj().transpose().shape)

	return(flat_index)		
