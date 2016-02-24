import numpy
import numpy as np
import matplotlib.pyplot as plt

import obspy
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees
from obspy.taup import TauPyModel


from array_util import get_coords, attach_coordinates_to_traces

import datetime
import scipy as sp
import scipy.signal as signal

"""
A collection of useful functions for handling the fk_filter and seismic data.

Author: S. Schneider 2015

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details: http://www.gnu.org/licenses/
"""

def create_sine( no_of_traces=10, len_of_traces=30000, samplingrate = 30000,
                 no_of_periods=1):
    
    deltax = 2*np.pi/len_of_traces
    signal_len = len_of_traces * no_of_periods
    period_time = 1 / samplingrate
    data_temp = np.array([np.zeros(signal_len)])
    t = []
    
    # first trace
    for i in range(signal_len):
        data_temp[0][i] = np.sin(i*deltax)
        t.append((float(i) + float(i)/signal_len)*2*np.pi/signal_len)
        data = data_temp
	
    # other traces
    for i in range(no_of_traces)[1:]:
       #np.array([np.zeros(len_of_traces)])
       #new_trace[0] = data[0]
       data = np.append(data, data_temp, axis=0)
       
       
    return(data, t)



def create_deltasignal(no_of_traces=10, len_of_traces=30000,
                       multiple=False, multipdist=2, no_of_multip=1, slowness=None,
                       zero_traces=False, no_of_zeros=0,
                       noise_level=0,
                       non_equi=False):
	"""
	function that creates a delta peak signal
	slowness = 0 corresponds to shift of 1 to each trace
	"""
	if slowness:
		slowness = slowness-1
	data = np.array([noise_level * np.random.rand(len_of_traces)])
	
	if multiple:
		dist = multipdist
		data[0][0] = 1
		for i in range(no_of_multip):
			data[0][dist+i*dist] = 1
	else:
		data[0][0] = 1
  
	data_temp = data
	for i in range(no_of_traces)[1:]:
		if slowness:
			new_trace = np.roll(data_temp, slowness*i)
		else:
			new_trace = np.roll(data_temp,i)
		data = np.append(data, new_trace, axis=0)

	if zero_traces:
		first_zero=len(data)/no_of_zeros
		while first_zero <= len(data):
			data[first_zero-1] = 0
			first_zero = first_zero+len(data)/no_of_zeros
	
	if non_equi:
		for i in [5, 50, 120]:
			data = line_set_zero(data, i, 10)
		data, indices= extract_nonzero(data)
	else:
		indices = []

	return(data, indices)

def standard_test_signal(snes1=1, snes2=3, noise=0, nonequi=False):
        y, yindices = create_deltasignal(no_of_traces=200, len_of_traces=200, 
        						multiple=True, multipdist=5, no_of_multip=1, 
        						slowness=snes1, noise_level=noise,
        						non_equi=nonequi)

        x, xindices = create_deltasignal(no_of_traces=200, len_of_traces=200, 
        						multiple=True, multipdist=5, no_of_multip=5, 
        						slowness=snes2, noise_level=noise,
        						non_equi=nonequi)
        a = x + y
        y_index = np.sort(np.unique(np.append(yindices, xindices)))
        return(a, y_index)

def shift_array(array, shift_value=0, y_dist=False):
	array_shift = array
	try:
		for i in range(len(array)):
			array_shift[i] = np.roll(array[i], -shift_value*y_dist[i])
	except (AttributeError, TypeError):
		for i in range(len(array)):
			array_shift[i] = np.roll(array[i], -shift_value*i)
	return(array_shift)

def maxrow(array):
	rowsum=0
	for i in range(len(array)):
		if array[i].sum() > rowsum:
			rowsum = array[i].sum()
			max_row_index = i
	return(max_row_index)

def plot(st, inv=None, cat=None, zoom=1, y_dist=1, yinfo=False, markphase=None):
	"""
	Alpha Version!
	
	Needs inventory and catalog for yinfo using

	param st: 	stream
	type st:	obspy.core.stream.Stream

	param inv:	inventory
	type inv:

	param cat:	event catalog
	type cat:

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

	if yinfo:
		if inv and cat:
			attach_coordinates_to_traces(st, inv, cat[0])
			depth = cat[0].origins[0]['depth']/1000.
			#Calculates y-axis info using epidistance information of the stream
			epidist = epidist_inv(inv, cat) 
			no_x,no_t = data.shape

			for j in range(no_x):
				station = st[j].meta.network + "." + st[j].meta.station
				y_dist = epidist[station]['epidist']
				#for i in range(no_t):
				plt.plot(t_axis,zoom*data[j]+ y_dist, color='black')

				if markphase:
					origin = cat[0].origins[0]['time']
					m = TauPyModel('ak135')
					dist = st[j].meta.distance
					time = m.get_travel_times(depth, dist)
					for k in range(len(time)):
						if time[k].name != markphase:
							continue
   						t = time[k].time
					phase_time = origin + t - st[j].stats.starttime
					Phase_npt = int(phase_time/st[j].stats.delta)
					Phase = Phase_npt * st[j].stats.delta
					plt.plot( (Phase,Phase),(-1+y_dist,1+y_dist) )			

		else:
			print("no inventory and catalog given")
			raise ValueError

	else:
		for i in range(len(data)):
			if type(y_dist) == int:
				plt.plot(t_axis,zoom*data[i]+ y_dist*i, color='black')
			else:
				print("No y_dist given.")

	plt.show()
  
def plot_fft(x, logscale=False, fftshift=False, scaling=1):
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


def plot_data_im(x, color='Greys', scaling=30):
	plt.imshow(x, origin='lower', cmap=color, interpolation='nearest', aspect=scaling)
	plt.show()

	
def multplot(data1, data2, Log, scale1, scale2):
	
	plt.subplot(2,1,1)
	plot_fft_subplot(data1, logscale=Log, scaling=scale1)
	plt.subplot(2,1,2)
	plot_fft_subplot(data2, logscale=Log, scaling=scale2)
	plt.show()

def stream2array(stream, normalize=False):

	st = stream

	if normalize:
		ArrayData = np.array([st[0].data])/float(max(np.array([st[0].data])[0]))
	else:
		ArrayData = np.array([st[0].data])
	
	for i in range(len(st))[1:]:
		if normalize:
			next_st = np.array([st[i].data]) / float(max(np.array([st[i].data])[0]))
		else:
			next_st = np.array([st[i].data])

		ArrayData = np.append(ArrayData, next_st, axis=0)
	return(ArrayData)

def array2stream(ArrayData):
	
	traces = []
	
	for i in range(len(ArrayData)):
		newtrace = obspy.core.trace.Trace(ArrayData[i])
		traces.append(newtrace)
		
	stream = obspy.core.stream.Stream(traces)
	return(stream)

def read_file(stream, inventory, catalog, array=False):
	"""
	function to read data files, such as MSEED, station-xml and quakeml, in a way of obspy.read
	if need, pushes stream in an array for further processing
	"""
	st=obspy.read(stream)
	inv=obspy.read_inventory(inventory)
	cat=obspy.readEvents(catalog)

	#pushing the trace data in an array
	if array:
		ArrayData=stream2array(st)
		return(st, inv, cat, ArrayData)
	else:
		return(st, inv, cat)

  
def epidist_inv(inventory, catalog):
	"""
	Calculates the epicentral distance between event and stations of the inventory in degrees.

	param inventory: 
	type inventory:

	param catalog:
	type catalog:

	Returns

	param Array_Coords:
	type Array_Coords: dict
	"""
	#calc min and max epidist between source and receivers
	inv = inventory
	cat = catalog
	event=cat[0]
	Array_Coords = get_coords(inv)
	eventlat = event.origins[0].latitude
	eventlon = event.origins[0].longitude

	for network in inv:
		for station in network:
			scode = network.code + "." + station.code
			lat1 = Array_Coords[scode]["latitude"]
			lat2 = Array_Coords[scode]["longitude"]
			# calculate epidist in km
			# adds an epidist entry to the Array_coords dictionary 
			Array_Coords[scode]["epidist"] = kilometer2degrees(gps2dist_azimuth( lat1, lat2, eventlat, eventlon )[0]/1000)

	return(Array_Coords)

def epidist_stream(stream, inventory, catalog):
	"""
	Receives the epicentral distance of the station-source couple given in the stream. It uses just the
	coordinates of the used stations in stream.

	param stream:
	type stream:

	param inventory: 
	type inventory:

	param catalog:
	type catalog:

	"""
	Array_Coords = epidist_inv(inventory, catalog)

	epidist = {}
	for trace in stream:
		scode = trace.meta.network + "." + trace.meta.station
		epidist["%s" % (scode)] =  {"epidist" : Array_Coords[scode]["epidist"]}

	return(epidist)

def epidist2list(epidist):
	"""
	converts dictionary entries of epidist into a list
	"""
	epidist_list = []
	for scode in epidist:
		epidist_list.append(epidist[scode]["epidist"])

	return(epidist_list)

def epidist2nparray(epidist):
	epidist_np = []
	for scode in epidist:
		epidist_np = np.append(epidist_np, [epidist[scode]["epidist"]])
		
	return(epidist_np)

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

def transpose(array):
	arrayt = np.reshape(array, array.size, order='F').reshape(len(array[0]),len(array))
	arrayt = arrayt.astype('float')
	return(arrayt)

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

def part_stack(st, yinfo):
	ps_st=st
	return ps_st



