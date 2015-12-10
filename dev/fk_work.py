"""
perform 2d fft
ordne k richtig zu, Formel finden!
Synthetics mit sinus wavelet und multiplen dessen
damit erstmal testen
"""

import obspy
import numpy
import numpy as np
import matplotlib.pyplot as plt
import Muenster_Array_Seismology as MAS
from Muenster_Array_Seismology import get_coords
from obspy.core.util.geodetics import gps2DistAzimuth, kilometer2degrees



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
       data =np.append(data, data_temp, axis=0)
       
       
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
  
def plot_fft(x, logscale=False, fftshift=False, scaling=1):
	"""
	Doing an fk-trafo and plotting it.

	param x:	Data of the array
	type x:		np.array

	param logscale:	Sets scaling of the plot to logarithmic
	type logscale:	boolean

	param fftshift: Ir True shifts zero values of f and k into the center of the plot
	type fftshift:	boolean

	param scaling:	Sets the scaling of the plot in terms of aspectratio y/x
	type scaling:	int
	"""

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

def plot_data_im(x, color='Greys', scaling=30):
	plt.imshow(x, origin='lower', cmap=color, interpolation='nearest', aspect=scaling)
	plt.show()

def plot_data(st, inv=None, cat=None, zoom=1, y_dist=1, yinfo=False):
	"""
	Alpha Version!
	Time axis has no time-ticks
	
	Needs inventory and catalog for yinfo using

	param st: 	array of data or stream
	type st:	np.array obspy.core.stream.Stream

	param zoom: zoom factor of the traces
	type zoom:	float

	param y_dist:	separating distance between traces, for example equidistant with "1" 
					or import epidist-list via epidist
	type y_dist:	int or list
	"""
	if type(st) == obspy.core.stream.Stream:
		data = stream2array(st)
	else:
		data = st

	if yinfo:
		#Calculates y-axis info using epidistance information of the stream
		epidist = epidist_stream(st, inv, cat) 
		y_dist = epidist2list(epidist)

	for i in range(len(data)):
		if type(y_dist) == int:
			plt.plot(zoom*data[i]+ y_dist*i)
		if type(y_dist) == list or type(y_dist) == numpy.ndarray:
			plt.plot(zoom*data[i]+ y_dist[i])
	plt.show()

def fk_filter_extract_phase(data, snes=False, y_dist=False):
	"""
	Function to test the fk workflow with synthetic data
	param data:	data of the array
	type data:	numpy.ndarray

	param snes:	slownessvalue of the desired extracted phase
	type snes:	int
	"""
	if snes:
		ds = shift_array(data, snes, y_dist)
	else:
		ds = data
	
	dsfft = np.fft.fftn(ds)
	#max_k = maxrow(dsfft)
	#print("maximum wavenumber k is %f" % max_k)
	#dsfft = line_cut(dsfft, max_k)
	dsfft = line_cut(dsfft, 0)
	ds = np.fft.ifftn(dsfft)
	
	data_fk = shift_array(ds, -snes, y_dist)
	
	return(data_fk.real)

def fk_filter_eliminate_phase(data, snes=False, y_dist=False):
	"""
	Function to test the fk workflow with synthetic data
	param data:	data of the array
	type data:	numpy.ndarray

	param snes:	slownessvalue of the desired extracted phase
	type snes:	int
	"""

	if snes:
		ds = shift_array(data, snes, y_dist)
	else:
		ds = data
		
	dsfft = np.fft.fftn(ds)
	#max_k = maxrow(dsfft)
	#print("maximum wavenumber k is %f" % max_k)
	#dsfft = line_set_zero(dsfft, max_k)
	dsfft = line_set_zero(dsfft, 0)
	ds = np.fft.ifftn(dsfft)
	
	data_fk = shift_array(ds, -snes, y_dist)
	
	return(data_fk.real)


def fk_filter(st, inv=None, cat=None, phase=None, epi_dist=None, normalize=True):
	"""
	Import stream, inventory, catalog and phase you want to investigate.
	The function bins the data or interpolates them or adds zero-signal-stations for equidistant spacing, applies an 2D FFT, removes a certain window around the
	desired phase to surpress mutliples and applies an 2d iFFT
	Alternative is an nonequidistant 2D FFT.

	param st: Stream
	type st: obspy.core.stream.Stream

	param inv: inventory
	type inv: obspy.station.inventory.Inventory

	param cat: catalog
	type cat: obspy.core.event.Catalog

	param phase: name of the phase to be investigated
	type phase: string
	
	param epidist: list of epidistances, corresponding to st
	type epidist: list
	"""

	"""
	Correction of global(array) slowness of phase ##################################
	"""
	#stream_aligned = MAS.align_phases(st, cat[0], inv, phase)
	stream_aligned = st

	
	"""
	Calculate epicentral distances of station-receiver couples######################
	"""
	if not epi_dist:
		if st and inv and cat:
			epidist = epidist_stream(stream_aligned, inv, cat)
	else:
		print()

	"""
	2D FFT #########################################################################
	"""
	#Convert to numpy.ndarray, stream info still in st
	ArrayData = stream2array(stream_aligned, normalize)
	
	#Apply FFT
	fft_Data = np.fft.fftn(ArrayData)
	
	#Find row with maximum sum-value - corresponding to zero k phase
	max_k=maxrow(fft_Data)
	
	#mute area around |f| > eps, choose eps dependent on your phase/data/i dont know yet
	fft_corr= line_cut(fft_Data, max_k)
	
	#apply 2D iFFT
	ArrayData_filtered = np.fft.ifftn(fft_corr)

	#Create Stream object
	stream_filtered = array2stream(ArrayData_filtered)
	for i in range(len(stream_filtered)):
		stream_filtered[i].meta = stream_aligned[i].meta
	
	"""
	Undo global-correction of the phase ############################################
	""" 
	#stream_filtered = MAS.align_phases()
	"""
	return stream with filtered data ####################################
	"""
	return(stream_filtered)

def stream2array(stream, normalize=True):

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
	Calculates the epicentral distance between event and stations of the inventory

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
			Array_Coords[scode]["epidist"] = kilometer2degrees(gps2DistAzimuth( lat1, lat2, eventlat, eventlon )[0]/1000)

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

def line_cut(array, stat, radius=None):
	"""
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


stream="WORK_D.MSEED"
inventory="2011-03-11T05:46:23.MSEED_inv.xml"
catalog="2011-03-11T05:46:23.MSEED_cat.xml"
phase="PP"

#fk_filter(stream, inventory, catalog, phase)
#data=create_signal(no_of_traces=1,len_of_traces=12,multiple=False)
#data=create_sine(no_of_traces=1, no_of_periods=2)

