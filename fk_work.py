"""
perform 2d fft
ordne k richtig zu, Formel finden!
Synthetics mit sinus wavelet und multiplen dessen
damit erstmal testen
"""



import obspy
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
                       zero_traces=False, no_of_zeros=0):
	"""
	function that creates a delta peak signal
	slowness = 0 corresponds to shift of 1 to each trace
	"""
	
	slowness = slowness-1
	dist = multipdist
	data = np.array([np.zeros(len_of_traces)])
	if multiple:
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

	return(data)

def create_standard_test(snes1=1, snes2=3):
        y = create_deltasignal(no_of_traces=200, len_of_traces=200, multiple=True, multipdist=5, no_of_multip=1, slowness=snes1)
        x = create_deltasignal(no_of_traces=200, len_of_traces=200, multiple=True, multipdist=5, no_of_multip=5, slowness=snes2)
        a = x + y
	return(a)

def shift_array(array, shift_value=0):
	for i in range(len(array)):
		array[i] = np.roll(array[i],-shift_value*i)
	return(array)

def maxrow(array):
	rowsum=0
	for i in range(len(array)):
		if array[i].sum() > rowsum:
			rowsum = array[i].sum()
			max_row_index = i
	return(max_row_index)
  
def plot_fk(x, logscale=False, fftshift=False, scaling=1):
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

	fftx = np.fft.fftn(x)
	if fftshift:
		plt.imshow((np.abs(np.fft.fftshift(fftx))), origin='lower',cmap=None, aspect=scaling)
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

def plot_data(st, inv, cat, zoom=1, y_dist=1, yinfo=False):
	"""
	Alpha Version!
	Time axis has no time-ticks

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

	if yinfo:
		#Calculates y-axis info using epidistance information of the stream
		epidist = epidist_stream(st, inv, cat) 
		y_dist = epidist2list(epidist)

	for i in range(len(data)):
		if type(y_dist) == int:
			plt.plot(zoom*data[i]+ y_dist*i)
		if type(y_dist) == list:
			plt.plot(zoom*data[i]+ y_dist[i])
	plt.show()

def fk_filter_synth(data, snes):
	"""
	Function to test the fk workflow with synthetic data
	param data:	data of the array
	type data:	numpy.ndarray

	param snes:	slownessvalue of the desired extracted phase
	type snes:	int
	"""

	ds = shift_array(data, snes)
	
	dsfft = np.fft.fftn(ds)
	max_k = maxrow(dsfft)
	dsfft = set_zero(dsfft, max_k)
	ds = np.fft.ifftn(dsfft)
	
	data = shift_array(ds, -snes)
	
	return(data)


def fk_filter(st, inv, cat, phase, normalize=True):
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
	"""
	
	"""
	Example workflow:
	example work flow for filtering
	snes muss noch korrekt umgerechnet werden

	import fk_work as fk
	import matplotlib.pyplot as plt
	import numpy as np


	snes = 1
	snes2 = 3
	y = fk.create_deltasignal(no_of_traces=200, len_of_traces=200, multiple=True, multipdist=5, no_of_multip=1, slowness=snes)

	x = fk.create_deltasignal(no_of_traces=200, len_of_traces=200, multiple=True, multipdist=5, no_of_multip=5, slowness=snes2)

	a = x + y

	work = fk.shift_array(a, snes)

	work_fft = np.fft.fftn(work)

	max_k=fk.maxrow(work_fft)

	newfft = fk.set_zero(work_fft, stat=max_k)

	new = np.fft.ifftn(newfft)

	data = fk.shift_array(new, -snes)
	
	"""
	

	"""
	Sorting of the data ############################################################
	"""
	ArrayData = stream2array(st, normalize)
	epidist, Array_Coords = epidist(inv, cat)

	st_aligned = MAS.align_phases()


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

	fft_data = np.fft.fft2(a, s=None, axes=(-2, -1))
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

	#Amplitudespectra: |F(x,y)| = ( Re(x,y)^2 + Im(x,y)^2 )^1/2
      # write loop through all elements
      #Amp =  np.sqrt(fft_data[i][j].real**2 + fft_data[i][j].imag**2)

	#mute area around |f| > eps, choose eps dependent on your phase/data/i dont know yet

	#apply 2D iFFT

	"""
	Undo global-correction of the phase ############################################
	""" 

	"""
	return stream with filtered and binned data ####################################
	"""

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

def set_zero(array, stat):
	x = len(array[0])
	y = len(array)
	
	new_array = np.array([ np.array(np.zeros(x), dtype=np.complex) ])
	new_line = new_array
	for i in range(y)[1:]:
		new_array = np.append(new_array, new_line, axis=0)
	
	#new_array[stat-1] = array[stat-1]
	new_array[stat] = array[stat]
	#new_array[stat+1] = array[stat+1]
	
	return(new_array)

stream="2011-03-11T05:46:23.MSEED"
inventory="2011-03-11T05:46:23.MSEED_inv.xml"
catalog="2011-03-11T05:46:23.MSEED_cat.xml"
phase="PP"

#fk_filter(stream, inventory, catalog, phase)
#data=create_signal(no_of_traces=1,len_of_traces=12,multiple=False)
#data=create_sine(no_of_traces=1, no_of_periods=2)

