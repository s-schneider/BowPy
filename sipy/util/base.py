from __future__ import absolute_import
import numpy as np
import scipy as sp
import math
import obspy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Stream, Inventory, Trace, read
from obspy.core.inventory.network import Network
from obspy.core.util.attribdict import AttribDict
import warnings


"""
Basic collection of fundamental functions for the SiPy lib
Author: S. Schneider 2016
"""


def array2stream(ArrayData, st_original=None, network=None):
	"""
	param network: Network, of with all the station information
	type network: obspy.core.inventory.network.Network
	"""		
	traces = []
	
	for i, trace in enumerate(ArrayData):
		newtrace = obspy.core.trace.Trace(trace)
		traces.append(newtrace)
		
	stream = Stream(traces)
	
	# Just writes the network information, if possible input original stream
	
	if isinstance(st_original, Stream):
		st_tmp = st_original.copy()
		# Checks length of ArrayData and st_original, if needed,
		# corrects trace.stats.npts value of new generated Stream-object.
		if ArrayData.shape[1] == len(st_tmp[0]):

			for i, trace in enumerate(stream):
				trace.stats = st_tmp[i].stats

		else:

			for i, trace in enumerate(stream):
				trace.stats = st_tmp[i].stats
				trace.stats.npts = ArrayData.shape[1]
			

	elif isinstance(network, Network) and not isinstance(st_tmp, Stream):

		for trace in stream:
			trace.meta.network = network.code
			trace.meta.station = network[0].code


	return stream


def array2trace(ArrayData, st_original=None):
	if ArrayData.ndim != 1:
		try:
			stream = array2stream(ArrayData, st_original)
			return stream
		except:
			msg='Dimension do not fit'
			raise IOError(msg)
	else:		
		trace = obspy.core.trace.Trace(ArrayData)

	if isinstance(st_original, Stream):
		trace.stats = st_original[0].stats
	elif isinstance(st_original, Trace):
		trace.stats = st_original.stats

	return trace


def cat4stream(stream, client_name, stime=None, etime=None, minmag=None, lat=None, lon=None, mindepth=None):

	client = Client(client_name)
	try:
		eventinfo 	= stream[0].stats.sh
		depth 		= eventinfo['DEPTH']+10
		lat 		= eventinfo['LAT']
		lon 		= eventinfo['LON']
		origin 		= eventinfo['ORIGIN']

		etime = origin + 300
		stime = origin - 300
		cat = client.get_events(starttime=stime, endtime=etime, maxdepth=depth, latitude=lat, longitude=lon, maxradius=0.5, mindepth=mindepth)

		return cat
	except:
		try:
			cat = client.get_events(starttime=stime, endtime=etime, latitude=lat, longitude=lon, minmagnitude=minmag, mindepth=mindepth)
			return cat
		except:
			print('No Catalog found')
			return


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


def create_ricker(n_of_samples, n_of_traces, delta_traces = 1,  slope=0, n_of_ricker_samples = 100., width_of_ricker=2., shift_of_ricker=0):
	"""
	Creates n_of_traces Traces with a Ricker wavelet
	:param n_of_samples: No of samplesw
	:type  n_of_samples: int
	
	:param n_of_traces: No of traces
	:type  n_of_traces: int

	:param slope: Indexshift of the traces, shift is applied by the relation delta_t = delta_traces * slope
	:type  slope: int

	:param width_of_ricker: width_of_ricker parameter of Ricker-wavelet, default 2
	:type  width_of_ricker: float

	:param n_of_ricker_samples: Number of samples for ricker
	:type  n_of_ricker_samples: int
	"""

	if n_of_samples < n_of_ricker_samples:
		msg='Number of tracesamples lower than number of ricker samples'
		raise IOError(msg)

	data = np.zeros((n_of_traces, n_of_samples))	

	trace = np.zeros(n_of_samples)
	ricker_tmp = sp.signal.ricker(n_of_ricker_samples, width_of_ricker)
	ricker = ricker_tmp/ricker_tmp.max()

	trace[shift_of_ricker:shift_of_ricker+n_of_ricker_samples] = ricker

	if slope != 0:
		for i in range(data.shape[0]):
			delta = np.floor( i * float(abs(slope) / float(delta_traces))).astype('int')
			data[i] = np.roll(trace, delta)[:n_of_samples]	
		if slope < 0:
			data = np.flipud(data)	
	elif slope == 0:	
		for i, dt in enumerate(data):
			data[i] = trace	

	return data


def create_sine( no_of_traces=10, len_of_traces=30000, samplingrate = 30000,
                 no_of_periods=1):
    
	deltax = 2*np.pi/len_of_traces
	signal_len = len_of_traces * no_of_periods
	data_temp = np.array([np.zeros(signal_len)])
	t = []

	# first trace
	for i in range(signal_len):
		data_temp[0][i] = np.sin(i*deltax)
		t.append((float(i) + float(i)/signal_len)*2*np.pi/signal_len)
		data = data_temp

	# other traces
	for i in range(no_of_traces)[1:]:
		data = np.append(data, data_temp, axis=0)
       
       
	return(data, t)


def cut2shortest(stream):
	"""
	Cuts traces in stream to the same length. Looks for the latest beginning and the earliest ending of traces in stream,
	which will be the new reference times.
	"""
	start = stream[0].stats.starttime
	end   = stream[0].stats.endtime
	for trace in stream:
		if trace.stats.starttime > start:
			start = trace.stats.starttime
		if trace.stats.endtime < end:
			end = trace.stats.endtime

	stream.trim(start, end)
	return stream


def inv4stream(stream, network, client_name):

	start 	= stream[0].stats.starttime
	end 	= stream[0].stats.endtime
	client 	= Client(client_name)
	inv 	= client.get_stations(network=network, starttime=start, endtime=end)

	return inv

def list2stream(list):

	stream = Stream()
	for station in list:
		for trace in station:
			stream.append(trace)

	return stream


def maxrow(array):
	rowsum=0
	for i in range(len(array)):
		if array[i].sum() > rowsum:
			rowsum = array[i].sum()
			max_row_index = i
	return(max_row_index)

def merge_or_keep_longest(stream):

	try:
		stream.merge()
	except:
		channels = AttribDict()

		for i, tr in enumerate(stream):
			if tr.stats.channel in channels:
				if tr.stats.npts < channels[tr.stats.channel][0]:
					stream.remove(tr)
				else:
					stream.remove(stream[channels[tr.stats.channel][1]])

			else:
				#Append the name of channel, samplingpoints and number of trace so channels
				channels[tr.stats.channel] = [tr.stats.npts, i]

	return stream


def nextpow2(i):
	#See Matlab documentary
	n = 1
	count = 0
	while n < abs(i):
		n *= 2
		count+=1
	return count


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


def split2stations(stream, merge=True, keep_masked=False):
	"""
	Splits a stream in a list of streams, sorted by the stations inside stream object. Merges traces with the same ID to one trace.
	"""
	stream.sort(['station'])

	stream_list = []
	st_tmp = Stream()

	statname = stream[0].stats.station
	for trace in stream:
		#Collect traces from same station
		if trace.stats.station == statname:
			st_tmp.append(trace)

		else:

			if merge: merge_or_keep_longest(st_tmp)
			stream_list.append(st_tmp)
			statname = trace.stats.station
			st_tmp = Stream()
			st_tmp.append(trace)

	if merge: merge_or_keep_longest(st_tmp)
	stream_list.append(st_tmp)

	if not keep_masked:
		for station in stream_list:
			station.sort(['channel'])
			for trace in station:
				if type(trace.data) == numpy.ma.core.MaskedArray:
					stream_list.remove(station)
					break


	return(stream_list)


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


def stats(stream):
	"""
	Prints stats of the stream
	"""

	for trace in stream:
		print(trace.stats)

	return


def stream2array(stream, normalize=False):
	sx = stream.copy()
	x = np.zeros((len(sx), len(sx[0].data)))
	for i, traces in enumerate(sx):
		x[i] = traces.data

	if normalize:
		if x.max()==0:
			print('Maximum value is 0')
			return(x)

		elif math.isnan(x.max()):
			print('Maximum values are NaN, set to 0')
			n   = np.isnan(x)
			x[n]= 0.
		
		x = x / x.max()
	return(x)


def LCM(a,b):
	"""
	Calculates the least common multiple of two values
	"""
	import fractions
	return abs(a * b) / fractions.gcd(a,b) if a and b else 0

