from __future__ import absolute_import
from collections import defaultdict
import tempfile
import numpy
import numpy as np
import obspy
import os
import shutil
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
from matplotlib.ticker import MaxNLocator
from obspy import UTCDateTime, Stream
from obspy.core import AttribDict
from obspy.geodetics.base import locations2degrees, gps2DistAzimuth, \
   kilometer2degrees
from obspy.taup import getTravelTimes
import scipy.interpolate as spi
import scipy as sp
import matplotlib.cm as cm
from obspy.signal.util import utlGeoKm,nextpow2
import ctypes as C
from obspy.core import Stream
import math
import warnings
from scipy.integrate import cumtrapz
from obspy.core import Stream
from obspy.signal.headers import clibsignal
from obspy.signal.invsim import cosTaper
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel


"""
Collection of useful functions for processing seismological array data

Author: S. Schneider 2016
"""

def __coordinate_values(inventory):
    geo = get_coords(inventory, returntype="dict")
    lats, lngs, hgt = [], [], []
    for coordinates in list(geo.values()):
        lats.append(coordinates["latitude"]),
        lngs.append(coordinates["longitude"]),
        hgt.append(coordinates["elevation"])
    return lats, lngs, hgt

def get_coords(inventory, returntype="dict"):
    """
    Get the coordinates of the stations in the inventory, independently of the channels,
    better use for arrays, than the channel-dependent core.inventory.inventory.Inventory.get_coordinates() .
    returns the variable coords with entries: elevation (in km), latitude and longitude.
    :param inventory: Inventory to get the coordinates from
    :type inventory: obspy.core.inventory.inventory.Inventory

    :param coords: dictionary with stations of the inventory and its elevation (in km), latitude and longitude
    :type coords: dict

    :param return: type of desired return
    :type return: dictionary or numpy.array

    """
    if returntype == "dict":
        coords = {}
        for network in inventory:
            for station in network:
                coords["%s.%s" % (network.code, station.code)] = \
                    {"latitude": station.latitude,
                     "longitude": station.longitude,
                     "elevation": float(station.elevation) / 1000.0}

    if returntype == "array":
        nstats = len(inventory[0].stations)
        coords = np.empty((nstats, 3))
        if len(inventory.networks) == 1:
            i=0
            for network in inventory:
                for station in network:
                    coords[i,0] = station.latitude
                    coords[i,1] = station.longitude
                    coords[i,2] = float(station.elevation) / 1000.0
                    i += 1

    return coords

def trace2array(trace):
	x = np.zeros(trace.size)
	for index, data in trace.data:
		x[index] = data

	return x 


def stream2array(stream, normalize=False):

	x = np.zeros((len(stream), len(stream[0].data)))
	for i, traces in enumerate(stream):
		for j, data in enumerate(traces):
			x[i,j]=data

	if normalize:
		x = x / x.max()

	return(x)

def array2stream(ArrayData, st_original=None, network=None):
	"""
	param network: Network, of with all the station information
	type network: obspy.core.inventory.network.Network
	"""
	traces = []
	
	for i in range(len(ArrayData)):
		newtrace = obspy.core.trace.Trace(ArrayData[i])
		traces.append(newtrace)
		
	stream = obspy.core.stream.Stream(traces)
	
	# Just writes the network information, if possible input original stream
	if st_original:
		i=0
		for trace in stream:
			trace.meta = st_original[i].meta
			i += 1	
		

	elif network and st_original == None:
		i=0
		for trace in stream:
			trace.meta.network = network.code
			trace.meta.station = network[0].code
			i += 1


	return(stream)


def attach_network_to_traces(stream, network):
	"""
	Attaches the network-code of the inventory to each trace of the stream
	"""
	for trace in stream:
		trace.meta.network = network.code

def attach_coordinates_to_traces(stream, inventory, event=None):
    """
    Function to add coordinates to traces.

    It extracts coordinates from a :class:`obspy.station.inventory.Inventory`
    object and writes them to each trace's stats attribute. If an event is
    given, the distance in degree will also be attached.

    :param stream: Waveforms for the array processing.
    :type stream: :class:`obspy.core.stream.Stream`
    :param inventory: Station metadata for waveforms
    :type inventory: :class:`obspy.station.inventory.Inventory`
    :param event: If the event is given, the event distance in degree will also
     be attached to the traces.
    :type event: :class:`obspy.core.event.Event`
    """
    # Get the coordinates for all stations
    coords = {}
    for network in inventory:
        for station in network:
            coords["%s.%s" % (network.code, station.code)] = \
                {"latitude": station.latitude,
                 "longitude": station.longitude,
                 "elevation": station.elevation}

    # Calculate the event-station distances.
    if event:
		event_lat = event.origins[0].latitude
		event_lng = event.origins[0].longitude
		event_dpt = event.origins[0].depth/1000.
		for value in coords.values():
			value["distance"] = locations2degrees(
				value["latitude"], value["longitude"], event_lat, event_lng)
			value["depth"] = event_dpt

    # Attach the information to the traces.
    for trace in stream:
        station = ".".join(trace.id.split(".")[:2])
        value = coords[station]
        trace.stats.coordinates = AttribDict()
        trace.stats.coordinates.latitude = value["latitude"]
        trace.stats.coordinates.longitude = value["longitude"]
        trace.stats.coordinates.elevation = value["elevation"]
        if event:
			trace.stats.distance = value["distance"]
			trace.stats.depth = value["depth"]


def center_of_gravity(inventory):
    lats, lngs, hgts = __coordinate_values(inventory)
    return {
        "latitude": np.mean(lats),
        "longitude": np.mean(lngs),
        "elevation": np.mean(hgts)}

def geometrical_center(inventory):
    lats, lngs, hgt = __coordinate_values(inventory)

    return {
        "latitude": (np.max(lats) +
                     np.min(lats)) / 2.0,
        "longitude": (np.max(lngs) +
                      np.min(lngs)) / 2.0,
        "absolute_height_in_km":
        (np.max(hgt) +
         np.min(hgt)) / 2.0
    }

def aperture(inventory):
    """
    The aperture of the array in kilometers.
    Method:find the maximum of the calculation of  distance of every possible combination of stations
    """
    lats, lngs, hgt = __coordinate_values(inventory)
    distances = []
    for i in range(len(lats)):
        for j in range(len(lats)):
            if lats[i] == lats[j]:
                continue
            distances.append(gps2DistAzimuth(lats[i],lngs[i],
                lats[j],lngs[j])[0] / 1000.0)
    return max(distances)

def find_closest_station(inventory, latitude, longitude,
                         absolute_height_in_km=0.0):
    """
    Calculates closest station to a given latitude, longitude and absolute_height_in_km
    param latitude: latitude of interest, in degrees
    type latitude: float
    param longitude: longitude of interest, in degrees
    type: float
    param absolute_height_in_km: altitude of interest in km
    type: float
    """
    min_distance = None
    min_distance_station = None

    lats, lngs, hgt = __coordinate_values(inventory)
    
    x = latitude
    y = longitude
    z = absolute_height_in_km

    for i in range(len(lats)):
        distance = np.sqrt( ((gps2DistAzimuth(lats[i], lngs[i], x, y)[0]) / 1000.0) ** 2  + ( np.abs( np.abs(z) - np.abs(hgt[i]))) ** 2 )
        if min_distance is None or distance < min_distance:
            min_distance = distance
            min_distance_station = inventory[0][i].code
    return min_distance_station


def epidist_inv(inventory, event):
	"""
	Calculates the epicentral distance between event and stations of the inventory in degrees.

	param inventory: 
	type inventory:

	param event: event of catalog
	type event:

	Returns

	param Array_Coords:
	type Array_Coords: dict
	"""
	#calc min and max epidist between source and receivers
	inv = inventory
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
			Array_Coords[scode]["epidist"] = locations2degrees( lat1, lat2, eventlat, eventlon )

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
	
	epidist_list.sort()

	return(epidist_list)

def epidist2nparray(epidist):
	epidist_np = []
	for scode in epidist:
		epidist_np = np.append(epidist_np, [epidist[scode]["epidist"]])
	
	epidist_np.sort()	
	return(epidist_np)

def alignon(st, inv, event, phase, ref=0 , maxtimewindow=None, taup_model='ak135'):
	"""
	Aligns traces on a given phase
	
	:param st: stream
	
	:param inv: inventory

	:param event: Eventdata

	:param phase: Phase to align the traces on
	:type phase: str

	:param ref: name or index of reference station, to which the others are aligned
	:type ref: int or str

	:param maxtimewindow: Timewindow around the theoretical arrivaltime
	:type maxtimewindow: int or float
	
	:param taup_model: model used by TauPyModel to calculate arrivals, default is ak135
	:type taup_model: str

	returns:
	:param st_align: Aligned stream on Phase
	:type st_align:

	"""
	
	# Calculate depth and distance of receiver and event.
	# Set some variables.

	attach_coordinates_to_traces(st, inv, event)
	depth = event.origins[0]['depth']/1000.
	origin = event.origins[0]['time']
	m = TauPyModel(taup_model)

	# Prepare Array of data.
	st_tmp = st.copy()
	data = stream2array(st_tmp)
	shifttimes=np.zeros(data.shape[0])

	if type(ref) == int:
		ref_dist = st[ref].stats.distance
		ref_start = st[ref].stats.starttime
		delta = st[ref].stats.delta
		iref = ref

	elif type(ref) == str:
		for i, trace in enumerate(st):
			if trace.stats['station'] != 'ref':
				continue
			ref_dist = trace.stats.distance
			iref = i
		ref_start = trace.stats.starttime
		delta = trace.stats.delta

	# Calculating reference arriving time/index of phase.
	ref_t = origin + m.get_travel_times(depth, ref_dist, phase_list=[phase])[0].time - ref_start
	ref_n = int(ref_t/delta)
	
	for no_x, data_x in enumerate(data):
		if no_x == iref:
			continue
	
		dist = st[no_x].stats.distance
		t = m.get_travel_times(depth, dist, phase_list=[phase])[0].time

		# Calculate arrivals, and shift times/indicies.
		phase_time = origin + t - st[no_x].stats.starttime
		phase_n = int(phase_time/delta)
		datashift, shift_index = shift2ref(data[no_x,:], ref_n, phase_n, mtw=maxtimewindow)
		shifttimes[no_x]=delta*shift_index
		data[no_x,:] = datashift

	st_align = array2stream(data, st)

	for i, trace in enumerate(st_align):
		if i == iref:
			continue
		trace.stats.starttime = trace.stats.starttime - shifttimes[i]	

	return st_align

def shift2ref(data, tref, tshift, mtw=None):
	
	if mtw:
		tmin = tref - int(mmtw/2.)
		tmax = tref + int(mtw/2.)
		stmax = data[tref]
		mtw_index = tref
		for k in range(tmin,tmax+1):
			if data[k] > stmax:
					stmax=data[k]
					mtw_index = k
		shift_value = tref - mtw_index
		shift_data = np.roll(data, shift_value)

	else:
		shift_value = tref - tshift
		shift_data = np.roll(data, shift_value)

	return shift_data, shift_value


def partial_stack(st, yinfo, no_of_bins, phase, order=None, maxtimewindow=None, taup_model='ak135'):
	"""
	Will sort the traces into equally distributed bins and stack the bins.
	The stacking is just an addition of the traces, more advanced schemes might follow.
	The uniform distribution is useful for FK-filtering, SSA and every method that requires
	a uniform distribution.
	
	Needs depth information attached to the stream, array_util.see attach_coordinates_to_stream()
	
	input:
	:param st: obspy stream object
	:type st:

	:param yinfo: list of distances of the traces, sorted to match st
	:type yinfo: list

	:param no_of_bins: number of bins, that should be used 
	:type no_of_bins: int

	:param order: Order of Nth-root stacking, default None
	:type order: float

	returns: 
	:param ps_st: partial stacked data of the array in no_of_bins uniform distributed stacks
	:type ps_st:

	:param bin_distribution: distribution  traces in bins
	:type: numpy.ndarray

	:param L: Location of bin borders
	:type L: numpy.ndarray

	:param y_resample: resampled yinfo of the stacked traces
	:type y_resample: numpy.ndarray
	"""
	
	# Checking for correct input.
	if type(yinfo) != list or type(order) == int: 
		msg="wrong input type of variables!"
		raise TypeError(msg)

	data = stream2array(st, normalize=True)

	# Calculate the border of each bin 
	# and the new yinfo values.
	L = np.linspace(min(yinfo), max(yinfo), no_of_bins)
	delta_L = abs(L[0] - L[1])

	

	# Resample the y-axis information to new, equally distributed ones.
	y_resample = np.linspace( L[0] + delta_L/2., L[len(L)-1]-delta_L/2., no_of_bins-1)
	bin_distribution = np.zeros(len(y_resample))
	y_len, t_len = data.shape

	# Preallocate some space in memory.
	ps_st = np.zeros((len(y_resample),t_len))
	yr_sampletimes = np.zeros(len(y_resample)).astype('int')
	yi_sampletimes = np.zeros(len(yinfo)).astype('int')
	
	m = TauPyModel(taup_model)
	depth = st[0].meta.depth
	delta = st[0].meta.delta

	# Calculate theoretical arrivals
	for i, e in enumerate(yr_sampletimes):
		yr_sampletimes[i] = int(m.get_travel_times(depth, y_resample[i], phase_list=[phase])[0].time / delta)
	
	for i, e in enumerate(yi_sampletimes):
		yi_sampletimes[i] = int(m.get_travel_times(depth, yinfo[i], phase_list=[phase])[0].time / delta)




	# Loop through all traces.
	for i in range(len(L))[1:]:
		count=0.
		for j in range(y_len):
			if j==0 and i==1:

				tref = yr_sampletimes[i-1]
				Phase_npt = yi_sampletimes[j]

				if maxtimewindow:
					tmin = Phase_npt - int( (maxtimewindow/2.)/delta )
					tmax = Phase_npt + int( (maxtimewindow/2.)/delta )
					stmax = data[j][Phase_npt]
					mtw_index = Phase_npt
					for k in range(tmin,tmax+1):
						if data[j][k] > stmax:
								stmax=data[j][k]
								mtw_index = k
					shift = tref - mtw_index
					print( "Bin %i,   Trace %i, Arrivaltime %i, t-Resample %i, Shiftingtime %i, EpiDist %f, y_resampled %f " % (i, j, Phase_npt,tref,shift, yinfo[j], y_resample[i-1]))
					data[j,:] =  np.roll(data[j,:], shift)

				else:

					shift = tref - Phase_npt
					data[j,:] =  np.roll(data[j,:], shift)					
					
				if order:
					for k in range(t_len):
						sgnps = np.sign(ps_st[i-1,k])
						sgnst = np.sign(data[j,k])
						ps_st[i-1,k] = sgnps * (abs(ps_st[i-1,k])**(1./order)) + \
										sgnst *(abs(data[j,k])**(1./order))
					count += 1.
				else:
					ps_st[i-1,:] = ps_st[i-1,:] + data[j,:]
					count += 1.
				
				continue

			if yinfo[j] > L[i-1] and yinfo[j] <= L[i]:

				tref = yr_sampletimes[i-1]
				Phase_npt = yi_sampletimes[j]

				if maxtimewindow:
					tmin = Phase_npt - int( (maxtimewindow/2.)/delta )
					tmax = Phase_npt + int( (maxtimewindow/2.)/delta )
					stmax = data[j][Phase_npt]
					mtw_index = Phase_npt
					for k in range(tmin,tmax+1):
						if data[j][k] > stmax:
								stmax=data[j][k]
								mtw_index = k
					shift = tref - mtw_index
					print( "Bin %i,   Trace %i, Arrivaltime %i, t-Resample %i, Shiftingtime %i, EpiDist %f, y_resampled %f " % (i, j, Phase_npt,tref,shift, yinfo[j], y_resample[i-1]))
					data[j,:] =  np.roll(data[j,:], shift)

				else:
					shift = tref - Phase_npt
					data[j,:] =  np.roll(data[j,:], shift)

				if order:
					for k in range(t_len):
						sgnps = np.sign(ps_st[i-1,k])
						sgnst = np.sign(data[j,k])
						ps_st[i-1,k] = sgnps * (abs(ps_st[i-1,k])**(1./order)) + \
										sgnst *(abs(data[j,k])**(1./order))
					count += 1.
				else:
					ps_st[i-1,:] = ps_st[i-1,:] + data[j,:]
					count += 1.

		if count != 0.:	
			if order:
				sgn = np.sign(ps_st[i-1,:])
				ps_st[i-1,:] = sgn * ( ( ps_st[i-1,:]/ count )**(order) )
				bin_distribution[i-1] += count
			else:
				ps_st[i-1,:] = ps_st[i-1,:] / count
				bin_distribution[i-1] += count
		
	return ps_st,bin_distribution, L, y_resample


def plot(inventory, projection="local"):
    """
    Function to plot the geometry of the array, 
    including its center of gravity and geometrical center

    :type inventory: obspy.core.inventory.inventory.Inventory
    :param inventory: Inventory to be plotted

    :type projection: strg, optional
    :param projection: The map projection. Currently supported are:

    * ``"global"`` (Will plot the whole world.)
    * ``"ortho"`` (Will center around the mean lat/long.)
    * ``"local"`` (Will plot around local events)   
    """
    if inventory:
        inventory.plot(projection, show=False)
        bmap = plt.gca().basemap

        grav = center_of_gravity(inventory)
        x, y = bmap(grav["longitude"], grav["latitude"])
        bmap.scatter(x, y, marker="x", c="red", s=40, zorder=20)
        plt.text(x, y, "Center of Gravity", color="red")

        geo = geometrical_center(inventory)
        x, y = bmap(geo["longitude"], geo["latitude"])
        bmap.scatter(x, y, marker="x", c="green", s=40, zorder=20)
        plt.text(x, y, "Geometrical Center", color="green")

        plt.show()





def plot_transfer_function(stream, inventory, sx=(-10, 10), sy=(-10, 10), sls=0.5, freqmin=0.1, freqmax=4.0,
                           numfreqs=10):
    """
    Plot transfer function (uses array transfer function as a function of
    slowness difference and frequency).

    :param sx: Min/Max slowness for analysis in x direction.
    :type sx: (float, float)
    :param sy: Min/Max slowness for analysis in y direction.
    :type sy: (float, float)
    :param sls: step width of slowness grid
    :type sls: float
    :param freqmin: Low corner of frequency range for array analysis
    :type freqmin: float
    :param freqmax: High corner of frequency range for array analysis
    :type freqmax: float
    :param numfreqs: number of frequency values used for computing array
     transfer function
    :type numfreqs: int
    """
    sllx, slmx = sx
    slly, slmy = sy
    sllx = kilometer2degrees(sllx)
    slmx = kilometer2degrees(slmx)
    slly = kilometer2degrees(slly)
    slmy = kilometer2degrees(slmy)
    sls = kilometer2degrees(sls)

    stepsfreq = (freqmax - freqmin) / float(numfreqs)
    transff = array_transff_freqslowness(stream, inventory, (sllx, slmx, slly, slmy),
                                               sls, freqmin, freqmax,
                                               stepsfreq)

    sllx = degrees2kilometers(sllx)
    slmx = degrees2kilometers(slmx)
    slly = degrees2kilometers(slly)
    slmy = degrees2kilometers(slmy)
    sls = degrees2kilometers(sls)

    slx = np.arange(sllx, slmx + sls, sls)
    sly = np.arange(slly, slmy + sls, sls)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # ax.pcolormesh(slx, sly, transff.T)
    ax.contour(sly, slx, transff.T, 10)
    ax.set_xlabel('slowness [s/deg]')
    ax.set_ylabel('slowness [s/deg]')
    ax.set_ylim(slx[0], slx[-1])
    ax.set_xlim(sly[0], sly[-1])
    plt.show()


def plot_gcp(slat, slon, qlat, qlon, plat, plon, savefigure=None):
    
    global m
    # lon_0 is central longitude of projection, lat_0 the central latitude.
    # resolution = 'c' means use crude resolution coastlines, 'l' means low, 'h' high etc.
    # zorder is the plotting level, 0 is the lowest, 1 = one level higher ...   
    #m = Basemap(projection='nsper',lon_0=20, lat_0=25,resolution='c')
    m = Basemap(projection='kav7',lon_0=-45, resolution='c')   
    qx, qy = m(qlon, qlat)
    sx, sy = m(slon, slat)
    px, py = m(plon, plat)
    m.drawmapboundary(fill_color='#B4FFFF')
    m.fillcontinents(color='#00CC00',lake_color='#B4FFFF', zorder=0)
    #import event coordinates, with symbol (* = Star)
    m.scatter(qx, qy, 80, marker='*', color= '#004BCB', zorder=2)
    #import station coordinates, with symbol (^ = triangle)
    m.scatter(sx, sy, 80, marker='^', color='red', zorder=2)
    #import bouncepoints coord.
    m.scatter(px, py, 10, marker='d', color='yellow', zorder=2)

    m.drawcoastlines(zorder=1)
    #greatcirclepath drawing from station to event
    #Check if qlat has a length
    try:
        for i in range(len(qlat)):
            m.drawgreatcircle(qlon[i], qlat[i], slon[i], slat[i], linewidth = 1, color = 'black', zorder=1)
    except TypeError:       
        m.drawgreatcircle(qlon, qlat, slon, slat, linewidth = 1, color = 'black', zorder=1)
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,120.,30.), zorder=1)
    m.drawmeridians(np.arange(0.,420.,60.), zorder=1)
    plt.title("")
    
    if savefigure:
        plt.savefig('plot_gcp.png', format="png", dpi=900)
    else:
        plt.show()

