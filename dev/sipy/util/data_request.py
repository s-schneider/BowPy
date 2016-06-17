from __future__ import print_function
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy import Stream
import obspy
import numpy as np
from obspy.geodetics import locations2degrees, gps2dist_azimuth
from obspy.taup import TauPyModel
import sys

from sipy.util.array_util import center_of_gravity, plot_map, attach_network_to_traces, attach_coordinates_to_traces, geometrical_center

def data_request(client_name, start, end, minmag, net=None, scode="*", channels="*", minlat=None,
                 maxlat=None,minlon=None,maxlon=None, station_minlat=None,
                 station_maxlat=None, station_minlon=None, station_maxlon=None, mindepth=None, maxdepth=None, 
                 radialcenterlat=None, radialcenterlon=None, minrad=None, maxrad=None,
                 station_radialcenterlat=None, station_radialcenterlon=None, station_minrad=None, station_maxrad=None,
                 azimuth=None, baz=False, t_before_first_arrival=1, t_after_first_arrival=9, savefile=False, file_format='SAC'):
	"""
	Searches in a given Database for seismic data. Restrictions in terms of starttime, endtime, network etc can be made.
	If data is found it returns a stream variable, with the waveforms, an inventory with all station and network information
	and a catalog with the event information.

	:param client_name: Name of desired fdsn client, for a list of all clients see: 
		                https://docs.obspy.org/tutorial/code_snippets/retrieving_data_from_datacenters.html
	:type  client_name:  string

	:param start, end: starttime, endtime
	:type : UTCDateTime

	:param minmag: Minimum magnitude of event
	:type  minmag: float

	:param net: Network code for which to search data for
	:type  net: string

	:param scode: Station code for which to search data for
	:type  scode: string

	:param channels: Used channels of stations 
	:type  channels: string

	:param minlat, maxlat, minlon, maxlon: Coordinate-window of interest
	:type : float

	:param mindepth, maxdepth: depth information of event in km
	:type : float

	:param radialcenterlat, radialcenterlon: Centercoordinates of a radialsearch, if radialsearch=True
	:type : float

	:param minrad, maxrad: Minimum and maximum radii for radialsearch
	:type : float

	:param azimuth: Desired range of azimuths of event, station couples in deg as a list [minimum azimuth, maximum azimuth]
	:type  azimuth: list

	:param baz: Desired range of back-azimuths of event, station couples in deg as a list [minimum back azimuth, maximum back azimuth]
	:type  baz: list

	:param t_before_first_arrival, t_before_after_arrival: Length of the seismograms, startingpoint, minutes before 1st arrival and
															minutes after 1st arrival.
	:type  t_before_first_arrival, t_before_after_arrival: float, int
	
	:param savefile: if True, Stream, Inventory and Catalog will be saved local, in the current directory.
	:type  savefile: bool

	:param format: File-format of the data, for supported formats see: https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.write.html#obspy.core.stream.Stream.write
	:type  format: string
	
	returns

	:param: list_of_stream, Inventory, Catalog
	:type: list, obspy, obspy 

	### Example ###

	from obspy import UTCDateTime
	start = UTCDateTime(2003,1,1,0,0)
	end = UTCDateTime(2013,12,31,0,0)
	list_of_stream, inventory, cat = data_request('IRIS', start, end, 8.6, 'TA', minlat=34., maxlat=50., minlon=-125., maxlon=-116.)
	
	st = list_of_stream[2]
	st.normalize()
	inv = inventory[2]

	from obspy import UTCDateTime
	start = UTCDateTime(2016,01,01,0,0)
	end = UTCDateTime(2016,04,27,0,0)
	list_of_stream, inventory, cat = data_request('IRIS', start, end, 3.5, station_radialcenterlat=54., 
	station_radialcenterlon=-117., station_minrad=0., station_maxrad=2., minlat=49., maxlat=59., minlon=-125., maxlon=-110.)
	"""

	data =[]
	stream = Stream()
	streamall = []
	client = Client(client_name)

	try:
		catalog = client.get_events(starttime=start, endtime=end, minmagnitude=minmag, maxdepth=maxdepth, mindepth=mindepth, latitude=radialcenterlat, longitude=radialcenterlon, minradius=minrad, maxradius=maxrad,minlatitude=minlat, maxlatitude=maxlat, minlongitude=minlon, maxlongitude=maxlon)

	except:
		print("No events found for given parameters.")
		return
	
	print("Following events found: \n")
	print(catalog)
	m = TauPyModel(model="ak135")
	Plist = ["P", "Pdiff", "p"]
	for event in catalog:
		print("\n")
		print("########################################")
		print("Looking for available data for event: \n")
		print(event.short_str())
		print("\n")

		origin_t = event.origins[0].time
		station_stime = UTCDateTime(origin_t - 3600*24)
		station_etime = UTCDateTime(origin_t + 3600*24)

		try:
			inventory = client.get_stations(network=net, station=scode, level="station", starttime=station_stime, endtime=station_etime,
			 								minlatitude=station_minlat, maxlatitude=station_maxlat, minlongitude=station_minlon, maxlongitude=station_maxlon,
			 								latitude=station_radialcenterlat, longitude=station_radialcenterlon, minradius=station_minrad, maxradius=station_maxrad)
			print(inventory)
		except:
			print("No Inventory found for given parameters")
			return
		
		for network in inventory:

			elat = event.origins[0].latitude
			elon = event.origins[0].longitude
			depth = event.origins[0].depth/1000.

			array_fits = True
			if azimuth or baz:
				cog=center_of_gravity(network)
				slat = cog['latitude']
				slon = cog['longitude']			
				epidist = locations2degrees(slat,slon,elat,elon)
				arrivaltime = m.get_travel_times(source_depth_in_km=depth, distance_in_degree=epidist,
							                        phase_list=Plist)

				P_arrival_time = arrivaltime[0]

				Ptime = P_arrival_time.time
				tstart = UTCDateTime(event.origins[0].time + Ptime - t_before_first_arrival * 60)
				tend = UTCDateTime(event.origins[0].time + Ptime + t_after_first_arrival * 60)


				center = geometrical_center(inv)
				clat = center['latitude']
				clon = center['longitude']
				if azimuth:
					print("Looking for events in the azimuth range of %f to %f" % (azimuth[0], azimuth[1]) )
					center_az = gps2dist_azimuth(clat, clon, elat, elon)[1]
					if center_az > azimuth[1] and center_az < azimuth[0]: 
						print("Geometrical center of Array out of azimuth bounds, \ncheking if single stations fit")
						array_fits = False

				elif baz:
					print("Looking for events in the back azimuth range of %f to %f" %(baz[0], baz[1]))
					center_baz = gps2dist_azimuth(clat, clon, elat, elon)[2]
					if center_baz > baz[1] and center_baz < baz[0]: 
						print("Geometrical center of Array out of back azimuth bounds, \ncheking if single stations fit")
						array_fits = False

			# If array fits to azimuth/back azimuth or no azimuth/back azimuth is given
			no_of_stations = 0
			if array_fits:

				for station in network:

					epidist = locations2degrees(station.latitude,station.longitude,elat,elon)
					arrivaltime = m.get_travel_times(source_depth_in_km=depth, distance_in_degree=epidist,
								                        phase_list=Plist)

					P_arrival_time = arrivaltime[0]

					Ptime = P_arrival_time.time
					tstart = UTCDateTime(event.origins[0].time + Ptime - t_before_first_arrival * 60)
					tend = UTCDateTime(event.origins[0].time + Ptime + t_after_first_arrival * 60)

					try:
						streamreq = client.get_waveforms(network=network.code, station=station.code, location='*', channel=channels, starttime=tstart, endtime=tend, attach_response=True)
						no_of_stations += 1
						print("Downloaded data for %i of %i available stations!" % (no_of_stations, network.selected_number_of_stations), end='\r' )
						sys.stdout.flush()
						stream 		   += streamreq
						try:
							if inventory_used:
								inventory_used 	+= client.get_stations(network=net, station=scode, level="station", starttime=station_stime, endtime=station_etime,
			 								minlatitude=station_minlat, maxlatitude=station_maxlat, minlongitude=station_minlon, maxlongitude=station_maxlon,
			 								latitude=station_radialcenterlat, longitude=station_radialcenterlon, minradius=station_minrad, maxradius=station_maxrad)
									
						except:
								inventory_used 	 = client.get_stations(network=net, station=scode, level="station", starttime=station_stime, endtime=station_etime,
			 								minlatitude=station_minlat, maxlatitude=station_maxlat, minlongitude=station_minlon, maxlongitude=station_maxlon,
			 								latitude=station_radialcenterlat, longitude=station_radialcenterlon, minradius=station_minrad, maxradius=station_maxrad)
					except:
						continue


			# If not checking each station individually.
			else:
				for station in network:
					epidist = locations2degrees(station.latitude,station.longitude,elat,elon)
					arrivaltime = m.get_travel_times(source_depth_in_km=depth, distance_in_degree=epidist,
								                        phase_list=Plist)


					P_arrival_time = arrivaltime[0]

					Ptime = P_arrival_time.time
					tstart = UTCDateTime(event.origins[0].time + Ptime - t_before_first_arrival * 60)
					tend = UTCDateTime(event.origins[0].time + Ptime + t_after_first_arrival * 60)

					fit = False
					if azimuth:
						stat_az = gps2dist_azimuth(station.latitude, station.longitude, elat, elon)[1]
						if stat_az > azimuth[1] and stat_az < azimuth[0]: fit = True
					elif baz:
						stat_baz = gps2dist_azimuth(station.latitude, station.longitude, elat, elon)[2]
						if stat_baz > baz[1] and stat_baz < baz[0]: fit = True
					if fit:
						try:
							streamreq = client.get_waveforms(network = network.code, station = station.code, location='*', channel = channels, startime = tstart, endtime = tend, attach_response = True)
							no_of_stations += 1
							print("Downloaded data for %i of %i available stations!" % (no_of_stations, network.selected_number_of_stations), end='\r' )
							sys.stdout.flush()
							stream 		+= streamreq
							try:
								if inventory_used:
									inventory_used 	+= client.get_stations(network=net, station=scode, level="station", starttime=station_stime, endtime=station_etime,
			 								minlatitude=station_minlat, maxlatitude=station_maxlat, minlongitude=station_minlon, maxlongitude=station_maxlon,
			 								latitude=station_radialcenterlat, longitude=station_radialcenterlon, minradius=station_minrad, maxradius=station_maxrad)
							except:
									inventory_used 	 = client.get_stations(network=net, station=scode, level="station", starttime=station_stime, endtime=station_etime,
			 								minlatitude=station_minlat, maxlatitude=station_maxlat, minlongitude=station_minlon, maxlongitude=station_maxlon,
			 								latitude=station_radialcenterlat, longitude=station_radialcenterlon, minradius=station_minrad, maxradius=station_maxrad)
						except:

							continue

		try:
			if invall:
				invall += inventory
		except:
			invall 		= inventory

		attach_network_to_traces(stream, inventory)
		attach_coordinates_to_traces(stream, inventory, event)
		streamall.append(stream)
		stream = Stream()

	if savefile:
		 stname = str(origin_t).split('.')[0] + ".MSEED"
		 invname = stname + "_inv.xml"
		 catname = stname + "_cat.xml"
		 stream.write(stname, format=file_format)
		 inventory.write(invname, format="STATIONXML")
		 catalog.write(catname, format="QUAKEML")

	plt.ion()
	#invall.plot()
	#catalog.plot()
	plt.ioff()
	inventory = invall
	list_of_stream = streamall
	return(list_of_stream, inventory, catalog)

def create_insta_from_invcat(network, event):
	import obspy
	from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees, locations2degrees
	from obspy import read as read_st
	from obspy import read_inventory as read_inv
	from obspy import read_events as read_cat
	from obspy.taup import TauPyModel

	import numpy
	import numpy as np
	import matplotlib.pyplot as plt
	import matplotlib as mpl
	import scipy as sp
	import scipy.signal as signal
	from numpy import genfromtxt

	from sipy.util.array_util import get_coords

	import os
	import datetime

	import sipy.filter.fk as fk
	from sipy.filter.fk import fk_filter
	import sipy.util.fkutil as fku
	import instaseis as ins


	db 		= ins.open_db("/Users/Simon/dev/instaseis/10s_PREM_ANI_FORCES")

	tofe 	= event.origins[0].time
	lat 	= event.origins[0].latitude 
	lon 	= event.origins[0].longitude
	depth 	= event.origins[0].depth

	source = ins.Source(
	latitude=lat, longitude=lon, depth_in_m=depth,
	m_rr = 0.526e26 / 1E7,
	m_tt = -2.1e26 / 1E7,
	m_pp = -1.58e26 / 1E7,
	m_rt = 1.08e+26 / 1E7,
	m_rp = 2.05e+26 / 1E7,
	m_tp = 0.607e+26 / 1E7,
	origin_time=tofe
	)

	stream = Stream()
	tmp = []
	for station in network:
		rec = ins.Receiver(latitude=str(station.latitude), longitude=str(station.longitude), network=str(network.code), station=str(station.code) )
		tmp.append(db.get_seismograms(source=source, receiver=rec))

	for x in tmp:
		stream += x

	return stream



