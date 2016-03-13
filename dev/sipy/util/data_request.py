from __future__ import print_function
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy import Stream
import obspy
import numpy as np
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

from sipy.util.array_util import center_of_gravity, plot_gcp

def data_request(client_name, start, end, minmag, net, scode="*", channels="BHZ", minlat=None,
                 maxlat=None,minlon=None,maxlon=None, mindepth=None, maxdepth=None, 
                 radialcenterlat=None, radialcenterlon=None, minrad=None, maxrad=None, 
                 azimuth=None, radialsearch=False, savefile=False):
    """
    Searches in a given Database for seismic data. Restrictions in terms of starttime, endtime, network etc can be made.

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

    :param azimuth: Desired azimuth of event, station couples in deg
    :type  azimuth: float

    :param radialsearch: Sets radialsearch on or off
    :type  radialsearch: bool

    :param savefile: if True, Stream, Inventory and Catalog will be saved local, default directory is the current.
    :type  savefile: bool.


    returns

    :param: data as a list of tuples in form (Stream, Inventory, Catalog)
    :type: list

    ### Example ###

    from obspy import UTCDateTime
    dl="IRIS"
    start = UTCDateTime("2011-01-01T00:00:00")
    end= UTCDateTime("2011-12-31T23:59:59")
    minmag=9.0
    nw="TA"
    stats="*"

    data = data_request(client_name = dl, start = start, end = end, minmag = minmag, net = nw, scode = stats)

    """

    data =[]
    stream = Stream()
    client = Client(client_name)

    try:
        if radialsearch:
        	catalog = client.get_events(starttime=start, endtime=end, minmagnitude=minmag, maxdepth=maxdepth, mindepth=mindepth, latitude=radialcenterlat, longitude=radialcenterlon, minradius=minrad, maxradius=maxrad)
        else:
            catalog = client.get_events(starttime=start, endtime=end, minmagnitude=minmag, maxdepth=maxdepth, mindepth=mindepth)
    except:
        print("No events found for given parameters.")
        return

    print(catalog)
    m = TauPyModel(model="ak135")
    Plist = ["P", "Pdiff"]
    for event in catalog:
        origin_t = event.origins[0].time
        station_stime = UTCDateTime(origin_t - 3600*24)
        station_etime = UTCDateTime(origin_t + 3600*24)
        try:
            inventory = client.get_stations(network=net, station=scode, level="station", starttime=station_stime, endtime=station_etime, minlatitude=minlat, maxlatitude=maxlat, minlongitude=minlon, maxlongitude=maxlon)
        except:
            print("No Inventory found for given parameters")
            return

        for network in inventory:
            cog=center_of_gravity(network)
            slat = cog['latitude']
            slon = cog['longitude']
            elat = event.origins[0].latitude
            elon = event.origins[0].longitude
            depth = event.origins[0].depth/1000.
            epidist = locations2degrees(slat,slon,elat,elon)
            arrivaltime = m.get_travel_times(source_depth_in_km=depth, distance_in_degree=epidist,
                                                phase_list=Plist)
            P_arrival_time = arrivaltime[0]
            Ptime = P_arrival_time.time
            tstart = UTCDateTime(event.origins[0].time + Ptime - 3 * 60)
            tend = UTCDateTime(event.origins[0].time + Ptime + 10 * 60)

            for station in network:
                try:
                    streamreq = client.get_waveforms(network.code, station.code, '*', channels,
                                      tstart, tend)
                    print("Downloaded data for station %s ... \n" % station.code )
                    stream += streamreq
                except:
                    print("No data for station %s ... \n" % station.code )
                    continue

        if savefile:
                 stname = str(origin_t).split('.')[0] + ".MSEED"
                 invname = stname + "_inv.xml"
                 catname = stname + "_cat.xml"
                 st[i].write(stname, format="MSEED")
                 inventory[i].write(invname, format="STATIONXML")
                 catalog[i].write(catname, format="QUAKEML")
        

        stream_inv_event = (stream, inventory, event)
        data.append(stream_inv_event)

    return(data)

