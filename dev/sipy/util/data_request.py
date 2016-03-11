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
                 maxlat=None,minlon=None,maxlon=None, mind=None, maxd=None, 
                 lat=None, lon=None, minrad=None, maxrad=None, azimuth=None, radialsearch=False, savefile=False):

    client = Client(client_name)


    if radialsearch:
    	catalog = client.get_events(starttime=start, endtime = end,
                            		minmagnitude = minmag, maxdepth=maxd, mindepth=mind,
                            		latitude=lat, longitude=lon, minradius = minrad, maxradius = maxrad)
    else:
    	catalog = client.get_events(starttime=start, endtime = end,
                            minmagnitude = minmag, maxdepth=maxd, mindepth=mind)	                         

    inventory = []
    st = []
    event = []
    origin = []
    slat = []
    slon = []
    elat = []
    elon = []
    depth = []
    epidist = []
    tstart = []
    tend = []

    m = TauPyModel(model="ak135")
    Plist = ["P"]
    for i in range(len(catalog)):
        event.append(catalog[i])
        origin.append(event[i].origins)
        station_stime = UTCDateTime(origin[i][0].time - 3600*24)
        station_etime = UTCDateTime(origin[i][0].time + 3600*24)
        inventory.append(client.get_stations(network=net, station=scode, level="station", 
                            starttime=station_stime, endtime=station_etime, 
                            minlatitude=minlat, maxlatitude=maxlat, minlongitude=minlon, maxlongitude=maxlon))

        cog=center_of_gravity(inventory[i])
        slat.append(cog['latitude'])
        slon.append(cog['longitude'])
        elat.append(origin[i][0].latitude)
        elon.append(origin[i][0].longitude)
        depth.append(origin[i][0].depth/1000)
        epidist.append(locations2degrees(slat[i],slon[i],elat[i],elon[i]))
        arrivaltime = m.get_travel_times(source_depth_in_km=depth[i],distance_in_degree=epidist[i],
                                            phase_list=Plist)
        P_arrival_time = arrivaltime[0]
        Ptime = P_arrival_time.time
        tstart.append(UTCDateTime(origin[i][0].time + Ptime - 3 * 60))
        tend.append(UTCDateTime(origin[i][0].time + Ptime + 10 * 60))

        network = inventory[i]
        stations = []
        stations_str = ""
        for station in network[0]:
            stations.append(station.code)
        stations_str = ','.join(map(str,stations))
        location = "*"
        st.append(client.get_waveforms(network[0].code, stations_str, location, channels,
                                  tstart[i], tend[i]))
        if savefile:
                 stname = str(event[i].origins[0].time).split('.')[0] + ".MSEED"
                 invname = stname + "_inv.xml"
                 catname = stname + "_cat.xml"
                 st[i].write(stname, format="MSEED")
                 inventory[i].write(invname, format="STATIONXML")
                 catalog[i].write(catname, format="QUAKEML")
    return(st, inventory, catalog)

"""
Example
dl="IRIS"
starttime = UTCDateTime("2011-01-01T00:00:00")
endtime = UTCDateTime("2011-12-31T23:59:59")
minmag=9.0
nw="TA"
stats="*"

st, inv, cat  = data_request(client_name = dl, start = s, end = e, minmag = mm, net = nw, scode = stats, savefile=True)
"""

