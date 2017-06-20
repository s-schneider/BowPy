import obspy
from obspy.clients.syngine import Client as synClient
from obspy.clients.fdsn import Client as irisClient

"""
# Example
from obspy.clients.syngine import Client as synClient
from obspy.clients.fdsn import Client as irisClient
model='ak135f_1s'
eventid="GCMT:122604A"
iriseventid=1916079
irisclient = irisClient('IRIS')
inv = irisclient.get_stations(network='TA', starttime=UTCDateTime(2017,1,1),
                              endtime=UTCDateTime(2018,1,1), maxlatitude=50)
streams, cat = get_syngine_data(model, eventid, iriseventid, inv)
"""


def get_syngine_data(model, eventid, iriseventid, inv):
    irisclient = irisClient('IRIS')
    synclient = synClient()
    streams = {}

    for network in inv:
        stream = obspy.Stream()

        for station in network:
            stream_tmp = synclient.get_waveforms(model=model,
                                                 network=network.code,
                                                 station=station.code,
                                                 eventid=eventid)
            stream.append(stream_tmp[0])

        streams[network.code] = stream

    cat = irisclient.get_events(eventid=iriseventid)

    return streams, cat
