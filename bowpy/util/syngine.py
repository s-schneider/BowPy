import obspy
from obspy import Stream
from obspy.clients.syngine import Client as synClient
from obspy.clients.fdsn import Client as fdsnClient
from obspy.core.util.attribdict import AttribDict
from obspy.geodetics.base import degrees2kilometers, gps2dist_azimuth
from bowpy.util.array_util import dist_azimuth2gps, geometrical_center

"""
:param sourcedoublecouple: Specify a source as a double couple. The
    list of values are ``strike``, ``dip``, ``rake`` [, ``M0`` ],
    where strike, dip and rake are in degrees and M0 is the scalar
    seismic moment in Newton meters (Nm). If not specified, a value
    of *1e19* will be used as the scalar moment.

:param sourcemomenttensor: Specify a source in moment tensor
    components as a list: ``Mrr``, ``Mtt``, ``Mpp``, ``Mrt``, ``Mrp``,
    ``Mtp`` with values in Newton meters (*Nm*).

# Example 1
from obspy.clients.syngine import Client as synClient
from obspy.clients.fdsn import Client as fdsnClient
from bowpy.util.syngine import get_syngine_data

model='ak135f_1s'
eventid="GCMT:201305240544A"
irisclient = fdsnClient('IRIS')
inv = irisclient.get_stations(network='TA', station='121A',
                              starttime=UTCDateTime(2017,1,1),
                              endtime=UTCDateTime(2018,1,1), maxlatitude=50)
streams, cat = get_syngine_data(model, client="IRIS", eventid=eventid, inv=inv)
st = streams.TA

# Example 2

from obspy.clients.syngine import Client as synClient
from obspy.clients.fdsn import Client as fdsnClient
from bowpy.util.syngine import get_syngine_data
from obspy.core import AttribDict

model='ak135f_1s'
irisclient = fdsnClient('IRIS')


origins = AttribDict()
origins['latitude'] = 54.61
origins['longitude'] = 153.77
origins['depth'] = 611000.0
origins['time'] = UTCDateTime(2013, 5, 24, 5, 45, 8.3)
moment_tensor = [-1.670, 0.382, 1.280, -0.784, -3.570, 0.155]
exponent = 1E28
moment_tensor[:] = [x * exponent for x in moment_tensor]
sourcedoublecouple = [189, 11, -93, 3.95e+28]

inv = irisclient.get_stations(network='TA', station='121A',
                              starttime=UTCDateTime(2017,1,1),
                              endtime=UTCDateTime(2018,1,1),
                              maxlatitude=50)

streams, cat_syn = get_syngine_data(model, client='IRIS', inv=inv,
                                    origins=origins, m_tensor=moment_tensor)
st = streams.TA
"""


def get_syngine_data(model, client=None, reclat=None, reclon=None, inv=None,
                     eventid=None, origins=None, m_tensor=None,
                     source_dc=None):
    """
    param reclat:
    type reclat: list of floats
    param reclon:
    type reclon: list of floats
    """
    if client:
        client = fdsnClient(client)
    synclient = synClient()

    if inv:
        streams = AttribDict()
        for network in inv:
            stream = obspy.Stream()

            for station in network:
                print(station)
                if eventid:
                    stream_tmp = synclient.get_waveforms(model=model,
                                                         network=network.code,
                                                         station=station.code,
                                                         eventid=eventid
                                                         )
                else:
                    stream_tmp = synclient.get_waveforms(model=model,
                                                         network=network.code,
                                                         station=station.code,
                                                         origintime=origins.time,
                                                         sourcelatitude=origins.latitude,
                                                         sourcelongitude=origins.longitude,
                                                         sourcedepthinmeters=origins.depth,
                                                         sourcemomenttensor=m_tensor,
                                                         sourcedoublecouple=source_dc
                                                         )
                stream.append(stream_tmp[0])
            streams[network.code] = stream

    if reclat and reclon:
        stream = obspy.Stream()
        for rlat, rlon in zip(reclat, reclon):
            if eventid:
                stream_tmp = synclient.get_waveforms(model=model,
                                                     receiverlatitude=rlat,
                                                     receiverlongitude=rlon,
                                                     eventid=eventid
                                                     )
            else:
                stream_tmp = synclient.get_waveforms(model=model,
                                                     receiverlatitude=rlat,
                                                     receiverlongitude=rlon,
                                                     origintime=origins.time,
                                                     sourcelatitude=origins.latitude,
                                                     sourcelongitude=origins.longitude,
                                                     sourcedepthinmeters=origins.depth,
                                                     sourcemomenttensor=m_tensor,
                                                     sourcedoublecouple=source_dc
                                                     )
            stream.append(stream_tmp[0])
        streams = stream

    if origins:
        starttime = origins.time - 120
        endtime = starttime + 120
        if client:
            cat = client.get_events(starttime, endtime,
                                    minlatitude=origins.latitude-.5,
                                    maxlatitude=origins.latitude+.5)
        else:
            cat = None
    else:
        cat = None

    return streams, cat


def get_ref_data(stream, inv, model='ak135f_1s', eventid=None, origins=None,
                 m_tensor=None, source_dc=None):

    ref_stream = Stream()

    rlats = []
    rlons = []
    geom = geometrical_center(inv)
    d, az, baz = gps2dist_azimuth(origins.latitude, origins.longitude,
                                  geom.latitude, geom.longitude)
    for i, trace in enumerate(stream):
        dist = degrees2kilometers(trace.stats.distance)*1000.

        rlat, rlon = dist_azimuth2gps(origins.latitude, origins.longitude,
                                      az, dist)
        if rlon > 180:
            rlon = 180. - rlon

        print(rlat, rlon)
        rlats.append(rlat)
        rlons.append(rlon)
        print('Receiving trace %i of %i.' % (i+1, len(stream)))
        stream_tmp, cat_void = get_syngine_data(model, reclat=rlats, reclon=rlons,
                                                eventid=eventid, origins=origins,
                                                m_tensor=m_tensor, source_dc=source_dc
                                                )

        trace_tmp = stream_tmp[0].copy()
        trace_tmp.stats.station = trace.stats.station
        trace_tmp.stats.starttime = trace.stats.starttime
        trace_tmp.stats.distance = trace.stats.distance
        ref_stream.append(trace_tmp)

    return ref_stream
