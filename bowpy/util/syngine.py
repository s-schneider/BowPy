import obspy
from obspy.clients.syngine import Client as synClient
from obspy.clients.fdsn import Client as fdsnClient
from obspy.core.util.attribdict import AttribDict
from obspy.core import UTCDateTime

"""
# Example
from obspy.clients.syngine import Client as synClient
from obspy.clients.fdsn import Client as fdsnClient

model='ak135f_1s'
eventid="GCMT:201305240544A"
iriseventid=1916079
irisclient = fdsnClient('IRIS')
inv = irisclient.get_stations(network='TA', starttime=UTCDateTime(2017,1,1),
                              endtime=UTCDateTime(2018,1,1), maxlatitude=50)
streams, cat = get_syngine_data(model, eventid, iriseventid, inv)

or

:param sourcedoublecouple: Specify a source as a double couple. The
    list of values are ``strike``, ``dip``, ``rake`` [, ``M0`` ],
    where strike, dip and rake are in degrees and M0 is the scalar
    seismic moment in Newton meters (Nm). If not specified, a value
    of *1e19* will be used as the scalar moment.

:param sourcemomenttensor: Specify a source in moment tensor
    components as a list: ``Mrr``, ``Mtt``, ``Mpp``, ``Mrt``, ``Mrp``,
    ``Mtp`` with values in Newton meters (*Nm*).

from obspy.clients.syngine import Client as synClient
from obspy.clients.fdsn import Client as fdsnClient

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

inv = irisclient.get_stations(network='TA', starttime=UTCDateTime(2017,1,1),
                              endtime=UTCDateTime(2018,1,1), maxlatitude=50)
streams, cat_syn = get_syngine_data(model, inv, 'IRIS', origins=origins,
                                m_tensor=moment_tensor)
"""


def get_syngine_data(model, inv, client, eventid=None, origins=None,
                     m_tensor=None, source_dc=None):
    client = fdsnClient(client)
    synclient = synClient()
    streams = AttribDict()

    for network in inv:
        stream = obspy.Stream()

        for station in network:
            if eventid:
                stream_tmp = synclient.get_waveforms(model=model,
                                                     network=network.code,
                                                     station=station.code,
                                                     eventid=eventid)
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

    starttime = origins.time - 120
    endtime = starttime + 120
    cat = client.get_events(starttime, endtime,
                            minlatitude=origins.latitude-.5,
                            maxlatitude=origins.latitude+.5)

    return streams, cat
