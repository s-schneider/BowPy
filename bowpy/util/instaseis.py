# get data with instaseis
from __future__ import absolute_import
import os
import instaseis as ins


def create_quake_origins(time=None, lat=None, lon=None, depth_in_m=None,
                         m_rr=None, m_tt=None, m_pp=None, m_rt=None, m_rp=None,
                         m_tp=None, event=None):
    """
    Routine to create the quake_origins file for dosynthetics.
    :param time: Origin time of the event
    :type  time: UTCDatetime

    m_rr = 0.526e26 / 1E7,
    m_tt = -2.1e26 / 1E7,
    m_pp = -1.58e26 / 1E7,
    m_rt = 1.08e+26 / 1E7,
    m_rp = 2.05e+26 / 1E7,
    m_tp = 0.607e+26 / 1E7,
    """

    # If event file is input, creating of quake_origins starts with from this
    # input, other inputs will be ignored!
    if event:
        time = event.origins[0].time
        lat = event.origins[0].latitude
        lon = event.origins[0].longitude
        depth_in_m = event.origins[0].depth

    quake_origins = {
        'tofe': 		time,
        'latitude': 	lat,
        'longitude': 	lon,
        'depth_in_m':	depth_in_m,
        'm_rr':			m_rr,
        'm_tt':			m_tt,
        'm_pp':			m_pp,
        'm_rt':			m_rt,
        'm_rp':			m_rp,
        'm_tp':			m_tp,
    }

    if None in quake_origins.viewvalues():
        msg = '"None" values in quake_origins are not allowed'
        raise IOError(msg)

    return quake_origins


def dosynthetics(database_path, quake_origins, inv=None, inv_list=None,
                 stream=None, stream_file_name='insta_stream.pickle'):
    """
    Routine to create synthetic data using instaseis.


    :param database_path:
    :type  database_path:

    :param quake_origins:
    :type  quake_origins: dict from create_quake_origins

        Example Honshu:

        m_rr = 1.730e29
        m_tt = -0.281e29
        m_pp = -1.450e29
        m_rt = 2.120e29
        m_rp = 4.550e29
        m_tp = -0.657e29

        event = read_cat('/Users/Simon/Documents/Studium/WWU/Master/Thesis/data_all/test_datasets/instaseis/honshu_data/HONSHU_cat.xml')[0]
        q_origins = create_quake_origins(None, None, None, None, m_rr, m_tt, m_pp, m_rt, m_rp, m_tp, event)
        dbpath = '~/dev/python/instaseis/10s_PREM_ANI_FORCES'
        stream, inv, cat = dosynthetics(db_path, q_origin, inv)
    """
    if None in quake_origins.viewvalues():
        msg = '"None" values in quake_origins are not allowed'
        raise IOError(msg)

    db = ins.open_db(database_path)
    source = ins.Source(
        latitude=quake_origins['latitude'],
        longitude=quake_origins['longitude'],
        depth_in_m=quake_origins['depth_in_m'],
        m_rr=quake_origins['m_rr'],
        m_tt=quake_origins['m_tt'],
        m_pp=quake_origins['m_pp'],
        m_rt=quake_origins['m_rt'],
        m_rp=quake_origins['m_rp'],
        m_tp=quake_origins['m_rp'],
        origin_time=quake_origins['tofe']
        )

    # Prepare synthetic stream and inv.
    if stream_file_name:
        prefix = stream_file_name.split('.')
        inv_file_name = prefix[0] + '_inv.xml'
        cat_file_name = prefix[0] + '_cat.xml'

    else:
        inv_file_name = 'inv_tmp.xml'
        cat_file_name = 'cat_tmp.xml'

    receiver_synth = []

    if stream:
        for trace in stream:
            receiver_synth.append(ins.Receiver(latitude=str(trace.stats.coordinates['latitude']), longitude=str(trace.stats.coordinates['longitude']),
                                 network=str(trace.stats.network), station=str(trace.stats.station)))

    elif inv and not stream:
        network = inv[0]
        for station in network:
            receiver_synth.append(ins.Receiver(latitude=str(station.latitude), longitude=str(station.longitude), network=str(network.code), station=str(station.code) ))

    elif inv_list:
        for i, station in enumerate(inv_list):
            receiver_synth.append(ins.Receiver(latitude=str(station[0]), longitude=str(station[1]), network=str('X'), station='X' + str(i) ))

    st_synth = []
    for i in range(len(x)):
        st_synth.append(db.get_seismograms(source=source, receiver=receiver_synth[i]))

    stream=st_synth[0]
    for i in range(len(st_synth))[1:]:
        stream.append(st_synth[i][0])

    writeQuakeML(cat_file_name, quake_origins['tofe'], quake_origins['latitude'], quake_origins['longitude'], quake_origins['depth_in_m'])

    if not inv:
        writeStationML(inv_file_name)
        inv = read_inv(inv_file_name)

    cat=read_cat(cat_file_name)

    if stream_file_name:
        stream.write(stream_file_name, format='pickle')
    else:
        os.remove(cat_file_name)
        if inv:
            os.remove(inv_file_name)



    return stream, inv, cat


def writeQuakeML(catfile, tofe, lat, lon, depth):

    with open( catfile, "w") as fh:
        fh.write("<?xml version=\'1.0\' encoding=\'utf-8\'?> \n")
        fh.write("<q:quakeml xmlns:q=\"http://quakeml.org/xmlns/quakeml/1.2\" xmlns:ns0=\"http://service.iris.edu/fdsnws/event/1/\" xmlns=\"http://quakeml.org/xmlns/bed/1.2\"> \n")
        fh.write("  <eventParameters publicID=\"smi:local/6b269cbf-6b00-4643-8c2c-cbe6274083ae\"> \n")
        fh.write("    <event publicID=\"smi:service.iris.edu/fdsnws/event/1/query?eventid=3279407\"> \n")
        fh.write("      <preferredOriginID>smi:service.iris.edu/fdsnws/event/1/query?originid=9933375</preferredOriginID> \n")
        fh.write("      <preferredMagnitudeID>smi:service.iris.edu/fdsnws/event/1/query?magnitudeid=16642444</preferredMagnitudeID> \n")
        fh.write("      <type>earthquake</type> \n")
        fh.write("      <description ns0:FEcode=\"228\"> \n")
        fh.write("        <text>NEAR EAST COAST OF HONSHU, JAPAN</text> \n ")
        fh.write("        <type>Flinn-Engdahl region</type> \n")
        fh.write("      </description> \n")
        fh.write("      <origin publicID=\"smi:service.iris.edu/fdsnws/event/1/query?originid=9933375\" ns0:contributor=\"ISC\" ns0:contributorOriginId=\"02227159\" ns0:catalog=\"ISC\" ns0:contributorEventId=\"16461282\"> \n")
        fh.write("        <time> \n")
        fh.write("          <value>%s</value> \n" % tofe)
        fh.write("        </time> \n")
        fh.write("        <latitude> \n")
        fh.write("          <value>%f</value> \n" %lat)
        fh.write("        </latitude>\n")
        fh.write("        <longitude> \n")
        fh.write("          <value>%f</value> \n" %lon)
        fh.write("        </longitude>\n")
        fh.write("        <depth> \n")
        fh.write("          <value>%f</value> \n" %depth)
        fh.write("        </depth>\n")
        fh.write("        <creationInfo> \n")
        fh.write("          <author>Simon</author> \n")
        fh.write("        </creationInfo> \n")
        fh.write("      </origin> \n")
        fh.write("      <magnitude publicID=\"smi:service.iris.edu/fdsnws/event/1/query?magnitudeid=16642444\"> \n")
        fh.write("        <mag> \n")
        fh.write("          <value>9.1</value> \n")
        fh.write("        </mag> \n")
        fh.write("        <type>MW</type> \n")
        fh.write("        <originID>smi:service.iris.edu/fdsnws/event/1/query?originid=9933383</originID> \n")
        fh.write("        <creationInfo> \n")
        fh.write("          <author>Simon</author> \n")
        fh.write("        </creationInfo> \n")
        fh.write("      </magnitude> \n")
        fh.write("    </event> \n")
        fh.write("  </eventParameters> \n")
        fh.write("</q:quakeml>")

    return

def writeStationML(invfile, lats, lons):

    with open( invfile, "w") as fh:
        fh.write("<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n")
        fh.write("<FDSNStationXML schemaVersion=\"1.0\" xmlns=\"http://www.fdsn.org/xml/station/1\">\n")
        fh.write("  <Source>IRIS-DMC</Source>\n")
        fh.write("  <Sender>IRIS-DMC</Sender>\n")
        fh.write("  <Created>2015-11-05T18:22:28+00:00</Created>\n")
        fh.write("  <Network code=\"LA\" endDate=\"2500-12-31T23:59:59+00:00\" restrictedStatus=\"open\" startDate=\"2003-01-01T00:00:00+00:00\">\n")
        fh.write("    <Description>Synthetic Array - Linear Array</Description>\n")
        fh.write("    <TotalNumberStations>20</TotalNumberStations>\n")
        fh.write("    <SelectedNumberStations>20</SelectedNumberStations>\n")

        j=0
        ncount = 0
        for i in lats:
            for j in lons:
                slat=i
                slon=j
                ncount += 1
                name="X" + str(ncount)
                fh.write("    <Station code=\"%s\" endDate=\"2011-11-17T23:59:59+00:00\" restrictedStatus=\"open\" startDate=\"2010-01-08T00:00:00+00:00\">\n" % name)
                fh.write("      <Latitude unit=\"DEGREES\">%f</Latitude>\n" % slat)
                fh.write("      <Longitude unit=\"DEGREES\">%f</Longitude>\n" % slon)
                fh.write("      <Elevation>0.0</Elevation>\n")
                fh.write("      <Site>\n")
                fh.write("        <Name> %s </Name>\n" % name)
                fh.write("      </Site>\n")
                fh.write("    <CreationDate>2010-01-08T00:00:00+00:00</CreationDate>\n")
                fh.write("    </Station>\n")
                j += 1
        fh.write("  </Network>\n")
        fh.write("</FDSNStationXML>")

    return
