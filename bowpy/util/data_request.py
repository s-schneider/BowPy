from __future__ import absolute_import, print_function
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy import Stream
from obspy.geodetics import locations2degrees, gps2dist_azimuth
from obspy.taup import TauPyModel
from obspy.core.event import Catalog, Event, Magnitude, Origin, MomentTensor
import sys
from bowpy.util.array_util import (center_of_gravity, attach_network_to_traces,
                                   attach_coordinates_to_traces,
                                   geometrical_center)
try:
    import instaseis
except:
    pass


def data_request(client_name, start=None, end=None, minmag=None, cat=None,
                 inv=None, cat_client_name=None, net=None, scode="*",
                 channels="*", minlat=None, maxlat=None, minlon=None,
                 maxlon=None, station_minlat=None, station_maxlat=None,
                 station_minlon=None, station_maxlon=None,
                 mindepth=None, maxdepth=None, radialcenterlat=None,
                 radialcenterlon=None, minrad=None, maxrad=None,
                 station_radcenlat=None, station_radcenlon=None,
                 station_minrad=None, station_maxrad=None,
                 azimuth=None, baz=False, t_before_first_arrival=1,
                 t_after_first_arrival=9, savefile=False, file_format='SAC',
                 normal_mode_data=False):
    """
    Searches in a given Database for seismic data. Restrictions in terms of
    starttime, endtime, network etc can be made. If data is found it returns a
    stream variable, with the waveforms, an inventory with all station and
    network information and a catalog with the event information.

    :param client_name: Name of desired fdsn client,
                        for a list of all clients see:
                        https://docs.obspy.org/tutorial/code_snippets/retrieving_data_from_datacenters.html
    :type  client_name:  string

    :param start, end: starttime, endtime
    :type : UTCDateTime

    :param minmag: Minimum magnitude of event
    :type  minmag: float

    :param cat_client_name: Name of Event catalog, default is "None", resulting
                            in catalog search, defined by client_name

    :type  cat_client_name: string

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

    :param radialcenterlat, radialcenterlon:
    Centercoordinates of a radialsearch, if radialsearch=True
    :type : float

    :param minrad, maxrad: Minimum and maximum radii for radialsearch
    :type : float

    :param azimuth: Desired range of azimuths of event, station couples in deg
                    as a list [minimum azimuth, maximum azimuth]
    :type  azimuth: list

    :param baz: Desired range of back-azimuths of event, station couples in deg
                as a list [minimum back azimuth, maximum back azimuth]
    :type  baz: list

    :param t_before_first_arrival, t_before_after_arrival:
    Length of the seismograms, startingpoint, minutes before 1st arrival and
    minutes after 1st arrival.
    :type  t_before_first_arrival, t_before_after_arrival: float, int

    :param savefile: if True, Stream, Inventory and Catalog will be saved
                     local, in the current directory.
    :type  savefile: bool

    :param format: File-format of the data, for supported formats see:
    https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.write.html#obspy.core.stream.Stream.write
    :type  format: string

    returns

    :param: list_of_stream, Inventory, Catalog
    :type: list, obspy, obspy



    ### Example 1 ###

    from obspy import UTCDateTime
    from bowpy.util.data_request import data_request

    start = UTCDateTime(2010,1,1,0,0)
    end = UTCDateTime(2010,12,31,0,0)
    minmag = 8
    station = '034A'
    list_of_stream, inventory, cat = data_request('IRIS', start, end, minmag,
                                                  net='TA', scode=station)

    st = list_of_stream[0]
    st = st.select(channel='BHZ')
    st.normalize()
    inv = inventory[0]

    st.plot()
    inv.plot()
    cat.plot()

    ### Example 2 ###

    from obspy import UTCDateTime
    from bowpy.util.data_request import data_request
    import mechanize

    start = UTCDateTime(2010,1,1,0,0)
    end = UTCDateTime(2010,12,31,0,0)
    minmag = 8
    station = '034A'
    client = 'IRIS'
    cat_client = 'globalcmt'
    list_of_stream, inventory, cat = data_request(client, cat_client, start,
                                                  end, minmag, net='TA',
                                                  scode=station)

    st = list_of_stream[0]
    st = st.select(channel='BHZ')
    st.normalize()
    inv = inventory[0]

    st.plot()
    inv.plot()
    cat.plot()

    """
    if not cat and not inv:
        if not start and not end and not minmag:
            print('Neither catalog and inventory specified nor dates')
            return

    stream = Stream()
    streamall = []

    # build in different approach for catalog search, using urllib
    if cat:
        catalog = cat
        client = Client(client_name)
    else:
        if cat_client_name == 'globalcmt':
            catalog = request_gcmt(starttime=start, endtime=end,
                                   minmagnitude=minmag, mindepth=mindepth,
                                   maxdepth=maxdepth, minlatitude=minlat,
                                   maxlatitude=maxlat, minlongitude=minlon,
                                   maxlongitude=maxlon)
            client = Client(client_name)
        else:
            client = Client(client_name)
            try:
                catalog = client.get_events(starttime=start, endtime=end,
                                            minmagnitude=minmag,
                                            mindepth=mindepth,
                                            maxdepth=maxdepth,
                                            latitude=radialcenterlat,
                                            longitude=radialcenterlon,
                                            minradius=minrad,
                                            maxradius=maxrad,
                                            minlatitude=minlat,
                                            maxlatitude=maxlat,
                                            minlongitude=minlon,
                                            maxlongitude=maxlon)

            except:
                print("No events found for given parameters.")
                return

    print("Following events found: \n")
    print(catalog)
    m = TauPyModel(model="ak135")
    for event in catalog:
        if inv:
            inventory = inv

        else:
            print("\n")
            print("########################################")
            print("Looking for available data for event: \n")
            print(event.short_str())
            print("\n")

            origin_t = event.origins[0].time
            station_stime = UTCDateTime(origin_t - 3600*24)
            station_etime = UTCDateTime(origin_t + 3600*24)

            try:
                inventory = client.get_stations(network=net, station=scode,
                                                level="station",
                                                starttime=station_stime,
                                                endtime=station_etime,
                                                minlatitude=station_minlat,
                                                maxlatitude=station_maxlat,
                                                minlongitude=station_minlon,
                                                maxlongitude=station_maxlon,
                                                latitude=station_radcenlat,
                                                longitude=station_radcenlon,
                                                minradius=station_minrad,
                                                maxradius=station_maxrad)

                msg = "Inventory with %i networks, " +\
                      "containing %i stations found."
                print(msg % (len(inventory),
                             len(inventory.get_contents()['stations'])))
            except:
                print("No Inventory found for given parameters")
                return

        for net in inventory:

            print("Searching in network: %s" % net.code)
            elat = event.origins[0].latitude
            elon = event.origins[0].longitude
            depth = event.origins[0].depth/1000.

            array_fits = True
            if azimuth or baz:
                cog = center_of_gravity(net)
                slat = cog['latitude']
                slon = cog['longitude']
                epidist = locations2degrees(slat, slon, elat, elon)
                arr_time = m.get_travel_times(source_depth_in_km=depth,
                                              distance_in_degree=epidist)

                # Checking for first arrival time
                P_arrival_time = arr_time[0]

                Ptime = P_arrival_time.time
                tstart = UTCDateTime(event.origins[0].time + Ptime -
                                     t_before_first_arrival * 60)
                tend = UTCDateTime(event.origins[0].time + Ptime +
                                   t_after_first_arrival * 60)

                center = geometrical_center(net)
                clat = center['latitude']
                clon = center['longitude']
                if azimuth:
                    print("Looking for events in the azimuth range of %f to %f\
                          " % (azimuth[0], azimuth[1]))
                    center_az = gps2dist_azimuth(clat, clon, elat, elon)[1]
                    if center_az > azimuth[1] and center_az < azimuth[0]:
                        print("Geometrical center of Array out of azimuth"
                              + " bounds, \nchecking if single stations fit")
                        array_fits = False

                elif baz:
                    print("Looking for events in the back azimuth " +
                          "range of %f to %f" % (baz[0], baz[1]))
                    center_baz = gps2dist_azimuth(clat, clon, elat, elon)[2]
                    if center_baz > baz[1] and center_baz < baz[0]:
                        print("Geometrical center of Array out of back " +
                              "azimuth bounds, \nchecking if " +
                              "single stations fit")
                        array_fits = False

            # If array fits to azimuth/back azimuth or no azimuth/back azimuth
            # is given
            no_of_stations = 0
            if array_fits:

                for station in net:

                    epidist = locations2degrees(station.latitude,
                                                station.longitude,
                                                elat, elon)
                    arr_time = m.get_travel_times(source_depth_in_km=depth,
                                                  distance_in_degree=epidist)
                    P_arrival_time = arr_time[0]

                    Ptime = P_arrival_time.time
                    tstart = UTCDateTime(event.origins[0].time + Ptime -
                                         t_before_first_arrival * 60)
                    if normal_mode_data:
                        tend = UTCDateTime(event.origins[0].time +
                                           Ptime + 50 * 60 * 60)
                    else:
                        tend = UTCDateTime(event.origins[0].time + Ptime +
                                           t_after_first_arrival * 60)

                    try:
                        if normal_mode_data:
                            st_req = client.get_waveforms(network=net.code,
                                                          station=station.code,
                                                          location='*',
                                                          channel=channels,
                                                          starttime=tstart,
                                                          endtime=tend,
                                                          attach_response=True,
                                                          longestonly=True)
                        else:
                            st_req = client.get_waveforms(network=net.code,
                                                          station=station.code,
                                                          location='*',
                                                          channel=channels,
                                                          starttime=tstart,
                                                          endtime=tend,
                                                          attach_response=True)
                        no_of_stations += 1
                        msg = "Downloaded data for %i of %i available " +\
                              "stations!"
                        print(msg % (no_of_stations,
                                     net.selected_number_of_stations),
                              end='\r')

                        sys.stdout.flush()
                        stream += st_req
                    except:
                        continue
                print('\n')

            # If not, checking each station individually.
            else:
                for station in net:
                    epidist = locations2degrees(station.latitude,
                                                station.longitude, elat, elon)
                    arr_time = m.get_travel_times(source_depth_in_km=depth,
                                                  distance_in_degree=epidist)

                    # Checking for first arrival time
                    P_arrival_time = arr_time[0]

                    Ptime = P_arrival_time.time
                    tstart = UTCDateTime(event.origins[0].time + Ptime -
                                         t_before_first_arrival * 60)
                    tend = UTCDateTime(event.origins[0].time + Ptime +
                                       t_after_first_arrival * 60)

                    fit = False
                    if azimuth:
                        stat_az = gps2dist_azimuth(station.latitude,
                                                   station.longitude,
                                                   elat, elon)[1]
                        if stat_az > azimuth[1] and stat_az < azimuth[0]:
                            fit = True
                    elif baz:
                        stat_baz = gps2dist_azimuth(station.latitude,
                                                    station.longitude,
                                                    elat, elon)[2]
                        if stat_baz > baz[1] and stat_baz < baz[0]:
                            fit = True
                    if fit:
                        try:
                            st_req = client.get_waveforms(network=net.code,
                                                          station=station.code,
                                                          location='*',
                                                          channel=channels,
                                                          startime=tstart,
                                                          endtime=tend,
                                                          attach_response=True)
                            no_of_stations += 1
                            msg = "Downloaded data for %i of %i available " +\
                                  "stations!"
                            print(msg % (no_of_stations,
                                         net.selected_number_of_stations),
                                  end='\r')

                            sys.stdout.flush()
                            stream += st_req
                        except:
                            continue

        invall = inventory

        attach_network_to_traces(stream, inventory)
        attach_coordinates_to_traces(stream, inventory, event)
        streamall.append(stream)
        stream = Stream()

    if savefile:
        stname = str(origin_t).split('.')[0] + "." + file_format
        invname = stname + "_inv.xml"
        catname = stname + "_cat.xml"
        stream.write(stname, format=file_format)
        inventory.write(invname, format="STATIONXML")
        catalog.write(catname, format="QUAKEML")

    plt.ion()
    plt.ioff()
    inventory = invall
    list_of_stream = streamall

    return(list_of_stream, inventory, catalog)


def create_insta_from_invcat(network, event, database):
    """
    This function creates synthetic data using the given network and
    event information, with the database of instaseis

    :param network: Desired Network, for which the data is generated
    :type  network: obspy.core.inventory.Network

    :param event: Event, for wich the data is generated. The event must have
    stored the moment tensor (e.g. given by glogalcmt.org)
    :type  event: obspy.core.event.Event

    :param database: Link to the database, e.g. the path on your harddrive
    :type  database: str
    """

    db = instaseis.open_db(database)

    tofe = event.origins[0].time
    lat = event.origins[0].latitude
    lon = event.origins[0].longitude
    depth = event.origins[0].depth

    source = instaseis.Source(latitude=lat, longitude=lon, depth_in_m=depth,
                              m_rr=event.MomentTensor.m_rr,
                              m_tt=event.MomentTensor.m_tt,
                              m_pp=event.MomentTensor.m_pp,
                              m_rt=event.MomentTensor.m_rt,
                              m_rp=event.MomentTensor.m_rp,
                              m_tp=event.MomentTensor.m_tp,
                              origin_time=tofe
                              )

    stream = Stream()
    tmp = []
    for station in network:
        rec = instaseis.Receiver(latitude=str(station.latitude),
                                 longitude=str(station.longitude),
                                 network=str(network.code),
                                 station=str(station.code))
        tmp.append(db.get_seismograms(source=source, receiver=rec))

    for x in tmp:
        stream += x

    return stream


def request_gcmt(starttime, endtime, minmagnitude=None, mindepth=None,
                 maxdepth=None, minlatitude=None, maxlatitude=None,
                 minlongitude=None, maxlongitude=None):
    from mechanize import Browser
    import re

    """
    Description
    I am using mechanize. My attempt is just preliminary, for the current
    globalcmt.org site. It is possible to store Moment Tensor information
    in the catalog file.
    """

    # Split numbers and text
    r = re.compile("([a-zA-Z]+)([0-9]+)")
    br = Browser()
    br.open('http://www.globalcmt.org/CMTsearch.html')
    # Site has just one form
    br.select_form(nr=0)

    br.form['yr'] = str(starttime.year)
    br.form['mo'] = str(starttime.month)
    br.form['day'] = str(starttime.day)
    br.form['oyr'] = str(endtime.year)
    br.form['omo'] = str(endtime.month)
    br.form['oday'] = str(endtime.day)
    br.form['list'] = ['4']
    br.form['itype'] = ['ymd']
    br.form['otype'] = ['ymd']

    if minmagnitude:
        br.form['lmw'] = str(minmagnitude)
    if minlatitude:
        br.form['llat'] = str(minlatitude)
    if maxlatitude:
        br.form['ulat'] = str(maxlatitude)
    if minlongitude:
        br.form['llon'] = str(minlongitude)
    if maxlongitude:
        br.form['ulon'] = str(maxlongitude)
    if mindepth:
        br.form['lhd'] = str(mindepth)
    if maxdepth:
        br.form['uhd'] = str(maxdepth)

    print("Submitting parameters to globalcmt.org ")
    req = br.submit()
    print("Retrieving data, creating catalog.")

    data = []
    for line in req:
        data.append(line)

    data_chunked = _chunking_list(keyword='\n', list=data)
    origins = []
    magnitudes = []
    tensor = []

    for line in data_chunked:
        for element in line:
            if 'event name' in element:
                try:
                    org = line[1].split()
                    year = int(r.match(org[0]).groups()[1])
                    mon = int(org[1])
                    day = int(org[2])
                    hour = int(org[3])
                    minute = int(org[4])
                    sec_temp = int(org[5].split('.')[0])
                    msec_temp = int(org[5].split('.')[1])

                except:
                    org = line[1].split()
                    year = int(org[1])
                    mon = int(org[2])
                    day = int(org[3])
                    hour = int(org[4])
                    minute = int(org[5])
                    sec_temp = int(org[6].split('.')[0])
                    msec_temp = int(org[6].split('.')[1])

                origins_temp = UTCDateTime(year, mon, day, hour, minute,
                                           sec_temp, msec_temp)
                # adding time shift located in line[3]
                origin = origins_temp + float(line[3].split()[2])
                magnitude = float(line[1].split()[10])
                latitude = float(line[5].split()[1])
                longitude = float(line[6].split()[1])
                depth = 1000. * float(line[7].split()[1])
                m_rr = float(line[8].split()[1])
                m_tt = float(line[9].split()[1])
                m_pp = float(line[10].split()[1])
                m_rt = float(line[11].split()[1])
                m_rp = float(line[12].split()[1])
                m_tp = float(line[13].split()[1])

                magnitudes.append(("Mw", magnitude))
                origins.append((latitude, longitude, depth, origin))
                tensor.append((m_rr, m_tt, m_pp, m_rt, m_rp, m_tp))

    cat = Catalog()

    for mag, org, ten in zip(magnitudes, origins, tensor):
        # Create magnitude object.
        magnitude = Magnitude()
        magnitude.magnitude_type = mag[0]
        magnitude.mag = mag[1]
        # Write origin object.
        origin = Origin()
        origin.latitude = org[0]
        origin.longitude = org[1]
        origin.depth = org[2]
        origin.time = org[3]
        # Create event object and append to catalog object.
        event = Event()
        event.magnitudes.append(magnitude)
        event.origins.append(origin)

        event.MomentTensor = MomentTensor()
        event.MomentTensor.m_rr = ten[0]
        event.MomentTensor.m_tt = ten[1]
        event.MomentTensor.m_pp = ten[2]
        event.MomentTensor.m_rt = ten[3]
        event.MomentTensor.m_rp = ten[4]
        event.MomentTensor.m_tp = ten[5]

        cat.append(event)

    return cat


def _chunking_list(keyword, list):
    """
    taken from
    http://stackoverflow.com/questions/19575702/pythonhow-to-split-
    file-into-chunks-by-the-occurrence-of-the-header-word
    """
    chunks = []
    current_chunk = []

    for line in list:

        if line.startswith(keyword) and current_chunk:
            chunks.append(current_chunk[:])
            current_chunk = []

        current_chunk.append(line)
    chunks.append(current_chunk)

    return chunks
