#!/usr/bin/env python
from collections import defaultdict
import tempfile
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from obspy import UTCDateTime, Stream
from obspy.core import AttribDict
from obspy.core.util.geodetics import locations2degrees, gps2DistAzimuth, \
    kilometer2degrees
from obspy.taup import getTravelTimes
import scipy.interpolate as spi
import scipy as sp
import matplotlib.cm as cm
from obspy.signal.util import utlGeoKm,nextpow2,utlLonLat
import ctypes as C
from obspy.core import Stream
import math
import warnings
from scipy.integrate import cumtrapz
from obspy.core import Stream
from obspy.signal.headers import clibsignal
from obspy.signal.invsim import cosTaper
from obspy.taup import TauPyModel
from obspy.taup import getTravelTimes
from mpl_toolkits.basemap import Basemap

KM_PER_DEG = 111.1949
os.system('clear')  # clear screen
model =  TauPyModel(model="ak135")

def vespagram(stream, ev, inv, method, scale, nthroot=4,
              static3D=False, vel_corr=4.8, sl=(0.0, 10.0, 0.1),
              plot_trace=True, phase_shift=0, phase = ["PP"],
              plot_max_beam_trace=False, save_fig=False, plot_circle_path=False,
              plot_stations=False,vespagram_show=True,vespa_iter=0,
              name_vespa=True, static_correction = False):
    """
    :param stream: Waveforms for the array processing.
    :type stream: :class:`obspy.core.stream.Stream`
    :param inventory: Station metadata for waveforms
    :type inventory: :class:`obspy.station.inventory.Inventory`
    :param method: Method used for the array analysis
        (one of "DLS": Delay and Sum, "PWS": Phase Weighted Stack).
    :type method: str
    :param frqlow: Low corner of frequency range for array analysis
    :type frqlow: float
    :param frqhigh: High corner of frequency range for array analysis
    :type frqhigh: float
    :param baz: pre-defined (theoretical or calculated) backazimuth used for calculation
    :type baz_plot: float
    :param scale: scale for plotting
    :type scale: float
    :param nthroot: estimating the nthroot for calculation of the beam
    :type nthroot: int
    :param filter: Whether to bandpass data to selected frequency range
    :type filter: bool
    :param static3D: static correction of topography using `vel_corr` as
        velocity (slow!)
    :type static3D: bool
    :param vel_corr: Correction velocity for static topography correction in
        km/s.
    :type vel_corr: float
    :param sl: Min/Max and stepwidth slowness for analysis
    :type sl: (float, float,float)
    :param plot_trace: if True plot the vespagram as wiggle plot, if False as density map
    :phase_shift: time shifting for ploting theoretical taup-traveltimes phases
    """
    
    # choose the maximun of the starting times
    starttime = max([tr.stats.starttime for tr in stream])
    # choose the ninumun of the ending times
    endtime = min([tr.stats.endtime for tr in stream])
    # keep only the shortest window lenght of the whole seismograms
    stream.trim(starttime,endtime)
    # remove the trend
    #stream.detrend('simple')

    # print in the screen
    # print(starttime)
    # print(endtime)
    # closeInput = raw_input("Press ENTER to exit")
    
    org = ev.preferred_origin() or ev.origins[0]
    ev_lat = org.latitude
    ev_lon = org.longitude
    ev_depth = org.depth/1000.  # in km
    ev_otime = org.time

    # print(org)
    # print(ev_lat)
    # print(ev_lon)
    # print(ev_depth)
    # print(ev_otime)
    # closeInput = raw_input("Press ENTER to exit")

    sll, slm, sls = sl
    # print sl
    # print sll
    # print slm
    # print sls
    # closeInput = raw_input("Press ENTER to exit")
    
    sll /= KM_PER_DEG
    slm /= KM_PER_DEG
    sls /= KM_PER_DEG
    center_lon = 0.
    center_lat = 0.
    center_elv = 0.
    seismo = stream
    #seismo.attach_response(inv)
    #seismo.merge()
    sz = Stream()
    i = 0
    for tr in seismo:
        for station in inv[0].stations:
            if tr.stats.station == station.code:
                #print(tr.stats.station)
                #print(station.code)
                tr.stats.coordinates = \
                    AttribDict({'latitude': station.latitude,
                                'longitude': station.longitude,
                                'elevation': station.elevation,
                                'name': station.code})
                center_lon += station.longitude
                center_lat += station.latitude
                center_elv += station.elevation
                i += 1
                # print(station.network)
        sz.append(tr)

    for network in inv:
	    array_name = network.code
    array_name = array_name.encode('utf8')

    # print(array_name)
    # print(type(array_name))

    if i == 0:
        msg = 'Stations can not be found!'
        raise ValueError(msg)

    #sz.plot()
    #stream.plot()

    center_lon /= float(i)
    center_lat /= float(i)
    center_elv /= float(i)
 
    # calculate the back azimuth
    great_cricle_dist, baz, az2 = gps2DistAzimuth(center_lat,center_lon,ev_lat,ev_lon)
    great_circle_dist_deg = great_cricle_dist/ (1000*KM_PER_DEG)
    # print("baz")
    # print(baz)
    # print("az2")
    # print(az2)

    if plot_circle_path:
       plot_great_circle_path(ev_lon,ev_lat,ev_depth,center_lon,center_lat,baz,great_cricle_dist,model)
              
    if plot_stations:
       plot_array_stations(sz,center_lon,center_lat,array_name)

    # print(center_lon)
    # print(center_lat)
    # print(center_elv)
    
    #closeInput = raw_input("Press ENTER to exit")

    # trim it again?!?!
    stt = starttime
    e = endtime
    nut = 0.
    max_amp = 0.
    # sz.trim(stt, e)
    # sz.detrend('simple')
    # print sz

    # compute the number of traces in the vespagram
    nbeam = int((slm - sll)/sls + 0.5) + 1
    # print("nbeam")
    # print(nbeam)

    # arguments to compute the vespagram
    kwargs = dict(
        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll=sll, slm=slm, sls=sls, baz=baz, stime=starttime, etime=endtime, source_depth = ev_depth, 
        distance = great_circle_dist_deg, static_correction = static_correction, phase = phase, method=method,
        nthroot=nthroot, correct_3dplane=False, static_3D=static3D,vel_cor=vel_corr)
    
    # date to compute total time in routine
    start = UTCDateTime()

    # compute the vespagram
    slow, beams, max_beam, beam_max, mini, maxi = vespagram_baz(sz, **kwargs)
    # print slow
    # print total time in routine
    print "Total time in routine: %.2f\n" % (UTCDateTime() - start)
    
    # Plot the seismograms
    # sampling rate
    df = sz[0].stats.sampling_rate
    npts = len(beams[0])
    #print("npts")
    #print(npts)
    # time vector
    T = np.arange(0, npts/df, 1/df)
    # reconvert slowness to degrees
    sll *= KM_PER_DEG
    slm *= KM_PER_DEG
    sls *= KM_PER_DEG
    # slowness vector
    slow = np.arange(sll, slm, sls)
    max_amp = np.max(beams[:, :])
    #min_amp = np.min(beams[:, :])
    scale *= sls
    
    # initialize the figure
    fig = plt.figure(figsize=(12, 8))
    
#    print("sl")
#    print(sl)
#    print("sll")
#    print(sll)
#    print("slm")
#    print(slm)
#    print("sls")
#    print(sls)

    # get taup points for ploting the phases
    phase_name_info,phase_slowness_info,phase_time_info = get_taupy_points(center_lat,center_lon,ev_lat,ev_lon,ev_depth, \
                                                                    starttime,endtime,mini,maxi,ev_otime,phase_shift,sll,slm)

    # print(phase_name_info)
    # print(phase_slowness_info)
    # print(phase_time_info)

    if plot_trace:
        ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        for i in xrange(nbeam):
            if plot_max_beam_trace:
                if i == max_beam:
                    ax1.plot(T, sll + scale*beams[i]/max_amp + i*sls, 'r',zorder=1)
                else:
                    ax1.plot(T, sll + scale*beams[i]/max_amp + i*sls, 'k',zorder=-1)
            else:    
                ax1.plot(T, sll + scale*beams[i]/max_amp + i*sls, 'k',zorder=-1)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('slowness (s/deg)')
        ax1.set_xlim(T[0], T[-1])
        data_minmax = ax1.yaxis.get_data_interval()
        minmax = [min(slow[0], data_minmax[0]), max(slow[-1], data_minmax[1])]
        ax1.set_ylim(*minmax)
        # plot the phase info
        ax1.scatter(phase_time_info,phase_slowness_info,s=2000,marker=u'|',lw=2,color='g')
        for i, txt in enumerate(phase_name_info):
          ax1.annotate(txt,(phase_time_info[i],phase_slowness_info[i]),fontsize=18,color='r')
    #####
    else:
        #step = (max_amp - min_amp)/100.
        #level = np.arange(min_amp, max_amp, step)
        #beams = beams.transpose()
        cmap = cm.hot_r
        #cmap = cm.rainbow

        ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        #ax1.contour(slow,T,beams,level)
        #extent = (slow[0], slow[-1], \
        #               T[0], T[-1])
        extent = (T[0], T[-1], slow[0] - sls * 0.5, slow[-1] + sls * 0.5)

        ax1.set_ylabel('slowness (s/deg)')
        ax1.set_xlabel('T (s)')
        beams = np.flipud(beams)
        ax1.imshow(beams, cmap=cmap, interpolation="nearest",extent=extent, aspect='auto')
        # plot the phase info
        ax1.scatter(phase_time_info,phase_slowness_info,s=2000,marker=u'|',lw=2,color='g')
        for i, txt in enumerate(phase_name_info):
          ax1.annotate(txt,(phase_time_info[i],phase_slowness_info[i]),fontsize=18,color='r')

    ####
    result = "BAZ: %.2f Time: %s" % (baz, stt)
    ax1.set_title(result)

    if vespagram_show:
        plt.show()

    # save the figure
    if save_fig:
        fig_name = 'vespagram_%s.pdf' % vespa_iter 
        plt.savefig(fig_name, format='pdf', dpi=None)
    
    return slow, beams, max_beam, beam_max

def vespagram_baz(stream, sll, slm, sls, baz, stime, etime, source_depth, distance, static_correction, 
                  phase, verbose=False,coordsys='lonlat',timestamp='mlabday', method="DLS",nthroot=1,
                  store=None,correct_3dplane=False,static_3D=False,
                  vel_cor=4.):

    """
    Estimating the azimuth or slowness vespagram

    :param stream: Stream object, the trace.stats dict like class must
        contain a obspy.core.util.AttribDict with 'latitude', 'longitude' (in
        degrees) and 'elevation' (in km), or 'x', 'y', 'elevation' (in km)
        items/attributes. See param coordsys
    :type sll: Float
    :param sll: slowness  min (lower)
    :type slm: Float
    :param slm: slowness max
    :type sls: Float
    :param sls: slowness step
    :type baz: Float
    :param baz: given backazimuth
    :type stime: UTCDateTime
    :param stime: Starttime of interest
    :type etime: UTCDateTime
    :param etime: Endtime of interest
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :type timestamp: string
    :param timestamp: valid values: 'julsec' and 'mlabday'; 'julsec' returns
        the timestamp in secons since 1970-01-01T00:00:00, 'mlabday'
        returns the timestamp in days (decimals represent hours, minutes
        and seconds) since '0001-01-01T00:00:00' as needed for matplotlib
        date plotting (see e.g. matplotlibs num2date)
    :type store: function
    :param store: A custom function which gets called on each iteration. It is
        called with the relative power map and the time offset as first and
        second arguments and the iteration number as third argument. Useful for
        storing or plotting the map for each iteration. For this purpose the
        dump function of this module can be used.
    :return: numpy.ndarray of beams with different slownesses
    """
    # print(static_correction)
    # compare th original trace with the traces used for computing the vespagram
    #stream.plot()
    #closeInput = raw_input("Press ENTER to exit")

    # check that sampling rates do not vary
    # sampling rate
    fs = stream[0].stats.sampling_rate

    # number of stations
    nstat = len(stream)
    # print("nstat")
    # print(nstat)

    if len(stream) != len(stream.select(sampling_rate=fs)):
        msg = 'in sonic sampling rates of traces in stream are not equal'
        raise ValueError(msg)

    # maximum lenght of the seismogram that can be used
    ndat = int((etime - stime)*fs)
    #print ndat
    
    # number of beam traces
    nbeams = int(((slm - sll) / sls + 0.5) + 1)
    # print("nbeams")
    # print nbeams
    # closeInput = raw_input("Press ENTER to exit")

    geometry = get_geometry(stream,coordsys=coordsys,verbose=verbose)
    #stream.plot()
    #print("geometry:")
    # print(geometry)
    if verbose:
        print("geometry: %s") % geometry
        # print(geometry)
        print("Stream contains the following traces: %s") % stream
        # print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    time_shift_table = get_timeshift_baz(geometry, sll, slm, sls, baz, source_depth, 
                                         distance, phase, vel_cor=vel_cor, 
                                         static_3D=static_3D, model=model, 
                                         static_correction=static_correction)

    # calculate the overlaping lenght of the traces
    mini = np.min(time_shift_table[:, :])
    maxi = np.max(time_shift_table[:, :])
    # print("mini")
    # print(mini)
    # print("maxi")
    # print(maxi)
    #print("stime")
    #print(stime)
    #print("etime")
    #print(etime)
    spoint, _epoint = get_spoint(stream, (stime - mini), (etime - maxi))
    
    # print("spoint")
    # print(spoint)
    # print("epoint")
    # print(_epoint)

    # recalculate the maximum possible trace length
    ndat = int(((etime - maxi) - (stime - mini)) * fs) + 1
    seconds = (etime - maxi) - (stime - mini)
    # print("ndat recalculated")
    # print(ndat)
    print("Total seconds of the vespagram: %.2f") % seconds
    # print(seconds)
    number_total_points = len(stream[0]) 
    biggest_left = int(np.abs(mini) * fs) + 1
    biggest_right = number_total_points - int(np.abs(maxi) * fs) - 1
    efective_trace_lenght = biggest_right - biggest_left 

    # vespagram matrix
    beams = np.zeros((nbeams, ndat), dtype='f8')
    
    # initialize variabes
    max_beam = 0.
    slow = 0.
    beam_max = 0.   
    
    # print("efective_trace_lenght")
    # print(efective_trace_lenght)
    # print("number total efective points")
    # print(number_total_points)
    # print("biggest_left")
    # print(biggest_left)
    # print("ndat")
    # print(ndat)
    # print("sampling_rate")
    # print(fs)
    # print("biggest_right")
    # print(biggest_right)
    # print("stime")
    # print(stime)
    # print("mini")
    # print(mini)
    # print("maxi")
    # print(maxi)

    for x in xrange(nbeams):
        singlet = 0.
        if method == 'DLS':
            for i in xrange(nstat):
                # check the nthroot used
                #print("nthroot", nthroot)
                
                # correct way to do it!
                starting_point =  biggest_left + int(time_shift_table[i, x] * fs)
                ending_point = starting_point + ndat
                
                # original implementation
                s = spoint[i] + int(time_shift_table[i, x]*fs + 0.5)
                # shifted = stream[i].data[s: s + ndat]
                
                # our implementation
                shifted = stream[i].data[starting_point : ending_point]
                #print(shifted)
                
                singlet += 1. / nstat * np.sum(shifted * shifted)
                
                # compute the vespagram
                beams[x] += 1. / nstat * np.power(np.abs(shifted), 1. / nthroot) * shifted / np.abs(shifted)
            
            beams[x] = np.power(np.abs(beams[x]), nthroot) * beams[x] / np.abs(beams[x])
            bs = np.sum(beams[x]*beams[x])
            bs /= singlet
            #bs = np.abs(np.max(beams[x]))
            if bs > max_beam:
                max_beam = bs
                beam_max = x
                slow = np.abs(sll + x * sls)
                if (slow) < 1e-8:
                    slow = 1e-8
                    
        if method == 'PWS':
           stack = np.zeros(ndat, dtype='c8')
           phase = np.zeros(ndat, dtype='f8')
           coh = np.zeros(ndat, dtype='f8')
           for i in xrange(nstat):
               s = spoint[i] + int(time_shift_table[i, x] * fs +0.5)
               try:
                  shifted = sp.signal.hilbert(stream[i].data[s : s + ndat])
               except IndexError:
                  break
               phase = np.arctan2(shifted.imag, shifted.real)
               stack.real += np.cos(phase)
               stack.imag += np.sin(phase)
           coh = 1. / nstat * np.abs(stack)
           for i in xrange(nstat):
               s = spoint[i]+int(time_shift_table[i, x] * fs + 0.5)
               shifted = stream[i].data[s: s + ndat]
               singlet += 1. / nstat * np.sum(shifted * shifted)
               beams[x] += 1. / nstat * shifted * np.power(coh, nthroot)
           bs = np.sum(beams[x]*beams[x])
           bs = bs / singlet
           if bs > max_beam:
              max_beam = bs
              beam_max = x
              slow = np.abs(sll + x * sls)
              if (slow) < 1e-8:
                  slow = 1e-8


    return(slow, beams, beam_max, max_beam, mini, maxi)

def get_spoint(stream, stime, etime):
    """
    Calculates start and end offsets relative to stime and etime for each
    trace in stream in samples.

    :type stime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param stime: Start time
    :type etime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param etime: End time
    :returns: start and end sample offset arrays
    """
    spoint = np.empty(len(stream), dtype=np.int32, order="C")
    epoint = np.empty(len(stream), dtype=np.int32, order="C")
    for i, tr in enumerate(stream):
        if tr.stats.starttime > stime:
            msg = "Specified stime %s is smaller than starttime %s in stream"
            raise ValueError(msg % (stime, tr.stats.starttime))
        if tr.stats.endtime < etime:
            msg = "Specified etime %s is bigger than endtime %s in stream"
            raise ValueError(msg % (etime, tr.stats.endtime))
        # now we have to adjust to the beginning of real start time
        spoint[i] = int((stime - tr.stats.starttime) *
                        tr.stats.sampling_rate + .5)
        epoint[i] = int((tr.stats.endtime - etime) *
                        tr.stats.sampling_rate + .5)
    return spoint, epoint

def get_timeshift_baz(geometry, sll, slm, sls, baze, source_depth, distance, phase, vel_cor=4.,static_3D=False,
                      model=model, static_correction=False):
    """
    Returns timeshift table for given array geometry and a pre-definded
    backazimut

    :param geometry: Nested list containing the arrays geometry, as returned by
            get_group_geometry
    :param sll_x: slowness x min (lower)
    :param slm_y: slowness x max (lower)
    :param sl_s: slowness step
    :param baze:  backazimuth applied
    :param vel_cor: correction velocity (upper layer) in km/s
    :param static_3D: a correction of the station height is applied using
        vel_cor the correction is done according to the formula:
        t = rxy*s - rz*cos(inc)/vel_cor
        where inc is defined by inv = asin(vel_cor*slow)
    """
    nstat = len(geometry)  # last index are center coordinates
    baz = math.pi * baze / 180. # BAZ converted from degrees to radiants
    nbeams = int((slm - sll) / sls + 0.5) + 1

    # print(static_correction)
    # check the correct values
    # print(slm*KM_PER_DEG)
    # print(sll*KM_PER_DEG)
    # print(nbeams)

    # time shift table is given by the number of staions and number of beam traces
    time_shift_tbl = np.empty((nstat, nbeams), dtype="float32")

    arrivals = model.get_travel_times(source_depth, distance, phase_list = phase)
    inc_ang = arrivals[0].incident_angle
    inc_ang_rad = inc_ang * np.pi/180
    print('The incidence angle is %.2f deg and %.2f rad') % (inc_ang, inc_ang_rad)

    for k in xrange(nbeams):
        sx = sll + k * sls
        #print(sx)
        if vel_cor*sx < 1.:
            # print("Im in velocity correction - timeshift!!!")
            inc = np.arcsin(vel_cor*sx)
        else:
            inc = np.pi/2.
        
        if static_correction:
            time_shift_tbl[:, k] = - sx * (geometry[:, 0] * math.sin(baz) + geometry[:, 1] * math.cos(baz))  \
                                   + sx * geometry[:, 2]/1000 * 1./np.tan(inc_ang_rad)
            # print(1./np.tan(inc_ang_rad))
            # print(time_shift_tbl)

        else:
            # time shift table matrix    
            time_shift_tbl[:, k] = - sx * (geometry[:, 0] * math.sin(baz) + geometry[:, 1] * math.cos(baz))
            # print(time_shift_tbl)

            if static_3D:
	          time_shift_tbl[:, k] += geometry[:, 2] * np.cos(inc) / vel_cor
    
    #print("TIME SHIFT TABLE")
    # print(time_shift_tbl)

    return time_shift_tbl

def get_geometry(stream,coordsys='lonlat',return_center=True,verbose=False):
    """
    Method to calculate the array geometry and the center coordinates in km

    :param stream: Stream object, the trace.stats dict like class must
        contain an :class:`~obspy.core.util.attribdict.AttribDict` with
        'latitude', 'longitude' (in degrees) and 'elevation' (in km), or 'x',
        'y', 'elevation' (in km) items/attributes. See param ``coordsys``
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :param return_center: Returns the center coordinates as extra tuple
    :return: Returns the geometry of the stations as 2d :class:`numpy.ndarray`
            The first dimension are the station indexes with the same order
            as the traces in the stream object. The second index are the
            values of [lat, lon, elev] in km
            last index contains center [lat, lon, elev] in degrees and km if
            return_center is true
    """
    nstat = len(stream)
    center_lat = 0.
    center_lon = 0.
    center_h = 0.
    geometry = np.empty((nstat, 3))

    if isinstance(stream, Stream):
        for i, tr in enumerate(stream):
            if coordsys == 'lonlat':
                geometry[i, 0] = tr.stats.coordinates.longitude
                geometry[i, 1] = tr.stats.coordinates.latitude
                geometry[i, 2] = tr.stats.coordinates.elevation
                #print("latitude",geometry[i, 1])
                #print("longitude",geometry[i, 0])
            elif coordsys == 'xy':
                geometry[i, 0] = tr.stats.coordinates.x
                geometry[i, 1] = tr.stats.coordinates.y
                geometry[i, 2] = tr.stats.coordinates.elevation
    elif isinstance(stream, np.ndarray):
        print("Im not here!")
        geometry = stream.copy()
    else:
        raise TypeError('only Stream or numpy.ndarray allowed')

    if verbose:
        print("coordsys = " + coordsys)

    if coordsys == 'lonlat':
        center_lon = geometry[:, 0].mean()
        center_lat = geometry[:, 1].mean()
        # print(center_lon)
        # print(center_lat)
        center_h = geometry[:, 2].mean()
        
        for i in np.arange(nstat):
            x, y = utlGeoKm(center_lon, center_lat, geometry[i, 0],geometry[i, 1])  # original: utlGeoKm
            geometry[i, 0] = x
            geometry[i, 1] = y
            geometry[i, 2] -= center_h
    elif coordsys == 'xy':
        geometry[:, 0] -= geometry[:, 0].mean()
        geometry[:, 1] -= geometry[:, 1].mean()
        geometry[:, 2] -= geometry[:, 2].mean()
    else:
        raise ValueError("Coordsys must be one of 'lonlat', 'xy'")
    
    if return_center:
        return np.c_[geometry.T,np.array((center_lon, center_lat, center_h))].T
    else:
        return geometry


def get_taupy_points(center_lat,center_lon,ev_lat,ev_lon,ev_depth,
                     stime,etime,mini,maxi,ev_otime,phase_shift,sll,slm):
  
  distance = locations2degrees(center_lat,center_lon,ev_lat,ev_lon)
  #print(distance)

  model =  TauPyModel(model="ak135")
  arrivals = model.get_pierce_points(ev_depth,distance)
  #arrivals = earthmodel.get_pierce_points(ev_depth,distance,phase_list=('PP','P^410P'))  

  # compute the vespagram window
  start_vespa = stime - mini
  end_vespa = etime - maxi

  # compare the arrival times with the time window
  count = 0
  k = 0
  phase_name_info = []
  phase_slowness_info = []
  phase_time_info = []

  for i_elem in arrivals:
    #print(i_elem)
    dummy_phase = arrivals[count]
    #print(dummy_phase)
    # phase time in seconds
    taup_phase_time = dummy_phase.time
    #print(taup_phase_time)
    # slowness of the phase
    taup_phase_slowness = dummy_phase.ray_param_sec_degree
    # compute the UTC travel phase time 
    taup_phase_time2 = ev_otime + taup_phase_time + phase_shift
    
    # print(start_vespa)
    # print(end_vespa)
    # print(taup_phase_time2)

    if start_vespa <= taup_phase_time2 <= end_vespa: # time window    
      if sll <= taup_phase_slowness <= slm: # slowness window
      
        # seconds inside the vespagram
        taup_mark = taup_phase_time2 - start_vespa
        # store the information
        phase_name_info.append(dummy_phase.name)
        phase_slowness_info.append(dummy_phase.ray_param_sec_degree)
        phase_time_info.append(taup_mark)
        #print(phases_info[k])
        k += 1

    count += 1  
    
  #print(phase_name_info)
  
  phase_slowness_info = np.array(phase_slowness_info)
  phase_time_info = np.array(phase_time_info)

  return phase_name_info, phase_slowness_info, phase_time_info


def plot_great_circle_path(ev_lon,ev_lat,ev_depth,center_lon,center_lat,baz,great_cricle_dist, model):

  plt.figure(num=1,figsize=(17,10),dpi=100) # define plot size in inches (width, height) & resolution(DPI)

  distance = locations2degrees(center_lat,center_lon,ev_lat,ev_lon)
  #print(distance)
  # earthmodel =  TauPyModel(model="ak135")
  arrivals = model.get_pierce_points(ev_depth,distance,phase_list=["PP"])
  #print(arrivals)
  arrival = arrivals[0]
  pierce_info = arrival.pierce
  #print(pierce_info)
  max_index = 0.
  count = 0.
  max_val = 0.
  for i_index in pierce_info:
    #print(i_index)
    count += 1
    if i_index[3] > max_val: 
      max_val = i_index[3] 
      max_index = count - 1
      #print(max_index)
  
  #print(max_index)
  bounce_vect = pierce_info[max_index]
  bounce_dist = bounce_vect[2] / 0.017455053237912375 # convert from radians to degrees
  # print("bounce_dist")
  # print(bounce_dist)

  # print("ev_lat")
  # print(ev_lat)
  # print("ev_lon")
  # print(ev_lon)
  # print("center_lon")
  # print(center_lon)
  # print("center_lat")
  # print(center_lat)
  # print("backazimuth")
  # print(baz)

  # bounce point approximation
  bounce_lat_appx, bounce_lon_appx = midpoint(ev_lat,ev_lon,center_lat,center_lon)

  # putting everything into a vector
  lons = [ev_lon, center_lon]
  lats = [ev_lat, center_lat]

  # trick - the basemap functions does not like the arguments that math gives
  resolution = 0.0001
  bounce_lon_appx = np.round(bounce_lon_appx/resolution)*resolution
  bounce_lat_appx = np.round(bounce_lat_appx/resolution)*resolution

  # print(bounce_lon_appx)
  # print(bounce_lat_appx)

  # plot results
  map = Basemap(projection='hammer',lon_0=bounce_lon_appx,lat_0=bounce_lat_appx,resolution='c')
  map.drawcoastlines()
  # map.fillcontinents()
  # map.drawmapboundary()
  map.fillcontinents(color='#cc9966',lake_color='#99ffff')
  map.drawmapboundary(fill_color='#99ffff')
  great_cricle_dist_deg = great_cricle_dist/ (1000*KM_PER_DEG)
  #plt.title('Bounce point plot',fontsize=26)
  msg = 'Great circle distance is %.2f deg' % great_cricle_dist_deg
  plt.title(msg)

  # draw great circle path
  map.drawgreatcircle(ev_lon,ev_lat,center_lon,center_lat,linewidth=3,color='g')

  # plot event
  x, y = map(ev_lon,ev_lat)
  map.scatter(x, y, 200, marker='*',color='k',zorder=10)
  
  # plot receiver
  x, y = map(center_lon,center_lat)
  map.scatter(x, y, 100, marker='^',color='k',zorder=10)
  
  # plot the bounce point approximated
  x, y = map(bounce_lon_appx,bounce_lat_appx)
  map.scatter(x, y, 100, marker='D',color='k',zorder=10)


def midpoint(lat1,lon1,lat2,lon2):
  
  # compute the mid point between two coordinates in degrees
  assert -90 <= lat1 <= 90
  assert -90 <= lat2 <= 90
  assert -180 <= lon1 <= 180
  assert -180 <= lon2 <= 180
  lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))

  dlon = lon2 - lon1
  dx = math.cos(lat2) * math.cos(dlon)
  dy = math.cos(lat2) * math.sin(dlon)
  lat3 = math.atan2(math.sin(lat1) + math.sin(lat2), math.sqrt((math.cos(lat1) + dx) * (math.cos(lat1) + dx) + dy * dy))
  lon3 = lon1 + math.atan2(dy, math.cos(lat1) + dx)
  
  return(math.degrees(lat3), math.degrees(lon3))

def align_phases(stream, event, inventory, phase_name, method="simple"):
    """
    Aligns the waveforms with the theoretical travel times for some phase. The
    theoretical travel times are calculated with obspy.taup.

    :param stream: Waveforms for the array processing.
    :type stream: :class:`obspy.core.stream.Stream`
    :param event: The event for which to calculate phases.
    :type event: :class:`obspy.core.event.Event`
    :param inventory: Station metadata.
    :type inventory: :class:`obspy.station.inventory.Inventory`
    :param phase_name: The name of the phase you want to align. Must be
        contained in all traces. Otherwise the behaviour is undefined.
    :type phase_name: str
    :param method: Method is either `simple` or `fft`. Simple will just shift
     the starttime of Trace, while 'fft' will do the shift in the frequency
     domain. Defaults to `simple`.
    :type method: str
    """
    method = method.lower()
    if method not in ['simple', 'fft']:
        msg = "method must be 'simple' or 'fft'"
        raise ValueError(msg)

    stream = stream.copy()
    attach_coordinates_to_traces(stream, inventory, event)

    stream.traces = sorted(stream.traces, key=lambda x: x.stats.distance)[::-1]

    tr_1 = stream[-1]
    tt_1 = getTravelTimes(tr_1.stats.distance,
                          event.origins[0].depth / 1000.0,
                          "ak135")
    
    cont = 0.
    for tt in tt_1:
        if tt["phase_name"] != phase_name:
            continue
        if tt["phase_name"] == phase_name:
            cont = 1.
        tt_1 = tt["time"]
        break
   
    if cont == 0:    
      msg = "The selected phase is not present in your seismograms!!!"
      raise ValueError(msg)
        
    for tr in stream:
        tt = getTravelTimes(tr.stats.distance,
                            event.origins[0].depth / 1000.0,
                            "ak135")
        for t in tt:
            if t["phase_name"] != phase_name:
                continue
            tt = t["time"]
            break
        if method == "simple":
            tr.stats.starttime -= (tt - tt_1)
        else:
            shifttrace_freq(Stream(traces=[tr]), [- ((tt - tt_1))])
    return stream

def shifttrace_freq(stream, t_shift):
    if isinstance(stream, Stream):
        for i, tr in enumerate(stream):
            ndat = tr.stats.npts
            samp = tr.stats.sampling_rate
            nfft = nextpow2(ndat)
            nfft *= 2
            tr1 = np.fft.rfft(tr.data, nfft)
            for k in xrange(0, nfft / 2):
                tr1[k] *= np.complex(
                    np.cos((t_shift[i] * samp) * (k / float(nfft))
                           * 2. * np.pi),
                    -np.sin((t_shift[i] * samp) *
                            (k / float(nfft)) * 2. * np.pi))

            tr1 = np.fft.irfft(tr1, nfft)
            tr.data = tr1[0:ndat]

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
            # coords["%s.%s" % (network.code, station.code)] = \
            coords[".%s" % (station.code)] = \
                {"latitude": station.latitude,
                 "longitude": station.longitude,
                 "elevation": station.elevation}

    # Calculate the event-station distances.
    if event:
        event_lat = event.origins[0].latitude
        event_lng = event.origins[0].longitude
        for value in coords.values():
            value["distance"] = locations2degrees(
                value["latitude"], value["longitude"], event_lat, event_lng)

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


def show_distance_plot(stream, event, inventory, starttime, endtime,
                       plot_travel_times=True):
    """
    Plots distance dependent seismogramm sections.

    :param stream: The waveforms.
    :type stream: :class:`obspy.core.stream.Stream`
    :param event: The event.
    :type event: :class:`obspy.core.event.Event`
    :param inventory: The station information.
    :type inventory: :class:`obspy.station.inventory.Inventory`
    :param starttime: starttime of traces to be plotted
    :type starttime: UTCDateTime
    :param endttime: endttime of traces to be plotted
    :type endttime: UTCDateTime
    :param plot_travel_times: flag whether phases are marked as traveltime plots
     in the section obspy.taup is used to calculate the phases
    :type pot_travel_times: bool
    """
    stream = stream.slice(starttime=starttime, endtime=endtime).copy()
    event_depth_in_km = event.origins[0].depth / 1000.0
    event_time = event.origins[0].time

    attach_coordinates_to_traces(stream, inventory, event=event)

    cm = plt.cm.jet

    stream.traces = sorted(stream.traces, key=lambda x: x.stats.distance)[::-1]

    # One color for each trace.
    colors = [cm(_i) for _i in np.linspace(0, 1, len(stream))]

    # Relative event times.
    times_array = stream[0].times() + (stream[0].stats.starttime - event_time)

    distances = [tr.stats.distance for tr in stream]
    min_distance = min(distances)
    max_distance = max(distances)
    distance_range = max_distance - min_distance
    stream_range = distance_range / 10.0

    # Normalize data and "shift to distance".
    stream.normalize()
    for tr in stream:
        tr.data *= stream_range
        tr.data += tr.stats.distance

    plt.figure(figsize=(18, 10))
    for _i, tr in enumerate(stream):
        plt.plot(times_array, tr.data, label="%s.%s" % (tr.stats.network,
                 tr.stats.station), color=colors[_i])
    plt.grid()
    plt.ylabel("Distance in degree to event")
    plt.xlabel("Time in seconds since event")
    plt.legend()

    dist_min, dist_max = plt.ylim()

    if plot_travel_times:

        distances = defaultdict(list)
        ttimes = defaultdict(list)

        for i in np.linspace(dist_min, dist_max, 1000):
            tts = getTravelTimes(i, event_depth_in_km, "ak135")
            for phase in tts:
                name = phase["phase_name"]
                distances[name].append(i)
                ttimes[name].append(phase["time"])

        for key in distances.iterkeys():
            min_distance = min(distances[key])
            max_distance = max(distances[key])
            min_tt_time = min(ttimes[key])
            max_tt_time = max(ttimes[key])

            if min_tt_time >= times_array[-1] or \
                    max_tt_time <= times_array[0] or \
                    (max_distance - min_distance) < 0.8 * (dist_max - dist_min):
                continue
            ttime = ttimes[key]
            dist = distances[key]
            if max(ttime) > times_array[0] + 0.9 * times_array.ptp():
                continue
            plt.scatter(ttime, dist, s=0.5, zorder=-10, color="black", alpha=0.8)
            plt.text(max(ttime) + 0.005 * times_array.ptp(),
                     dist_max - 0.02 * (dist_max - dist_min),
                     key)

    plt.ylim(dist_min, dist_max)
    plt.xlim(times_array[0], times_array[-1])

    plt.title(event.short_str())

    plt.show()

def plot_array_stations(stream,center_lon,center_lat,array_name):
    
    plt.figure(num=2,figsize=(17,10),dpi=100) # define plot size in inches (width, height) & resolution(DPI)
    
    nstat = len(stream)
    array = np.empty((nstat, 2))
    
    for i, tr in enumerate(stream):
        array[i, 0] = tr.stats.coordinates.longitude
        array[i, 1] = tr.stats.coordinates.latitude
    
    # minimum 
    minlon = np.min(array[:,0])
    maxlon = np.max(array[:,0])
    minlat = np.min(array[:,1])
    maxlat = np.max(array[:,1])

    lon_dif = maxlon - minlon
    lat_dif = maxlat - minlat
    percentage = 0.2

    minlon = minlon - percentage*lon_dif
    maxlon = maxlon + percentage*lon_dif
    minlat = minlat - percentage*lat_dif
    maxlat = maxlat + percentage*lat_dif
        
    # plot results
    map = Basemap(projection='tmerc',llcrnrlon=minlon,llcrnrlat=minlat,
        urcrnrlon=maxlon,urcrnrlat=maxlat,lon_0=center_lon,lat_0=center_lat,
        resolution='i')
    map.drawcoastlines()
    # map.fillcontinents()
    # map.drawmapboundary()
    map.fillcontinents(color='#cc9966',lake_color='#99ffff')
    map.drawmapboundary(fill_color='#99ffff')
    # plt.title('Array plot',fontsize=26)
    plt.title('Array %s' % array_name, fontsize = 24)

    # draw parallels.
    parallels = np.arange(-90.,90.,1.)
    map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    # draw meridians
    meridians = np.arange(-180.,180.,2.)
    map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)

    # plot corrdinates
    x, y = map(array[:, 0],array[:, 1])
    map.scatter(x, y, 200, marker='^',color='k',zorder=10)
    
    # plot the geometrical center
    x, y = map(center_lon,center_lat)
    map.scatter(x, y, 200, marker='o',color='r',zorder=10)

    for i,tr in enumerate(stream):
        x, y = map(array[i,0],array[i,1])
        # print(x,y)
        plt.annotate(tr.stats.coordinates.name,(x*1.02,y*1.02),fontsize=18,color='k')

      