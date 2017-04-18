from __future__ import absolute_import
from collections import defaultdict
import tempfile
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
from matplotlib.ticker import MaxNLocator
from obspy import UTCDateTime, Stream
from obspy.core import AttribDict
from obspy.core.util.geodetics import locations2degrees, gps2DistAzimuth, \
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
#from obspy.clients.fdsn import Client
from obspy.fdsn.client import Client



KM_PER_DEG = 111.1949


def array_transfer_helper(stream, inventory, sx=(-10, 10), sy=(-10, 10),
                          sls=0.5, freqmin=0.1, freqmax=4.0, numfreqs=10,
                          coordsys='lonlat', correct3dplane=False,
                          static3D=False, velcor=4.8):
    """
    Array Response wrapper routine for MESS 2014.

    :param stream: Waveforms for the array processing.
    :type stream: :class:`obspy.core.stream.Stream`
    :param inventory: Station metadata for waveforms
    :type inventory: :class:`obspy.station.inventory.Inventory`
    :param slx: Min/Max slowness for analysis in x direction.
    :type slx: (float, float)
    :param sly: Min/Max slowness for analysis in y direction.
    :type sly: (float, float)
    :param sls: step width of slowness grid
    :type sls: float
    :param frqmin: Low corner of frequency range for array analysis
    :type frqmin: float
    :param frqmax: High corner of frequency range for array analysis
    :type frqmax: float
    :param numfreqs: number of frequency values used for computing array
     transfer function
    :type numfreqs: int
    :param coordsys: defined coordingate system of stations (lonlat or km)
    :type coordsys: string
    :param correct_3dplane: correct for an inclined surface (not used)
    :type correct_3dplane: bool
    :param static_3D: correct topography
    :type static_3D: bool
    :param velcor: velocity used for static_3D correction
    :type velcor: float
    """

    for tr in stream:
        for station in inventory[0].stations:
            if tr.stats.station == station.code:
                tr.stats.coordinates = \
                    AttribDict(dict(latitude=station.latitude,
                               longitude=station.longitude,
                               elevation=station.elevation))
                break

    sllx, slmx = sx
    slly, slmy = sx
    sllx /= KM_PER_DEG
    slmx /= KM_PER_DEG
    slly /= KM_PER_DEG
    slmy /= KM_PER_DEG
    sls = sls/KM_PER_DEG

    stepsfreq = (freqmax - freqmin) / float(numfreqs)
    transff = array_transff_freqslowness(
        stream, (sllx, slmx, slly, slmy), sls, freqmin, freqmax, stepsfreq,
        coordsys=coordsys, correct_3dplane=False, static_3D=static3D,
        vel_cor=velcor)

    sllx *= KM_PER_DEG
    slmx *= KM_PER_DEG
    slly *= KM_PER_DEG
    slmy *= KM_PER_DEG
    sls *= KM_PER_DEG

    slx = np.arange(sllx, slmx+sls, sls)
    sly = np.arange(slly, slmy+sls, sls)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    #ax.pcolormesh(slx, sly, transff.T)
    ax.contour(sly, slx, transff.T, 10)
    ax.set_xlabel('slowness [s/deg]')
    ax.set_ylabel('slowness [s/deg]')
    ax.set_ylim(slx[0], slx[-1])
    ax.set_xlim(sly[0], sly[-1])
    plt.show()

def get_geometry(stream,coordsys='lonlat',return_center=False,verbose=False):
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
            elif coordsys == 'xy':
                geometry[i, 0] = tr.stats.coordinates.x
                geometry[i, 1] = tr.stats.coordinates.y
                geometry[i, 2] = tr.stats.coordinates.elevation
    elif isinstance(stream, np.ndarray):
        geometry = stream.copy()
    else:
        raise TypeError('only Stream or numpy.ndarray allowed')

    if verbose:
        print("coordsys = " + coordsys)

    if coordsys == 'lonlat':
        center_lon = geometry[:, 0].mean()
        center_lat = geometry[:, 1].mean()
        center_h = geometry[:, 2].mean()
        for i in np.arange(nstat):
            x, y = utlGeoKm(center_lon, center_lat, geometry[i, 0],
                               geometry[i, 1])
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
        return np.c_[geometry.T,
                     np.array((center_lon, center_lat, center_h))].T
    else:
        return geometry

def array_transff_freqslowness(stream, inventory, slim, sstep, fmin, fmax, fstep, correct_3dplane=False,
                               static_3D=False, vel_cor=4.):
    """
    Returns array transfer function as a function of slowness difference and
    frequency.

    :type coords: numpy.ndarray
    :param coords: coordinates of stations in longitude and latitude in degrees
        elevation in km, or x, y, z in km
    :type coordsys: string
    :param coordsys: valid values: 'lonlat' and 'xy', choose which coordinates
        to use
    :param slim: either a float to use symmetric limits for slowness
        differences or the tupel (sxmin, sxmax, symin, symax)
    :type fmin: double
    :param fmin: minimum frequency in signal
    :type fmax: double
    :param fmin: maximum frequency in signal
    :type fstep: double
    :param fmin: frequency sample distance
    """
    # geometry = get_geometry(stream, coordsys=coordsys,
    #                         correct_3dplane=correct_3dplane, verbose=False)

    #geometry = get_geometry(stream,verbose=False)
    geometry = get_coords(inventory, returntype="array")

    if isinstance(slim, float):
        sxmin = -slim
        sxmax = slim
        symin = -slim
        symax = slim
    elif isinstance(slim, tuple):
        if len(slim) == 4:
            sxmin = slim[0]
            sxmax = slim[1]
            symin = slim[2]
            symax = slim[3]
    else:
        raise TypeError('slim must either be a float or a tuple of length 4')

    nsx = int(np.ceil((sxmax + sstep / 10. - sxmin) / sstep))
    nsy = int(np.ceil((symax + sstep / 10. - symin) / sstep))
    nf = int(np.ceil((fmax + fstep / 10. - fmin) / fstep))

    transff = np.empty((nsx, nsy))
    buff = np.zeros(nf)

    for i, sx in enumerate(np.arange(sxmin, sxmax + sstep / 10., sstep)):
        for j, sy in enumerate(np.arange(symin, symax + sstep / 10., sstep)):
            for k, f in enumerate(np.arange(fmin, fmax + fstep / 10., fstep)):
                _sum = 0j
                for l in np.arange(len(geometry)):
                    _sum += np.exp(complex(
                        0., (geometry[l, 0] * sx + geometry[l, 1] * sy) *
                        2 * np.pi * f))
                buff[k] = abs(_sum) ** 2
            transff[i, j] = cumtrapz(buff, dx=fstep)[-1]

    transff /= transff.max()
    return transff

def array_analysis_helper(stream, inventory, method, frqlow, frqhigh,
                          filter=True, baz_plot=True, static3D=False,
                          vel_corr=4.8, wlen=-1, slx=(-10, 10),
                          sly=(-10, 10), sls=0.5, array_response=True):
    """
    Array analysis wrapper routine for MESS 2014.

    :param stream: Waveforms for the array processing.
    :type stream: :class:`obspy.core.stream.Stream`
    :param inventory: Station metadata for waveforms
    :type inventory: :class:`obspy.station.inventory.Inventory`
    :param method: Method used for the array analysis
     (one of "FK": Frequnecy Wavenumber, "DLS": Delay and Sum,
     "PWS": Phase Weighted Stack, "SWP": Slowness Whitened Power).
    :type method: str
    :param filter: Whether to bandpass data to selected frequency range
    :type filter: bool
    :param frqlow: Low corner of frequency range for array analysis
    :type frqlow: float
    :param frqhigh: High corner of frequency range for array analysis
    :type frqhigh: float
    :param baz_plot: Whether to show backazimuth-slowness map (True) or
     slowness x-y map (False).
    :type baz_plot: str
    :param static3D: static correction of topography using `vel_corr` as
     velocity (slow!)
    :type static3D: bool
    :param vel_corr: Correction velocity for static topography correction in
     km/s.
    :type vel_corr: float
    :param wlen: sliding window for analysis in seconds, use -1 to use the
     whole trace without windowing.
    :type wlen: float
    :param slx: Min/Max slowness for analysis in x direction.
    :type slx: (float, float)
    :param sly: Min/Max slowness for analysis in y direction.
    :type sly: (float, float)
    :param sls: step width of slowness grid
    :type sls: float
    :param array_response: superimpose array reponse function in plot (slow!)
    :type array_response: bool
    """

    if method not in ("FK", "DLS", "PWS", "SWP"):
        raise ValueError("Invalid method: ''" % method)

    sllx, slmx = slx
    slly, slmy = sly

    starttime = max([tr.stats.starttime for tr in stream])
    endtime = min([tr.stats.endtime for tr in stream])
    stream.trim(starttime, endtime)

    #stream.attach_response(inventory)
    stream.merge()
    for tr in stream:
        for station in inventory[0].stations:
            if tr.stats.station == station.code:
                tr.stats.coordinates = \
                    AttribDict(dict(latitude=station.latitude,
                                    longitude=station.longitude,
                                    elevation=station.elevation))
                break

    if filter:
        stream.filter('bandpass', freqmin=frqlow, freqmax=frqhigh,
                      zerophase=True)

    print(stream)
    spl = stream.copy()

    tmpdir = tempfile.mkdtemp(prefix="obspy-")
    filename_patterns = (os.path.join(tmpdir, 'pow_map_%03d.npy'),
                         os.path.join(tmpdir, 'apow_map_%03d.npy'))

    def dump(pow_map, apow_map, i):
        np.save(filename_patterns[0] % i, pow_map)
        np.save(filename_patterns[1] % i, apow_map)

    try:
        # next step would be needed if the correction velocity needs to be
        # estimated
        #
        sllx /= KM_PER_DEG
        slmx /= KM_PER_DEG
        slly /= KM_PER_DEG
        slmy /= KM_PER_DEG
        sls /= KM_PER_DEG
        vc = vel_corr
        if method == 'FK':
            kwargs = dict(
                #slowness grid: X min, X max, Y min, Y max, Slow Step
                sll_x=sllx, slm_x=slmx, sll_y=slly, slm_y=slmy, sl_s=sls,
                # sliding window properties
                win_len=wlen, win_frac=0.8,
                # frequency properties
                frqlow=frqlow, frqhigh=frqhigh, prewhiten=0,
                # restrict output
                store=dump,
                semb_thres=-1e9, vel_thres=-1e9, verbose=False,
                timestamp='julsec', stime=starttime, etime=endtime,
                method=0, correct_3dplane=False, vel_cor=vc,
                static_3D=static3D)

            # here we do the array processing
            start = UTCDateTime()
            out = array_processing(stream, **kwargs)
            print("Total time in routine: %f\n") % (UTCDateTime() - start)

            # make output human readable, adjust backazimuth to values
            # between 0 and 360
            t, rel_power, abs_power, baz, slow = out.T

        else:
            kwargs = dict(
                # slowness grid: X min, X max, Y min, Y max, Slow Step
                sll_x=sllx, slm_x=slmx, sll_y=slly, slm_y=slmy, sl_s=sls,
                # sliding window properties
                # frequency properties
                frqlow=frqlow, frqhigh=frqhigh,
                # restrict output
                store=dump,
                win_len=wlen, win_frac=0.5,
                nthroot=4, method=method,
                verbose=False, timestamp='julsec',
                stime=starttime, etime=endtime, vel_cor=vc,
                static_3D=False)

            # here we do the array processing
            start = UTCDateTime()
            out = beamforming(stream, **kwargs)
            print("Total time in routine: %f\n") % (UTCDateTime() - start)

            # make output human readable, adjust backazimuth to values
            # between 0 and 360
            trace = []
            t, rel_power, baz, slow_x, slow_y, slow = out.T

            # calculating array response
        if array_response:
            stepsfreq = (frqhigh - frqlow) / 10.
            tf_slx = sllx
            tf_smx = slmx
            tf_sly = slly
            tf_smy = slmy
            transff = array_transff_freqslowness(
                stream, (tf_slx, tf_smx, tf_sly, tf_smy), sls, frqlow,
                frqhigh, stepsfreq, coordsys='lonlat',
                correct_3dplane=False, static_3D=False, vel_cor=vc)

        # now let's do the plotting
        cmap = cm.rainbow

        #
        # we will plot everything in s/deg
        slow *= KM_PER_DEG
        sllx *= KM_PER_DEG
        slmx *= KM_PER_DEG
        slly *= KM_PER_DEG
        slmy *= KM_PER_DEG
        sls *= KM_PER_DEG

        numslice = len(t)
        powmap = []
        slx = np.arange(sllx-sls, slmx, sls)
        sly = np.arange(slly-sls, slmy, sls)
        if baz_plot:
            maxslowg = np.sqrt(slmx*slmx + slmy*slmy)
            bzs = np.arctan2(sls, np.sqrt(slmx*slmx + slmy*slmy))*180/np.pi
            xi = np.arange(0., maxslowg, sls)
            yi = np.arange(-180., 180., bzs)
            grid_x, grid_y = np.meshgrid(xi, yi)
        # reading in the rel-power maps
        for i in xrange(numslice):
            powmap.append(np.load(filename_patterns[0] % i))
            if method != 'FK':
                trace.append(np.load(filename_patterns[1] % i))

        npts = stream[0].stats.npts
        df = stream[0].stats.sampling_rate
        T = np.arange(0, npts / df, 1 / df)

        # if we choose windowlen > 0. we now move through our slices
        for i in xrange(numslice):
            slow_x = np.sin((baz[i]+180.)*np.pi/180.)*slow[i]
            slow_y = np.cos((baz[i]+180.)*np.pi/180.)*slow[i]
            st = UTCDateTime(t[i]) - starttime
            if wlen <= 0:
                en = endtime
            else:
                en = st + wlen
            print(UTCDateTime(t[i]))
            # add polar and colorbar axes
            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_axes([0.1, 0.87, 0.7, 0.10])
            # here we plot the first trace on top of the slowness map
            # and indicate the possibiton of the lsiding window as green box
            if method == 'FK':
                ax1.plot(T, spl[0].data, 'k')
                if wlen > 0.:
                    try:
                        ax1.axvspan(st, en, facecolor='g', alpha=0.3)
                    except IndexError:
                        pass
            else:
                T = np.arange(0, len(trace[i])/df, 1 / df)
                ax1.plot(T, trace[i], 'k')

            ax1.yaxis.set_major_locator(MaxNLocator(3))

            ax = fig.add_axes([0.10, 0.1, 0.70, 0.7])

            # if we have chosen the baz_plot option a re-griding
            # of the sx,sy slowness map is needed
            if baz_plot:
                slowgrid = []
                transgrid = []
                pow = np.asarray(powmap[i])
                for ix, sx in enumerate(slx):
                    for iy, sy in enumerate(sly):
                        bbaz = np.arctan2(sx, sy)*180/np.pi+180.
                        if bbaz > 180.:
                            bbaz = -180. + (bbaz-180.)
                        slowgrid.append((np.sqrt(sx*sx+sy*sy), bbaz,
                                         pow[ix, iy]))
                        if array_response:
                            tslow = (np.sqrt((sx+slow_x) *
                                     (sx+slow_x)+(sy+slow_y) *
                                     (sy+slow_y)))
                            tbaz = (np.arctan2(sx+slow_x, sy+slow_y) *
                                    180 / np.pi + 180.)
                            if tbaz > 180.:
                                tbaz = -180. + (tbaz-180.)
                            transgrid.append((tslow, tbaz,
                                              transff[ix, iy]))

                slowgrid = np.asarray(slowgrid)
                sl = slowgrid[:, 0]
                bz = slowgrid[:, 1]
                slowg = slowgrid[:, 2]
                grid = spi.griddata((sl, bz), slowg, (grid_x, grid_y),
                                    method='nearest')
                ax.pcolormesh(xi, yi, grid, cmap=cmap)

                if array_response:
                    level = np.arange(0.1, 0.5, 0.1)
                    transgrid = np.asarray(transgrid)
                    tsl = transgrid[:, 0]
                    tbz = transgrid[:, 1]
                    transg = transgrid[:, 2]
                    trans = spi.griddata((tsl, tbz), transg,
                                         (grid_x, grid_y),
                                         method='nearest')
                    ax.contour(xi, yi, trans, level, colors='k',
                               linewidth=0.2)

                ax.set_xlabel('slowness [s/deg]')
                ax.set_ylabel('backazimuth [deg]')
                ax.set_xlim(xi[0], xi[-1])
                ax.set_ylim(yi[0], yi[-1])
            else:
                ax.set_xlabel('slowness [s/deg]')
                ax.set_ylabel('slowness [s/deg]')
                slow_x = np.cos((baz[i]+180.)*np.pi/180.)*slow[i]
                slow_y = np.sin((baz[i]+180.)*np.pi/180.)*slow[i]
                ax.pcolormesh(slx, sly, powmap[i].T)
                ax.arrow(0, 0, slow_y, slow_x, head_width=0.005,
                         head_length=0.01, fc='k', ec='k')
                if array_response:
                    tslx = np.arange(sllx+slow_x, slmx+slow_x+sls, sls)
                    tsly = np.arange(slly+slow_y, slmy+slow_y+sls, sls)
                    try:
                        ax.contour(tsly, tslx, transff.T, 5, colors='k',
                                   linewidth=0.5)
                    except:
                        pass
                ax.set_ylim(slx[0], slx[-1])
                ax.set_xlim(sly[0], sly[-1])
            new_time = t[i]

            result = "BAZ: %.2f, Slow: %.2f s/deg, Time %s" % (
                baz[i], slow[i], UTCDateTime(new_time))
            ax.set_title(result)

            plt.show()
    finally:
        shutil.rmtree(tmpdir)

def get_spoint(stream, stime, etime):
    """
    Calculates start and end offsets relative to stime and etime for each
    trace in stream in samples.

    :param stime: UTCDateTime to start
    :param etime: UTCDateTime to end
    :returns: start and end sample offset arrays
    """
    slatest = stream[0].stats.starttime
    eearliest = stream[0].stats.endtime
    for tr in stream:
        if tr.stats.starttime >= slatest:
            slatest = tr.stats.starttime
        if tr.stats.endtime <= eearliest:
            eearliest = tr.stats.endtime

    nostat = len(stream)
    spoint = np.empty(nostat, dtype="int32", order="C")
    epoint = np.empty(nostat, dtype="int32", order="C")
    # now we have to adjust to the beginning of real start time
    if (slatest - stime) > stream[0].stats.delta/2.:
        msg = "Specified start-time is smaller than starttime in stream"
        raise ValueError(msg)
    if (eearliest - etime) < -stream[0].stats.delta/2.:
        msg = "Specified end-time bigger is than endtime in stream"
        print(eearliest, etime)
        raise ValueError(msg)
    for i in xrange(nostat):
        offset = int(((stime - slatest) / stream[i].stats.delta + 1.))
        negoffset = int(((eearliest - etime) / stream[i].stats.delta + 1.))
        diffstart = slatest - stream[i].stats.starttime
        frac, ddummy = math.modf(diffstart)
        spoint[i] = int(ddummy)
        if frac > stream[i].stats.delta * 0.25:
            msg = "Difference in start times exceeds 25% of samp rate"
            warnings.warn(msg)
        spoint[i] += offset
        diffend = stream[i].stats.endtime - eearliest
        frac, ddummy = math.modf(diffend)
        epoint[i] = int(ddummy)
        epoint[i] += negoffset
    return spoint, epoint

@staticmethod
def get_stream_offsets(stream, stime, etime):
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
            msg = "Specified stime %s is smaller than starttime %s " \
                  "in stream"
            raise ValueError(msg % (stime, tr.stats.starttime))
        if tr.stats.endtime < etime:
            msg = "Specified etime %s is bigger than endtime %s in stream"
            raise ValueError(msg % (etime, tr.stats.endtime))
        # now we have to adjust to the beginning of real start time
        spoint[i] = int(
            (stime - tr.stats.starttime) * tr.stats.sampling_rate + .5)
        epoint[i] = int(
            (tr.stats.endtime - etime) * tr.stats.sampling_rate + .5)
    return spoint, epoint

def beamforming(stream, sll_x, slm_x, sll_y, slm_y, sl_s, frqlow, frqhigh,
                stime, etime,   win_len=-1, win_frac=0.5,
                verbose=False, coordsys='lonlat', timestamp='mlabday',
                method="DLS", nthroot=1, store=None, correct_3dplane=False,
                static_3D=False, vel_cor=4.):
    """
    Method for Delay and Sum/Phase Weighted Stack/Whitened Slowness Power

    :param stream: Stream object, the trace.stats dict like class must
        contain a obspy.core.util.AttribDict with 'latitude', 'longitude' (in
        degrees) and 'elevation' (in km), or 'x', 'y', 'elevation' (in km)
        items/attributes. See param coordsys
    :type sll_x: Float
    :param sll_x: slowness x min (lower)
    :type slm_x: Float
    :param slm_x: slowness x max
    :type sll_y: Float
    :param sll_y: slowness y min (lower)
    :type slm_y: Float
    :param slm_y: slowness y max
    :type sl_s: Float
    :param sl_s: slowness step
    :type stime: UTCDateTime
    :param stime: Starttime of interest
    :type etime: UTCDateTime
    :param etime: Endtime of interest
    :type win_len: Float
    :param window length for sliding window analysis, default is -1 which means
        the whole trace;
    :type win_frac: Float
    :param fraction of win_len which is used to 'hop' forward in time
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :type timestamp: string
    :param timestamp: valid values: 'julsec' and 'mlabday'; 'julsec' returns
        the timestamp in secons since 1970-01-01T00:00:00, 'mlabday'
        returns the timestamp in days (decimals represent hours, minutes
        and seconds) since '0001-01-01T00:00:00' as needed for matplotlib
        date plotting (see e.g. matplotlibs num2date)
    :type method: string
    :param method: the method to use "DLS" delay and sum; "PWS" phase weigted
        stack; "SWP" slowness weightend power spectrum
    :type nthroot: Float
    :param nthroot: nth-root processing; nth gives the root (1,2,3,4), default
        1 (no nth-root)
    :type store: function
    :param store: A custom function which gets called on each iteration. It is
        called with the relative power map and the time offset as first and
        second arguments and the iteration number as third argument. Useful for
        storing or plotting the map for each iteration. For this purpose the
        dump function of this module can be used.
    :type correct_3dplane: Boolean
    :param correct_3dplane: if Yes than a best (LSQ) plane will be fitted into
        the array geometry.
        Mainly used with small apature arrays at steep flanks
    :type static_3D: Boolean
    :param static_3D: if yes the station height of am array station is taken
        into account accoring the formula:
            tj = -xj*sxj - yj*syj + zj*cos(inc)/vel_cor
        the inc angle is slowness dependend and thus must
        be estimated for each grid-point:
            inc = asin(v_cor*slow)
    :type vel_cor: Float
    :param vel_cor: Velocity for the upper layer (static correction) in km/s
    :return: numpy.ndarray of timestamp, relative relpow, absolute relpow,
        backazimut, slowness, maximum beam (for DLS)
    """
    res = []
    eotr = True

    # check that sampling rates do not vary
    fs = stream[0].stats.sampling_rate
    nstat = len(stream)
    if len(stream) != len(stream.select(sampling_rate=fs)):
        msg = 'in sonic sampling rates of traces in stream are not equal'
        raise ValueError(msg)

    # loop with a sliding window over the dat trace array and apply bbfk

    grdpts_x = int(((slm_x - sll_x) / sl_s + 0.5) + 1)
    grdpts_y = int(((slm_y - sll_y) / sl_s + 0.5) + 1)

    abspow_map = np.empty((grdpts_x, grdpts_y), dtype='f8')
    geometry = get_geometry(stream, coordsys=coordsys,
                            correct_3dplane=correct_3dplane, verbose=verbose)
    #geometry = get_geometry(stream, coordsys=coordsys, verbose=verbose)

    if verbose:
        print("geometry:")
        print(geometry)
        print("stream contains following traces:")
        print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    time_shift_table = get_timeshift(geometry, sll_x, sll_y, sl_s, grdpts_x,
                                     grdpts_y, vel_cor=vel_cor,
                                     static_3D=static_3D)

    mini = np.min(time_shift_table[:, :, :])
    maxi = np.max(time_shift_table[:, :, :])
    spoint, _epoint = get_spoint(stream, (stime-mini), (etime-maxi))
    minend = np.min(_epoint)
    maxstart = np.max(spoint)

    # recalculate the maximum possible trace length
    #    ndat = int(((etime-maxi) - (stime-mini))*fs)
    if(win_len < 0):
            nsamp = int(((etime-maxi) - (stime-mini))*fs)
    else:
        #nsamp = int((win_len-np.abs(maxi)-np.abs(mini)) * fs)
        nsamp = int(win_len * fs)

    if nsamp <= 0:
        print('Data window too small for slowness grid')
        print('Must exit')
        quit()

    nstep = int(nsamp * win_frac)

    stream.detrend()
    newstart = stime
    slow = 0.
    offset = 0
    count = 0
    while eotr:
        max_beam = 0.
        if method == 'DLS':
            for x in xrange(grdpts_x):
                for y in xrange(grdpts_y):
                    singlet = 0.
                    beam = np.zeros(nsamp, dtype='f8')
                    shifted = np.zeros(nsamp, dtype='f8')
                    for i in xrange(nstat):
                        s = spoint[i]+int(time_shift_table[i, x, y] * fs + 0.5)
                        try:
                            shifted = stream[i].data[s + offset:s + nsamp + offset]
                            if len(shifted) < nsamp:
                                shifted = np.pad(shifted,(0,nsamp-len(shifted)),'constant',constant_values=(0,1))
                            singlet += 1./nstat*np.sum(shifted*shifted)
                            beam += 1. / nstat * np.power(np.abs(shifted), 1. / nthroot) * shifted/np.abs(shifted)
                        except IndexError:
                            break
                    beam = np.power(np.abs(beam), nthroot) * beam / np.abs(beam)
                    bs = np.sum(beam*beam)
                    abspow_map[x, y] = bs / singlet
                    if abspow_map[x, y] > max_beam:
                        max_beam = abspow_map[x, y]
                        beam_max = beam
        if method == 'PWS':
            for x in xrange(grdpts_x):
                for y in xrange(grdpts_y):
                    singlet = 0.
                    beam = np.zeros(nsamp, dtype='f8')
                    stack = np.zeros(nsamp, dtype='c8')
                    phase = np.zeros(nsamp, dtype='f8')
                    shifted = np.zeros(nsamp, dtype='f8')
                    coh = np.zeros(nsamp, dtype='f8')
                    for i in xrange(nstat):
                        s = spoint[i] + int(time_shift_table[i, x, y] * fs +
                                            0.5)
                        try:
                            shifted = sp.signal.hilbert(stream[i].data[
                                s + offset: s + nsamp + offset])
                            if len(shifted) < nsamp:
                                shifted = np.pad(shifted,(0,nsamp-len(shifted)),'constant',constant_values=(0,1))
                        except IndexError:
                            break
                        phase = np.arctan2(shifted.imag, shifted.real)
                        stack.real += np.cos(phase)
                        stack.imag += np.sin(phase)
                    coh = 1. / nstat * np.abs(stack)
                    for i in xrange(nstat):
                        s = spoint[i]+int(time_shift_table[i, x, y] * fs + 0.5)
                        shifted = stream[i].data[s+offset: s + nsamp + offset]
                        singlet += 1. / nstat * np.sum(shifted * shifted)
                        beam += 1. / nstat * shifted * np.power(coh, nthroot)
                    bs = np.sum(beam*beam)
                    abspow_map[x, y] = bs / singlet
                    if abspow_map[x, y] > max_beam:
                        max_beam = abspow_map[x, y]
                        beam_max = beam
        if method == 'SWP':
            # generate plan for rfftr
            nfft = nextpow2(nsamp)
            deltaf = fs / float(nfft)
            nlow = int(frqlow / float(deltaf) + 0.5)
            nhigh = int(frqhigh / float(deltaf) + 0.5)
            nlow = max(1, nlow)  # avoid using the offset
            nhigh = min(nfft / 2 - 1, nhigh)  # avoid using nyquist
            nf = nhigh - nlow + 1  # include upper and lower frequency

            beam = np.zeros((grdpts_x, grdpts_y, nf), dtype='f16')
            steer = np.empty((nf, grdpts_x, grdpts_y, nstat), dtype='c16')
            spec = np.zeros((nstat, nf), dtype='c16')
            time_shift_table *= -1.
            clibsignal.calcSteer(nstat, grdpts_x, grdpts_y, nf, nlow,
                                 deltaf, time_shift_table, steer)
            try:
                for i in xrange(nstat):
                    dat = stream[i].data[spoint[i] + offset:
                                         spoint[i] + offset + nsamp]
                    dat = (dat - dat.mean()) * tap
                    spec[i, :] = np.fft.rfft(dat, nfft)[nlow: nlow + nf]
            except IndexError:
                break

            for i in xrange(grdpts_x):
                for j in xrange(grdpts_y):
                    for k in xrange(nf):
                        for l in xrange(nstat):
                            steer[k, i, j, l] *= spec[l, k]

            beam = np.absolute(np.sum(steer, axis=3))
            less = np.max(beam, axis=1)
            max_buffer = np.max(less, axis=1)

            for i in xrange(grdpts_x):
                for j in xrange(grdpts_y):
                    abspow_map[i, j] = np.sum(beam[:, i, j] / max_buffer[:],
                                              axis=0) / float(nf)

            beam_max = stream[0].data[spoint[0] + offset:
                                      spoint[0] + nsamp + offset]

        ix, iy = np.unravel_index(abspow_map.argmax(), abspow_map.shape)
        abspow = abspow_map[ix, iy]
        if store is not None:
            store(abspow_map, beam_max, count)
        count += 1
        print(count)
        # here we compute baz, slow
        slow_x = sll_x + ix * sl_s
        slow_y = sll_y + iy * sl_s

        slow = np.sqrt(slow_x ** 2 + slow_y ** 2)
        if slow < 1e-8:
            slow = 1e-8
        azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
        baz = azimut % -360 + 180
        res.append(np.array([newstart.timestamp, abspow, baz, slow_x, slow_y,
                             slow]))
        if verbose:
            print(newstart, (newstart + (nsamp / fs)), res[-1][1:])
        if (newstart + (nsamp + nstep)/fs ) > etime:
            eotr = False
        offset += nstep

        newstart += nstep / fs
    res = np.array(res)
    if timestamp == 'julsec':
        pass
    elif timestamp == 'mlabday':
       # 719162 == hours between 1970 and 0001
        res[:, 0] = res[:, 0] / (24. * 3600) + 719162
    else:
        msg = "Option timestamp must be one of 'julsec', or 'mlabday'"
        raise ValueError(msg)
    return np.array(res)


#    return(baz,slow,slow_x,slow_y,abspow_map,beam_max)

def get_timeshift_baz(geometry, sll, slm, sls, baze, vel_cor=4.,
                      static_3D=False):
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
    baz = math.pi*baze/180.
    nbeams = int((slm - sll) / sls + 0.5) + 1
    time_shift_tbl = np.empty((nstat, nbeams), dtype="float32")
    for k in xrange(nbeams):
        sx = sll + k * sls
        if vel_cor*sx < 1.:
            inc = np.arcsin(vel_cor*sx)
        else:
            inc = np.pi/2.
        time_shift_tbl[:, k] = sx * (geometry[:, 0] * math.sin(baz) +
                                     geometry[:, 1] * math.cos(baz))
        if static_3D:
            time_shift_tbl[:, k] += geometry[:, 2] * np.cos(inc) / vel_cor

    return time_shift_tbl

def vespagram_baz(stream, sll, slm, sls, baz, stime, etime, verbose=False,
                  coordsys='lonlat',timestamp='mlabday', method="DLS",nthroot=1,
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
    # check that sampling rates do not vary
    fs = stream[0].stats.sampling_rate
    nstat = len(stream)
    if len(stream) != len(stream.select(sampling_rate=fs)):
        msg = 'in sonic sampling rates of traces in stream are not equal'
        raise ValueError(msg)

    ndat = int((etime - stime)*fs)

    nbeams = int(((slm - sll) / sls + 0.5) + 1)

    #geometry = get_geometry(stream,coordsys=coordsys,correct_3dplane=False,verbose=verbose)
    geometry = get_geometry(stream,coordsys=coordsys,verbose=verbose)

    if verbose:
        print("geometry:")
        print(geometry)
        print("stream contains following traces:")
        print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    time_shift_table = get_timeshift_baz(geometry, sll, slm, sls, baz,
                                         vel_cor=vel_cor, static_3D=static_3D)

    mini = np.min(time_shift_table[:, :])
    maxi = np.max(time_shift_table[:, :])
    spoint, _epoint = get_spoint(stream, (stime - mini), (etime - maxi))

    # recalculate the maximum possible trace length
    ndat = int(((etime - maxi) - (stime - mini)) * fs)
    beams = np.zeros((nbeams, ndat), dtype='f8')

    stream.detrend()
    max_beam = 0.
    slow = 0.

    for x in xrange(nbeams):
        singlet = 0.
        if method == 'DLS':
            for i in xrange(nstat):
                s = spoint[i]+int(time_shift_table[i, x]*fs + 0.5)
                shifted = stream[i].data[s: s + ndat]
                singlet += 1. / nstat * np.sum(shifted * shifted)
                beams[x] += 1. / nstat * np.power(np.abs(shifted), 1. / nthroot) \
                    * shifted / np.abs(shifted)
            beams[x] = np.power(np.abs(beams[x]), nthroot) * beams[x] / \
                np.abs(beams[x])
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


    return(slow, beams, beam_max, max_beam)

def array_processing(stream, win_len, win_frac, sll_x, slm_x, sll_y, slm_y,
                     sl_s, semb_thres, vel_thres, frqlow, frqhigh, stime,
                     etime, prewhiten, verbose=False, coordsys='lonlat',
                     timestamp='mlabday', method=0, correct_3dplane=False,
                     vel_cor=4., static_3D=False, store=None):
    """
    Method for FK-Analysis/Capon

    :param stream: Stream object, the trace.stats dict like class must
        contain a obspy.core.util.AttribDict with 'latitude', 'longitude' (in
        degrees) and 'elevation' (in km), or 'x', 'y', 'elevation' (in km)
        items/attributes. See param coordsys
    :type win_len: Float
    :param win_len: Sliding window length in seconds
    :type win_frac: Float
    :param win_frac: Fraction of sliding window to use for step
    :type sll_x: Float
    :param sll_x: slowness x min (lower)
    :type slm_x: Float
    :param slm_x: slowness x max
    :type sll_y: Float
    :param sll_y: slowness y min (lower)
    :type slm_y: Float
    :param slm_y: slowness y max
    :type sl_s: Float
    :param sl_s: slowness step
    :type semb_thres: Float
    :param semb_thres: Threshold for semblance
    :type vel_thres: Float
    :param vel_thres: Threshold for velocity
    :type frqlow: Float
    :param frqlow: lower frequency for fk/capon
    :type frqhigh: Float
    :param frqhigh: higher frequency for fk/capon
    :type stime: UTCDateTime
    :param stime: Starttime of interest
    :type etime: UTCDateTime
    :param etime: Endtime of interest
    :type prewhiten: int
    :param prewhiten: Do prewhitening, values: 1 or 0
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :type timestamp: string
    :param timestamp: valid values: 'julsec' and 'mlabday'; 'julsec' returns
        the timestamp in secons since 1970-01-01T00:00:00, 'mlabday'
        returns the timestamp in days (decimals represent hours, minutes
        and seconds) since '0001-01-01T00:00:00' as needed for matplotlib
        date plotting (see e.g. matplotlibs num2date)
    :type method: int
    :param method: the method to use 0 == bf, 1 == capon
    :param vel_cor: correction velocity (upper layer) in km/s
    :param static_3D: a correction of the station height is applied using
        vel_cor the correction is done according to the formula:
        t = rxy*s - rz*cos(inc)/vel_cor
        where inc is defined by inv = asin(vel_cor*slow)
    :type store: function
    :param store: A custom function which gets called on each iteration. It is
        called with the relative power map and the time offset as first and
        second arguments and the iteration number as third argument. Useful for
        storing or plotting the map for each iteration. For this purpose the
        dump function of this module can be used.
    :return: numpy.ndarray of timestamp, relative relpow, absolute relpow,
        backazimut, slowness
    """
    BF, CAPON = 0, 1
    res = []
    eotr = True

    # check that sampling rates do not vary
    fs = stream[0].stats.sampling_rate
    if len(stream) != len(stream.select(sampling_rate=fs)):
        msg = ('in array-processing sampling rates of traces in stream are '
               'not equal')
        raise ValueError(msg)

    grdpts_x = int(((slm_x - sll_x) / sl_s + 0.5) + 1)
    grdpts_y = int(((slm_y - sll_y) / sl_s + 0.5) + 1)

    #geometry = get_geometry(stream, coordsys=coordsys,
    #                        correct_3dplane=correct_3dplane, verbose=verbose)

    geometry = get_geometry(stream,coordsys=coordsys,verbose=verbose)

    if verbose:
        print("geometry:")
        print(geometry)
        print("stream contains following traces:")
        print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    time_shift_table = get_timeshift(geometry, sll_x, sll_y, sl_s, grdpts_x,
                                     grdpts_y, vel_cor=vel_cor,
                                     static_3D=static_3D)
    # offset of arrays
    mini = np.min(time_shift_table[:, :, :])
    maxi = np.max(time_shift_table[:, :, :])

    spoint, _epoint = get_spoint(stream, stime, etime)

    # loop with a sliding window over the dat trace array and apply bbfk
    nstat = len(stream)
    fs = stream[0].stats.sampling_rate
    if win_len < 0.:
        nsamp = int((etime - stime)*fs)
        print(nsamp)
        nstep = 1
    else:
        nsamp = int(win_len * fs)
        nstep = int(nsamp * win_frac)

    # generate plan for rfftr
    nfft = nextpow2(nsamp)
    deltaf = fs / float(nfft)
    nlow = int(frqlow / float(deltaf) + 0.5)
    nhigh = int(frqhigh / float(deltaf) + 0.5)
    nlow = max(1, nlow)  # avoid using the offset
    nhigh = min(nfft / 2 - 1, nhigh)  # avoid using nyquist
    nf = nhigh - nlow + 1  # include upper and lower frequency

    # to spead up the routine a bit we estimate all steering vectors in advance
    steer = np.empty((nf, grdpts_x, grdpts_y, nstat), dtype='c16')
    clibsignal.calcSteer(nstat, grdpts_x, grdpts_y, nf, nlow,
                         deltaf, time_shift_table, steer)
    R = np.empty((nf, nstat, nstat), dtype='c16')
    ft = np.empty((nstat, nf), dtype='c16')
    newstart = stime
    tap = cosTaper(nsamp, p=0.22)  # 0.22 matches 0.2 of historical C bbfk.c
    offset = 0
    count = 0
    relpow_map = np.empty((grdpts_x, grdpts_y), dtype='f8')
    abspow_map = np.empty((grdpts_x, grdpts_y), dtype='f8')
    while eotr:
        try:
            for i, tr in enumerate(stream):
                dat = tr.data[spoint[i] + offset:
                              spoint[i] + offset + nsamp]
                dat = (dat - dat.mean()) * tap
                ft[i, :] = np.fft.rfft(dat, nfft)[nlow:nlow + nf]
        except IndexError:
            break
        ft = np.require(ft, 'c16', ['C_CONTIGUOUS'])
        relpow_map.fill(0.)
        abspow_map.fill(0.)
        # computing the covariances of the signal at different receivers
        dpow = 0.
        for i in xrange(nstat):
            for j in xrange(i, nstat):
                R[:, i, j] = ft[i, :] * ft[j, :].conj()
                if method == CAPON:
                    R[:, i, j] /= np.abs(R[:, i, j].sum())
                if i != j:
                    R[:, j, i] = R[:, i, j].conjugate()
                else:
                    dpow += np.abs(R[:, i, j].sum())
        dpow *= nstat
        if method == CAPON:
            # P(f) = 1/(e.H R(f)^-1 e)
            for n in xrange(nf):
                R[n, :, :] = np.linalg.pinv(R[n, :, :], rcond=1e-6)

        # errcode = clibsignal.generalizedBeamformer(relpow_map,abspow_map,steer,R,
        #                                            nsamp,nstat,prewhiten,grdpts_x,
        #                                            grdpts_y,nfft,nf,dpow,method)

        errcode = clibsignal.generalizedBeamformer(relpow_map,abspow_map,steer,R,nstat,prewhiten,
                                                   grdpts_x,grdpts_y,nf,dpow,method)

        if errcode != 0:
            msg = 'generalizedBeamforming exited with error %d'
            raise Exception(msg % errcode)
        ix, iy = np.unravel_index(relpow_map.argmax(), relpow_map.shape)
        relpow, abspow = relpow_map[ix, iy], abspow_map[ix, iy]
        if store is not None:
            store(relpow_map, abspow_map, count)
        count += 1

        # here we compute baz, slow
        slow_x = sll_x + ix * sl_s
        slow_y = sll_y + iy * sl_s

        slow = np.sqrt(slow_x ** 2 + slow_y ** 2)
        if slow < 1e-8:
            slow = 1e-8
        azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
        baz = azimut % -360 + 180
        if relpow > semb_thres and 1. / slow > vel_thres:
            res.append(np.array([newstart.timestamp, relpow, abspow, baz,
                                 slow]))
            if verbose:
                print(newstart, (newstart + (nsamp / fs)), res[-1][1:])
        if (newstart + (nsamp + nstep) / fs) > etime:
            eotr = False
        offset += nstep

        newstart += nstep / fs
    res = np.array(res)
    if timestamp == 'julsec':
        pass
    elif timestamp == 'mlabday':
        # 719162 == hours between 1970 and 0001
        res[:, 0] = res[:, 0] / (24. * 3600) + 719162
    else:
        msg = "Option timestamp must be one of 'julsec', or 'mlabday'"
        raise ValueError(msg)
    return np.array(res)

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

    for tt in tt_1:
        if tt["phase_name"] != phase_name:
            continue
        tt_1 = tt["time"]
        break

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


def vespagram(stream, ev, inv, method, frqlow, frqhigh, baz, scale, nthroot=4,
              filter=True, static3D=False, vel_corr=4.8, sl=(0.0, 10.0, 0.5),
              align=False, align_phase=['P', 'Pdiff'], plot_trace=True):
    """
    vespagram wrapper routine for MESS 2014.

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
    :param sl: Min/Max and stepwidthslowness for analysis
    :type sl: (float, float,float)
    :param align: whether to align the vespagram to a certain phase
    :type align: bool
    :param align_phase: phase to be aligned with (might be a list if simulateneous arivials are expected (P,PcP,Pdif)
    :type align: str
    :param plot_trace: if True plot the vespagram as wiggle plot, if False as density map
    :type align: bool
    """

    starttime = max([tr.stats.starttime for tr in stream])
    endtime = min([tr.stats.endtime for tr in stream])
    stream.trim(starttime, endtime)

    org = ev.preferred_origin() or ev.origins[0]
    ev_lat = org.latitude
    ev_lon = org.longitude
    ev_depth = org.depth/1000.  # in km
    ev_otime = org.time

    sll, slm, sls = sl
    sll /= KM_PER_DEG
    slm /= KM_PER_DEG
    sls /= KM_PER_DEG
    center_lon = 0.
    center_lat = 0.
    center_elv = 0.
    seismo = stream
    seismo.attach_response(inv)
    seismo.merge()
    sz = Stream()
    i = 0
    for tr in seismo:
        for station in inv[0].stations:
            if tr.stats.station == station.code:
                tr.stats.coordinates = \
                    AttribDict({'latitude': station.latitude,
                                'longitude': station.longitude,
                                'elevation': station.elevation})
                center_lon += station.longitude
                center_lat += station.latitude
                center_elv += station.elevation
                i += 1
        sz.append(tr)

    center_lon /= float(i)
    center_lat /= float(i)
    center_elv /= float(i)

    starttime = max([tr.stats.starttime for tr in stream])
    stt = starttime
    endtime = min([tr.stats.endtime for tr in stream])
    e = endtime
    stream.trim(starttime, endtime)

    #nut = 0
    max_amp = 0.
    sz.trim(stt, e)
    sz.detrend('simple')

    print(sz)
    fl, fh = frqlow, frqhigh
    if filter:
        sz.filter('bandpass', freqmin=fl, freqmax=fh, zerophase=True)

    if align:
        deg = []
        shift = []
        res = gps2DistAzimuth(center_lat, center_lon, ev_lat, ev_lon)
        deg.append(kilometer2degrees(res[0]/1000.))
        tt = getTravelTimes(deg[0], ev_depth, model='ak135')
        for item in tt:
            phase = item['phase_name']
            if phase in align_phase:
                try:
                    travel = item['time']
                    travel = ev_otime.timestamp + travel
                    dtime = travel - stt.timestamp
                    shift.append(dtime)
                except:
                    break
        for i, tr in enumerate(sz):
            res = gps2DistAzimuth(tr.stats.coordinates['latitude'],
                                  tr.stats.coordinates['longitude'],
                                  ev_lat, ev_lon)
            deg.append(kilometer2degrees(res[0]/1000.))
            tt = getTravelTimes(deg[i+1], ev_depth, model='ak135')
            for item in tt:
                phase = item['phase_name']
                if phase in align_phase:
                    try:
                        travel = item['time']
                        travel = ev_otime.timestamp + travel
                        dtime = travel - stt.timestamp
                        shift.append(dtime)
                    except:
                        break
        shift = np.asarray(shift)
        shift -= shift[0]
        shifttrace_freq(sz, -shift)

    baz += 180.
    nbeam = int((slm - sll)/sls + 0.5) + 1
    kwargs = dict(
        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll=sll, slm=slm, sls=sls, baz=baz, stime=stt, method=method,
        nthroot=nthroot, etime=e, correct_3dplane=False, static_3D=static3D,
        vel_cor=vel_corr)

    start = UTCDateTime()
    slow, beams, max_beam, beam_max = vespagram_baz(sz, **kwargs)
    print("Total time in routine: %f\n") % (UTCDateTime() - start)

    df = sz[0].stats.sampling_rate
    # Plot the seismograms
    npts = len(beams[0])
    print(npts)
    T = np.arange(0, npts/df, 1/df)
    sll *= KM_PER_DEG
    slm *= KM_PER_DEG
    sls *= KM_PER_DEG
    slow = np.arange(sll, slm, sls)
    max_amp = np.max(beams[:, :])
    #min_amp = np.min(beams[:, :])
    scale *= sls

    fig = plt.figure(figsize=(12, 8))

    if plot_trace:
        ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        for i in xrange(nbeam):
            if i == max_beam:
                ax1.plot(T, sll + scale*beams[i]/max_amp + i*sls, 'r',
                         zorder=1)
            else:
                ax1.plot(T, sll + scale*beams[i]/max_amp + i*sls, 'k',
                         zorder=-1)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('slowness [s/deg]')
        ax1.set_xlim(T[0], T[-1])
        data_minmax = ax1.yaxis.get_data_interval()
        minmax = [min(slow[0], data_minmax[0]), max(slow[-1], data_minmax[1])]
        ax1.set_ylim(*minmax)
    #####
    else:
        #step = (max_amp - min_amp)/100.
        #level = np.arange(min_amp, max_amp, step)
        #beams = beams.transpose()
        #cmap = cm.hot_r
        cmap = cm.rainbow

        ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        #ax1.contour(slow,T,beams,level)
        #extent = (slow[0], slow[-1], \
        #               T[0], T[-1])
        extent = (T[0], T[-1], slow[0] - sls * 0.5, slow[-1] + sls * 0.5)

        ax1.set_ylabel('slowness [s/deg]')
        ax1.set_xlabel('T [s]')
        beams = np.flipud(beams)
        ax1.imshow(beams, cmap=cmap, interpolation="nearest",
                   extent=extent, aspect='auto')

    ####
    result = "BAZ: %.2f Time %s" % (baz-180., stt)
    ax1.set_title(result)

    plt.show()
    return slow, beams, max_beam, beam_max


def get_timeshift(geometry, sll_x, sll_y, sl_s, grdpts_x,
                  grdpts_y, vel_cor=4., static_3D=False):
    """
    Returns timeshift table for given array geometry

    :param geometry: Nested list containing the arrays geometry, as returned by
            get_group_geometry
    :param sll_x: slowness x min (lower)
    :param sll_y: slowness y min (lower)
    :param sl_s: slowness step
    :param grdpts_x: number of grid points in x direction
    :param grdpts_x: number of grid points in y direction
    :param vel_cor: correction velocity (upper layer) in km/s
    :param static_3D: a correction of the station height is applied using
        vel_cor the correction is done according to the formula:
        t = rxy*s - rz*cos(inc)/vel_cor
        where inc is defined by inv = asin(vel_cor*slow)
    """
    if static_3D:
        nstat = len(geometry)  # last index are center coordinates
        time_shift_tbl = np.empty((nstat, grdpts_x, grdpts_y), dtype="float32")
        for k in xrange(grdpts_x):
            sx = sll_x + k * sl_s
            for l in xrange(grdpts_y):
                sy = sll_y + l * sl_s
                slow = np.sqrt(sx*sx + sy*sy)
                if vel_cor*slow <= 1.:
                    inc = np.arcsin(vel_cor*slow)
                else:
                    print ("Warning correction velocity smaller than apparent "
                           "velocity")
                    inc = np.pi/2.
                time_shift_tbl[:, k, l] = sx * geometry[:, 0] + sy * \
                    geometry[:, 1] + geometry[:, 2] * np.cos(inc) / vel_cor
        return time_shift_tbl
    # optimized version
    else:
        mx = np.outer(geometry[:, 0], sll_x + np.arange(grdpts_x) * sl_s)
        my = np.outer(geometry[:, 1], sll_y + np.arange(grdpts_y) * sl_s)
        return np.require(
            mx[:, :, np.newaxis].repeat(grdpts_y, axis=2) +
            my[:, np.newaxis, :].repeat(grdpts_x, axis=1),
            dtype='float32')

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


"""
NEW STUFF, WHATS WORKING RIGHT NOW!
"""
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


def __coordinate_values(inventory):
    geo = get_coords(inventory, returntype="dict")
    lats, lngs, hgt = [], [], []
    for coordinates in list(geo.values()):
        lats.append(coordinates["latitude"]),
        lngs.append(coordinates["longitude"]),
        hgt.append(coordinates["elevation"])
    return lats, lngs, hgt

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

"""
def plot_data(stream, zoom=1, t_window=None):
    plot_inc = 0.1
    window = []
    for i in range(0,stream.count(),4):
        tr = stream[i]
        trdata_plot = tr.data * zoom + i * plot_inc
        plt.plot(tr.times(), trdata_plot, 'black')


    if savefigure:
        plt.savefig('plot_data.png', format="png", dpi=900)
    else:
        plt.show()

"""
