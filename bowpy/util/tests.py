from __future__ import absolute_import, print_function

import numpy
import numpy as np
from numpy.random import randint
import matplotlib
import matplotlib.pyplot as plt
import sys

from bowpy.util.base import stream2array, array2stream
from bowpy.filter.fk import pocs_recon
from bowpy.util.array_util import stack
from bowpy.util.fkutil import plot
# If using a Mac Machine, otherwitse comment the next line out:
matplotlib.use('TkAgg')


def qtest_pocs(st_rec, st_orginal, alpharange, irange):
    """
    Runs the selected method in a certain range of parameters
    (iterations and alpha), returns a table of Q values ,defined as:

    Q = 10 * log( || d_org || ^2 _2  / ||  d_org - d_rec || ^2 _2 )

    The highest Q value is the one to be chosen.
    """
    Qall = []
    dmethod = 'reconstruct'
    method = 'linear'

    st_org = st_orginal.copy()
    data_org = stream2array(st_org, normalize=True)

    for alpha in alpharange:

        print("########## CURRENT ALPHA %f  ##########\n" % alpha)
        for i in irange:

            print('POCS RECON WITH %i ITERATIONS' % i, end="\r")
            sys.stdout.flush()
            srs = st_rec.copy()
            st_pocsrec = pocs_recon(srs, maxiter=int(i), method=method,
                                    dmethod=dmethod, alpha=alpha)
            drec = stream2array(st_pocsrec, normalize=True)
            Q_tmp = np.linalg.norm(data_org, 2)**2. / np.linalg.norm(data_org
                                                                     - drec,
                                                                     2)**2.
            Q = 10.*np.log(Q_tmp)
            Qall.append([alpha, i, Q])

    Qmax = [0, 0, 0]
    for i in Qall:
        if i[2] > Qmax[2]:
            Qmax = i

    return Qall


def qtest_plot(ifile, alpharange, irange, ifile_path=None, ofile=None, fs=20,
               cmap='Blues', cbarlim=None):

    if isinstance(alpharange, numpy.ndarray):
        yrange = alpharange
        yextent = np.zeros(yrange.size)
    else:
        msg = 'Wrong alpharange input'
        raise IOError(msg)

    if isinstance(irange, numpy.ndarray):
        xrange = irange
    else:
        msg = 'Wrong irange input'
        raise IOError(msg)

    Qraw = ifile
    Qmat = np.zeros((xrange.size, yrange.size))
    for xi, x in enumerate(xrange):
        for yi, y in enumerate(yrange):
            for j, item in enumerate(Qraw):
                if item[1] == x and item[0] == y:
                    Qmat[xi, yi] = item[2]

    fig, ax = plt.subplots(frameon=False)
    Qplot = Qmat.copy()
    maxindex = Qmat.argmax()
    Qmax = np.zeros(Qplot.shape)
    # Qmax[:,:]= np.float64('nan')
    Qmax[np.unravel_index(maxindex, Qmat.shape)] = 10000

    extent = (alpharange.min(), alpharange.max(), irange.min(), irange.max())
    im = ax.imshow(Qplot, aspect='auto', origin='lower', interpolation='none',
                   cmap=cmap, extent=extent)
    ax.set_ylabel('No of iterations', fontsize=fs)
    ax.set_xlabel(r'$\alpha$', fontsize=fs)
    ax.tick_params(axis='both', which='both', labelsize=fs)
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('Q', fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)
    if cbarlim:
        cbar.set_clim(cbarlim[0], cbarlim[1])

    # Customize major tick labels

    # ax.yaxis.set_major_locator(ticker.FixedLocator(np.linspace(1.5,10.5,10)))
    # ax.yaxis.set_major_formatter(ticker.FixedFormatter(np.linspace(1,10,10).astype('int')))

    for vi, value in enumerate(yrange):
        yextent[vi] = "{:.2f}".format(value)
    # ax.xaxis.set_major_locator(ticker.FixedLocator(np.linspace(.05,.95,10)))#np.linspace(1.5,99.5,9)))
    # ax.xaxis.set_major_formatter(ticker.FixedFormatter(yextent))
    # fig.set_size_inches(12,10)
    # fig.savefig(plotname, dpi=300)
    # plt.close("all")
    plt.show()
    return


def set_zero(st, name):

    for station in name:
        for i, trace in enumerate(st):
            if trace.stats.station == station:
                trace.data = np.zeros(trace.data.shape)
                st[i].stats.zerotrace = 'True'
                st[i].stats.station = 'empty'

    return


def bootstrap_stream(stream, n, fs=20, ylimit=None):
    data = stream2array(stream)
    sigma = bootstrap(data, n)
    plot_sigma(sigma, stream, fs=fs, ylimit=ylimit)
    return


def bootstrap(data, n):
    noft = data.shape[0]
    d_stack = stack(data)
    bootsum = 0.

    for n_i in range(n):
        b = np.zeros(data.shape)

        for noft_i in range(noft):
            r = randint(0, noft, size=1)[0]
            b[noft_i] = data[r]

        b_stack = stack(b, 1)
        boot = np.square((d_stack-b_stack))
        bootsum = bootsum + boot

    sigma = np.sqrt(bootsum / float(n*(n-1)))
    return sigma


def plot_sigma(sigma, stream, fs=20, ylimit=None):
    si_stream = array2stream(sigma, stream)
    plot(si_stream[0], ylabel='sigma', yticks=True, fs=fs, ylimit=ylimit)
    return
