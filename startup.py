try:
	import numpy
	import numpy as np
	from numpy import genfromtxt
	import math

	import matplotlib

	matplotlib.use('TkAgg')

	import matplotlib.pyplot as plt
	import matplotlib as mpl
	import scipy as sp
	import scipy.signal as signal
	import scipy.io as sio

	import os
	import datetime

	import obspy
	from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees, locations2degrees
	from obspy.taup import TauPyModel
	from obspy import read_inventory as read_inv
	from obspy import read_events as read_cat
	from obspy import UTCDateTime

	import bowpy
	import bowpy.filter.fk as fk
	import bowpy.filter.radon as radon
	import bowpy.util.fkutil as fku
	import bowpy.util.base as base
	import bowpy.util.array_util as au

	from bowpy.misc.read import read_st
	from bowpy.filter.fk import pocs_recon
	from bowpy.util.data_request import data_request
	from bowpy.util.base import cat4stream, inv4stream
	from bowpy.filter.fk import fk_filter, fk_reconstruct
	from bowpy.util.fkutil import  plot, nextpow2, slope_distribution, makeMask, create_iFFT2mtx, plotfk, fktrafo
	from bowpy.util.picker import get_polygon
	from bowpy.util.array_util import get_coords, attach_network_to_traces, attach_coordinates_to_traces,\
	stream2array, array2stream, attach_network_to_traces, attach_coordinates_to_traces, attach_epidist2coords, epidist2nparray, epidist2list, \
	alignon, gaps_fill_zeros, vespagram, rm, cut, plot_vespa

	print('Imported all modules, including bowpy and Obspy')

except ImportError:

	import numpy
	import numpy as np
	from numpy import genfromtxt
	import math

	import matplotlib

	matplotlib.use('TkAgg')

	import matplotlib.pyplot as plt
	import matplotlib as mpl
	import scipy as sp
	import scipy.signal as signal
	import scipy.io as sio

	import os
	import datetime

	print("Couldn't find bowpy modules, imported standard set of modules")
