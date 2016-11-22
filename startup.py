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

	import sipy
	import sipy.misc.Muenster_Array_Seismology_Vespagram as MAS
	import sipy.filter.fk as fk
	import sipy.filter.radon as radon
	import sipy.util.fkutil as fku
	import sipy.util.base as base
	import sipy.util.array_util as au
	import sipy.util.tests as tests

	from sipy.misc.read import read_st
	from sipy.filter.fk import pocs_recon
	from sipy.util.data_request import data_request
	from sipy.util.base import cat4stream, inv4stream
	from sipy.filter.fk import fk_filter, fk_reconstruct
	from sipy.util.fkutil import  plot, nextpow2, find_subsets, slope_distribution, makeMask, create_iFFT2mtx, plotfk, fktrafo
	from sipy.util.array_util import get_coords, attach_network_to_traces, attach_coordinates_to_traces,\
	stream2array, array2stream, attach_network_to_traces, attach_coordinates_to_traces, attach_epidist2coords, epidist2nparray, epidist2list, \
	alignon, partial_stack, gaps_fill_zeros, vespagram, rm, cut, plot_vespa
	from sipy.util.picker import get_polygon

	print('Imported all modules, including SiPy and Obspy')

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

	print("Couldn't find SiPy modules, imported standard set of modules")
