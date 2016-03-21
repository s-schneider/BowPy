from __future__ import absolute_import, print_function

import numpy
import numpy as np
from numpy import genfromtxt
import math

import matplotlib

# If using a Mac Machine, otherwitse comment the next line out:
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
from obspy import read as read_st
from obspy import read_inventory as read_inv
from obspy import read_events as read_cat

import sipy
import sipy.misc.Muenster_Array_Seismology_Vespagram as MAS
import sipy.filter.fk as fk
import sipy.filter.radon as radon
import sipy.util.fkutil as fku
import sipy.util.base as base

from sipy.util.data_request import data_request
from sipy.filter.fk import fk_filter, fktrafo, fk_reconstruct
from sipy.util.fkutil import  nextpow2, find_subsets, slope_distribution, makeMask, create_iFFT2mtx
from sipy.util.array_util import get_coords, attach_network_to_traces, attach_coordinates_to_traces,\
stream2array, array2stream, attach_network_to_traces, attach_coordinates_to_traces, attach_epidist2coords, epidist2nparray, epidist2list, \
alignon, partial_stack, gaps_fill_zeros, vespagram
from sipy.util.picker import get_polygon


testricker = ["../data/test_datasets/ricker/SR20.QHD","../data/test_datasets/ricker/SR50.QHD", "../data/test_datasets/ricker/SR80.QHD", "../data/test_datasets/ricker/SR90.QHD"]
testinsta = ["../data/test_datasets/instaseis/STS20.QHD","../data/test_datasets/instaseis/STS50.QHD", "../data/test_datasets/instaseis/STS80.QHD", "../data/test_datasets/instaseis/STS90.QHD"]


"""
FK_recon Reconstruction , with different shapes
FK_recon denoise, with different shapes

SSA Reconstruction, different ranks
SSA denoise, different ranks


FK_fil PP Extraction, different shapes
	-original
	-denoised
	-partial stacked

FK_fil P/PDiff Elimination different shapes
	-original
	-denoised
	-partial stacked
