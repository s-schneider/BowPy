from __future__ import absolute_import
import obspy
import numpy as np
from numpy import genfromtxt
from fk_work import fk_filter


st = obspy.read("FKOUT.QHD")

epi = genfromtxt("epidist.txt", dtype=None, usecols=0)

st_fil = fk_filter(st, epi_dist=epi)

st_fil.write("FKOUT.QHD", format="Q")
