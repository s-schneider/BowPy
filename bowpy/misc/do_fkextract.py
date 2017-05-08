from __future__ import absolute_import
import os
import obspy
#from numpy import genfromtxt
from fk_work import fk_filter_extract_phase, stream2array, array2stream


wdir = os.environ.get('wdir')

ifile = wdir + '/' + 'FKOUT.QHD'
ofile = wdir + '/' +  'FKIN.QHD'

st=obspy.read(ifile)

#epi = genfromtxt("epidist.txt", dtype=None, usecols=1)

data = stream2array(st)

data_fil = fk_filter_extract_phase(data)

st_fil = array2stream(data_fil)

st_fil.write(ofile, format="Q")
