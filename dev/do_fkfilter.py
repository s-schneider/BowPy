import obspy
import numpy as np
from numpy import genfromtxt
import fk_work as fkw
from fk_work import fk_filter
import os

#wdir = os.environ.get('x')
wdir = "/local/s_schn42/Thesis/FK-filter/data"

ifile = wdir + '/' + 'FKOUT.QHD'
ofile = wdir + '/' +  'FKIN.QHD'

st=obspy.read(ifile)

#epi = genfromtxt("epidist.txt", dtype=None, usecols=1)

#data = fkw.stream2array(st)
data_fil = fkw.fk_filter(st, ftype="FFT", fktype="eliminate")

#data_fil = _fk_fft_filter_eliminate_phase(data)

#st_fil = fkw.array2stream(data_fil)
#st_fil.write(ofile, format="Q")