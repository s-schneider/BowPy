"""
import fk_work
import fk_work as fkw
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
import scipy.signal as signal
import obspy
from numpy import genfromtxt
import fk_work as fkw
from fk_work import fk_filter
import os

stream="../data/WORK_D.MSEED"
inventory="../data/2011-03-11T05:46:23.MSEED_inv.xml"
catalog="../data/2011-03-11T05:46:23.MSEED_cat.xml"
st, inv, cat = fkw.read_file(stream, inventory, catalog)
ad = fkw.stream2array(st)
adt = fkw.transpose(ad)
epid = fkw.epidist2nparray(fkw.epidist_stream(st, inv, cat))
fkspectra, periods = fkw.fk_filter(st, ftype='LS', inv=inv, cat=cat, fktype="eliminate")
fkfft = abs(np.fft.fftn(ad))
samplingrate = 0.025

#Example data flow 20.01.2016
trace = ad[0]
xrange = np.linspace(0, trace.size*0.025, trace.size)




tracefft = np.fft.rfft(trace)
freq = np.fft.rfftfreq(trace.size, samplingrate)
freq = freq * 2. * np.pi
fftnorm=tracefft/max(tracefft)

frange_new = np.linspace(freq[1], max(freq), trace.size/2 + 1)
epidist = np.linspace(0, trace.size, trace.size) * 0.025
tracels_new = signal.lombscargle(epidist, trace.astype('float'), frange_new)


tracels = fkw.ls2ifft_prep(tracels_new, trace)
fls = fkw.convert_lsindex(frange_new, 0.025)



plt.plot(freq, abs((tracefft/max(tracefft)).real))
plt.plot(frange_new,tracels_new/max(tracels_new))
plt.plot(fls, tracels_new/max(tracels_new))
plt.plot(abs((tracefft/max(tracefft)).real))
plt.show()



plt.plot(tracels/max(tracels))
plt.plot(abs((tracefft/max(tracefft)).real))
plt.show()

plt.plot(np.fft.irfft(tracels))
plt.show()



"""

"""
Test Sinus
A = 2.
w = 1.
phi = 0.5 * np.pi

nin = 1000
nout = 100000
x = np.linspace(0.01, 10*np.pi, nin)
y = A * np.sin(w*x+phi)

yfft = np.fft.rfft(y)
yfft = yfft/max(yfft)

steps= max(x)/y.size

f_fft = np.fft.rfftfreq(y.size, steps) * 2. * np.pi

#frange = np.linspace(0.01, 10, nin/2)
frange = np.linspace(f_fft[1], max(f_fft), nin/2)
yls = signal.lombscargle(x, y, frange)
yls = yls/max(yls)
yls2ifft = fkw.ls2ifft_prep(yls)



"""

#Example data flow
"""

#fk_filter(stream, inventory, catalog, phase)
#data=create_signal(no_of_traces=1,len_of_traces=12,multiple=False)
#datatest=fkw.create_sine(no_of_traces=1, no_of_periods=2)

"""
##################INSTASEIS###############################

"""
get data with instaseis

import instaseis as ins
import obspy


db = ins.open_db("/Users/Simon/dev/instaseis/10s_PREM_ANI_FORCES")

source = ins.Source(
latitude=0.0, longitude=0.0, depth_in_m=100000,
m_rr = 3.71e23 / 1E7,
m_tt = 7.81e21 / 1E7,m_pp =8.26e23 / 1E7,
m_rt = 1.399e23 / 1E7,
m_rp =6.95e22 / 1E7,
m_tp =3.177e24 / 1E7,
origin_time=obspy.UTCDateTime(2016,2,9,18,41,1)
)

for i in range(20):
    lon = str(i*0.1)
    name="x"+lon
    x.append(ins.Receiver(latitude="0", longitude=lon, network="LA", station=lon ))
    print(lon)
for i in range(20):
    st.append(db.get_seismograms(source=source, receiver=x[i]))
"""
