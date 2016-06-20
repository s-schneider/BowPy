# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:30:44 2016

@author: patrick
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema


name1 = 'growth_rate/rrbc/no_slip/E_1e-4/OUT01';
A = np.genfromtxt(name1,comments = '#', dtype = np.float64);

t=A[:,1];
urms=A[:,3];


print 'Beginning time'
inta=raw_input();
inta=np.float(inta);


print'Enter maximum time for which growth rate should be evaluated'
intb=raw_input();
intb=np.float(intb);



#tnew=np.array([i for i in t if i < intb])
#newsize=np.size(tnew)
tnew=np.array([i for i in t if (i < intb and i > inta)]);
urmsnew=urms[np.array([i for i, item in enumerate(t) if item in tnew])];
#urmsmax=urms[argrelextrema(urms[:newsize],np.greater)];
#tmax=t[argrelextrema(urms[:newsize],np.greater)];

#tdiffrms=tmax[1:-1]-tmax[:-2];

#average_tdiff=np.average(tdiffrms);
#period=2.0*average_tdiff;
#freq=2.*np.pi/period;

#print 'Period and Frequency is'
#print period,freq;


def funcexp(x,a,b):
    return a*np.exp(b*x);

#fitparam,fitcov = curve_fit(funcexp,tmax,urmsmax,p0=(0.01,0.01));
fitparam,fitcov = curve_fit(funcexp,tnew,urmsnew,p0=(0.01,0.01));


print 'The growth rate is'
print fitparam[1];


scale_pow=-1;

def powerbase10(x,pos):
    return '%1.2f' % (x*(10**scale_pow));


formatter=tick.FuncFormatter(powerbase10);
#==============================================================================
print 'plot the plot'
fig,ax=plt.subplots();
ax.get_yaxis().set_major_formatter(formatter);
ax.set_ylabel('root mean square velocity $u_{rms}$  '+ '$10^{{{0:d}}}$'.format(scale_pow));
plt.plot(t,urms,'r.');
plt.plot(tnew,funcexp(tnew,fitparam[0],fitparam[1]),'k-');
#plt.plot(tmax,urmsmax,'b.');
plt.xlabel(r'non-dimensional time $t$');
plt.show();
plt.pause(10)
plt.close();
#==============================================================================
