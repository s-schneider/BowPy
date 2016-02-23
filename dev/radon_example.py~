#Example script for Radon transform

import numpy as np
import scipy.io as sio
import datetime
import math
from radon import nextpow2
from scipy.signal import hilbert
from scipy import sparse
import matplotlib.pyplot as plt

import radon


data = sio.loadmat("mtz_radon/Radon/data.mat")

# t        - time axis.
# Delta    - distance (offset) axis. --> epidist vector!
# M        - Amplitudes of phase arrivals.
# indicies - list of indicies relevent to the S670S phase.

# Define some variables for RT.

t = data['t']
IDelta = data['Delta']
M = data['M']
indicies = data['indicies']

mu=[5e-2]
P_axis=np.arange(-1,1.01,0.01)
meandelta = np.mean(IDelta)

# Invert to Radon domain using unweighted L2 inversion, linear path
# functions and an average distance parameter.

tic = datetime.datetime.now()
R=radon.radon_inverse(t, IDelta, M, P_axis, np.ones(IDelta.size), meandelta, "Linear", "L2", mu)
toc = datetime.datetime.now()
time = toc-tic
print( "Elapsed time is %s seconds." % str(time))

# Mute all phases except the S670S arrival.
R670=np.zeros(R.shape)
R670.conj().transpose().flat[ indicies.astype('int').tolist() ]=1
R670=R*R670



# Apply forward operator to the muted Radon domain.
# Resample distance vector, for correct time shifting
Delta_resampled = np.arange( int(math.floor(min(IDelta[0]))), int(math.ceil(max(IDelta[0])))+1, (int(math.ceil(max(IDelta[0]))) - int(math.floor(min(IDelta[0]))))/20.)
M670=radon.radon_forward(t, P_axis, R670, Delta_resampled, meandelta, 'Linear')





# Plot figures.
plt.subplot(3,1,1)
plt.title('Aligned SS
')
yticksorg =  np.arange(int(math.ceil(min(IDelta[0]/10)))*10, int(math.ceil(max(IDelta[0]/10)))*10 + 10,10)[::1]
xticksorg =  np.arange(int(math.ceil(min(t[0]/100)))*100, int(math.ceil(max(t[0]/100)))*100 + 100,100)[::2]
plt.yticks(yticksorg)
plt.xticks(xticksorg)
plt.imshow(M, extent=[max(t[0]), min(t[0]), min(IDelta[0]), max(IDelta[0])], aspect=16)

plt.subplot(3,1,2)
ytickshilbert = np.arange(-1, 1, 0.5)[::-1]
plt.yticks(ytickshilbert)
plt.xticks(xticksorg)
plt.imshow( abs(hilbert(abs(R).conj().transpose())).conj().transpose() ,extent=[min(t[0]), max(t[0]), -1, 1], aspect=256, interpolation='nearest' )


plt.subplot(3,1,3)
yticksradon = np.arange(int(math.ceil(min(Delta_resampled/10)))*10, int(math.ceil(max(Delta_resampled/10)))*10 + 10,10)[::-1]
plt.yticks(yticksradon)
plt.xticks(xticksorg)
plt.imshow(abs(M670), extent=[min(t[0]), max(t[0]), min(Delta_resampled), max(Delta_resampled)], aspect=16 )

plt.show()



