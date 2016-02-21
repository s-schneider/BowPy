import numpy as np
from numpy import dot
import scipy as sp
import average_anti_diag as aad


def ssa(d,nw,p,ssa_flag):
	"""
	SSA: 1D Singular Spectrum Analysis for snr enhancement

	  dp,sing,R = ssa(d,nw,p,ssa_flag);

	  IN   d:   1D time series (column)
	       nw:  view used to make the Hankel matrix
	       p:   number of singular values used to reconstuct the data
	       ssa_flag = 0 do not compute R

	  OUT  dp:  predicted (clean) data
	       R:   matrix consisting of the data predicted with
	            the first eof (R(:,1)), the second eof (R(:,2)) etc
	       sing: singular values of the Hankel matrix

	  Example:

	       d = cos(2*pi*0.01*[1:1:200]') + 0.5*randn(200,1)
	       [dp,sing,R] = ssa(d,100,2,1); plot(d); hold on; plot(dp+3);

	  Based on: 

	  M.D.Sacchi, 2009, FX SSA, CSEG Annual Convention, Abstracts,392-395.
	                    http://www.geoconvention.org/2009abstracts/194.pdf

	  Copyright (C) 2008, Signal Analysis and Imaging Group.
	  For more information: http://www-geo.phys.ualberta.ca/saig/SeismicLab
	  Author: M.D.Sacchi
	  Translated to Python by: S. Schneider, 2016

	  This program is free software: you can redistribute it and/or modify
	  it under the terms of the GNU General Public License as published
	  by the Free Software Foundation, either version 3 of the License, or
	  any later version.

	  This program is distributed in the hope that it will be useful,
	  but WITHOUT ANY WARRANTY; without even the implied warranty of
	  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	  GNU General Public License for more details: http://www.gnu.org/licenses/

	"""



	# Check for Data type of variables.
	if not type(d) == numpy.ndarray:
		print( "Wrong input type of d, must be numpy.ndarray" )
		raise TypeError

 	nt = d.size
 	N = nt-nw+1
 	l = np.arange(0,nw,1)

 	# Make Hankel Matrix.
 	M = np.zeros((nt,N))

 	for k in range(N):
 		M[:,k] = d[0][k+l]

 	# Eigenimage decomposition

 	U,S,V = sp.linalg.svd(M)

 	"""
	MATLAB ORG
	Reconstruct with one oscillatory component at the time

	if ssa_flag~=0;
	 for k=1:p
	  u = U(:,k);
	   Mp =  (u*u')*M;;
	  R(:,k) = average_anti_diag(Mp);
	 end;
	 dp = sum(d,2);

	 else
	 Mp = zeros(size(M));
	 for k=1:p
	  u = U(:,k);
	   Mp = Mp + (u*u')*M;;
	 end;
	  dp = average_anti_diag(Mp);
	end


	 sing = diag(S);

	 """

	 # Reconstruct with one oscillatory component at the time.

	 if not ssa_flag == 0:
	 	for k in range(p):
	 		u = U[:,k]
	 		Mp = dot( dot(u, u.conj().transpose()), M )
	 		R[:,k] = aad(Mp)
	 	dp = sum(d)

	 else:
	 	Mp = M
	 	for k in range(p):
	 		u = U[:,k]
	 		Mp = Mp + dot( dot(u, u.conj().transpose(), M )

	 	dp = aad(Mp)

	 sing = S.diagonal()

	return(dp,sing,R)














