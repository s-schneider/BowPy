import numpy as np


def average_anti_diag(A):
	"""
	Given a Hankel matrix A,  this program retrieves
	the signal that was used to make the Hankel matrix
	by averaging along the antidiagonals of A.

	M.D.Sacchi
	2008
	SAIG - Physics - UofA
	msacchi@ualberta.ca


	In    A: A hankel matrix

	Out   s: signal (column vector)
	"""

	"""
	MATLAB
	[m,n] = size(A);
	N = m+n-1;

	 s = zeros(N,1);

	 for i = 1 : N

	  a = max(1,i-m+1);
	  b = min(n,i);

	   for k = a : b
	    s(i,1) = s(i,1) + A(i-k+1,k);
	   end

	 s(i,1) = s(i,1)/(b-a+1);

	 end;
 	"""

 	m,n = A.shape

 	N = m+n-1

 	s = np.zeros((N,1))

 	for i in range(N):
 		a = max(1,i-m+1)
 		b = min(n,1)

 		for k in range(a,b):
 			s[i,0] = s[i,1] + A[i-k+1,k]

 		s[i,1]= s[i,1] / (b-a+1)
 		
	return(s)







