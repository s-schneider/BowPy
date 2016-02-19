function s = average_anti_diag(A);
%
% Given a Hankel matrix A,  this program retrieves
% the signal that was used to make the Hankel matrix
% by averaging along the antidiagonals of A.
%
% M.D.Sacchi
% 2008
% SAIG - Physics - UofA
% msacchi@ualberta.ca
% 
% 
% In    A: A hankel matrix
%
% Out   s: signal (column vector)
%


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
