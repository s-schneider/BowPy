function  [dp,sing,R] = ssa(d,nw,p,ssa_flag);
%SSA: 1D Singular Spectrum Analysis for snr enhancement
%
%  [dp,sing,R] = ssa(d,nw,p,ssa_flag);
%
%  IN   d:   1D time series (column)
%       nw:  view used to make the Hankel matrix
%       p:   number of singular values used to reconstuct the data
%       ssa_flag = 0 do not compute R
%
%  OUT  dp:  predicted (clean) data
%       R:   matrix consisting of the data predicted with
%            the first eof (R(:,1)), the second eof (R(:,2)) etc
%       sing: singular values of the Hankel matrix
%
%  Example:
%
%       d = cos(2*pi*0.01*[1:1:200]') + 0.5*randn(200,1)
%       [dp,sing,R] = ssa(d,100,2,1); plot(d); hold on; plot(dp+3);
%
%  Based on: 
%
%  M.D.Sacchi, 2009, FX SSA, CSEG Annual Convention, Abstracts,392-395.
%                    http://www.geoconvention.org/2009abstracts/194.pdf
%
%  Copyright (C) 2008, Signal Analysis and Imaging Group.
%  For more information: http://www-geo.phys.ualberta.ca/saig/SeismicLab
%  Author: M.D.Sacchi
%
%  This program is free software: you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published
%  by the Free Software Foundation, either version 3 of the License, or
%  any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details: http://www.gnu.org/licenses/
%


 [nt] = length(d);
 N = nt-nw+1;
 l = 1:1:nw;

 M = [];   

% Make Hankel Matrix 


 for k=1:N
  M(:,k) = d(k+l-1,1);
 end;

% Eigenimage decomposition

 [U,S,V] = svd(M);

% Reconstruct with one oscillatory component at the time

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

return

