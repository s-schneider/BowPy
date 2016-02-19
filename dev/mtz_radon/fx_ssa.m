function [DATA_f] = fx_ssa(DATA,dt,p,flow,fhigh);
%FX_SSA: Singular Spectrum Analysis in the fx domain for snr enhancement
%
%
% [DATA_f] = fx_ssa(DATA,dt,p,flow,fhigh);
%
%  IN   d:      data (traces are columns)
%       dt:     samplimg interval
%       p:      number of singular values used to reconstuct the data
%       flow:   min  freq. in the data in Hz
%       fhigh:  max  freq. in the data in Hz
%
%
%  OUT  DATA_f:  filtered data
%
%  Example:
%
%        d = linear_events;
%        [df] = fx_ssa(d,0.004,4,1,120);
%        wigb([d,df]);
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



 [nt,ntraces] = size(DATA);
 nf = 2*2^nextpow2(nt);
 
 DATA_FX_f = zeros(nf,ntraces);

% First and last samples of the DFT.

 ilow  = floor(flow*dt*nf)+1; 

  if ilow<1; 
   ilow=1; 
  end;

 ihigh = floor(fhigh*dt*nf)+1;

  if ihigh > floor(nf/2)+1; 
   ihigh=floor(nf/2)+1; 
  end

% Transform to FX

 DATA_FX = fft(DATA,nf,1);
 DATA_FX_f = zeros(size(DATA_FX));


 nw = floor(ntraces/2);

 for k = ilow:ihigh;

	 tmp  = DATA_FX(k,:).';

for j = 1:10
[tmp_out] = ssa(tmp,nw,p,0);
tmp = tmp_out;

end;


 DATA_FX_f(k,:) = tmp_out;

end;

% Honor symmetries

 for k=nf/2+2:nf
  DATA_FX_f(k,:) = conj(DATA_FX_f(nf-k+2,:));
 end

% Back to TX (the output) 

 DATA_f = real(ifft(DATA_FX_f,[],1));
 DATA_f = DATA_f(1:nt,:);

return
