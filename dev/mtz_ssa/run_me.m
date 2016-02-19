clear all; close all; clc;

load ../slant/test_ssa.mat;

pstack = pstack(1:500,:);
t = t(1:500);


%% Apply taper to decrease amplitude of PP 

din = taper(pstack, 20, 1);
din = din/max(din(:));

% Compute the Sampling Operator S

T = ones(size(din));

[nt,nx] = size(din);

% Compute the Sampling Operator S

for ix = 1:nx
 a = sum(din(:,ix)); if a ==0; T(:,ix)=0;end;
end;

% Parameters for SSA/CAZDOW reconstruction

rank_p =5;
alpha = 0.4;
n_iter = 8;

dt = 1;
fmin = 0.01;
fmax = 0.6;
 
% Reconstruction 

[d] = reconstruction(din,T,dt,fmin,fmax,rank_p,alpha,n_iter);

% Compare slant

[S, ~, ~] = rfun_slant_stack(din, t, p*111.4, 0.2);
[S2, tau, deg] =rfun_slant_stack(d, t, p*111.4, 0.2);

%%
figure();
pcolor(clip([din, d], 50, 50)); shading interp; axis ij;

figure(2);
pcolor(deg, tau, S'); shading interp; axis ij; axis([-4 4 -10 100]); caxis([0 0.1]);
hold on;
load peaks.mat
plot(ys, xs, '.k');

figure(3);
pcolor(deg, tau, S2'); shading interp; axis ij; axis([-4 4 -10 100]); caxis([0 0.1]);
hold on;
load peaks.mat
plot(ys, xs, '.k');
