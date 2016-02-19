% This script is intended to give a sample of applications for the Radon
% transform.

clear;

% Load variables.

load('data.mat','-mat');
% t        - time axis.
% Delta    - distance (offset) axis.
% M        - Amplitudes of phase arrivals.
% indicies - list of indicies relevent to the S670S phase.

% Define some variables for RT.
mu=5e-2;
P_axis=-1:0.01:1;
delta=mean(Delta);

% Invert to Radon domain using unweighted L2 inversion, linear path
% functions and an average distance parameter.
tic;
R=Radon_inverse(t, Delta, M, P_axis, ones(size(Delta)), delta, 'Linear', 'L2', mu);
toc

% Mute all phases except the S670S arrival.
R670=zeros(size(R));
R670(indicies)=1;
R670=R.*R670;

% Apply forward operator to the muted Radon domain.
Delta_resampled=floor(min(Delta)):(ceil(max(Delta))-floor(min(Delta)))/20:ceil(max(Delta));
M670=Radon_forward(t, P_axis, R670, Delta_resampled, delta, 'Linear');


% Plot figures.
figure(2); clf;

subplot(311); imagesc(t,Delta,M);
title('Aligned SS'); xlabel('Time (s)'); ylabel('Distance (deg)');

subplot(312); imagesc(t, P_axis,  abs(hilbert(R'))');
title('L2 Radon Inversion (Hilbert Envelope)'); ylabel('Ray Parameter (s/deg)'); xlabel('Time (s)');

subplot(313); imagesc(t,Delta_resampled, M670);
title('S670S Seismic Energy'); xlabel('Time (s)'); ylabel('Distance (deg)');

