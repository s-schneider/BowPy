clear all; close all; clc;
addpath('/home/contenti/raid5/contenti/mcodes/');

deg = 30:90;
dt = 0.2;

lead = 15;
ilead = round(lead ./ dt);

for n = 1:length(deg)
    
    P = taupTime('prem', 0, 'P', 'deg', deg(n));
    P = P(1).time;
    
    P200s = taupTime('prem', 0, 'P200s', 'deg', deg(n));
    P200s = P200s(1).time;
    
    P410s = taupTime('prem', 0, 'P400s', 'deg', deg(n));
    P410s = P410s(1).time;
    
    P670s = taupTime('prem', 0, 'P670s', 'deg', deg(n));
    P670s = P670s(1).time;
    
    dt200 = P200s - P;
    dt410 = P410s - P;
    dt670 = P670s - P;
    
    i200 = round(dt200 ./ dt);
    i410 = round(dt410 ./ dt);
    i670 = round(dt670 ./ dt);
    
    d( (ilead + 1), n) = 1;
    d( (ilead+i200), n) = 0.25;    
    d( (ilead+i410), n) = 0.25;
    d( (ilead+i670), n) = 0.25;

end

% Add buffer to end
d = [d; zeros(ilead, size(d, 2))];
[w,tw] = ricker(10,0.004);
d = conv2(d, w, 'same');

% Time axis
t = (0:size(d, 1)-1).*dt - lead;

% Add noise
SNR = 3;
n = 2*rand(size(d))-1;
n = repmat(n, 3, 3);
n = conv2(n, ones(10, 10), 'same');
n = n ./ max(max(abs(n)));
n = n((length(t)+1):2*length(t), (length(deg)+1):2*length(deg));

d = d + (n ./ SNR);

p = linspace(-0.04, 0.04, 200);
p = linspace(-0.015, 0.015, 200);
% p = linspace(-0.005, 0.005, 200);


% Invert to Radon domain using unweighted L2 inversion, linear path
% functions and an average distance parameter.
% muv=logspace(0, -3, 10);
% for l = 1:length(muv)
%     mu = muv(l);
%     tic;
%     R=Radon_inverse(t, deg, d', p, ones(size(deg)), 60, 'Parabolic', 'L2', mu);
%     toc
%     dh=Radon_forward(t, p, R, deg, 60, 'Parabolic');
%     misfit(l) = norm(d' - dh)^2;
%     dnorm(l) = norm(R)^2;
% end
% [k_corner, info] = corner(misfit, dnorm, 200);

%%
mu = muv(6);
tic;
R=Radon_inverse(t, deg, d', p, ones(size(deg)), 60, 'Parabolic', 'L2', mu);
toc

dh=Radon_forward(t, p, R, deg, 60, 'Parabolic');

%% Reference times
P = taupTime('prem', 0, 'P', 'deg', 60);
P = P(1).time;

P200s = taupTime('prem', 0, 'P200s', 'deg', 60);
P200s = P200s(1).time;

P410s = taupTime('prem', 0, 'P400s', 'deg', 60);
P410s = P410s(1).time;

P670s = taupTime('prem', 0, 'P670s', 'deg', 60);
P670s = P670s(1).time;

dt200 = P200s - P;
dt410 = P410s - P;
dt670 = P670s - P;

%% Plots
close all;
S = sign(R');
figure();
pcolor(p, t, S.*clip((R').^2, 5, 5)); shading interp
ylabel('Delay Time [s]');
xlabel('Moveout [s/km]');
hold on;
hline(dt200)
hline(dt410)
hline(dt670)
axis ij;

figure();
subplot(121);
pcolor(deg, t, clip(taper(d, 25, 10), 30, 30)); shading interp; axis ij; colormap(seismic(3));
axis([min(deg) max(deg) 0 85])
title(['Data']);
xlabel('Distance [deg]');
ylabel('Time [s]')

subplot(122);
pcolor(deg, t, clip(taper(dh', 25, 10), 30, 30)); shading interp; axis ij; colormap(seismic(3));
axis([min(deg) max(deg) 0 85])
title(['Modeled Data']);
xlabel('Distance [deg]');
ylabel('Time [s]')

