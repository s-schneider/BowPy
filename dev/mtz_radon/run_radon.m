clear all; close all; clc;

% load /home/alex/work/CRANE_recfxn/new_rfun/run_CA/idecon/MTZ_final/idecon_withdata.mat
% load /home/alex/work/CRANE_recfxn/new_rfun/apply_cor/corrected_mtz_new_fine.mat
load /home/alex/sean/contenti_work/work/CRANE_recfxn/new_rfun/apply_cor/corrected_mtz_new_fine.mat

% addpath('/home/contenti/raid5/contenti/mcodes/');
% addpath('/home/contenti/mcodes/');
% addpath('./Radon/');
% setpath;
% addpath('/home/contenti/raid5/contenti/mcodes/regu');
% addpath('/home/contenti/raid5/contenti/mcodes/extrema/');
% out = '/home/contenti/work/CRANE_recfxn/new_rfun/images/by_station/';
% addpath('~/raid5/contenti/mcodes/');

addpath('/home/alex/sean/contenti_work/raid5/mcodes/');
addpath('/home/alex/sean/contenti_work/work/mcodes/');
addpath('./Radon/');
setpath;
addpath('/home/alex/sean/contenti_work/raid5/mcodes/regu');
addpath('/home/alex/sean/contenti_work/raid5/mcodes/extrema/');
out = '/home/alex/sean/contenti_work/work/CRANE_recfxn/new_rfun/images/by_station/';
addpath('/home/alex/sean/contenti_work/raid5/mcodes/');
addpath('/home/alex/sean/contenti_work/raid5/mcodes/FMI/matTaup');
javaaddpath('/home/alex/sean/contenti_work/raid5/mcodes/FMI/lib/FMI.jar');


stalist = unique({log.sta});

t = log(1).tax;
dt = mean(diff(t));
lead = -t(1);
ilead = round(lead ./ dt);

% Select data subset
fthresh = 80;
snrthresh = 2;
ikeep = intersect(find( [log.idecon_dfit] > fthresh ) , find( [log.snr] > 2) );
% ikeep = find( [log.idecon_dfit] > 80 );
log = log(ikeep);
ND = length(log);

% Parameters for SSA/CAZDOW reconstruction
rank_p =5;
alpha = 0.4;
n_iter = 8;
dt = 1;
fmin = 0.01;
fmax = 0.6;
%%%%

dd = 1;
d = 30:dd:90;

% stalist = {'WALA'};
mindat = 25;
stalist = {'overall'}; % to trick code for overall result.

% for n = 1:length(stalist)
n=1;

    tic;
    disp(['Begin Station: ', stalist{n}]);
    this = log(strcmp({log.sta}, stalist{n}));
    
%      index = strcmp({log.sta}, 'CLA') | strcmp({log.sta}, 'LYA');
%      index = strcmp({log.sta}, 'WALA');
%      this = log(index);
 imSta = {'WALA'};
%     
       this = log;
    
    if length(this) < mindat
        disp('Breaking loop execution');
        bad(n) = 1;
        continue;
    end
    
    slat(n) = mean([this.slat]);
    slon(n) = mean([this.slon]);
    cor(n) = mean([this.cor]);
    
    % Partial stack for some regularization
    t = log(1).tax;
    pstack = zeros(length(t), length(d));
taup_arrival = zeros(length(d), length(imSta));
taup_arrival410s = zeros(length(d), length(imSta));
taup_arrival660s = zeros(length(d), length(imSta));

    for id = 1:length(d)
        mind = d(id) - 0.5*dd;
        maxd = d(id) + 0.5*dd;
        sub = this( [this.dist] > mind & [this.dist] <= maxd );
        if length(sub) >= 1
            pstack(:, id) = sum([sub.itr_dat_cor], 2) ./ length(sub);
            h(:, id) = imag(hilbert(pstack(:, id)));
            
                        % Get PREM arrival
            t1 = taupTime('prem', mean([sub.edep]), 'P', 'deg', mean([sub.dist]));
            t2 = taupTime('prem', mean([sub.edep]), 'Pms', 'deg', mean([sub.dist]));
            t3 = taupTime('prem', mean([sub.edep]), 'P410s', 'deg', mean([sub.dist]));
            t4 = taupTime('prem', mean([sub.edep]), 'P660s', 'deg', mean([sub.dist]));
            
            taup_arrival(id, n) = t2(1).time - t1(1).time;
            taup_arrival410s(id, n) = t3(1).time - t1(1).time;
            taup_arrival660s(id, n) = t4(1).time - t1(1).time;
            
        end
    end
                                                                                                                                                                 
    disp('Partial Stacking Complete!');
    toc
    
       
        %% Interpolate and denoise
% rank_p =3;
% alpha = 0.05;
% n_iter = 20;
% dt = 0.2;%%%%%% raw =1%%%
% fmin = 0.01;
% fmax = 2.0;
%     
    pstack = pstack(1:475,:);
    t = t(1:475);
    % Apply taper to decrease amplitude of P
    din = taper(pstack, 20, 1);
    din = din/max(din(:));
    % Compute the Sampling Operator S
    T = ones(size(din));
    [nt,nx] = size(din);
    for ix = 1:nx
        a = sum(din(:,ix)); if a ==0; T(:,ix)=0;end;
    end;
    % Reconstruction
    [pstack_interp] = reconstruction(din,T,dt,fmin,fmax,rank_p,alpha,n_iter);
    pstack_interp = taper(pstack_interp, 10, 5);
    % Notify me
    disp('SSA Denoising and Interpolation Complete!');
    toc
    
    %% Radon inversion
    % Define some variables for RT.
    muv=logspace(0, -3, 30);
    dp = 0.0004;
%     p = -0.2:dp:0.2;
%     p = linspace(-(0.05^2), (0.05^2), 200);
    p = linspace(-0.015, 0.015, 400);
    
    tstack = pstack_interp(151:end, :);
%     tstack = pstack(151:end, :);
%       tstack = taper(tstack, 5, 0); tstack = [zeros(150, length(d)); tstack];
       tstack = pstack_interp; % No muting

    % Invert to Radon domain using unweighted L2 inversion, parabolic path
    % functions and an average distance parameter.
    for l = 1:length(muv)
        mu = muv(l);
        R=Radon_inverse(t, d, tstack', p, ones(size(d)), 60, 'Parabolic', 'L2', mu);
        dh=Radon_forward(t, p, R, d, 60, 'Parabolic');
        misfit(l) = norm(tstack' - dh)^2;
        dnorm(l) = norm(R)^2;
        disp(['Radon Inversion Iteration: ', num2str(l)]);
        toc
    end
    
    [k_corner, info] = corner(dnorm, misfit);
    mu = muv(k_corner);
    muf(n) = mu;
    
    R=Radon_inverse(t, d, tstack', p, ones(size(d)), 60, 'Parabolic', 'L2', mu);
    dh=Radon_forward(t, p, R, d, 60, 'Parabolic');
%     
%     save([out, stalist{n}, '/', stalist{n},  '_PRT_SSA_cor_new_fine.mat'], 'p', 't', 'R', 'd', 'dh');
%     load([out, stalist{n}, '/', stalist{n},  '_PRT_SSA_cor_new_fine.mat']);
    
    
    %% Images
    addpath('/home/alex/sean/contenti_work/work/mcodes/cm_and_cb_utilities/');
    addpath('/home/alex/sean/contenti_work/work/mcodes/freezeColors/');
%     close all;
    s = sign(R);
    dmax = max(max(R.^2));
    [zmax, imax, ~, ~] = extrema2((s.*(R.^2))');
    [i, j] = ind2sub(size((s.*(R.^2))'), imax); j = j(1:25); i = i(1:25);
    
%     figure(1);clf
    set(figure(1),'color','white');
    subplot(2, 2, 3);
%     imagesc(p, t, (s.*(R.^2))'); 
%     imagesc(p, t, R'); 
   imagesc(p, t, clip(taper(R', 25, 10), 30, 30));  axis ij; 
    
    
    axis ij;
    hline(43.67);
    hline(68.34);
    axis([-0.003 0.003 0 85]);
%     axis([ -0.0030    0.0030   -5.0000   85.0000]); % For overall
    set(gca, 'FontSize', 17);
    xlabel('Squared Ray Parameter');
    ylabel('Intercept Time [s]');
     title([stalist{n}, ' Radon spectrum']);
    hold on;
    plot(-0.00035, 43.75, 'xk', 'MarkerSize', 10);
    plot(-0.00035, 43.75, 'ok', 'MarkerSize', 10);
    plot(-0.00118, 68.5, 'ok', 'MarkerSize', 10);
    plot(-0.00118, 68.5, 'xk', 'MarkerSize', 10);
    
    plot(p(j), t(i), 'xk');
    
    a = ((p(j) + 0.00035)./0.01).^2;
    b = ((t(i)' -  43.75)./30).^2;
    d410 = sqrt( a + b );
    
    a = ((p(j) + 0.00118)./0.01).^2;
    b = ((t(i)' -   68.5)./30).^2;
    d660 = sqrt( a + b );
    
    [~, i410] = min(d410);
    [~, i660] = min(d660);
    
    plot(p(j(i410)), t(i(i410)), 'xr');
    plot(p(j(i660)), t(i(i660)), 'xr');
  
    myd410(n) = depth_convert('time', t(i(i410)));
    myd660(n) = depth_convert('time', t(i(i660)));
    
    myp410(n) = p(j(i410));
    myp660(n) = p(j(i660));
    
    myA410(n) = R(j(i410), i(i410));
     myA660(n) = R(j(i410), i(i410));

    c = caxis;
    caxis([-0.75*dmax dmax])
%     caxis(0.15*c)
    caxis([-myA410(n), myA410(n)]);

%     caxis(0.005*c)
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperPosition', [0.25 0.25 4 8]);
    set(gcf, 'PaperPosition', [2 1 6.4 8.25]);  
    colorbar
%     print(gcf, '-depsc', '-painters', [out, stalist{n}, '/', stalist{n},  '_PRT_SSA_cor_new.eps']);

    
%    set(figure(2),'color','white');
    subplot(2, 2, 1);
    imagesc(d, t, clip(taper(pstack, 25, 10), 30, 30));  axis ij; colormap(seismic(3));
    axis([min(d) max(d) 0 85]);
    hold on;
    plot(d, taup_arrival(:, n), '.g');
    plot(d, taup_arrival410s(:, n), '.g');
    plot(d, taup_arrival660s(:, n), '.g'); 
    set(gca, 'FontSize', 17);
    title([stalist{n}, ' Raw+Cor']);
    xlabel('Distance [deg]');
    ylabel('Time [s]');
    
    subplot(2, 2, 2);
%     imagesc(d, t, clip(taper(pstack_interp, 25, 10), 30, 30));  axis ij; colormap(seismic(3));
    imagesc(d, t, clip(taper(pstack_interp, 25, 10), 30, 30));  axis ij; colormap(seismic(3));
     set(gca, 'FontSize', 17);
    axis([min(d) max(d) 0 85]);
    hold on;
    plot(d, taup_arrival(:, n), '.g');
    plot(d, taup_arrival410s(:, n), '.g');
    plot(d, taup_arrival660s(:, n), '.g'); 
    title([stalist{n}, ' Raw+cor+SSA']);
    xlabel('Distance [deg]');
    ylabel('Time [s]');
    
    
    subplot(2, 2, 4);
    imagesc(d, t, clip(taper(dh', 25, 10), 30, 30));  axis ij; colormap(seismic(3));
     set(gca, 'FontSize', 17);
    axis([min(d) max(d) 0 85]);
    hold on;
    plot(d, taup_arrival(:, n), '.g');
    plot(d, taup_arrival410s(:, n), '.g');
    plot(d, taup_arrival660s(:, n), '.g'); 
    title([stalist{n}, ' Raw+cor+SSA+LSPRT']);
    xlabel('Distance [deg]');
    ylabel('Time [s]');
%     set(gcf, 'PaperPositionMode', 'manual');
%     set(gcf, 'PaperUnits', 'inches');
%     set(gcf, 'PaperPosition', [2 1 6.4 8.25]);
%     print(gcf, '-depsc', '-painters', [out, stalist{n}, '/', stalist{n},  '_PRT_SSA_cor_input_new.eps']);
% % print(gcf, '-depsc', '-painters', ['group5_PRT_SSA_cor_input.eps']);

    disp(['Station Complete: ', stalist{n}]);
%     toc
% end