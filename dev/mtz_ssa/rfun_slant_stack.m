function [S, tau, deg] = rfun_slant_stack(d, t, p, dt)

% Setup intial parameters
p0 = taupTime('prem', 0, 'P', 'deg', 60); p0 = p0(1).rayParam; p0 = p0 ./ (180 / pi);
% p0 = 7;

[~, leadi] = min(abs(t));
nt = size(d, 1);

% Construct axes
ddeg = 0.05;
deg = -5:ddeg:5;
tau = t;

S = zeros(length(deg), length(tau));

for n = 1:size(d, 2)
    for l = 1:length(tau)
        for m = 1:length(deg)
            thist = tau(l) + deg(m)*(p(n) - p0);
            
            thisi = leadi + round(thist ./ dt);
            
            if thisi > 0 & thisi < nt
                S(m, l) = S(m, l) + d(thisi, n);
            end
        end
    end
end

S = S ./ size(d, 2);