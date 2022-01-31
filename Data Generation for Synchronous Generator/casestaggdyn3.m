%% casestaggdyn3.m

% Modified parameter values from default casestaggdyn.m found in MatDyn1.2 archive [1], 
% within \MatDyn\Cases\Dynamic.
% [1] https://www.esat.kuleuven.be/electa/teaching/matdyn/MatDyn1.2

function [gen,exc,gov,freq,stepsize,stoptime] = casestaggdyn3

%% General data

freq = 50;
stepsize = 0.01;
stoptime = 20;

%% Generator data

% The first line corresponds to external grid/infinite bus, while second line is associated with 
% the SG whose dynamics are investigated.

% [genmodel excmodel govmodel   H     D       xd      xq       xd_tr   xq_tr   Td_tr Tq_tr]
gen = [1      1        1        1e3   0       0.005   0.005    0       0       0     0;
       2      1        1        1     0.01    0.93    0.77     0.50    0.50    3.2   0.81];
   
%% Exciter data

% The first line corresponds to constant excitation model. No field voltage feedback control is 
% utilized from either external grid/infinite bus or SG (excmodel is set to 1 for both in gen
% structure above).

%   [gen  Ka  Ta    Ke       Te     Kf      Tf  Aex     Bex    Ur_min  Ur_max]
exc = [1  0   0     0        0      0       0   0       0      0       0;
       2  50  0.05  -0.17    0.95   0.04    1   0.014   1.55   -1.7    1.7];
 
%% Governor data

% The first line corresponds to constant mechanical power input model. No governor feedback control 
% is utilized for either external grid/infinite bus or SG (govmodel is set to 1 for both in 
% previously defined gen structure).

%   [gen K  T1  T2  T3  Pup  Pdown  Pmax  Pmin]
gov = [1 0  0   0   0   0    0      0     0;
       2 1  100 0   0.1 0.1 -0.1    1     0];

return;