%% casestagg2.m

% Modifications from default casestagg.m found in MatDyn1.2 archive [1], within 
% \MatDyn\Cases\Powerflow.
% [1] https://www.esat.kuleuven.be/electa/teaching/matdyn/MatDyn1.2
% Virtual bus 3 is added to represent the fault point in one of the transmission lines. Also, fault
% location, i.e. loc, is explicitly incorporated in branch structure to control for the resistance
% r, reactance x, and susceptance b of the two line segments into which one of the transmission
% lines virtually gets split into once the fault appears.

function [baseMVA, bus, gen, branch, area, gencost] = casestagg2

global loc;

%%-----  Power Flow Data  -----%%
%% system MVA base
baseMVA = 100;

%% bus data
%      bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
bus = [1        3       0	0	0	0	1       1	0	345     1   	1.1     0.9;
       2       	1       20	10	0	0	1       1	0	345     1   	1.1     0.9;
       3        2       0   0   0   0   1       1   0   345     1       1.1     0.9];

%% generator data
%      bus	Pg	Qg	Qmax	Qmin	Vg      mBase	status	Pmax	Pmin
gen = [1	0	0	300     -300	1.06	100     1   	250 	10;
       2	40	30	300     -300	1.06	100     1       300     10];

%% branch data
%         fbus	tbus	r               x               b               rateA	rateB	rateC	ratio	angle	status
branch = [1     3   	loc*0.04	    loc*0.1         loc*0.030*2	    250     250     250     0       0   	1;
          3     2       (1-loc)*0.04	(1-loc)*0.1     (1-loc)*0.030*2	250     250     250     0       0   	1;
          1     2       0.04            0.1             0.030*2         250     250     250     0       0   	1];

area=[];
gencost=[];

return;