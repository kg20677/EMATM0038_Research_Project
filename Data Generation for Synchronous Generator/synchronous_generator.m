%% synchronous_generator.m

% Code for generating time series of state and exogenous variables based on the response 
% of synchronous generator (SG) due to the emergence of three phase faults in one of the parallel 
% transmission lines which connect the SG via a transformer with the external grid.
% This code stems from additions and modifications to Test1.m found in MatDyn1.2 archive [1].
% [1] https://www.esat.kuleuven.be/electa/teaching/matdyn/MatDyn1.2

clc; clear all; close all;

global loc Rf;
global t1 t2;
mdopt = Mdoption;

t1 = 1; % time point pertaining to the inception of three phase fault (start of disturbance)
t2 = 1.12; % time point corresponding to clearance of three phase fault (end of disturbance)

Angles_ = [];
Speeds_ = [];
Eq_tr_ = [];
Ed_tr_ = [];
Efd_ = [];
PM_ = [];
Voltages_ = []; 

% Three phase faults are applied at different locations of one of the parallel transmission lines,
% in increments of 5% starting from 5% and ending at 95% of distance from the terminal bus.

for loc = 0.05:0.05:0.95

    % By sampling fault resistance values for each three phase fault, additional variability is 
    % induced among the experiments beyond the different fault location.

    Rf = 1e0 + (1e4-1e1) .* rand;
    loc
    
    % Settings according to MatDyn1.2 documentation [2].
    % [2] https://www.esat.kuleuven.be/electa/teaching/matdyn/MatDyn1.2manual

    mdopt(1)=5; % integration method, set as modified Euler with interface error control
    mdopt(2)=1e-4; % tolerance
    mdopt(3)=1e-4; % minimum step size
    mdopt(4)=1e-4; % maximum step size
    mdopt(5)=1; % output progress
    mdopt(6)=0; % no plots
    
    % casestagg2, casestaggdyn3, and staggevent2 are also provided in the repository. These files 
    % stem from modifications of default casestagg, casestaggdyn, and staggevent files. 
    % casestagg2: involves network topology and parameters, as well as the provision for a virtual 
    % bus to resemple the fault point in the transmission line;
    % casestaggdyn3: involves parameter values of SG, infinite bus, and controllers (both field
    % voltage and mechanical input power were assumed constant in the absence of any related 
    % feedback control mechanism);
    % staggevent2: defines the event (three phase fault) based on its start (t1) and end (t2) time 
    % points, as well as fault resistance Rf.

    [Angles,Speeds,Eq_tr,Ed_tr,Efd,PM,Voltages,Stepsize,Errest,Time]=rundyn('casestagg2','casestaggdyn3','staggevent2',mdopt);
    
    % Aggregation of time series data in structures of the form:
    % (number of time points) x (number of experiments). 
    % Index 2 in Angles(:,2), Speeds(:,2), etc correspond to respective variables associated with
    % the SG, while index 1 correspond to the same variables pertaining to the infinite bus/external
    % grid.

    Angles_ = [Angles_ Angles(:,2)];
    Speeds_ = [Speeds_ Speeds(:,2)];
    Eq_tr_ = [Eq_tr_ Eq_tr(:,2)];
    Ed_tr_ = [Ed_tr_ Ed_tr(:,2)];
    Efd_ = [Efd_ Efd(:,2)];
    PM_ = [PM_ PM(:,2)];
    Voltages_ = [Voltages_ Voltages(:,2)]; 
        
end

% Computation of time series of terminal voltage phasor angle, d- and q- axis voltages and stator 
% currents, and electric power output.

xd_tr = 0.50;
xq_tr = 0.50;
theta = angle(Voltages_)/pi*180;
vd = -abs(Voltages_).*sin(Angles_-theta);
vq = abs(Voltages_).*cos(Angles_-theta);
Id = (vq - Eq_tr_)./xd_tr;
Iq =-(vd - Ed_tr_)./xq_tr;
PE_ = Eq_tr_.*Iq + Ed_tr_.*Id + (xd_tr - xq_tr).*Id.*Iq;

% Removal of duplicates from time series data.

[uniqueTime i j] = unique(Time,'first');
indexToDupes = find(not(ismember(1:numel(Time),i)));
Time(indexToDupes) = [];
Angles_(indexToDupes,:) = [];
Speeds_(indexToDupes,:) = [];
Eq_tr_(indexToDupes,:) = [];
Ed_tr_(indexToDupes,:) = [];
Efd_(indexToDupes,:) = [];
Voltages_(indexToDupes,:) = [];
PM_(indexToDupes,:) = [];
PE_(indexToDupes,:) = [];
vd(indexToDupes,:) = [];
vq(indexToDupes,:) = [];
Id(indexToDupes,:) = [];
Iq(indexToDupes,:) = [];

% Segments of time series pertaining to the time-frame during which the three phase fault persists 
% are pruned and only the post-fault response is retained. 

Angles_(Time < 1.13,:) = [];
Speeds_(Time < 1.13,:) = [];
Eq_tr_(Time < 1.13,:) = [];
Ed_tr_(Time < 1.13,:) = [];
Efd_(Time < 1.13,:) = [];
Voltages_(Time < 1.13,:) = [];
PM_(Time < 1.13,:) = [];
PE_(Time < 1.13,:) = [];
vd(Time < 1.13,:) = [];
vq(Time < 1.13,:) = [];
Id(Time < 1.13,:) = [];
Iq(Time < 1.13,:) = [];
Time(Time < 1.13,:) = [];

% Since steady-state is attained well before 8 seconds, any further segments of time series beyond 
% 8 seconds are also pruned, since they do not contain any informative content.

Angles_(Time > 8,:) = [];
Speeds_(Time > 8,:) = [];
Eq_tr_(Time > 8,:) = [];
Ed_tr_(Time > 8,:) = [];
Efd_(Time > 8,:) = [];
Voltages_(Time > 8,:) = [];
PM_(Time > 8,:) = [];
PE_(Time > 8,:) = [];
vd(Time > 8,:) = [];
vq(Time > 8,:) = [];
Id(Time > 8,:) = [];
Iq(Time > 8,:) = [];
Time(Time > 8,:) = [];

Time = Time - 1.13;

% Time series are saved as .mat in order to be imported into the respective Python and Julia codes 
% for conducting SINDy and UDE-assisted SINDy.

save('time_series.mat','Angles_','Speeds_','Eq_tr_','Ed_tr_','Efd_','PM_','PE_','Voltages_','Time');
save('time_series_with_algebraic.mat','Angles_','Speeds_','Eq_tr_','Ed_tr_','Efd_','PM_','PE_','Voltages_','vd','vq','Id','Iq','Time');

%% 2D phase portraits

figure;
plot(Eq_tr_(:,:),Ed_tr_(:,:));
grid on; grid minor;
xlabel('q-axis transient voltage, E_q (pu)');
ylabel('d-axis transient voltage, E_d (pu)');

figure;
plot(Eq_tr_(:,:),Efd_(:,:));
grid on; grid minor;
xlabel('q-axis transient voltage, E_q (pu)');
ylabel('field excitation voltage, E_{fd} (pu)');

figure;
plot(Ed_tr_(:,:),Efd_(:,:));
grid on; grid minor;
xlabel('d-axis transient voltage, E_d (pu)');
ylabel('field excitation voltage, E_{fd} (pu)');

figure;
plot(Eq_tr_(:,:),Speeds_(:,:));
grid on; grid minor;
xlabel('q-axis transient voltage, E_q (pu)');
ylabel('angular velocity, \omega (pu)');

figure;
plot(Ed_tr_(:,:),Speeds_(:,:));
grid on; grid minor;
xlabel('d-axis transient voltage, E_d (pu)');
ylabel('angular velocity, \omega (pu)');

figure;
plot(Efd_(:,:),Speeds_(:,:));
grid on; grid minor;
xlabel('field excitation voltage, E_{fd} (pu)');
ylabel('angular velocity, \omega (pu)');

%% Gradual depiction of evolution associated with a pair of state variables

figure;
h = plot(Eq_tr_(:,1),Ed_tr_(:,1),'LineWidth',2);
ax = gca;
ax.XLabel.String = 'q-axis transient voltage, E_q (pu)';
ax.YLabel.String = 'd-axis transient voltage, E_d (pu)';
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.XMinorGrid = 'on';
ax.YMinorGrid = 'on';
axis([min(Eq_tr_(:,1)) max(Eq_tr_(:,1)) min(Ed_tr_(:,1)) max(Ed_tr_(:,1))])
for i = 1:length(Time)
    set(h,'Xdata',Eq_tr_(1:i,1),'Ydata',Ed_tr_(1:i,1));
    drawnow;
end