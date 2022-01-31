%% measurements.m

clc; clear all; close all;

% Selection between IEEE 14 and 30 bus systems.

% loadcase is a built-in function from MatPower [1].
% [1] R. D. Zimmerman, C. E. Murillo-Sanchez, and R. J. Thomas, "MATPOWER: SteadyState Operations, 
% Planning, and Analysis Tools for Power Systems Research and Education," IEEE Transactions on 
% Power Systems, 26 (2011), pp. 12–19.

mpc = loadcase('case14');
% mpc = loadcase('case30');

% Location of PMU sensors.

pmu_loc = [2,7,11,13];
% pmu_loc = [1, 2, 6, 9, 10, 12, 15, 18, 25, 27];

% Number of system buses.

n_bus = size(mpc.bus,1);
lines = mpc.branch;

% Computation of measurement Jacobian matrix H. In order to establish linear relationship 
% between measurements and state vector, voltage phasors are converted into Cartesian 
% coordinates.

[H, names] = calc_Jacobian_H(n_bus, lines, pmu_loc);

meas_ts = [];

for m = 1:1:1000

    % Generation of measurement segment pertaining to 250 time-steps. Sampling rate is assumed to be
    % equal to 1/50 sec = 0.02 sec and thus 250 time-steps correspond to 5 sec.

    Vm_ts = [];
    Va_ts = [];

    for t = 1:1:250

        % Real and reactive loads are perturbed by drawing random samples from normal distribution.

        mpc.bus(:,3) = normrnd(mpc.bus(:,3),0.1);
        mpc.bus(:,4) = normrnd(mpc.bus(:,4),0.1);

        % Power flow is run to solve for voltage phasors (magnitude and angle) at all system buses.
        
        % runpf is a built-in function from MatPower [1].
        [result, success] = runpf(mpc);

        Vm = result.bus(:,8);
        Va = result.bus(:,9)/180*pi;

        Vm_ts = [Vm_ts Vm];
        Va_ts = [Va_ts Va];

    end

    % Computation of measurement, where H*[V_x_ts;V_y_ts] is a (number of measurements x number of 
    % time steps) matrix.

    V_x_ts = Vm_ts.*cos(Va_ts);
    V_y_ts = Vm_ts.*sin(Va_ts);

    meas_ts = [meas_ts; normrnd(H*[V_x_ts;V_y_ts],0.001)];

end

% The number of rows in meas_ts is a multiple of the number of measurements. By uniformly sampling 
% 10% of measurement segments, each selected segment is corrupted for a uniformly sampled number 
% of consecutive time-steps (between 10 and 50). The first time-step which is associated with a
% corruption is also uniformly sampled (between index 1 and 200).

n_meas = size(H,1);
n_segments = size(meas_ts,1)/n_meas;

idx_corrupted = datasample([1:n_segments],round(0.1*n_segments),'replace',false).*n_meas;

for k = 1:1:length(idx_corrupted)
    idx_meas = randi([1 n_meas]);
    idx_start = randi([1 200]);
    n_consecutive = randi([10 50]);

    B = H*inv(H.'*H)*H.'-eye(size(H,1));

    % Since Cartesian coordinates are utilized for measurements, every 2 rows in H correspond to the
    % same measured phasor (current or voltage) and thus we corrupt both Cartesian coordinates of
    % the same measured quantity. This necessitates the following adjustment in the index of
    % corrupted measurement.
    
    if (mod(idx_meas,2) == 0)
        idx_meas = idx_meas - 1;
    end 
    attack = idx_meas;
    
    b = zeros(1,size(H,1));
    b(attack) = 1;
    d = zeros(size(H,1)+1,1);
    
    % Different levels of corruptions are applied to coordinates pertaining to voltage phasor 
    % and current measurements. The former are injected with 0.1 in each respective Cartesian
    % coordinate, while the latter are perturbed by 0.5 again in each respective Cartesian
    % coordinate.
    
    if k > size(H,1)-2*length(pmu_loc)
        size_attack = [0.1 0.1];
    else
        size_attack = [0.5 0.5];
    end
    d(size(H,1)+1) = size_attack(1);
    C = [B;b];
    C = [B];
    d(end) = [];
    [alpha,result, result2] = compute_alpha(H, C, d, attack, size_attack(1));

    meas_ts(idx_corrupted(k):(idx_corrupted(k)+n_meas-1),idx_start:(idx_start+n_consecutive-1)) = meas_ts(idx_corrupted(k):(idx_corrupted(k)+n_meas-1),idx_start:(idx_start+n_consecutive-1)) + repmat(alpha,1,n_consecutive);
    
end

list_corrupted_segments = idx_corrupted/size(H,1);

save('measurement_time_series.mat','meas_ts','list_corrupted_segments');

% Display pattern of attack vectors congingent upon the measurement which is a priori modified by
% the malicious actor.

alpha_ = [];
sum_nnz = [];
result2_ = [];

B = H*inv(H.'*H)*H.'-eye(size(H,1));

% Since each row of measurement Jacobian matrix pertains to measurements in Cartesian coordinates,
% the index is advanced by 2.

for k = 1:2:size(H,1)
    disp(k);
    attack = k;
    b = zeros(1,size(H,1));
    b(attack) = 1;
    d = zeros(size(H,1)+1,1);
    if k > size(H,1)-2*length(pmu_loc)
        size_attack = [0.1 0.1];
    else
        size_attack = [0.5 0.5];
    end
    d(size(H,1)+1) = size_attack(1);
    C = [B;b];
    C = [B];
    d(end) = [];
    [alpha, result, result2] = compute_alpha(H, C, d, attack, size_attack(1));
    alpha_ = [alpha_ alpha];
    result2_ = [result2_ result2];
end

alpha_pattern = (abs(alpha_)>=1e-4);
A = alpha_pattern(1:2:end,:);
B = alpha_pattern(2:2:end,:);
C = A | B;
alpha_pool = sum(C,1);

print_pattern(alpha_pattern);
% set(gcf,'renderer','Painters');
% print -depsc -tiff -r300 -painters figIEEE14.eps;
% print -depsc -tiff -r300 -painters figIEEE30.eps;