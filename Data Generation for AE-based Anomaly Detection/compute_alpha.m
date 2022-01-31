%% compute_alpha.m

% Computation of attack vector alpha. 

function [alpha, result, result2] = compute_alpha(H, C, d, attack, size_attack)

result2 = [];
[n_meas, n_bus] = size(H);

model.vtype = [repmat('C',2*n_meas,1);repmat('C',2*n_bus,1)];

model.obj = [ones(2*n_meas,1); zeros(2*n_bus,1)];
model.modelsense = 'min';

model.lb = [zeros(n_meas,1);zeros(n_meas,1);zeros(2*n_bus,1)];
model.ub = [+inf(n_meas,1);+inf(n_meas,1);+inf(2*n_bus,1)];

eye_ = eye(n_meas);

model.A = sparse([zeros(1,2*n_meas) H(attack,:) -H(attack,:);zeros(1,2*n_meas) H(attack+1,:) -H(attack+1,:);-eye(n_meas) eye(n_meas) H -H;-eye(n_meas) eye(n_meas) H -H]);
model.rhs = [size_attack; size_attack; zeros(n_meas,1);zeros(n_meas,1)];
model.sense = [repmat('=', 1, 1);repmat('=', 1, 1);repmat('=', n_meas, 1);repmat('=', n_meas, 1)];

params.timelimit = 600;

params.outputflag = 0;

% Gurobi optimizer [1] is utilized to solve the norm-1 based optimization 
% probem which is coded as an equivalent linear programming problem.
% [1] Gurobi Optimization, LLC, Gurobi Optimizer Reference Manual, 2021.

result = gurobi(model, params);
disp(result);

if strcmp(result.status, 'INFEASIBLE')
    result2 = gurobi_iis(model);
    alpha = [];
else
    u = result.x(1:n_meas);
    v = result.x(n_meas+1:2*n_meas);
    alpha = u - v;
end

end