# UDE_synchronous_generator.jl

# The following code is based on the Lotka Volterra example found in [1]. 
# Additions and modifications to accomondate the problem of deducing elusive 
# synchronous generator dynamics are made.
# [1] https://github.com/ChrisRackauckas/universal_differential_equations

using DifferentialEquations
using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra
using DiffEqSensitivity
using DiffEqFlux
using Flux
using Optim
using Sundials
using NLsolve
using Plots
gr()
using JLD2, FileIO
using Statistics
using Random
using MAT

# Simulated time series from MatDyn are loaded.

vars = matread("time_series_with_algebraic.mat")
Time = vars["Time"]
Angle = vars["Angles_"]/180*pi
Speed = vars["Speeds_"]*(2*pi*50)
Eq_tr = vars["Eq_tr_"]
Ed_tr = vars["Ed_tr_"]
Efd = vars["Efd_"]
Voltage = vars["Voltages_"]
V_q = vars["vq"]
V_d = vars["vd"]
I_d = vars["Id"]
I_q = vars["Iq"]
V_mag_ = abs.(Voltage)
V_ang_ = angle.(Voltage)
Pm = vars["PM_"]
Pe = vars["PE_"]

X = Array([Angle[:,1] Speed[:,1] Eq_tr[:,1] Ed_tr[:,1] V_d[:,1] V_q[:,1] I_d[:,1] I_q[:,1]])
t = Time

number_of_time_stamps = 50
X = X[1:number_of_time_stamps,:]
t = t[1:number_of_time_stamps]
tspan = (minimum(t),maximum(t))

# Gaussian noise is added to state measurements. 

x_mean = mean(X, dims = 1)
noise_magnitude = Float32(1e-4)
X_noisy = X .+ (noise_magnitude*x_mean) .* randn(eltype(X), size(X))
u0 = X_noisy[1,:]
X_noisy = transpose(X_noisy)

# Gaussian noise is added to exogenous measurements. 

Y = Array([V_mag_[:,1] V_ang_[:,1]])
Y = Y[1:number_of_time_stamps,:]
y_mean = mean(Y, dims = 1)
noise_magnitude = Float32(1e-4)
Y_noisy = Y .+ (noise_magnitude*y_mean) .* randn(eltype(Y), size(Y))

V_mag = Y_noisy[:,1]
V_ang = Y_noisy[:,2]

# Radial Basis Function (rbf).
rbf(x) = exp.(-(x.^2))

# The universal approximator (deep network embedding) is subsequently defined.

# Deep feedforward network. Input dimensionality is 6 since we have 4 state and 
# 2 exogenous variables. Since prediction is designated as scalar in this 
# experiment, output dimensionality is equal to 1. By manually changing the 
# number of hidden layer neurons in the respective FastDense(), the number of 
# hidden layers (by changing the number of FastDense() inside FastChain()), 
# and the activation function, several realizations of embedding U can be 
# attained and the experiment can be re-run. Some examples are explicitly 
# provided in comments.

# Case with 3 hidden layers, with 12 neurons each, and rbf activation functions.
U = FastChain(
    FastDense(6,12,rbf), FastDense(12,12,rbf), FastDense(12,12,rbf), FastDense(12,1)
)

# Case with 3 hidden layers, with 12 units each, and tanh activation functions.
#U = FastChain(
#    FastDense(6,12,tanh), FastDense(12,12,tanh), FastDense(12,12,tanh), FastDense(12,1)
#)

# Case with 3 hidden layers, with 12 units each, and relu activation functions.
#U = FastChain(
#    FastDense(6,12,relu), FastDense(12,12,relu), FastDense(12,12,relu), FastDense(12,1)
#)

# Case with 3 hidden layers, with 12 units each, and gelu activation functions.
#U = FastChain(
#    FastDense(6,12,gelu), FastDense(12,12,gelu), FastDense(12,12,gelu), FastDense(12,1)
#)

# Case with 4 hidden layers, with 12, 24, 24, and 12 units each, and rbf activation functions.
#U = FastChain(
#    FastDense(6,12,rbf), FastDense(12,24,rbf), FastDense(24,24,rbf), FastDense(24,12,rbf), FastDense(12,1)
#)

# Case with 2 hidden layers, with 12 units each, and rbf activation functions.
#U = FastChain(
#    FastDense(6,12,rbf), FastDense(12,12,rbf), FastDense(12,1)
#)

# Case with 1 hidden layern with 12 units, and rbf activation function.
#U = FastChain(
#    FastDense(6,12,rbf), FastDense(12,1)
#)

# Initial parameters of deep network embedding.
p = initial_params(U)

freq = 50 # electric grid fundamendal frequency in Hz
H = 1 # generator inertia in per unit
D = 0.01 # generator damping coefficient in per unit
omega_s = 2*pi*freq # synchronous angular speed in rad/sec
T_d_tr = 3.20 # d-axis transient time constant  
T_q_tr = 0.81 # q-axis transient time constant
x_d = 0.93 # d-axis reactance
x_d_tr = 0.50 # d-axis transient reactance
x_q = 0.77 # q-axis reactance
x_q_tr = 0.50 # q-axis transient reactance

p_true = Float64[H, D, x_q_tr, x_d_tr, x_q, x_d, T_q_tr, T_d_tr];

# UDE is subsequently defined in the presence of an unknown segment in right 
# hand side of differential equation associated with angular speed.

function ude_dynamics(du, u, p, t, p_true, U, Pm, Pe, V_mag, V_ang, Efd)

    H, D, x_q_tr, x_d_tr, x_q, x_d, T_q_tr, T_d_tr = p_true;
    f_s = 50;
    omega_s = 2*pi*f_s;
    v_d = u[5];
    v_q = u[6];
    I_d = u[7];
    I_q = u[8];

    # Predictions of deep network embedding.
    u_pred = U(vcat(u[1:4], V_mag[Int(round(100*t)+1)], V_ang[Int(round(100*t)+1)]), p);
    
    # u[1] pertains to SG angle: delta
    du[1] = u[2] - omega_s;

    # u[2] pertains to SG anglular velocity: omega
    # du[2] = (omega_s/(2*H))*(Pm[Int(round(100*t)+1)] - u[3]*I_d - u[4]*I_q - (x_d_tr - x_q_tr)*I_d*I_q - D*(u[2] - omega_s);
    du[2] = (omega_s/(2*H))*(Pm[Int(round(100*t)+1)] - u[3]*I_d - u[4]*I_q - u_pred[1] - D*(u[2] - omega_s));

    # u[3] pertains to SG q-axis transient voltage: Eq_tr
    du[3] = (1/T_d_tr)*(-u[3] - (x_d - x_d_tr)*I_d + Efd[Int(round(100*t)+1)]);
    
    # u[4] pertains to SG d-axis transient voltage: Ed_tr
    du[4] = (1/T_d_tr)*(-u[4] + (x_q - x_q_tr)*I_q);
    
    # v_d and v_q
    du[5] = -V_mag[Int(round(100*t)+1)]*sin(u[1]-V_ang[Int(round(100*t)+1)]) - v_d;
    du[6] = V_mag[Int(round(100*t)+1)]*cos(u[1]-V_ang[Int(round(100*t)+1)]) - v_q;

    # I_d and I_q, algebraic states u[7] and u[8], respectively.
    du[7] = (v_q-u[3])/x_d_tr - I_d;
    du[8] = -(v_d-u[4])/x_q_tr - I_q;

end

# Construction of mass matrix
M = zeros(8,8)
M[1,1] = 1
M[2,2] = 1
M[3,3] = 1
M[4,4] = 1

nn_dynamics(du, u, p, t) = ude_dynamics(du, u, p, t, p_true, U, Pm, Pe, V_mag, V_ang, Efd)
ude_func = ODEFunction(nn_dynamics, mass_matrix = M)
prob_nn = ODEProblem(ude_func, u0, tspan, p)

# Different choices for forward solver and adjoint method can be defined by 
# changing the following two arguments, respectively: alg and sensealg.
# For instance, Rodas5() pertains to a 5th order Rosenbrock method, while 
# InterpolatingAdjoint(checkpointing=true) corresponds to a checkpointed 
# interpolation approach for adjoint computation.
# Other examined choices for alg are Rosenbrock23, Rodas4P, Kvaerno5, 
# and KenCarp4, while examined sensealg options involve ForwardSensitivity, 
# ForwardDiffSensitivity, QuadratureAdjoint, and BacksolveAdjoint.

function predict(θ, X = X_noisy[:,1], T = t)
    Array(solve(prob_nn,RadauIIA5(autodiff=true),u0=X, p=θ, tspan = (T[1], T[end]), 
                saveat=t, sensealg = InterpolatingAdjoint()))                
end

# Loss function encompassing a regularization term.
function loss(θ)
    # Hyper-parameter alpha controlling the trade-off between goodness-of-fit 
    # and penalty on parameter values. To tune hyper-parameter alpha, a 
    # small plausible set of values is established with 20 logarithmically 
    # spaced values from 1e-3 to 1e-1
    alpha = 0.001
    X_pred = predict(θ)
    # Objective function: norm-2 loss on predicted versus noisy ground truth 
    # state variables, plus norm-1 penalty on parameter values.
    sum(abs2, X_noisy .- X_pred) + alpha*sum(abs, θ)

    # Objective function: norm-1 loss on predicted versus noisy ground truth 
    # state variables, plus norm-1 penalty on parameter values.
    #sum(abs, X_noisy .- X_pred) + lambda*sum(abs, θ)

    # Objective function: norm-2 loss on predicted versus noisy ground truth 
    # state variables, plus norm-2 penalty on parameter values.
    #sum(abs2, X_noisy .- X_pred) + lambda*sum(abs2, θ)    

    # Objective function: norm-1 loss on predicted versus noisy ground truth 
    # state variables, plus norm-2 penalty on parameter values.
    #sum(abs, X_noisy .- X_pred) + lambda*sum(abs2, θ)    

end

# Loss function based on norm-1 goodness-of-fit term, used when changing
# from norm-2 to norm-1 goodness-of-fit term for training.
function loss_change(θ)
    alpha = 0.001
    X_pred = predict(θ)
    sum(abs, X_noisy .- X_pred) + alpha*sum(abs, θ)
end

losses = Float64[]

callback2(θ,l) = begin
    push!(losses, l)
    if length(losses)%1==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false
end

# ADAM optimizer is employed for a pre-determined maximum number of iterations.

res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.01f0), cb=callback2, maxiters = 200)
#res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.005f0), cb=callback2, maxiters = 200)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

# Training continues with BFGS by utilizing as starting point the attained 
# parameters based on ADAM optimizer.
res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback2, maxiters = 100)
#res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.005f0), cb=callback2, maxiters = 100)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Training continues (when uncommented) with changing goodness-of-fit term from norm-2 to norm-1.
#res1_change = DiffEqFlux.sciml_train(loss_change, res2.minimizer, ADAM(0.01f0), cb=callback2, maxiters = 200)
#println("Training loss after $(length(losses)) iterations: $(losses[end])")
#res2 = DiffEqFlux.sciml_train(loss_change, res1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback2, maxiters = 100)
#res2 = DiffEqFlux.sciml_train(loss_change, res1.minimizer, BFGS(initial_stepnorm=0.005f0), cb=callback2, maxiters = 100)
#println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Predictions of state variables based on learnt UDE.
x_pred = predict(res2.minimizer, u0, t)
matwrite("x_pred.mat", Dict("x_pred" => x_pred))

# Predictions by deep network embedding.
missing_pred = U(vcat(x_pred, V_mag, V_ang), res2.minimizer)
matwrite("missing_pred.mat", Dict("missing_pred" => missing_pred))

# Sequential training based on additional trajectories (it did not improve training quality, i.e. there was no 
# escape from the local minimum).

for i = 2:19
  X = Array([Angle[:,1] Speed[:,1] Eq_tr[:,1] Ed_tr[:,1] V_d[:,1] V_q[:,1] I_d[:,1] I_q[:,1]])
  t = Time
  number_of_time_stamps = 50
  X = X[1:number_of_time_stamps,:]
  t = t[1:number_of_time_stamps]
  tspan = (minimum(t),maximum(t))

  # Gaussian noise is added to state measurements. 

  x_mean = mean(X, dims = 1)
  noise_magnitude = Float32(1e-4)
  X_noisy = X .+ (noise_magnitude*x_mean) .* randn(eltype(X), size(X))
  u0 = X_noisy[1,:]
  X_noisy = transpose(X_noisy)

  # Gaussian noise is added to exogenous measurements. 

  Y = Array([V_mag[:,i] V_angle[:,i]])
  Y = Y[1:number_of_time_stamps,:]
  y_mean = mean(Y, dims = 1)
  noise_magnitude = Float32(1e-4)
  Y_noisy = Y .+ (noise_magnitude*y_mean) .* randn(eltype(Y), size(Y))
  V_mag = Y_noisy[:,1]
  V_ang = Y_noisy[:,2]

  # ADAM optimizer is employed for a pre-determined maximum number of iterations.

  res1 = DiffEqFlux.sciml_train(loss, res2.minimizer, ADAM(0.01f0), cb=callback2, maxiters = 200)
  println("Training loss after $(length(losses)) iterations: $(losses[end])")

  # Training continues with BFGS by utilizing as starting point the attained 
  # parameters based on ADAM optimizer.

  res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback2, maxiters = 100)
  println("Final training loss after $(length(losses)) iterations: $(losses[end])")  

  # Predictions of state variables based on learnt UDE.

  x_pred = hcat(x_pred, predict(res2.minimizer, u0, t))

  # Predictions by deep network embedding.

  missing_pred = hcat(missing_pred, U(vcat(x_pred, V_mag, V_ang), res2.minimizer))

matwrite("x_pred_all.mat", Dict("x_pred_all" => x_pred))
matwrite("missing_pred_all.mat", Dict("missing_pred_all" => missing_pred))  

# Alternatively, by naive substitution the following UDE can be defined.
# The portion of dynamics pertaining to ~Iq*Id in domega/dt are again 
# replaced by a deep network embedding.

function ude_dynamics(du, u, p, t, p_true, U, Pm, Pe, V_mag, V_ang, Efd)

    H, D, x_q_tr, x_d_tr, x_q, x_d, T_q_tr, T_d_tr = p_true;
    f_s = 50;
    omega_s = 2*pi*f_s;

    # Predictions of deep network embedding.

    u_pred = U(vcat(u, V_mag[Int(round(100*t)+1)], V_ang[Int(round(100*t)+1)]), p);
        
    du[1] = u[2] - omega_s;
      
    du[2] = (omega_s/(2*H))*(- D*u[2] + D*omega_s + (1/x_d_tr)*u[3]^2 - (1/x_q_tr)*u[4]^2 
            + T_m - (1/x_q_tr)*u[3]*V*sin(u[1]-theta) + (1/x_d_tr)*u[4]*V*cos(u[1]-theta) + u_pred[1])             
    
    du[3] = (1/T_d_tr)*(- (x_d/x_q_tr)*u[4] + (x_d_tr/x_q_tr)*u[4] - u[3] + E_fd 
            - (x_d/x_q_tr)*V*sin(u[1]-theta) + (x_d_tr/x_q_tr)*V*sin(u[1]-theta))

    du[4] = (1/T_q_tr)*(- u[4] - (x_q/x_d_tr)*u[3] + (x_q_tr/x_d_tr)*u[3] 
            + (x_q/x_d_tr)*V*cos(u[1]-theta) - (x_q_tr/x_d_tr)*V*cos(u[1]-theta))

end

# In this case, ODEProblem is defined as follows and the same logic is applied, 
# albeit matrix X does not encompass algebraic variables.

nn_dynamics(du, u, p, t) = ude_dynamics(du, u, p, t, p_true, U, Pm, Pe, V_mag, V_ang, Efd)
prob_nn = ODEProblem(nn_dynamics, u0, tspan, p)

X = Array([Angle[:,1] Speed[:,1] Eq_tr[:,1] Ed_tr[:,1]])
X = X[1:number_of_time_stamps,:]
t = t[1:number_of_time_stamps]
tspan = (minimum(t),maximum(t))

# Gaussian noise is added to state measurements. 

x_mean = mean(X, dims = 1)
noise_magnitude = Float32(1e-4)
X_noisy = X .+ (noise_magnitude*x_mean) .* randn(eltype(X), size(X))
u0 = X_noisy[1,:]
X_noisy = transpose(X_noisy)

# ADAM optimizer is employed for a pre-determined maximum number of iterations.

res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.01f0), cb=callback2, maxiters = 200)
#res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.005f0), cb=callback2, maxiters = 200)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

# Training continues with BFGS by utilizing as starting point the attained 
# parameters based on ADAM optimizer.
res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback2, maxiters = 100)
#res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.005f0), cb=callback2, maxiters = 100)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Training continues (when uncommented) with changing goodness-of-fit term from norm-2 to norm-1.
#res1_change = DiffEqFlux.sciml_train(loss_change, res2.minimizer, ADAM(0.01f0), cb=callback2, maxiters = 200)
#println("Training loss after $(length(losses)) iterations: $(losses[end])")
#res2 = DiffEqFlux.sciml_train(loss_change, res1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback2, maxiters = 100)
#res2 = DiffEqFlux.sciml_train(loss_change, res1.minimizer, BFGS(initial_stepnorm=0.005f0), cb=callback2, maxiters = 100)
#println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Predictions of state variables based on learnt UDE.
x_pred = predict(res2.minimizer, u0, t)
matwrite("x_pred_alternative.mat", Dict("x_pred" => x_pred))

# Predictions by deep network embedding.
missing_pred = U(vcat(x_pred, V_mag, V_ang), res2.minimizer)
matwrite("missing_pred_alternative.mat", Dict("missing_pred" => missing_pred))