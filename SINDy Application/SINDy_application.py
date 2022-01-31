# SINDy_application.py

# The following code is based on examples found in the repository [1] of PySINDy [2]. 
# Additions and adjustments to accommodate the application of SINDy to synchronous 
# generator dynamics are made.
# [1] https://github.com/dynamicslab/pysindy
# [2] B. de Silva, K. Champion, M. Quade, J.-C. Loiseau, J. Kutz, and S. Brunton,
# "PySINDy: A Python package for the sparse identification of nonlinear dynamical systems
# from data," Journal of Open Source Software, 5 (2020), p. 2104.

import pysindy as ps
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import seaborn as sns
import pynumdiff.total_variation_regularization as tvr
sns.set_context('notebook')
sns.set_style('darkgrid')

def compute_aic_bic(model, mse, data):

    n_parameters = np.count_nonzero(model.coefficients())
    n_samples = np.array(data).reshape(-1).shape[0]
    aic = n_samples*np.log(mse) + 2*n_parameters
    bic = n_samples*np.log(mse) + n_parameters*np.log(n_samples)

    return -aic, -bic

mat = loadmat('time_series.mat')
t = mat['Time']
dt = 0.01
X = []
U = []
noise_magnitude_state = 1e-4
noise_magnitude_exogenous = 1e-4

# For state predictions emerging from deep network embedding, noise_magnitude_state is set to 0.
# noise_magnitude_state = 0. However, noise level injected to exogenous variables remains the same.

for i in range(0,19):
    X.append(np.transpose(np.vstack((mat['Angles_'][:,i]/180*np.pi + (noise_magnitude_state*np.mean(mat['Angles_'][:,i]/180*np.pi))*np.random.normal(),
                                     mat['Speeds_'][:,i]*(2*np.pi*50) + (noise_magnitude_state*np.mean(mat['Speeds_'][:,i]*(2*np.pi*50)))*np.random.normal(),
                                     mat['Eq_tr_'][:,i] + (noise_magnitude_state*np.mean(mat['Eq_tr_'][:,i]))*np.random.normal(),
                                     mat['Ed_tr_'][:,i] + (noise_magnitude_state*np.mean(mat['Ed_tr_'][:,i]))*np.random.normal()))))
    U.append(np.transpose(np.vstack((np.abs(mat['Voltages_'][:,i]) + (noise_magnitude_exogenous*np.mean(np.abs(mat['Voltages_'][:,i])))*np.random.normal(),
                                     np.angle(mat['Voltages_'][:,i]) + (noise_magnitude_exogenous*np.mean(np.angle(mat['Voltages_'][:,i])))*np.random.normal()))))    
    
X_train, X_test, U_train, U_test = train_test_split(X, U, test_size=0.2)

# Explicit specification of library terms involving sinusoidal terms.

library_functions = [
    lambda x,y : np.sin(x-y),
    lambda x,y : np.cos(x-y),
    
    lambda x,y,z : x*np.sin(y-z),
    lambda x,y,z : y*np.sin(x-z),
    lambda x,y,z : z*np.sin(y-x),
    lambda x,y,z : x*np.cos(y-z),
    lambda x,y,z : y*np.cos(x-z),
    lambda x,y,z : z*np.cos(y-x),
    
    lambda x,y,z : (x**2)*np.sin(y-z),
    lambda x,y,z,w : (x*y)*np.sin(z-w),
    lambda x,y,z,w : (x*z)*np.sin(y-w),
    lambda x,y,z,w : (x*w)*np.sin(y-z),    
    lambda x,y,z : (y**2)*np.sin(x-z),
    lambda x,y,z,w : (y*z)*np.sin(x-w),
    lambda x,y,z,w : (y*w)*np.sin(x-z),
    lambda x,y,z : (z**2)*np.sin(y-x),
    lambda x,y,z,w : (z*w)*np.sin(x-y),
    lambda x,y,z : (x**2)*np.cos(y-z),
    lambda x,y,z,w : (x*y)*np.cos(z-w),
    lambda x,y,z,w : (x*z)*np.cos(y-w),
    lambda x,y,z,w : (x*w)*np.cos(y-z),    
    lambda x,y,z : (y**2)*np.cos(x-z),
    lambda x,y,z,w : (y*z)*np.cos(x-w),    
    lambda x,y,z,w : (y*w)*np.cos(x-z),    
    lambda x,y,z : (z**2)*np.cos(y-x), 
    lambda x,y,z,w : (z*w)*np.cos(x-y),
        
    lambda x,y,z,w : np.sin(x-y)*np.cos(z-w), 
    lambda x,y,z,w : np.sin(z-w)*np.cos(x-y), 
    
    lambda x,y,z,w,k : x*np.sin(y-z)*np.cos(w-k),     
    lambda x,y,z,w,k : x*np.sin(w-k)*np.cos(y-z),     
    lambda x,y,z,w,k : y*np.sin(x-z)*np.cos(w-k),     
    lambda x,y,z,w,k : y*np.sin(w-k)*np.cos(x-z),     
    lambda x,y,z,w,k : z*np.sin(x-y)*np.cos(w-k),     
    lambda x,y,z,w,k : z*np.sin(w-k)*np.cos(x-y),     
    lambda x,y,z,w,k : w*np.sin(x-y)*np.cos(z-k),     
    lambda x,y,z,w,k : w*np.sin(z-k)*np.cos(x-y),     
    lambda x,y,z,w,k : k*np.sin(x-y)*np.cos(z-w),
    lambda x,y,z,w,k : k*np.sin(z-w)*np.cos(x-y),

    lambda x,y,z,w,m : (x**2)*np.sin(y-z)*np.cos(w-m),
    lambda x,y,z,w,m : (x**2)*np.sin(w-m)*np.cos(y-z),
    lambda x,y,z,w,m,n : (x*y)*np.sin(z-w)*np.cos(m-n),
    lambda x,y,z,w,m,n : (x*y)*np.sin(m-n)*np.cos(z-w),
    lambda x,y,z,w,m,n : (x*z)*np.sin(y-w)*np.cos(m-n),
    lambda x,y,z,w,m,n : (x*z)*np.sin(m-n)*np.cos(y-w),    
    lambda x,y,z,w,m,n : (x*w)*np.sin(y-z)*np.cos(m-n),
    lambda x,y,z,w,m,n : (x*w)*np.sin(m-n)*np.cos(y-z),
    lambda x,y,z,w,m,n : (x*m)*np.sin(y-z)*np.cos(w-n),
    lambda x,y,z,w,m,n : (x*m)*np.sin(w-n)*np.cos(y-z),    
    lambda x,y,z,w,m,n : (x*n)*np.sin(y-z)*np.cos(w-m),
    lambda x,y,z,w,m,n : (x*n)*np.sin(w-m)*np.cos(y-z),    

    lambda x,y,z,w,m : (y**2)*np.sin(x-z)*np.cos(w-m),
    lambda x,y,z,w,m : (y**2)*np.sin(w-m)*np.cos(x-z),
    lambda x,y,z,w,m,n : (y*z)*np.sin(x-w)*np.cos(m-n),
    lambda x,y,z,w,m,n : (y*z)*np.sin(m-n)*np.cos(x-w),
    lambda x,y,z,w,m,n : (y*w)*np.sin(x-z)*np.cos(m-n),
    lambda x,y,z,w,m,n : (y*w)*np.sin(m-n)*np.cos(x-z),    
    lambda x,y,z,w,m,n : (y*m)*np.sin(x-z)*np.cos(w-n),
    lambda x,y,z,w,m,n : (y*m)*np.sin(w-n)*np.cos(x-z),
    lambda x,y,z,w,m,n : (y*n)*np.sin(x-z)*np.cos(w-m),
    lambda x,y,z,w,m,n : (y*n)*np.sin(w-m)*np.cos(x-z),    
    
    lambda x,y,z,w,m : (z**2)*np.sin(x-y)*np.cos(w-m),
    lambda x,y,z,w,m : (z**2)*np.sin(w-m)*np.cos(x-y),
    lambda x,y,z,w,m,n : (z*w)*np.sin(x-y)*np.cos(m-n),
    lambda x,y,z,w,m,n : (z*w)*np.sin(m-n)*np.cos(x-y),
    lambda x,y,z,w,m,n : (z*m)*np.sin(x-y)*np.cos(w-n),
    lambda x,y,z,w,m,n : (z*m)*np.sin(w-n)*np.cos(x-y),    
    lambda x,y,z,w,m,n : (z*n)*np.sin(x-y)*np.cos(w-m),
    lambda x,y,z,w,m,n : (z*n)*np.sin(w-m)*np.cos(x-y),

    lambda x,y,z,w,m : (w**2)*np.sin(x-y)*np.cos(z-m),
    lambda x,y,z,w,m : (w**2)*np.sin(z-m)*np.cos(x-y),
    lambda x,y,z,w,m,n : (w*m)*np.sin(x-y)*np.cos(z-n),
    lambda x,y,z,w,m,n : (w*m)*np.sin(z-n)*np.cos(x-y),
    lambda x,y,z,w,m,n : (w*n)*np.sin(x-y)*np.cos(z-m),
    lambda x,y,z,w,m,n : (w*n)*np.sin(z-m)*np.cos(x-y),    

    lambda x,y,z,w,m : (m**2)*np.sin(x-y)*np.cos(z-w),
    lambda x,y,z,w,m : (m**2)*np.sin(z-w)*np.cos(x-y),
    lambda x,y,z,w,m,n : (m*n)*np.sin(x-y)*np.cos(z-w),
    lambda x,y,z,w,m,n : (m*n)*np.sin(z-w)*np.cos(x-y),
    
]

library_function_names = [
    lambda x,y : 'sin(' + x + '-' + y + ')',
    lambda x,y : 'cos(' + x + '-' + y + ')',
    
    lambda x,y,z : x+'*sin(' + y + '-' + z + ')',
    lambda x,y,z : y+'*sin(' + x + '-' + z + ')',
    lambda x,y,z : z+'*sin(' + y + '-' + x + ')',
    lambda x,y,z : x+'*cos(' + y + '-' + z + ')',
    lambda x,y,z : y+'*cos(' + x + '-' + z + ')',
    lambda x,y,z : z+'*cos(' + y + '-' + x + ')',
    
    lambda x,y,z : '('+x+'^2)*sin('+y+'-'+z+')',
    lambda x,y,z,w : '('+x+'*'+y+')*sin('+z+'-'+w+')',
    lambda x,y,z,w : '('+x+'*'+z+')*sin('+y+'-'+w+')',
    lambda x,y,z,w : '('+x+'*'+w+')*sin('+y+'-'+z+')',    
    lambda x,y,z : '('+y+'^2)*sin('+x+'-'+z+')',
    lambda x,y,z,w : '('+y+'*'+z+')*sin('+x+'-'+w+')',
    lambda x,y,z,w : '('+y+'*'+w+')*sin('+x+'-'+z+')',
    lambda x,y,z : '('+z+'^2)*sin('+y+'-'+x+')',
    lambda x,y,z,w : '('+z+'*'+w+')*sin('+x+'-'+y+')',
    lambda x,y,z : '('+x+'^2)*cos('+y+'-'+z+')',
    lambda x,y,z,w : '('+x+'*'+y+')*cos('+z+'-'+w+')',
    lambda x,y,z,w : '('+x+'*'+z+')*cos('+y+'-'+w+')',
    lambda x,y,z,w : '('+x+'*'+w+')*cos('+y+'-'+z+')',    
    lambda x,y,z : '('+y+'^2)*cos('+x+'-'+z+')',
    lambda x,y,z,w : '('+y+'*'+z+')*cos('+x+'-'+w+')',    
    lambda x,y,z,w : '('+y+'*'+w+')*cos('+x+'-'+z+')',    
    lambda x,y,z : '('+z+'^2)*cos('+y+'-'+x+')', 
    lambda x,y,z,w : '('+z+'*'+w+')*cos('+x+'-'+y+')',

    lambda x,y,z,w : 'sin('+x+'-'+y+')*cos('+z+'-'+w+')', 
    lambda x,y,z,w : 'sin('+z+'-'+w+')*cos('+x+'-'+y+')', 
    
    lambda x,y,z,w,k : x+'*sin('+y+'-'+z+')*cos('+w+'-'+k+')',     
    lambda x,y,z,w,k : x+'*sin('+w+'-'+k+')*cos('+y+'-'+z+')',     
    lambda x,y,z,w,k : y+'*sin('+x+'-'+z+')*cos('+w+'-'+k+')',     
    lambda x,y,z,w,k : y+'*sin('+w+'-'+k+')*cos('+x+'-'+z+')',     
    lambda x,y,z,w,k : z+'*sin('+x+'-'+y+')*cos('+w+'-'+k+')',     
    lambda x,y,z,w,k : z+'*sin('+w+'-'+k+')*cos('+x+'-'+y+')',     
    lambda x,y,z,w,k : w+'*sin('+x+'-'+y+')*cos('+z+'-'+k+')',     
    lambda x,y,z,w,k : w+'*sin('+z+'-'+k+')*cos('+x+'-'+y+')',     
    lambda x,y,z,w,k : k+'*sin('+x+'-'+y+')*cos('+z+'-'+w+')',
    lambda x,y,z,w,k : k+'*sin('+z+'-'+w+')*cos('+x+'-'+y+')',

    lambda x,y,z,w,m : '('+x+'^2)*sin('+y+'-'+z+')*cos('+w+'-'+m+')',
    lambda x,y,z,w,m : '('+x+'^2)*sin('+w+'-'+m+')*cos('+y+'-'+z+')',
    lambda x,y,z,w,m,n : '('+x+'*'+y+')*sin('+z+'-'+w+')*cos('+m+'-'+n+')',
    lambda x,y,z,w,m,n : '('+x+'*'+y+')*sin('+m+'-'+n+')*cos('+z+'-'+w+')',
    lambda x,y,z,w,m,n : '('+x+'*'+z+')*sin('+y+'-'+w+')*cos('+m+'-'+n+')',
    lambda x,y,z,w,m,n : '('+x+'*'+z+')*sin('+m+'-'+n+')*cos('+y+'-'+w+')',    
    lambda x,y,z,w,m,n : '('+x+'*'+w+')*sin('+y+'-'+z+')*cos('+m+'-'+n+')',
    lambda x,y,z,w,m,n : '('+x+'*'+w+')*sin('+m+'-'+n+')*cos('+y+'-'+z+')',
    lambda x,y,z,w,m,n : '('+x+'*'+m+')*sin('+y+'-'+z+')*cos('+w+'-'+n+')',
    lambda x,y,z,w,m,n : '('+x+'*'+m+')*sin('+w+'-'+n+')*cos('+y+'-'+z+')',    
    lambda x,y,z,w,m,n : '('+x+'*'+n+')*sin('+y+'-'+z+')*cos('+w+'-'+m+')',
    lambda x,y,z,w,m,n : '('+x+'*'+n+')*sin('+w+'-'+m+')*cos('+y+'-'+z+')',    

    lambda x,y,z,w,m : '('+y+'^2)*sin('+x+'-'+z+')*cos('+w+'-'+m+')',
    lambda x,y,z,w,m : '('+y+'^2)*sin('+w+'-'+m+')*cos('+x+'-'+z+')',
    lambda x,y,z,w,m,n : '('+y+'*'+z+')*sin('+x+'-'+w+')*cos('+m+'-'+n+')',
    lambda x,y,z,w,m,n : '('+y+'*'+z+')*sin('+m+'-'+n+')*cos('+x+'-'+w+')',
    lambda x,y,z,w,m,n : '('+y+'*'+w+')*sin('+x+'-'+z+')*cos('+m+'-'+n+')',
    lambda x,y,z,w,m,n : '('+y+'*'+w+')*sin('+m+'-'+n+')*cos('+x+'-'+z+')',
    lambda x,y,z,w,m,n : '('+y+'*'+m+')*sin('+x+'-'+z+')*cos('+w+'-'+n+')',
    lambda x,y,z,w,m,n : '('+y+'*'+m+')*sin('+w+'-'+n+')*cos('+x+'-'+z+')',
    lambda x,y,z,w,m,n : '('+y+'*'+n+')*sin('+x+'-'+z+')*cos('+w+'-'+m+')',
    lambda x,y,z,w,m,n : '('+y+'*'+n+')*sin('+w+'-'+m+')*cos('+x+'-'+z+')',
    
    lambda x,y,z,w,m : '('+z+'^2)*sin('+x+'-'+y+')*cos('+w+'-'+m+')',
    lambda x,y,z,w,m : '('+z+'^2)*sin('+w+'-'+m+')*cos('+x+'-'+y+')',
    lambda x,y,z,w,m,n : '('+z+'*'+w+')*sin('+x+'-'+y+')*cos('+m+'-'+n+')',
    lambda x,y,z,w,m,n : '('+z+'*'+w+')*sin('+m+'-'+n+')*cos('+x+'-'+y+')',
    lambda x,y,z,w,m,n : '('+z+'*'+m+')*sin('+x+'-'+y+')*cos('+w+'-'+n+')',
    lambda x,y,z,w,m,n : '('+z+'*'+m+')*sin('+w+'-'+n+')*cos('+x+'-'+y+')',    
    lambda x,y,z,w,m,n : '('+z+'*'+n+')*sin('+x+'-'+y+')*cos('+w+'-'+m+')',
    lambda x,y,z,w,m,n : '('+z+'*'+n+')*sin('+w+'-'+m+')*cos('+x+'-'+y+')',

    lambda x,y,z,w,m : '('+w+'^2)*sin('+x+'-'+y+')*cos('+z+'-'+m+')',
    lambda x,y,z,w,m : '('+w+'^2)*sin('+z+'-'+m+')*cos('+x+'-'+y+')',
    lambda x,y,z,w,m,n : '('+w+'*'+m+')*sin('+x+'-'+y+')*cos('+z+'-'+n+')',
    lambda x,y,z,w,m,n : '('+w+'*'+m+')*sin('+z+'-'+n+')*cos('+x+'-'+y+')',
    lambda x,y,z,w,m,n : '('+w+'*'+n+')*sin('+x+'-'+y+')*cos('+z+'-'+m+')',
    lambda x,y,z,w,m,n : '('+w+'*'+n+')*sin('+z+'-'+m+')*cos('+x+'-'+y+')',    

    lambda x,y,z,w,m : '('+m+'^2)*sin('+x+'-'+y+')*cos('+z+'-'+w+')',
    lambda x,y,z,w,m : '('+m+'^2)*sin('+z+'-'+w+')*cos('+x+'-'+y+')',
    lambda x,y,z,w,m,n : '('+m+'*'+n+')*sin('+x+'-'+y+')*cos('+z+'-'+w+')',
    lambda x,y,z,w,m,n : '('+m+'*'+n+')*sin('+z+'-'+w+')*cos('+x+'-'+y+')',
    
]

extra_library = ps.CustomLibrary(
    library_functions=library_functions, function_names=library_function_names
)

# Polynomial terms up to and including quadratic terms are encompassed in the library.

poly_library = ps.PolynomialLibrary(degree=2)

# Library Theta(.) comprises polynomial and previously defined sinusoidal terms.

custom_library = poly_library + extra_library

threshold_range = np.linspace(start=1e-6, stop=1e-1, num=100)
alpha_range = np.linspace(start=1e-4, stop=1e-1, num=100)

scores = dict()
rmse_out_of_sample = {}

# Linesearch associated with threshold hyper-parameter for STLSQ and SR3.

for threshold in threshold_range: 

# Linesearch associated with alpha hyper-parameter for LASSO.
# for alpha in alpha_range: 

    # Utilized optimizers pertain to STLSQ, SR3, and LASSO.

    optimizer = ps.STLSQ(threshold=threshold, max_iter=100, fit_intercept=True)
    #optimizer = ps.SR3(threshold=threshold, max_iter=100, fit_intercept=True)
    #optimizer=Lasso(alpha=alpha, fit_intercept=True)

    # According to https://github.com/dynamicslab/pysindy/blob/master/examples/5_differentiation.ipynb, 
    # the utilization of trend_filtered with order=0 pertains to total variational derivative computation.
    # Alternatively, computation and storage of derivatives via the utilization of PyNumDiff could be 
    # as follows:
    # dXdt = []
    # for i in range(0,len(X_train)):
    #     T = X_train[i].shape[0]
    #     dXdt_hat = np.zeros(T)
    #     for j in range(0,X_train[i].shape[1]):
    #         _, dxdt_hat = tvr.iterative_velocity(X_train[i][:,j], dt=0.01, params=[1,0.001])
    #         dXdt_hat = np.vstack((dXdt_hat, dxdt_hat))
    #     dXdt_hat = np.delete(dXdt_hat, 0, axis=0)
    #     dXdt.append(dXdt_hat.T)
    # In this case, derivatives could be subsequently passed as arguments to utilized PySINDy methods.

    score = dict()
    
    model = ps.SINDy(optimizer=optimizer, feature_library=custom_library, 
                     differentiation_method=ps.SINDyDerivative(kind='trend_filtered', order=0, alpha=1e-1))

    model.fit(x=X_train, u=U_train, t=dt, multiple_trajectories=True)

    # n_features = model.n_output_features_
    # print(f"Features ({n_features}):", model.get_feature_names())

    # Computation of out-of-sample RMSE.
    
    score['rmse'] = np.sqrt(model.score(x=X_test, u=U_test, t=dt, multiple_trajectories=True, metric=mean_squared_error))
    
    # Computation of AIC/BIC on training data.
      
    aic_bic = compute_aic_bic(model, model.score(x=X_train, u=U_train, t=dt, multiple_trajectories=True, metric=mean_squared_error), X)
    score['aic'] = aic_bic[0]
    score['bic'] = aic_bic[1]
    score['nnz_params'] = np.count_nonzero(model.coefficients())
    
    scores[threshold] = score    
    
df_scores = DataFrame.from_dict(scores).T
#best_threshold = threshold_range[np.argmin(np.array(df_scores.rmse))]
#best_alpha = alpha_range[np.argmin(np.array(df_rmse_out_of_sample.rmse))]
print(df_scores)

fig = plt.figure()
ax = fig.add_subplot()

sns.set_style(style='dark')
color = 'tab:blue'
ax.set_xlabel('Number of non-zero coefficients')
ax.set_ylabel('AIC', color=color)
ax.plot(df_scores.nnz_params, df_scores.aic, color=color)
ax.tick_params(axis='y', labelcolor=color)
ax1 = ax.twinx() 
color = 'tab:red'
ax1.set_ylabel('BIC', color=color)
ax1.plot(df_scores.nnz_params, df_scores.bic, color=color)
ax1.tick_params(axis='y', labelcolor=color)

fig = plt.figure()
ax = fig.add_subplot()

sns.set_style(style='dark')
color = 'tab:blue'
ax.set_xlabel('Number of non-zero parameters')
ax.set_ylabel('RMSE on Test Set', color=color)
ax.plot(df_scores.nnz_params, df_scores.rmse, color=color)
ax.tick_params(axis='y', labelcolor=color)

# Re-fit on entire dataset based on best threshold/alpha value based on out-of-sample RMSE and AIC/BIC.

best_threshold = 0.09798
#best_alpha = 0.5*1e-1
optimizer = ps.STLSQ(threshold=best_threshold, max_iter=100, fit_intercept=True)
#optimizer = ps.SR3(threshold=best_threshold, max_iter=100, fit_intercept=True)
#optimizer=Lasso(alpha=alpha, fit_intercept=True)
model = ps.SINDy(optimizer=optimizer, feature_library=custom_library, 
                 differentiation_method=ps.SINDyDerivative(kind='trend_filtered', order=0, alpha=1e-1))
model.fit(x=X, u=U, t=dt, multiple_trajectories=True)

print(model.print())