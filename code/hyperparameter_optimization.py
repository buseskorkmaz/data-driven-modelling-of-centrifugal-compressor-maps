# -*- coding: utf-8 -*-
"""
@author: Buse Korkmaz

Bayesian optimization implementation to obtain hyperparameters
minimize MSE.

usage:
    change data_name and specify data source such as being same 
    with GP2D_scaled.py 
    
    these functions is called by GP2_scaled.optimize_regression

"""
import os
import pandas as pd
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic,Matern, RBF,ConstantKernel as C,DotProduct, WhiteKernel,ExpSineSquared
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from util import *
import warnings
os.chdir("../data")

# to have reproducible results
np.random.seed(7)

observations_dict = {"sample_size": [],
                     "sample_method":[],
                     "kernel": [],
                     "alpha": [],
                     "MSE of GPR": [],
                     "MSE of Polynomial Fit": [],
                     "Integral Absolute Error of GPR": [],
                     "Integral Absolute Error of Polynomial Fit": []
                     }

pred_dict = { "true_val" : [],
             "gaussian_pred": [],
             "polynomial_pred": []
            }

# Load vector of scanned values (mass flow/pressure ratio/efficiency)
etas_full, garrett_full1, garrett_full2,garrett_to4b = read_data()
data_name = "to4b-h3-speed"
data = read_data(data_name)
data_length = len(data)


def regression(sample_size, sample_method, kernel, alpha, kernels=None, name=True, scaled=True, X= None, y=None, sample_indices=None):
    try:

        if sample_method == "systematic":
            # Prepare input (X) and output (y) vectors for training
            Xsize = (sample_size,2)
            ysize = (sample_size,1)
            
            sample_indices = []
            X , y = systematic_sampling(data,Xsize, ysize,sample_size, sample_indices,scaled)
  
        if scaled:
           
            x_scaler = StandardScaler()
            # transform input data
            x_scaled = x_scaler.fit_transform(X)
            y_scaler = StandardScaler()
            # transform output data
            y_scaled = y_scaler.fit_transform(y)
        
        # Setup GPR with the given kernel
        try:
            gp = GaussianProcessRegressor(kernel=kernel,alpha=alpha,n_restarts_optimizer=150)
        except:
            gp = GaussianProcessRegressor(kernel=kernel,alpha=alpha)            
        
        # Train GP
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Trigger a warning
            if scaled:
                gp.fit(x_scaled, y_scaled)
            else:
                gp.fit(X, y)
            
            if len(w) > 0 and 'sklearn.exceptions.ConvergenceWarning' in str(w[-1].message):
                 errorg = float('inf')
                 errorp = float('inf')
                 print("catched")
                 return float("inf")
             
            else:
                # Setup least squares regression pipeline
                Input=[('polynomial',PolynomialFeatures(degree=3)),('modal',LinearRegression())]
                pipe=Pipeline(Input)
        
                # fit the transformed features to Linear Regression
                if scaled:
                    pipe.fit(x_scaled, y_scaled)
                else:
                    pipe.fit(X,y)
              
    
                # Generate inputs from all scanned values
                Xfsize = (data_length-sample_size,2)
                Xf = np.zeros(Xfsize)
                idx = 0
                for i in range(data_length):
                    if(i not in sample_indices):
                        Xf[idx,0] = data[i,0]
                        Xf[idx,1] = data[i,1]
                        idx += 1    
                
                # Generate GP predictions for all scanned values    
                if scaled:
                    y_predgf, MSEf = gp.predict(x_scaler.transform(Xf), return_std=True)
                    y_predgf = y_scaler.inverse_transform(y_predgf)
                else:
                    y_predgf, MSEf = gp.predict(Xf, return_std=True)
                
                # Calculate GP prediction errors for all scanned values 
                size_errorg = (data_length-sample_size,1)
                errorg = np.zeros(size_errorg)
                idx = 0
                for i in range(data_length):
                    if i not in sample_indices:
                        errorg[idx,0]=y_predgf[idx,0]-data[i,2]
                        pred_dict["true_val"].append(data[i,2])
                        pred_dict["gaussian_pred"].append(y_predgf[idx,0])
                        idx += 1

                # Generate polynomial predictions for all scanned values  
                y_pfsize = (data_length-sample_size,1)
                size_errorp = (data_length-sample_size,1)
                y_pf = np.zeros(y_pfsize)
                errorp = np.zeros(size_errorp)
                idx = 0
                for i in range(data_length):
                    if (i not in sample_indices):
                        # Generate polynomial prediction errors for all scanned values  
                        if scaled:
                            y_pf[idx,0] = pipe.predict(x_scaler.transform(Xf[idx].reshape(1, -1)))
                            errorp[idx,0]= y_scaler.inverse_transform(y_pf[idx,0].reshape(-1,1))-data[i,2]
                            pred_dict["polynomial_pred"].append(y_scaler.inverse_transform(y_pf[idx,0].reshape(-1,1)))

                        else:
                            y_pf[idx,0] = pipe.predict(Xf[idx].reshape(1, -1))
                            errorp[idx,0]= y_pf[idx,0]-data[i,2]
                            pred_dict["polynomial_pred"].append(y_pf[idx,0])
                        
                        idx += 1
                                        
                if scaled:
                    y_pf = y_scaler.inverse_transform(y_pf.reshape(-1,1))
                else:
                    y_pf = y_pf.reshape(-1,1)
                
                # plt.plot(errorg)
                # plt.plot(errorp)
                # plt.show()
                
                # optional error metric
                # print('Integral Absolute Error of GPR')
                # print(np.sum(abs(errorg)))
                # print('Integral Absolute Error of Polynomial Fit')
                # print(np.sum(abs(errorp)))
                
                print('MSE of GPR')
                print(np.square(errorg).mean())
                print('MSE of Polynomial Fit')
                print(np.square(errorp).mean())
             
                
                observations_dict["sample_size"].append(sample_size)
                if name: 
                    kernel_name = list(kernels.keys())[list(kernels.values()).index(kernel)]   
                    observations_dict["kernel"].append(kernel_name)
                else:
                    observations_dict["kernel"].append(kernel)
                observations_dict["alpha"].append(alpha)
                observations_dict["sample_method"].append(sample_method)
                observations_dict["MSE of GPR"].append(np.square(errorg).mean())
                observations_dict["MSE of Polynomial Fit"].append(np.square(errorp).mean())
                observations_dict["Integral Absolute Error of GPR"].append(np.sum(abs(errorg)))
                observations_dict["Integral Absolute Error of Polynomial Fit"].append(np.sum(abs(errorp)))
    
        return np.square(errorg).mean()
    
    except:
        observations_dict["sample_size"].append(sample_size)
        if name: 
            kernel_name = list(kernels.keys())[list(kernels.values()).index(kernel)]   
            observations_dict["kernel"].append(kernel_name)
        else:
            observations_dict["kernel"].append(kernel)
        observations_dict["alpha"].append(alpha)
        observations_dict["sample_method"].append(sample_method)
        observations_dict["MSE of GPR"].append(float("inf"))
        observations_dict["MSE of Polynomial Fit"].append(float("inf"))
        observations_dict["Integral Absolute Error of GPR"].append(float("inf"))
        observations_dict["Integral Absolute Error of Polynomial Fit"].append(float("inf"))
    
        return float("inf")    

def design_RBF(coef, lower_bound, upper_bound):
    return coef * RBF(length_scale=(0.1, 1800000), length_scale_bounds=(lower_bound, upper_bound))

def design_RationalQuadratic(coef, lower_bound, upper_bound):
    return  coef * RationalQuadratic(length_scale=1.0, alpha=0.1, length_scale_bounds= (lower_bound, upper_bound))

def design_ExpSineSquared(coef, lower_bound, upper_bound):  
    return coef * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                    length_scale_bounds=(lower_bound, upper_bound),
                                    periodicity_bounds=(1.0, 10.0))
def design_DotProduct(coef, lower_bound, upper_bound):
    return C(coef, (lower_bound, upper_bound)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(lower_bound, upper_bound)) ** 2)    
                  
def design_Matern(coef, nu, lower_bound, upper_bound):
    return coef * Matern(length_scale=1.0, length_scale_bounds=(lower_bound, upper_bound), nu=nu)                  


def objective_function(params):
    coef = params["coef"]
    alpha = params["alpha"]
    min_kernel = params["min_kernel"]
    scaled = params["scaled"]
    nu = None
    if min_kernel == 'Matern':
        nu = params["nu"]
    lower_bound = params["lower_bound"]
    upper_bound = params["upper_bound"]
    sample_size = params["sample_size"]
    sample_method = params["sample_method"]
    if sample_method == "random":
        X = params["X"]
        y = params["y"]
        sample_indices = params["sample_indices"]

    kernels =  {'RBF': design_RBF(coef, lower_bound, upper_bound),
                'RationalQuadratic': design_RationalQuadratic(coef, lower_bound, upper_bound),
               'ExpSineSquared': design_ExpSineSquared(coef, lower_bound, upper_bound),
              'DotProduct': design_DotProduct(coef,lower_bound, upper_bound),
               'Matern': design_Matern(coef, nu, lower_bound, upper_bound)}
    
    print(min_kernel)
    # print(kernels)
    kernel = kernels[min_kernel]
    if sample_method == "systematic":
        return regression(sample_size, sample_method, kernel, alpha, name=False, scaled=scaled)
    else:
        return regression(sample_size, sample_method, kernel, alpha, name=False, scaled=scaled, X=X, y=y, sample_indices=sample_indices)

           
def bayesianopt_for_coef(coef_list, alpha_list, min_kernel, scaled, sample_size, sample_method, nu_list=None, X=None, y=None, sample_indices=None):
    
    if min_kernel == 'Matern' and sample_method == "systematic":
        space = {'coef': hp.choice('coef',coef_list),
                 'alpha': hp.choice('alpha',alpha_list),
                 'nu': hp.choice('nu',nu_list),
                 'min_kernel': hp.choice('min_kernel',[min_kernel]),
                 'scaled': hp.choice('scaled',[scaled]),
                 'sample_size': hp.choice('sample_size',[sample_size]),
                 'sample_method': hp.choice('sample_method',[sample_method]),
                 'lower_bound': hp.quniform('lower_bound',1e-10, 1e-1,5e-4),
                 'upper_bound': hp.quniform('upper_bound', 1e3, 1e10,2e3)
                 }
    
    elif min_kernel == 'Matern' and sample_method == "random":
        space = {'coef': hp.choice('coef',coef_list),
                 'alpha': hp.choice('alpha',alpha_list),
                 'nu': hp.choice('nu',nu_list),
                 'min_kernel': hp.choice('min_kernel',[min_kernel]),
                 'scaled': hp.choice('scaled',[scaled]),
                 'X': hp.choice('X',[X]),
                 'y': hp.choice('y',[y]),
                 'sample_indices': hp.choice('sample_indices',[sample_indices]),
                 'sample_size': hp.choice('sample_size',[sample_size]),
                 'sample_method': hp.choice('sample_method',[sample_method]),
                 'lower_bound': hp.quniform('lower_bound',1e-10, 1e-1,5e-4),
                 'upper_bound': hp.quniform('upper_bound', 1e3, 1e10,2e3)
                 }
    
    elif min_kernel != 'Matern' and sample_method == "random":
        space = {'coef': hp.choice('coef',coef_list),
                 'alpha': hp.choice('alpha',alpha_list),
                 'nu': hp.choice('nu',nu_list),
                 'min_kernel': hp.choice('min_kernel',[min_kernel]),
                 'scaled': hp.choice('scaled',[scaled]),
                 'X': hp.choice('X',[X]),
                 'y': hp.choice('y',[y]),
                 'sample_indices': hp.choice('sample_indices',[sample_indices]),
                 'sample_size': hp.choice('sample_size',[sample_size]),
                 'sample_method': hp.choice('sample_method',[sample_method]),
                 'lower_bound': hp.quniform('lower_bound',1e-10, 1e-1,5e-4),
                 'upper_bound': hp.quniform('upper_bound', 1e3, 1e10,2e3)
                 }
    
    else:
        space = {'coef': hp.choice('coef',coef_list),
                 'alpha': hp.choice('alpha',alpha_list),
                 'min_kernel': hp.choice('min_kernel',[min_kernel]),
                 'scaled': hp.choice('scaled',[scaled]),
                 'sample_size': hp.choice('sample_size',[sample_size]),
                 'sample_method': hp.choice('sample_method',[sample_method]),
                 'lower_bound': hp.quniform('lower_bound',1e-10, 1e-1,5e-4),
                 'upper_bound': hp.quniform('upper_bound', 1e3, 1e10,2e3)
                 }
    

    trials = Trials()
    best = fmin(fn= objective_function,
                space= space,
                algo= tpe.suggest,
                max_evals = 50,
                trials= trials)
    
    return best
