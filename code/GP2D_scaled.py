# -*- coding: utf-8 -*-
"""
@author: Buse Korkmaz & Mehmet Mercangoz

Python implementation of proposed compressor map modelling approach
in "Data Driven Modelling of Centrifugal Compressor Maps for Control 
and Optimization Applications" submitted to ECC22.

Output
    optimized_results: Hyperparamaters of each kernel type that
        produce minimize error and GP and Polynomial Regression
        error metrics by sample size and method
    
    predictions: Produced predictions over validation data with
        the best performed model in optimized_results

usage:
    specify following parameters:
        sample_size = training sample_size, default is 20
        sample_method = training instances sampling method, 
            default is "systematic". For further explanation 
            of sampling methods check implementation is under 
            util.py or the paper above
        data_name = optimized results and produced predictions
            are saved with this data name. For speed line 
            regressions ensure the data_name consists the word 
            of "speed".Choose different data names for different 
            data sources. Currently available datasets under ./data:
                -etas_full : EFR 91S74 efficiency
                -garrett_full1 : Garrett 3076R efficiency
                -garrett_full2 : Garrett 1544 efficiency
                -garrett_to4b : Garrett to4b efficiency
                -etas_speed : EFR 91S74 speed
                -gt 3076r speed : Garrett 3076R speed
                -garrett 1544 speed : Garrett 1544 speed
                -to4b-h3-speed : Garrett to4b speed
        data = use of the ready data sources or pass name of the 
            dataset to read_data
"""

import os
os.chdir("./code")
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic,Matern, RBF,ConstantKernel as C,DotProduct, WhiteKernel,ExpSineSquared
import pandas as pd
import warnings
from util import *
from hyperparameter_optimization import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
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

# Load vector of scanned efficiency values (mass flow/pressure ratio/efficiency)
etas_full, garrett_full1, garrett_full2,garrett_to4b = read_data()
data_name = "to4b-h3-speed"
data = read_data(data_name)

if "speed" in data_name:
    scaled=True
else:
    scaled=False
    
data_length = len(data)

def regression(sample_size, sample_method, kernel, alpha, kernels=None, name=True, scaled=True, X=None, y=None, sample_indices=None):
    try:
        try:
            if sample_indices !=None:
                # Prepare input (X) and output (y) vectors for training
                Xsize = (sample_size,2)
                ysize = (sample_size,1)
                
                sample_indices = []
                if sample_method == "systematic":
                    X , y = systematic_sampling(data,Xsize, ysize,sample_size, sample_indices,scaled)
                elif sample_method == "random":
                    X , y = random_sample(data,Xsize, ysize,sample_size, sample_indices)    
        except:
            pass
        
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
        
def optimize_regression(sample_size, sample_method, scaled, X=None, y=None, sample_indices=None):
    coef_list = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 200, 500]
    alpha_list = [1e-20, 1e-15, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    nu_list = [0.1,0.5, 1, 1.5, 2, 2.5,3, 3.5, 5, 7, 100, 1000]
     
    optimized_results = {
        # 'RBF': -1,
        #                  'RationalQuadratic': -1,
        #                  'ExpSineSquared': -1,
        #                  'DotProduct': -1,
                         'Matern': -1}
    kernel_names = list(optimized_results.keys())
    nu = None
    
    for kernel_name in kernel_names:
        if kernel_name == 'Matern' and sample_method == "random":
            best_params = bayesianopt_for_coef(coef_list, alpha_list, kernel_name, scaled, sample_size, sample_method, nu_list, X, y, sample_indices)
        elif kernel_name == 'Matern' and sample_method == "systematic":
            best_params = bayesianopt_for_coef(coef_list, alpha_list, kernel_name, scaled, sample_size, sample_method, nu_list)
        elif kernel_name != 'Matern' and sample_method == "random":
            best_params = bayesianopt_for_coef(coef_list, alpha_list, kernel_name, scaled, sample_size, sample_method, X, y, sample_indices)
        else:
            best_params = bayesianopt_for_coef(coef_list, alpha_list, kernel_name, scaled, sample_size, sample_method)        
        
        coef = coef_list[best_params["coef"]]
        alpha =  alpha_list[best_params["alpha"]]
        lower_bound = best_params["lower_bound"]
        upper_bound = best_params["upper_bound"]
        
        if kernel_name == 'Matern':
            nu = nu_list[best_params["nu"]]
        
        kernels =  {'RBF': design_RBF(coef, lower_bound, upper_bound),
                'RationalQuadratic': design_RationalQuadratic(coef, lower_bound, upper_bound),
               'ExpSineSquared': design_ExpSineSquared(coef, lower_bound, upper_bound),
              'DotProduct': design_DotProduct(coef,lower_bound, upper_bound),
               'Matern': design_Matern(coef, nu, lower_bound, upper_bound)}
  
        kernel = kernels[kernel_name]
        
        optimized_results[kernel_name] = regression(sample_size, sample_method, kernel, alpha, name=False, scaled=scaled, X=X, y=y)
    return optimized_results
    

sample_method = "systematic"
sample_size = 60
if sample_method == "systematic":
    optimized_results = optimize_regression(sample_size=sample_size, sample_method="systematic", scaled=scaled)
    print(optimized_results)

elif sample_method == "random":
    # Prepare input (X) and output (y) vectors for training
    Xsize = (sample_size,2)
    ysize = (sample_size,1)
    
    sample_indices = []
    X , y = random_sample(data,Xsize, ysize,sample_size, sample_indices)    
    
    optimized_results = optimize_regression(sample_size=sample_size, sample_method="random", scaled=scaled, X=X, y=y, sample_indices=sample_indices)


observations = pd.DataFrame.from_dict(observations_dict, orient='index').T
observations.to_csv(data_name + '_optimized_results.csv', index=False)

observations_sorted = observations.sort_values(by=['MSE of GPR'])
kernel = observations_sorted.kernel[0]
alpha = observations_sorted.alpha[0]

keys = list(pred_dict.keys())
for key in keys:
    pred_dict[key] = []
regression(sample_size, sample_method, kernel, alpha, name=False, scaled=scaled)

predictions = pd.DataFrame.from_dict(pred_dict, orient='index').T
predictions.to_csv(data_name + '_predictions.csv', index=False)

