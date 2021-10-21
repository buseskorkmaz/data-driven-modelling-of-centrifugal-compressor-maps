"""
@author: Buse Korkmaz
Polynomial Regression Implementation for compressor effiency
and pressure ratio approximation.
Output
    to see MSE values of different degrees use as below 
    create_polynomial_regression_model(sample_size, sample_method, 
                                       kernel, alpha, plot=False, 
                                       lower_degree=2, 
                                       upper_degree=3)
    
    to see fit plot with MSE values of GP and polynomial use as below
    create_polynomial_regression_model(sample_size, sample_method, 
                                       kernel, alpha, plot=True, 
                                       lower_degree=2, 
                                       upper_degree=5)
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
        kernel = use the best kernel in optimized results or any
            custom kernel
        alpha = use the best kernel in optimized results or any
            custom alpha
"""
import os
os.chdir("./code")
from util import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
os.chdir("../data")

etas_full, garrett_full1, garrett_full2, garrett_to4b = read_data()
data = pd.read_csv('garret to4b.csv').values
data_length = len(data)
data_name= 'garret to4b'
sample_size = 20
sample_method = "systematic"
kernel = 0.0316**2 * RBF(length_scale=[0.1, 1.8e+06])
alpha = 0.0001
           
def create_polynomial_regression_model(sample_size, sample_method, kernel, alpha, plot=True, lower_degree=2, upper_degree=5):
    "Creates a polynomial regression model for the given degree"
    if 'speed' in data_name:
        speed = True
    else:
        speed = False
    
    # Prepare input (X) and output (y) vectors for training
    Xsize = (sample_size,2)
    ysize = (sample_size,1)
    
    sample_indices = []
    if sample_method == "systematic":
        X , y = systematic_sampling(data,Xsize, ysize,sample_size, sample_indices,speed)
    elif sample_method == "random":
        X , y = random_sample(data,Xsize, ysize,sample_size, sample_indices)    
    
    if speed:
        x_scaler = StandardScaler()
        # transform input data
        x_scaled = x_scaler.fit_transform(X)
        y_scaler = StandardScaler()
        # transform output data
        y_scaled = y_scaler.fit_transform(y)

    data_length = len(data)
       
    
    if plot:
        plt.style.use('seaborn-bright')
        fig, ax = plt.subplots(figsize=(12,8))
    
        
        try:
            gp = GaussianProcessRegressor(kernel=kernel,alpha=alpha,n_restarts_optimizer=150)
        except:
            gp = GaussianProcessRegressor(kernel=kernel,alpha=alpha)            
       
        if speed:
            gp.fit(x_scaled, y_scaled)
        else:
            gp.fit(X, y)
        
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
        y_pfsize = (data_length-sample_size,1)
        size_errorp = (data_length-sample_size,1)
        y_pf = np.zeros(y_pfsize)
        errorp = np.zeros(size_errorp)
        y_true, xs = [] , []
        idx = 0
        if not speed:
            std_effs = []
        else:
            std_speeds = []
        for i in range(data_length):
            if (i not in sample_indices):
                if speed:
                    y_pf[idx,0], std_speed = gp.predict(x_scaler.transform(Xf[idx].reshape(1, -1)), return_std=True)
                    std_speeds.append(std_speed)
                else:
                    y_pf[idx,0], std_eff = gp.predict(Xf[idx].reshape(1, -1), return_std=True)
                    std_effs.append(std_eff)
                # Generate paraboloid prediction errors for all scanned values  
                if speed:
                    errorp[idx,0]= y_scaler.inverse_transform(y_pf[idx,0].reshape(1, -1))-data[i,2]
                else:
                    errorp[idx,0]= y_pf[idx,0]-data[i,2]
                y_true.append(data[i,2])
                xs.append(i)
                idx += 1
        if speed:
            gp_pred = y_scaler.inverse_transform(y_pf.reshape(-1,1))
        else:
            gp_pred = y_pf.reshape(-1,1)
    
        #plotting predictions
        plt.plot(xs,gp_pred,label='GPR' +  ', MSE {:.3E}'.format(np.square(errorp).mean()),lw=3)
        if speed:
            plt.fill_between(xs, gp_pred.ravel() - 1.96*np.array(std_speeds).ravel(), gp_pred.ravel() + 1.96*np.array(std_speeds).ravel(), 
                         alpha=0.2, color='b', label = "95% confidence interval of GPR")
        else:
             plt.fill_between(xs, gp_pred.ravel() - 1.96*np.array(std_effs).ravel(), gp_pred.ravel() + 1.96*np.array(std_effs).ravel(), 
                         alpha=0.2, color='b', label = "95% confidence interval of GPR")

   
    for degree in range(lower_degree,upper_degree):
        # creating pipeline and fitting it on data
        Input=[('polynomial',PolynomialFeatures(degree)),('modal',LinearRegression())]
        pipe=Pipeline(Input)
        
       
        # fit the transformed features to Linear Regression
        if speed and sample_size > 20:
            pipe.fit(x_scaled, y_scaled)
        else:
            pipe.fit(X,y)
        # pipe[0].get_feature_names(['a', 'b'])
        # pipe[1].coef_
         
        # Generate inputs from all scanned values
        Xfsize = (data_length-sample_size,2)
        Xf = np.zeros(Xfsize)
        idx = 0
        for i in range(data_length):
            if(i not in sample_indices):
                Xf[idx,0] = data[i,0]
                Xf[idx,1] = data[i,1]
                idx += 1    
                      
        # Generate polynomial predictions for all scanned values  
        y_pfsize = (data_length-sample_size,1)
        size_errorp = (data_length-sample_size,1)
        y_pf = np.zeros(y_pfsize)
        errorp = np.zeros(size_errorp)
        y_true, xs = [] , []
        idx = 0
        for i in range(data_length):
            if (i not in sample_indices):
                # Generate polynomial prediction errors for all scanned values  
                if speed and sample_size >20:
                    y_pf[idx,0] = pipe.predict(x_scaler.transform(Xf[idx].reshape(1, -1)))
                    errorp[idx,0]= y_scaler.inverse_transform(y_pf[idx,0].reshape(-1,1))-data[i,2]
                else:
                    y_pf[idx,0] = pipe.predict(Xf[idx].reshape(1, -1))
                    errorp[idx,0]= y_pf[idx,0]-data[i,2]

                y_true.append(data[i,2])
                xs.append(i)
                idx += 1
                
        x_poly, poly_pred = xs, y_pf.reshape(-1,1)
        
        #plotting predictions
        if speed and sample_size > 20:
            poly_pred = y_scaler.inverse_transform(y_pf.reshape(-1,1))
        else:
            poly_pred = y_pf.reshape(-1,1)
        
        if plot:
            plt.plot(x_poly,poly_pred,label='Degree ' + str(degree) + ', MSE {:.3E}'.format(np.square(errorp).mean()), lw=3)
            plt.xlabel('Test samples',fontsize=20)
            if not speed:
                plt.ylabel('Efficiency',fontsize=20)
                plt.title('Comparison of Efficiency Approximation (sample size = {0})'.format(sample_size), fontsize=20)
            else:
                plt.ylabel('Pressure Ratio', fontsize=20)
                plt.title('Comparison of Pressure Ratio Approximation (sample size = {0})'.format(sample_size), fontsize=20)
                     
        print('MSE of degree {}'.format(degree))
        print(np.square(errorp).mean())
        
        print('Max error of degree {}'.format(degree))
        print(np.max(errorp))
    
    if plot:
        plt.scatter(x_poly,y_true,s=15, c = 'red', label='True value')
        plt.legend(fontsize=16)
        if not speed:
            plt.savefig("Efficiency_Paraboloid_comparison_{}_sample_size_{}_95conf.pdf".format(data_name, sample_size))
        else:
            plt.savefig("Speed_Paraboloid_comparison_{}_sample_size_{}_95conf.pdf".format(data_name, sample_size))
        plt.show()
    
    else:
        return np.square(errorp).mean()

# to see MSE values of different degrees use as below 
# create_polynomial_regression_model(sample_size, sample_method, kernel, alpha, plot=False, lower_degree=2, upper_degree=3)

# to see fit plot with MSE values of GP and polynomial use as below
# create_polynomial_regression_model(sample_size, sample_method, kernel, alpha, plot=True, lower_degree=2, upper_degree=5)