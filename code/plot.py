# -*- coding: utf-8 -*-
"""
@author: Buse Korkmaz

Output
    - surface plots : 3d plot of Gaussian Process Regression
        and Polynomial fit. Use for efficiency
        
    - contour plots : Efficiency map with GPR and Polynomial 
        Regression.
        
    - scatter plot : True speed lines and approximated speed lines
        by pressure ratio regression.
    
    - reproduction : Reproduction of Garret TO4B which is used in
    the paper.
    
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
import numpy as np
import re
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import pandas as pd
from util import *
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic,Matern, RBF,ConstantKernel as C,DotProduct, WhiteKernel,ExpSineSquared
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from scipy.spatial import ConvexHull
import matplotlib 
from hyperparameter_optimization import design_DotProduct, design_ExpSineSquared, design_Matern, design_RBF, design_RationalQuadratic
np.random.seed(7)

os.chdir("../data")

data_name = "gt 3076R speed"
etas_full, garrett_full1, garrett_full2, garrett_to4b = read_data()
data = read_data(data_name)
#data = garrett_full2

if "speed" in data_name:
    scaled=True
    speed=True
else:
    scaled=False
    speed=False    

plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    

def create_surface_plots(sample_size, sample_method, kernel, alpha, save=False):
    
    # Prepare input (X) and output (y) vectors for training
    Xsize = (sample_size,2)
    ysize = (sample_size,1)
    sample_indices = []
    if sample_method == "systematic":
        X , y = systematic_sampling(data,Xsize, ysize,sample_size, sample_indices)
    elif sample_method == "random":
        X , y = random_sample(data,Xsize, ysize,sample_size, sample_indices)    
    
            
    # Setup GPR with a kernel
    gp = GaussianProcessRegressor(kernel=kernel,alpha=alpha, n_restarts_optimizer=150)
    # Train GP
    gp.fit(X, y)
    
    # Setup least squares regression matrix for polynomial
    Input=[('polynomial',PolynomialFeatures(3)),('modal',LinearRegression())]
    pipe=Pipeline(Input)
    # fit the transformed features to Linear Regression
    pipe.fit(X, y)
    
    # Generate mesh of inputs for plotting efficiency surfaces
    x1 = np.linspace(5, 60,num=30) #p
    x2 = np.linspace(1, 3.4,num=30) #q
    x1x2 = np.array(list(product(x1, x2)))
           
    # Generate mesh of inputs for plotting efficiency surfaces
    x1 = np.linspace(X[:,0].min(), X[:,0].max() ,num=20) #p
    x2 = np.linspace(X[:,1].min(), X[:,1].max() ,num=20) #q
    x1x2 = np.array(list(product(x1, x2)))
    
    # Generate mesh of GP outputs for plotting efficiency surfaces
    y_predg, MSE = gp.predict(x1x2, return_std=True)
    
    # Generate mesh of polynomial outputs for plotting efficiency surfaces
    y_predp = pipe.predict(x1x2)
    
    # Create surface plots
    plt.style.use('seaborn-bright')

    X0p, X1p = x1x2[:,0].reshape(20,20), x1x2[:,1].reshape(20,20)
    Zp = np.reshape(y_predg,(20,20))
    Zt = np.reshape(y_predp,(20,20))
    fig = plt.figure(figsize=(10,8))
    ax = fig.gca(projection='3d')     
    surf1 = ax.plot_surface(X0p, X1p, Zp, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
    surf2 = ax.plot_surface(X0p, X1p, Zt, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
    ax.set_xlabel('Compressor mass flow [kg/s]')
    ax.set_ylabel('Pressure ratio [-]')
    ax.set_zlabel('Efficiency [-]')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_facecolor('white')
    plt.rcParams['grid.color'] = "#000000"
    fig.colorbar(surf2, pad = 0.1)
    if save:
        plt.savefig("Surface_plot_of_{}.pdf".format(data_name))
    plt.show()
    

def contour_plots(sample_size, sample_method, kernel, alpha, speed=False, save=False,degree=3):
    
    # Prepare input (X) and output (y) vectors for training
    Xsize = (sample_size,2)
    ysize = (sample_size,1)
    
    sample_indices = []
    if sample_method == "systematic":
        X , y = systematic_sampling(data,Xsize, ysize,sample_size, sample_indices)
    elif sample_method == "random":
        X , y = random_sample(data,Xsize, ysize,sample_size, sample_indices)    
    
    if speed:
        x_scaler = StandardScaler()
        # transform input data
        x_scaled = x_scaler.fit_transform(X)
        y_scaler = StandardScaler()
        # transform output data
        y_scaled = y_scaler.fit_transform(y)

    # Setup GPR with a kernel
    try:
        gp = GaussianProcessRegressor(kernel=kernel,alpha=alpha,n_restarts_optimizer=150)
    except:
        gp = GaussianProcessRegressor(kernel=kernel,alpha=alpha)            
    
    # Train GP
    if speed:
        gp.fit(x_scaled, y_scaled)
    else:
        gp.fit(X, y)
    
    Input=[('polynomial',PolynomialFeatures(degree)),('modal',LinearRegression())]
    pipe=Pipeline(Input)
    # fit the transformed features to Linear Regression
    pipe.fit(X, y)
    
    # Generate mesh of inputs for plotting efficiency surfaces
    x1 = np.linspace(X[:,0].min()-1, X[:,0].max()+1,num=30) #p
    x2 = np.linspace(X[:,1].min()-0.1, X[:,1].max()+0.1,num=30) #q
    x1x2 = np.array(list(product(x1, x2)))
    # Generate mesh of GP outputs for plotting efficiency surfaces
    if speed:
        # Generate mesh of GP outputs for plotting efficiency surfaces
        y_predg, MSE = gp.predict(x_scaler.transform(x1x2), return_std=True)
    else:
        y_predg, MSE = gp.predict(x1x2, return_std=True)
    
    if speed:
        y_predg = y_scaler.inverse_transform(y_predg)
      
    # Generate mesh of paraboloid outputs for plotting efficiency surfaces
    y_predp = pipe.predict(x1x2)
    
    # Create surface plots
    plt.style.use('seaborn-bright')
    
    """Gaussian Contour Plot"""
    fig, ax = plt.subplots(figsize=(10,8))
    cm = matplotlib.cm.get_cmap('jet')

    cnorm = matplotlib.colors.Normalize(vmin=y_predp.min(), vmax=y_predp.max())
    smap = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cm)

    X0p, X1p = x1x2[:,0].reshape(30,30), x1x2[:,1].reshape(30,30)
    Zp = np.reshape(y_predg,(30,30))
    Zt = np.reshape(y_predp,(30,30))
    if data_name == "garret to4b":
        CS = ax.contour(X0p, X1p, Zp, levels = np.concatenate((np.arange(-0.08, 0.6, 0.04), [0.6,0.65,0.68,0.7, 0.72, 0.74]), axis=0), cmap = 'jet', vmin=y_predg.min(), vmax=y_predg.max())
    else:
        CS = ax.contour(X0p, X1p, Zp,levels = np.concatenate((np.arange(y_predg.min(), np.unique(data[:,2]).min(), 0.04), np.unique(data[:,2]))), cmap = 'jet', vmin=y_predg.min(), vmax=y_predg.max())
    ax.clabel(CS, inline=True, fontsize=10, colors='black', fmt = '%.2f')
    # CS = ax.contour(X0p, X1p, Zt, levels = [0.7, 0.74], cmap = 'jet', vmin=y_predg.min(), vmax=y_predg.max())
    # ax.clabel(CS, inline=True, fontsize=10, colors='black', fmt = '%.2f')
    
    if data_name == "garret to4b":
        points = pd.read_csv("gt-to4b-h3-operating-range.csv").values
        # PERFORM CONVEX HULL
        hull = ConvexHull(points)  
        
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'black', lw=4)
        
    ax.grid(True)
    ax.set_xlabel('Compressor mass flow [kg/s]', fontsize=20)
    ax.set_ylabel('Pressure ratio [-]',fontsize=20)
    plt.title('Contour Plot of Gaussian Process Regression',fontsize=20)
    if save:
        plt.savefig("Contour_plot_of_GPR_{}.pdf".format(data_name))
    plt.show()
    
    """Polynomial Contour Plot"""
    fig, ax = plt.subplots(figsize=(10,8))
    cm = matplotlib.cm.get_cmap('jet')

    cnorm = matplotlib.colors.Normalize(vmin=y_predp.min(), vmax=y_predp.max())
    smap = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cm)

    X0p, X1p = x1x2[:,0].reshape(30,30), x1x2[:,1].reshape(30,30)
    Zp = np.reshape(y_predg,(30,30))
    Zt = np.reshape(y_predp,(30,30))
    if data_name == "garret to4b":
        CS = ax.contour(X0p, X1p, Zt, levels = np.concatenate((np.arange(-4.8, -0.1, 0.2),np.arange(-0.1, 0.6, 0.08), [0.6,0.65,0.68,0.7, 0.72, 0.74]), axis=0), cmap = 'jet', vmin=y_predg.min(), vmax=y_predg.max())
    else:
        CS = ax.contour(X0p, X1p, Zt, levels = np.concatenate((np.arange(y_predp.min(), np.unique(data[:,2]).min(), 0.08), np.unique(data[:,2]))), cmap = 'jet', vmin=y_predg.min(), vmax=y_predg.max())
    ax.clabel(CS, inline=True, fontsize=10, colors='black', fmt = '%.2f')
   
    if data_name == "garret to4b":
        points = pd.read_csv("gt-to4b-h3-operating-range.csv").values
        # PERFORM CONVEX HULL
        hull = ConvexHull(points)  
        
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'black', lw=4)
    
    ax.grid(True)
    ax.set_xlabel('Compressor mass flow [kg/s]', fontsize=20)
    ax.set_ylabel('Pressure ratio [-]',fontsize=20)
    plt.title('Contour Plot of Polynomial Regression (degree = {})'.format(degree),fontsize=20)
    if save:
        plt.savefig("Contour_plot_of_Polynomial_{}.pdf".format(data_name))
    plt.show()


def scatter_plot(sample_size, sample_method, kernel, alpha,scaled=True):
    # use for only speed, contour plot is better for efficiency results
    
    plot_dict = {'pressure_ratio' : [],
                 'gaussian_pred': [],
                 'paraboloid_pred': [],
                 'speed': [],
                 'mass_flow':[]}
    
    data_length = len(data)
        
    # Prepare input (X) and output (y) vectors for training
    Xsize = (sample_size,2)
    ysize = (sample_size,1)
    # Randomly sample 25 input output pairs out of the scanned efficiency values
    sample_indices = []
    if sample_method == "systematic":
        X , y = systematic_sampling(data,Xsize, ysize,sample_size, sample_indices)
    elif sample_method == "random":
        X , y = random_sample(data,Xsize, ysize,sample_size, sample_indices)    
    
    if scaled:
        x_scaler = StandardScaler()
        # transform input data
        x_scaled = x_scaler.fit_transform(X)
        y_scaler = StandardScaler()
        # transform output data
        y_scaled = y_scaler.fit_transform(y)

    # Setup GPR with a kernel  
    try:
        gp = GaussianProcessRegressor(kernel=kernel,alpha=alpha,n_restarts_optimizer=150)
    except:
        gp = GaussianProcessRegressor(kernel=kernel,alpha=alpha)            
    
    # Train GP
    if scaled:
        gp.fit(x_scaled, y_scaled)
    else:
        gp.fit(X, y)
    
    
    # Generate inputs from all scanned efficiency values
    Xfsize = (data_length-sample_size,2)
    Xf = np.zeros(Xfsize)
    idx = 0
    for i in range(data_length):
        if(i not in sample_indices):
            Xf[idx,0] = data[i,0]
            Xf[idx,1] = data[i,1]
            idx += 1  
    
          
    # Generate GP predictions for all scanned efficiency values    
    if scaled:
        y_predgf, MSEf = gp.predict(x_scaler.transform(Xf), return_std=True)
        y_predgf = y_scaler.inverse_transform(y_predgf)
    else:
        y_predgf, MSEf = gp.predict(Xf, return_std=True)
    
    # Calculate GP prediction errors for all scanned efficiency values 
    size_errorg = (data_length-sample_size,1)
    errorg = np.zeros(size_errorg)
    idx = 0
    for i in range(data_length):
        if i not in sample_indices:
            errorg[idx,0]=y_predgf[idx,0]-data[i,2]
            plot_dict["mass_flow"].append(data[i,0])
            plot_dict["speed"].append(data[i,1])
            plot_dict["pressure_ratio"].append(data[i,2])
            plot_dict["gaussian_pred"].append(y_predgf[idx,0])
            idx += 1

    
    plt.style.use('seaborn-bright')

    plot_frame = pd.DataFrame.from_dict(plot_dict, orient='index').T
    fig, ax = plt.subplots()
    plot_frame.plot(figsize= (10,8), title="GPR Predictions for Pressure Ratio", kind= 'scatter', ax=ax,  x = 'mass_flow', y = 'gaussian_pred', c = 'red', s = 5)
    plot_frame.plot(kind= 'scatter', x = 'mass_flow', y = 'pressure_ratio', c = 'blue', s= 5, ax=ax)
    ax.legend(["True value", "GPR Predictions"])
    ax.set(ylabel = 'Pressure ratio', xlabel = 'Mass flow')
    plt.yscale("linear")
    plt.savefig("Scatter_plot_of_{}.pdf".format(data_name))
    plt.show()

def reproduction():
    # Create surface plots
    plt.style.use('seaborn-bright')

    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_ylim([1,3])
    ax.set_xlim([7,56])

    ax.set_yticks(np.arange(1, 3.40, step=0.25))
 
    eff = pd.read_csv('efficiency lines.csv').values
    
    points_60 = eff[np.where(eff[:,2] == 0.6)][:,:2]
    points_65 = eff[np.where(eff[:,2] == 0.65)][:,:2][:-6]
    points_70_h1 = pd.read_csv('0.7 first half.csv').values
    points_70_top = pd.read_csv('0.7 top.csv').values
    points_70_h2 = pd.read_csv('0.7 second half.csv').values
    points_70_bottom = pd.read_csv('0.7 bottom.csv').values[4:-1]
    points_70 = np.concatenate([points_70_h1, points_70_top, points_70_h2, points_70_bottom] , axis=0)
    points_72_h1 = pd.read_csv('0.72 first half.csv').values
    points_72_top = pd.read_csv('0.72 top.csv').values
    points_72_h2 = pd.read_csv('0.72 second half.csv').values
    points_72_bottom = pd.read_csv('0.72 bottom.csv').values
    points_72 = np.concatenate([points_72_h1, points_72_top, points_72_h2, points_72_bottom] , axis=0)
    points_74_h1 = pd.read_csv('0.74 first half.csv').values
    points_74_top = pd.read_csv('0.74 top.csv').values
    points_74_h2 = pd.read_csv('0.74 second half.csv').values
    points_74_bottom = pd.read_csv('0.74 bottom.csv').values
    points_74 = np.concatenate([points_74_h1, points_74_top, points_74_h2, points_74_bottom] , axis=0)
    surge_limit = pd.read_csv('surge limit.csv').values
    speed_line_1 = pd.read_csv('46200.csv').values
    speed_line_2 = pd.read_csv('69500.csv').values[:-12]
    speed_line_3 = pd.read_csv('84200.csv').values[:-7]
    speed_line_4 = pd.read_csv('96600.csv').values[:-5]
    speed_line_5 = pd.read_csv('105500.csv').values[:-4]
    speed_line_6 = pd.read_csv('114100.csv').values
    speed_line_7 = pd.read_csv('120400.csv').values
    speed_line_8 = np.concatenate([pd.read_csv('126000.csv').values[:-7],pd.read_csv('126000.csv').values[-2:]],axis=0)
    surge_near = pd.read_csv('surge near line.csv').values
    
    efficiency_points = [points_60, points_65]
    speed_lines = [speed_line_1, speed_line_2, speed_line_3, speed_line_4,
                   speed_line_5, speed_line_6, speed_line_7, speed_line_8]
    
    hull = ConvexHull(points_70)  
    
    for simplex in hull.simplices:
        ax.plot(points_70[simplex, 0], points_70[simplex, 1], 'black', lw=2)
    
    hull = ConvexHull(points_72)  
    
    for simplex in hull.simplices:
        ax.plot(points_72[simplex, 0], points_72[simplex, 1], 'black', lw=2)
    
    hull = ConvexHull(points_74)  
    
    for simplex in hull.simplices:
        ax.plot(points_74[simplex, 0], points_74[simplex, 1], 'black', lw=2)
    
    ax.plot(surge_limit[:-3,0], surge_limit[:-3,1],'k--', lw=2)
    ax.plot([surge_limit[0][0], speed_line_1[12][0]], [surge_limit[0][1], speed_line_1[12][1]],'k--', lw=2)
    ax.plot([surge_limit[6][0], speed_line_2[0][0]], [surge_limit[6][1], speed_line_2[0][1]],'k', lw=2)
    ax.plot([surge_limit[12][0], speed_line_3[0][0]], [surge_limit[12][1], speed_line_3[0][1]],'k', lw=2)
    ax.plot([surge_limit[17][0], speed_line_4[0][0]], [surge_limit[17][1], speed_line_4[0][1]],'k', lw=2)
    ax.plot([surge_limit[23][0], speed_line_5[0][0]], [surge_limit[23][1], speed_line_5[0][1]],'k', lw=2)
    ax.plot([surge_limit[28][0], speed_line_6[0][0]], [surge_limit[28][1], speed_line_6[0][1]],'k', lw=2)
    ax.plot([surge_limit[33][0], speed_line_7[0][0]], [surge_limit[33][1], speed_line_7[0][1]],'k', lw=2)
    ax.plot([surge_limit[36][0], speed_line_8[0][0]], [surge_limit[36][1], speed_line_8[0][1]],'k', lw=2)
    ax.plot([points_60[0][0], speed_line_1[9][0]], [points_60[0][1], speed_line_1[9][1]],'k', lw=2)
    ax.plot([surge_near[0][0], speed_line_1[16][0]], [surge_near[0][1], speed_line_1[16][1]],'k', lw=2)
    
    text0 = ax.annotate("60%", xy=(0.7, 0.3), xycoords=ax.transAxes)
    text1 = ax.annotate("65%", xy=(0.57, 0.25), xycoords=ax.transAxes)
    text2 = ax.annotate("70%", xy=(0.63, 0.35), xycoords=ax.transAxes)   
    text3 = ax.annotate("72%", xy=(0.63, 0.4), xycoords=ax.transAxes)   
    text4 = ax.annotate("74%", xy=(0.54, 0.4), xycoords=ax.transAxes)
    text5 = ax.annotate("SURGE LIMIT", xy=(0.28, 0.39), xycoords=ax.transAxes)
    text5.set_fontsize(14)
    text5.set_c('k')
    text5.set_rotation(35)
    text5.set_bbox(dict(boxstyle='square,pad=0',fc='white',edgecolor='white'))
 
    speed_line_1_text = ax.annotate("46,200", xy=(0.30, 0.05), xycoords=ax.transAxes)
    speed_line_2_text = ax.annotate("69,500", xy=(0.49, 0.13), xycoords=ax.transAxes)
    speed_line_3_text = ax.annotate("84,200", xy=(0.66, 0.23), xycoords=ax.transAxes)
    speed_line_4_text = ax.annotate("96,600", xy=(0.78, 0.31), xycoords=ax.transAxes)
    speed_line_5_text = ax.annotate("105,500", xy=(0.83, 0.37), xycoords=ax.transAxes)
    speed_line_6_text = ax.annotate("114,100", xy=(0.45, 0.65), xycoords=ax.transAxes)
    speed_line_7_text = ax.annotate("120,400", xy=(0.52, 0.74), xycoords=ax.transAxes)
    speed_line_8_text = ax.annotate("126,000", xy=(0.55, 0.81), xycoords=ax.transAxes)
    
    texts = [text0, text1, text2, text3, text4,
             speed_line_1_text, speed_line_2_text, speed_line_3_text, speed_line_4_text,
             speed_line_5_text, speed_line_6_text, speed_line_7_text, speed_line_8_text]
    
    for text in texts:
        text.set_fontsize(14)
        text.set_c('k')
        text.set_bbox(dict(boxstyle='square,pad=0',fc='white',edgecolor='white'))
   
    for speed_line in speed_lines:
        for i in range(0,len(speed_line)-1):
            ax.plot(speed_line[i:i+2,0], speed_line[i:i+2,1], 'black', lw=2)
        
    for p in efficiency_points:
        for i in range(0,len(p)-1):
            ax.plot(p[i:i+2,0], p[i:i+2,1], 'black',lw=2)
    
    for i in range(0,len(surge_near)-1):
        ax.plot(surge_near[i:i+2,0], surge_near[i:i+2,1], 'black',lw=2)
    
   
    ax.grid(True)
    ax.set_xlabel('Corrected mass flow [kg/s]', fontsize=20)
    ax.set_ylabel('Pressure ratio [-]',fontsize=20)
    plt.title('Garrett T04B-H3 Compressor Map',fontsize=20)
    plt.savefig("Garrett T04B-H3 Compressor Map.pdf")
    plt.show()

observations = pd.read_csv(data_name  +'_optimized_results.csv')

sample_size = observations["sample_size"].values[0]
sample_method = observations["sample_method"].values[0]
observations_sorted = observations.sort_values(by=['MSE of GPR'])
alpha = observations_sorted['alpha'].values[0]

# it will be in string form, convert to kernel object
kernel_str = observations_sorted['kernel'].values[0]
if 'Matern' in kernel_str:
    try:
        [coef, nu] = re.findall(r'\d+\.\d+', kernel_str)
    except:
        [coef] = re.findall(r'\d+\.\d+', kernel_str)
        nu = re.findall(r'\d+', kernel_str)[-1]
    coef , nu = float(coef)**2 , float(nu)
else:
    [coef, lower_bound, upper_bound] = re.findall(r'\d+\.\d+', kernel_str)
    coef , lower_bound, upper_bound = float(coef)**2 , float(lower_bound), float(upper_bound)

if 'RBF' in kernel_str:
    kernel = coef * RBF(length_scale=[lower_bound, upper_bound])

elif 'DotProduct' in kernel_str:
    kernel = C(coef, (lower_bound, upper_bound)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(lower_bound, upper_bound)) ** 2)

elif 'ExpSineSquared' in kernel_str:
    kernel = coef * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                    length_scale_bounds=(lower_bound, upper_bound),
                                    periodicity_bounds=(1.0, 10.0))
elif 'RationalQuadratic' in kernel_str:
    kernel = coef * RationalQuadratic(length_scale=1.0, alpha=0.1, length_scale_bounds= (lower_bound, upper_bound))

else:
    kernel = coef * Matern(length_scale=1.0, length_scale_bounds=(0.1, 1.8e6), nu=nu)                  

if data_name == 'etas_full':
    degree = 2
elif data_name == "garrett_full1":
    degree = 3
elif data_name == "garrett_full2":  
    degree = 4    

# create_surface_plots(sample_size, sample_method, kernel, alpha)
# contour_plots(sample_size, sample_method, kernel, alpha,save=True,degree=degree)
scatter_plot(sample_size, sample_method, kernel, alpha)

