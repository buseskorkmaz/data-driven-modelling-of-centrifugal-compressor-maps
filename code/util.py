# -*- coding: utf-8 -*-
"""
@author: Buse Korkmaz
"""

import pandas as pd
from numpy import loadtxt
import numpy as np
import random
import os


def read_data(filename=None):
    etas_full=loadtxt("etas_import_scaled_full.txt")
    garrett_full1 = pd.read_csv('garret 3076R.csv').values
    garrett_full2 = pd.read_csv('garret 1544.csv').values
    garrett_to4b = pd.read_csv('garret to4b.csv').values

    if filename == None:    
        return etas_full, garrett_full1, garrett_full2, garrett_to4b
    else:
        return pd.read_csv(filename+".csv").values

def random_sample(data, Xsize,ysize,sample_size,indices=[]):
    X = np.zeros(Xsize)
    y = np.zeros(ysize)
    j=random.sample(range(len(data)), sample_size)
    for i in range(sample_size):
        X[i,0] = data[j[i],0]
        X[i,1] = data[j[i],1]
        y[i,0] = data[j[i],2]
        indices.append(j[i])
    
    return X,y

def systematic_sampling(data, Xsize, ysize, sample_size=20, sample_indices=[],speed=False):
    X = np.zeros(Xsize)
    y = np.zeros(ysize)
    if not speed or (speed and sample_size <= 20):
        indices = np.arange(0, len(data) - 1, len(data)//sample_size)
    else:
        indices = np.arange(0, len(data) - 1, len(data)/sample_size)
    for i in range(sample_size):
        if speed and sample_size > 20:
            j=round(indices[i])
        else:
            j = indices[i]
        X[i,0] = data[j,0]
        X[i,1] = data[j,1]
        y[i,0] = data[j,2]
        sample_indices.append(j)
   
    return X,y
