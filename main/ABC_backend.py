#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries
import numpy as np
import pandas as pd
import numba
from numba import njit
import warnings

# Deactivate numba deprecation warnings
warnings.simplefilter('ignore', category=numba.NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=numba.NumbaPendingDeprecationWarning)

data = pd.read_csv(r"covid_19_clean_complete.csv")

##########################################################################

# Function to sort random number according to a given histogram
@njit(fastmath=True) # Numba pre-compilation decorator
def sort(n, hist, bins): 
    # n: how many numbers to sort;
    # hist: l-sized array with height of columns of normalized histogram;
    # bins: (l+1)-sized array with values of bins divisions
    
    d = bins[1] - bins[0] # Bin size
    
    dat = [] # List of sorted random numbers
    
    for i in range(n):
        
        x = np.random.uniform(0., 1.)
        
        # Conversion of 0-1 random number to number sorted according to the given histogram
        for j in range(len(hist)):
            
            if (x < np.sum(hist[:j+1])*d):
                
                dat.append(np.random.uniform(bins[j], bins[j+1]))
                break
    
    return np.array(dat) # Converts list of sorted random numbers to numpy array

##########################################################################

# Rejection ABC
@njit(fastmath=True)
def rejABC(model, prior_params, dat_t, dat_y, y0, eps, n_sample):
    # model: function to be fit; 
    # prior_params: list of ranges for uniform priors;
    # dat_t: data time;
    # dat_y: data points;
    # y0: initial conditions for model;
    # eps: tolerance;
    # n_sample: number of samples to be sorted
    
    n_mp = len(prior_params) # Number of model parameters to be fit
    
    p = np.zeros(n_mp+1) # Array of parameters
    
    post = np.zeros((1,n_mp+1)) # Array to build posterior distribution
    
    for i in range(n_sample):
        
        # Sort parameters according to given priors
        for i in range(n_mp):
        
            p[i] = np.random.uniform(prior_params[i,0], prior_params[i,1])
        
        d = np.sqrt(np.sum((dat_y-model(dat_t, p[:-1], y0))**2))/len(dat_t)
        p[-1] = d # Model-data distance
        
        # Check parameters and add sample to posterior distribution
        if (d < eps):
        
            post = np.concatenate((post, p.reshape((1,n_mp+1)))).reshape(len(post)+1, n_mp+1)
    
    return post[1:]