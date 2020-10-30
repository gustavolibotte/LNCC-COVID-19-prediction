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

##########################################################################

# # Function to sort random number according to a given histogram
# @njit(fastmath=True) # Numba pre-compilation decorator
# def sort(n, hist, bins): 
#     # n: how many numbers to sort;
#     # hist: l-sized array with height of columns of normalized histogram;
#     # bins: (l+1)-sized array with values of bins divisions
    
#     d = bins[1] - bins[0] # Bin size
    
#     dat = [] # List of sorted random numbers
    
#     for i in range(n):
        
#         x = np.random.uniform(0., 1.)
        
#         # Conversion of 0-1 random number to number sorted according to the given histogram
#         for j in range(len(hist)):
            
#             if (x < np.sum(hist[:j+1])*d):
                
#                 dat.append(np.random.uniform(bins[j], bins[j+1]))
#                 break
    
#     return np.array(dat) # Converts list of sorted random numbers to numpy array

# Function to sort random number according to a given histogram
# @njit(fastmath=True) # Numba pre-compilation decorator
def sort(n, hist, bins): 
    # n: how many numbers to sort;
    # hist: l-sized array with height of columns of normalized histogram;
    # bins: (l+1)-sized array with values of bins divisions
    
    d = bins[1] - bins[0] # Bin size
    
    cum = np.cumsum(hist) * d
    
    b = (bins[1:]+bins[:-1])/2
    
    dat = np.interp(np.random.uniform(0, 1, n), cum, b) + np.random.uniform(0, d, n)
    
    return dat

# # @njit(fastmath=True)
# def sort(n, hist, bins): 
#     a = np.cumsum(hist)
#     a /= a[-1]

#     x = np.random.uniform(0., 1., n)

#     j = np.searchsorted(a,x)

#     return bins[j-1] + (x-a[j-1])*(bins[j] - bins[j-1])/(a[j]-a[j-1])

##########################################################################

# Rejection ABC
# @njit(fastmath=True)
def rejABC(model, prior_params, dat_t, dat_y, y0, eps, n_sample, n_max):
    # model: function to be fit; 
    # prior_params: list of ranges for uniform priors;
    # dat_t: data time;
    # dat_y: data points;
    # y0: initial conditions for model;
    # eps: tolerance;
    # n_sample: number of samples to be sorted
    
    n_mp = len(prior_params) # Number of model parameters to be fit
    
    p = np.zeros(n_mp+1, dtype=np.float64) # Array of parameters
    
    post = np.zeros((1,n_mp+1)) # Array to build posterior distribution
    
    for i in range(n_sample):
        
        # Sort parameters according to given priors
        for j in range(n_mp):
        
            p[j] = prior_params[j][0](prior_params[j][1], prior_params[j][2])
        
        d = np.sqrt(np.sum((dat_y-model(dat_t, p[:-1], y0))**2))/len(dat_t)
        p[-1] = d # Model-data distance
        
        # Check parameters and add sample to posterior distribution
        if (d < eps):
        
            post = np.concatenate((post, p.reshape((1,n_mp+1)))).reshape(len(post)+1, n_mp+1)
        
        if (len(post) > n_max):
            break
        
    return post[1:]

def smcABC(model, hist, bins, n_bins, p_std, dat_t, dat_y, y0, eps, n_sample, n_max):
    # model: function to be fit; 
    # hist+bins: past posterior for new prior
    # n_bins: number of bins to be used to make new prior from last posterior
    # p_std: standard deviations of last posterior, to add noise to new posterior
    # dat_t: data time;
    # dat_y: data points;
    # y0: initial conditions for model;
    # eps: tolerance;
    # n_sample: number of samples to be sorted
    
    n_mp = len(hist) # Number of model parameters to be fit
    
    p = np.zeros(n_mp+1, dtype=np.float64) # Array of parameters
    
    post = np.zeros((1,n_mp+1)) # Array to build posterior distribution
    
    for i in range(n_sample):
        
        # Sort parameters according to given priors
        for j in range(n_mp):
        
            # p[j] = np.random.uniform(prior_params[j,0], prior_params[j,1])
            p[j] = sort(1, hist[j], bins[j]) + np.random.normal(scale=p_std[j]/n_bins)
        
        d = np.sqrt(np.sum((dat_y-model(dat_t, p[:-1], y0))**2))/len(dat_t)
        p[-1] = d # Model-data distance
        
        # Check parameters and add sample to posterior distribution
        if (d < eps):
        
            post = np.concatenate((post, p.reshape((1,n_mp+1)))).reshape(len(post)+1, n_mp+1)
            
        if (len(post) > n_max):
            break
            
    return post[1:]

# Akaike Information Criterion
def AIC(k, L):
    
    return 2*(k-np.log(L))