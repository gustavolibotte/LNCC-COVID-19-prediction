#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries
import numpy as np
import pandas as pd
import numba
from numba import njit, jit
import warnings

# Deactivate numba deprecation warnings
warnings.simplefilter('ignore', category=numba.NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=numba.NumbaPendingDeprecationWarning)

##########################################################################

# Function to sort random number according to a given histogram
@njit(fastmath=True) # Numba pre-compilation decorator
def sort(n, hist, bins): 
    # n: how many numbers to sort;
    # hist: l-sized array with height of columns of normalized histogram;
    # bins: (l+1)-sized array with values of bins divisions
    
    d = bins[1] - bins[0] # Bin size
    
    hist = np.concatenate((np.array([0]), hist))
    
    cum = np.cumsum(hist) * d
    
    # b = (bins[1:]+bins[:-1])/2
    
    dat = np.interp(np.random.uniform(0, 1, n), cum, bins) #+ np.random.uniform(0, d, n)
    
    return dat

##########################################################################

@njit 
def dist(d, args):
    
    if (d == "uniform"):
        
        return np.random.uniform(*args)
    
    if (dist == "lognormal"):
        
        return np.random.lognormal(*args)

@njit
def distance(x, y, weights):
    
    s = 0
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            
            s += (x[i,j]-y[i,j])**2*weights[i]
            
    return np.sqrt(s/np.sum(weights)/x.shape[0]/x.shape[1])

# Rejection ABC
def rejABC(model, weights, prior_func, prior_args, dat_t, dat_y, y0, eps, n_sample, n_max):
    # model: function to be fit; 
    # prior_params: list of ranges for uniform priors;
    # dat_t: data time;
    # dat_y: data points;
    # y0: initial conditions for model;
    # eps: tolerance;
    # n_sample: number of samples to be sorted
    
    n_mp = len(prior_func) # Number of model parameters to be fit
    
    p = np.zeros(n_mp+1, dtype=np.float64) # Array of parameters
    
    post = np.zeros((1,n_mp+1)) # Array to build posterior distribution
    
    samples = np.zeros((n_sample, n_mp), dtype=np.float64)
    for i in range(n_mp):
        samples[:,i] = dist(prior_func[i], prior_args[i]+(n_sample,))
    
    for i in range(n_sample):
        
        # Sort parameters according to given priors
        p[:-1] = samples[i]
        
        d = distance(dat_y, model(dat_t, p[:-1], y0), weights)
        p[-1] = d # Model-data distance
        
        # Check parameters and add sample to posterior distribution
        if (d < eps and not np.isnan(d)):
        
            post = np.concatenate((post, p.reshape((1,n_mp+1)))).reshape(len(post)+1, n_mp+1)
        
        if (len(post) > n_max):
            break
        
    return post[1:]

def smcABC(model, weights, prior_args, hist, bins, n_bins, p_std, dat_t, dat_y, y0, eps, n_sample, n_max):
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
    
    samples = np.zeros((n_sample, n_mp), dtype=np.float64)
    for i in range(n_mp):
        samples[:,i] = sort(n_sample, hist[i], bins[i]) + np.random.normal(scale=p_std[i], size=n_sample)
    
    lower_bounds, upper_bounds = np.array(prior_args).T
    
    for i in range(n_sample):
        
        # Sort parameters according to given priors
        p[:-1] = samples[i]
        
        if not check_box(lower_bounds, upper_bounds, p[:-1]):

                continue
        
        d = distance(dat_y, model(dat_t, p[:-1], y0), weights)
        p[-1] = d # Model-data distance
        
        # Check parameters and add sample to posterior distribution
        if (d < eps):
        
            post = np.concatenate((post, p.reshape((1,n_mp+1)))).reshape(len(post)+1, n_mp+1)
            
        if (len(post) > n_max):
            break

    return post[1:]

@njit
def check_box(a, b, x):
        
    return (a<x).all() and (x<b).all()    

@njit
def interpn(x_new, x, y):
    
    r = []
    
    for i in range(len(x)):
        
        r.append(np.interp(x_new[i], x[i], y[i]))
        
    return np.array(r)

@njit
def prod(x):
    
    s = 1
    
    for i in range(len(x)):
        
        s *= x[i]
        
    return s

@njit
def summ(x):
    
    s = 0
    
    for i in range(len(x)):
        
        s += x[i]
        
    return s

@njit(fastmath=True)
def smc_weights(p, p_std, prior, priors, priors_bins, prior_weights):
    
    s = 0
    
    for i in range(len(prior)):
        
        s += prior_weights[i] * np.exp(-summ(((prior[i,:-1]-p[:-1])/p_std)**2))
    
    return prod(interpn(p[:-1], priors_bins, priors)) / s #np.sum(prior_weights.reshape((-1,1)) * np.exp(-np.sum(((p[:-1]-prior[:,:-1])/p_std)**2)))

def smcABC2(model, weights, prior, prior_weights, prior_args, n_bins, dat_t, dat_y, y0, eps, n_sample):
    # model: function to be fit; 
    # hist+bins: past posterior for new prior
    # n_bins: number of bins to be used to make new prior from last posterior
    # p_std: standard deviations of last posterior, to add noise to new posterior
    # dat_t: data time;
    # dat_y: data points;
    # y0: initial conditions for model;
    # eps: tolerance;
    # n_sample: number of samples to be sorted
    
    priors = np.zeros((prior.shape[1]-1, n_bins))
    priors_bins = np.zeros((prior.shape[1]-1, n_bins))
    
    for i in range(len(priors)):
        
        priors[i], bins = np.histogram(prior[:,i], n_bins, density=True)
        priors_bins[i] = (bins[1:]+bins[:-1])/2

    n_mp = len(prior[0])-1 # Number of model parameters to be fit
    
    p = np.zeros(n_mp+1, dtype=np.float64) # Array of parameters
    
    post = np.zeros((1,n_mp+1)) # Array to build posterior distribution

    p_std = prior[:,:-1].std(axis=0)
    
    post_weights = []
    
    prior_items = np.random.choice(len(prior), n_sample, p=prior_weights)
    
    lower_bounds, upper_bounds = np.array(prior_args).T

    for i in prior_items:
        
        d = eps+1
        
        while (d > eps):
        
            # Sort parameters according to given priors
            p[:-1] = prior[i,:-1] + np.random.normal(scale=p_std)
            
            if not check_box(lower_bounds, upper_bounds, p[:-1]):
                # print("out")
                continue
            
            d = distance(dat_y, model(dat_t, p[:-1], y0), weights)
        
        # print("ok")
        
        p[-1] = d # Model-data distance
        post = np.concatenate((post, p.reshape((1,n_mp+1)))).reshape(len(post)+1, n_mp+1)
        post_weights.append(smc_weights(p, p_std, prior, priors, priors_bins, prior_weights))
            
    return post[1:], np.array(post_weights)

# Akaike Information Criterion
def AIC(k, L):
    
    return 2*(k-np.log(L))