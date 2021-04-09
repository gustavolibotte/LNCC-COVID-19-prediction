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

@njit(fastmath=True)
def dist(d, args):
    
    if (d == "uniform"):
        
        return np.random.uniform(*args)
    
    if (d == "lognormal"):
        
        return np.random.lognormal(*args)

from scipy import stats

@njit(fastmath=True)
def sign(x):
    
    return x/np.abs(x)

@njit(fastmath=True)
def pdf(d, args, x):
    
    if (d == "uniform"):
        
        return (sign(x-args[0])+sign(args[1]-x))/(2*(args[1]-args[0]))
    
    # if (d == "lognormal"):
        
    #     return stats.lognorm.pdf(x, scale=np.exp(args[0]), s=args[1])    

@njit(fastmath=True)
def distance(x, y, weights=None):
    
    if (weights == None):
        
        weights = np.ones(x.shape[0])
    
    s = np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            
            s[i] += (x[i,j]-y[i,j])**2
            
        s[i] = s[i]/x.shape[1]*weights[i]

    return np.sqrt(np.sum(s)/np.sum(weights))

# Rejection ABC

def rejABC(model, weights, prior_func, prior_args, dat_t, dat_y, y0, eps, n_sample, fixed_params=None):
    # model: function to be fit; 
    # prior_params: list of ranges for uniform priors;
    # dat_t: data time;
    # dat_y: data points;
    # y0: initial conditions for model;
    # eps: tolerance;
    # n_sample: number of samples to be sorted
    
    n_mp = len(prior_func) # Number of model parameters to be fit
    
    p = np.zeros(n_mp+1, dtype=np.float64) # Array of parameters
    
    if (type(fixed_params) != type(None)):
    
        for i in range(len(fixed_params)):
            
            if (np.isnan(fixed_params[i]) == False):
                
                p[i] = fixed_params[i]
                
    else:
        
        fixed_params = np.array([np.nan]*n_mp)
    
    post = np.zeros((n_sample,n_mp+1)) # Array to build posterior distribution
    
    post_trials = []
    
    trials = 0
    
    for i in range(n_sample):
        
        d = eps+1
        
        while (d > eps or np.isnan(d)):
            
            trials += 1
            
            # Sort parameters according to given priors
            for j in np.where(np.isnan(fixed_params))[0]:
                p[j] = dist(prior_func[j], tuple(prior_args[j]))
            d = distance(dat_y, model(dat_t, p[:-1], y0), weights)
            p[-1] = d
            post_trials.append(np.copy(p))
        
        p[-1] = d # Model-data distance
        post[i] = p
        
        print("\rSorting Rejection ABC samples... %i/%i" % (i+1, n_sample), end="")
    print("")
    
    return post, trials

# @njit(fastmath=True)
def gen_samples(model, weights, prior_func, prior_args, dat_t, dat_y, y0, n_sample, fixed_params=None):
    # model: function to be fit; 
    # prior_params: list of ranges for uniform priors;
    # dat_t: data time;
    # dat_y: data points;
    # y0: initial conditions for model;
    # eps: tolerance;
    # n_sample: number of samples to be sorted
    
    n_mp = len(prior_func) # Number of model parameters to be fit
    
    p = np.zeros(n_mp+1, dtype=np.float64) # Array of parameters
    
    if (type(fixed_params) != type(None)):
    
        for i in range(len(fixed_params)):
            
            if (np.isnan(fixed_params[i]) == False):
                
                p[i] = fixed_params[i]
    
    else:
        
        fixed_params = np.array([np.nan]*n_mp)
    
    post = np.zeros((n_sample, n_mp+1), dtype=np.float64)
    
    for i in range(n_sample):
        
        d = 1e301
        
        while (d > 1e300 or np.isnan(d)):
            
            for j in np.where(np.isnan(fixed_params))[0]:
                p[j] = dist(prior_func[j], tuple(prior_args[j])) 
                
            d = distance(dat_y, model(dat_t, p[:-1], y0), weights)
            
        p[-1] = d

        post[i] = p #.reshape(len(post)+1, n_mp+1)
        
        print("\rGenerating samples... %i/%i" % (i+1, n_sample), end="")
    print("")
    
    return post

def gen_samples_from_dist(model, weights, prior, prior_weights, prior_bounds, dat_t, dat_y, y0, n_sample, fixed_params=None):
    # model: function to be fit; 
    # prior_params: list of ranges for uniform priors;
    # dat_t: data time;
    # dat_y: data points;
    # y0: initial conditions for model;
    # eps: tolerance;
    # n_sample: number of samples to be sorted
    
    n_mp = len(prior[0])-1 # Number of model parameters to be fit
    
    p = np.zeros(n_mp+1, dtype=np.float64) # Array of parameters
    
    if (type(fixed_params) != type(None)):
    
        for i in range(len(fixed_params)):
            
            if (np.isnan(fixed_params[i]) == False):
                
                p[i] = fixed_params[i]
    
    else:
        
        fixed_params = np.array([np.nan]*n_mp, dtype=np.float64)
    
    nf_par_idx = np.where(np.isnan(fixed_params))[0]
    
    post = np.zeros((n_sample,n_mp+1)) # Array to build posterior distribution

    p_std = prior[:,:-1].std(axis=0)[nf_par_idx]
    
    lower_bounds, upper_bounds = np.array(prior_bounds)[nf_par_idx].T
    
    for i in range(n_sample):
        
        d = 1e301
        
        while (d > 1e300 or np.isnan(d)):
            
            # Sort parameters according to given priors
            p[nf_par_idx] = prior[np.random.choice(len(prior), p=prior_weights),nf_par_idx] + np.random.normal(scale=2*p_std)
            
            if not check_box(lower_bounds, upper_bounds, p[nf_par_idx]):
                # print("out")
                continue
            
            d = distance(dat_y, model(dat_t, p[:-1], y0), weights)

        p[-1] = d # Model-data distance
        post[i] = p
        
        print("\rGenerating samples... %i/%i" % (i+1, n_sample), end="")
    print("")
    
    return post

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
def smc_weights(p, p_std, prior, prior_func, prior_args, prior_weights):
    
    s = 0
    
    n = 1
    
    for i in range(len(p)):
        
        n *= pdf(prior_func[i], prior_args[i], p[i])
    
    for i in range(len(prior)):
        
        s += prior_weights[i] * np.exp(-summ(((prior[i]-p)/p_std)**2))
    
    return  n / s

def smcABC(model, weights, prior, prior_weights, prior_func, prior_args, prior_bounds, dat_t, dat_y, y0, eps, n_max=None, fixed_params=None, noise_scale=1.):
    # model: function to be fit; 
    # p_std: standard deviations of last posterior, to add noise to new posterior
    # dat_t: data time;
    # dat_y: data points;
    # y0: initial conditions for model;
    # eps: tolerance;
    # n_sample: number of samples to be sorted
        
    n_mp = len(prior[0])-1 # Number of model parameters to be fit
    
    p = np.zeros(n_mp+1, dtype=np.float64) # Array of parameters
    
    if (type(fixed_params) != type(None)):
    
        for i in range(len(prior_func)):
            
            if (np.isnan(fixed_params[i]) == False):
                
                p[i] = fixed_params[i]
    
    else:
        
        fixed_params = np.array([np.nan]*n_mp, dtype=np.float64)
    
    nf_par_idx = np.int64(np.where(np.isnan(fixed_params))[0])
    
    post = np.zeros((1,n_mp+1)) # Array to build posterior distribution

    p_std = prior[:,:-1].std(axis=0)[nf_par_idx]
    
    lower_bounds, upper_bounds = np.array(prior_bounds)[nf_par_idx].T
    
    if (n_max != None):
        
        n_sample = n_max
        
    else:
        
        n_sample = len(prior)
        
    post_weights = np.zeros(n_sample)
    
    trials = 0
    
    for i in range(n_sample):
        
        d = eps+1
        
        while (d > eps or np.isnan(d)):
            
            trials += 1
            
            # Sort parameters according to given priors
            p[nf_par_idx] = prior[np.random.choice(len(prior), p=prior_weights), nf_par_idx] + np.random.normal(scale=p_std/noise_scale)
            # print(p)
            if not check_box(lower_bounds, upper_bounds, p[nf_par_idx]):
                # print("out")
                continue
            
            d = distance(dat_y, model(dat_t, p[:-1], y0), weights)
        
        # print("ok")
        
        p[-1] = d # Model-data distance
        post = np.concatenate((post, p.reshape((1,n_mp+1)))).reshape(len(post)+1, n_mp+1)
        post_weights[i] = smc_weights(p[nf_par_idx], p_std, prior[:,nf_par_idx], prior_func[nf_par_idx], prior_args[nf_par_idx], prior_weights)
        
        print("\rSorting ABC-SMC samples... %i/%i"%(i+1, n_sample), end="")
    print("")
    
    return post[1:], np.array(post_weights), trials

# Akaike Information Criterion
def AIC(k, L):
    
    return 2*(k+L)