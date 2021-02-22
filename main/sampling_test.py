#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 23:05:13 2021

@author: joao-valeriano
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit(fastmath=True) # Numba pre-compilation decorator
def sort(n, hist, bins): 
    # n: how many numbers to sort;
    # hist: l-sized array with height of columns of normalized histogram;
    # bins: (l+1)-sized array with values of bins divisions
    
    hist = np.concatenate((np.array([0]), hist))
    
    d = bins[1] - bins[0] # Bin size
    # bins = np.concatenate(([bins[0]-d], bins))
    
    cum = np.cumsum(hist) * d
    
    # b = (bins[1:]+bins[:-1])/2
    # b = np.concatenate(([bins[0]], b))
    
    dat = np.interp(np.random.uniform(0, 1, n), cum, bins) + np.random.uniform(-d/2, d/2, n)
    
    # print(np.diff(b), np.diff(cum))
    
    # plt.plot(b, cum)
    # plt.show()
    
    return dat

x = np.random.uniform(-5, 5, size=3000)
y = np.random.uniform(-5, 5, size=3000)

t1 = np.linspace(-5, 5, 100)
t2 = np.linspace(-5, 5, 100)

def f(x, y):
    
    return x**2+y**2

t1_, t2_ = np.meshgrid(t1, t2)

plt.figure(figsize=(10,8))
plt.contourf(t1, t2, f(t1_, t2_).T, alpha=0.5)
plt.colorbar()
plt.scatter(x, y, s=10)
plt.show()

d = f(x,y)
d_ = np.concatenate((d, x, y)).reshape((3, 3000)).T
d_ = d_[np.argsort(d)]

cut = np.percentile(d, 50)
print("cut:", cut)

cut_idx = np.where(d_[:,0] <= cut)[0][-1]

def ellip_fit(params):
    
    a, b, c1, c2 = params
    
    s = np.sign((d_[:,1]-c1)**2/a**2 + (d_[:,2]-c2)**2/b**2 - 1)
    
    return np.sum(s[:cut_idx]) - np.sum(s[cut_idx:])

n_bins = 20

hist = np.zeros((2, n_bins))
bins = np.zeros((2, n_bins+1))

# Define new priors
for k in range(len(hist)):
    
    hist[k], bins[k] = np.histogram(d_[:cut_idx,k+1], n_bins, density=True)
    
x2 = sort(3000, hist[0], bins[0])
y2 = sort(3000, hist[1], bins[1])

plt.figure(figsize=(10,8))
plt.contourf(t1, t2, f(t1_, t2_).T, alpha=0.5)
plt.colorbar()
plt.scatter(x, y, s=10)
plt.scatter(x2, y2, s=5, c="red")
