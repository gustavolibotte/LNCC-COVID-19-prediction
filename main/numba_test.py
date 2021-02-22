#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 18:48:38 2021

@author: joao-valeriano
"""

import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt
import time

x = np.random.normal(size=(3,100))
y = np.random.normal(size=(3,100))

# t = time.time()
# for i in range(100000):
#     d = rmse(x, y, np.ones(3))
# print("rmse exec time:", (time.time()-t)/100000, "s")

# t = time.time()
# for i in range(100000):
#     d = rmse_numba(x, y, np.ones(3))
# print("rmse_numba exec time:", (time.time()-t)/100000, "s")

# t = time.time()
# for i in range(100000):
#     d = rmse_numba_no_np(x, y, np.ones(3))
# print("rmse_numba exec time:", (time.time()-t)/100000, "s")

n = 10**np.linspace(0, 6, 7)

exec_time = np.zeros((len(n), 3))

for i in range(len(n)):
    
    def rmse(x, y, weights):
    
        return np.sqrt(np.sum((x-y)**2*weights.reshape((-1,1))))/len(x)/np.sum(weights)
    
    @njit
    def rmse_numba(x, y, weights):
        
        return np.sqrt(np.sum((x-y)**2*weights.reshape((-1,1)))/len(x)/np.sum(weights))
    
    @njit
    def rmse_numba_no_np(x, y, weights):
        
        s = 0
        
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                
                s += (x[i,j] - y[i,j])**2 * weights[i]
                
        return np.sqrt(s/np.sum(weights)/x.shape[0]/x.shape[1])
    
    t = time.time()
    for j in range(int(n[i])):
        d = rmse(x, y, np.ones(3))
    exec_time[i,0] = time.time()-t
    
    t = time.time()
    for j in range(int(n[i])):
        d = rmse_numba(x, y, np.ones(3))
    exec_time[i,1] = time.time()-t
    
    t = time.time()
    for j in range(int(n[i])):
        d = rmse_numba_no_np(x, y, np.ones(3))
    exec_time[i,2] = time.time()-t
    
    print("\r", i, "calculated.", end="")

plt.figure(figsize=(10,7))    
plt.plot(n, exec_time)
plt.xlabel("Repetitions")
plt.ylabel("Execution time (s)")
plt.xscale("log")
plt.yscale("log")