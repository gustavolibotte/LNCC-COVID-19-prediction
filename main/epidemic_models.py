#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries
import numpy as np
import numba
from numba import njit
import warnings

# Deactivate numba deprecation warnings
warnings.simplefilter('ignore', category=numba.NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=numba.NumbaPendingDeprecationWarning)

##########################################################################

# 4th order Runge-Kutta integrator
@njit(fastmath=True)
def rk4(f, y0, t, args, h=1.): 
    # f: function to be integrated; y0: initial conditions;
    # t: time points for the function to be evaluated;
    # args: extra function parameters;
    # h: time step
    
    t_ = np.arange(t[0], t[-1], h)
    n = len(t_)
    y_ = np.zeros((n, len(y0)))
    y_[0] = y0
    
    for i in range(n-1):
        
        k1 = f(y_[i], t_[i], args)
        k2 = f(y_[i] + k1 * h / 2., t_[i] + h / 2., args)
        k3 = f(y_[i] + k2 * h / 2., t_[i] + h / 2., args)
        k4 = f(y_[i] + k3 * h, t_[i] + h, args)
        y_[i+1] = y_[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    
    y = np.zeros((len(t), len(y0)))
    
    for i in range(len(y0)):
        
        y[:,i] = np.interp(t, t_, y_[:,i]) # Interpolate solution to wished time points
    
    return y

##########################################################################

# SIR model differential equation
@njit(fastmath=True)
def SIR(y, t, params):
    
    S, I, R = y
    beta, N, gamma = params
    
    return np.array([-beta*I*S/N,
                     beta*I*S/N-gamma*I,
                     gamma*I])

#S SIR model solutions
@njit(fastmath=True)
def SIR_sol(t, params, y0):
    
    y0[0] = params[1] - (y0[1] + y0[2])
    
    return rk4(SIR, y0, t, params)[:,1:].T

@njit(fastmath=True)
def SIRD(y, t, params):

    S, R, I, D = y
    beta, N, gamma, mu = params

    return np.array([-betaIS/N,
                     gammaI,
                     betaIS/N-gammaI-muI,
                     muI])

#S SIRD model solutions
@njit(fastmath=True)
def SIRD_sol(t, params, y0):

    y0[0] = params[1] - (y0[1] + y0[2] + y0[3])

    sol = rk4(SIRD, y0, t, params)

    I_tot = np.sum(sol[:,1:], axis = 1)
    D = sol[:,3]
    return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))