#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:02:00 2020

@author: pedroc
"""

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

# SIRD model differential equation
@njit(fastmath=True)
def SIRD(y, t, params):
    
    S, R, I, D = y
    beta, N, gamma, mu = params
    
    return np.array([-beta*I*S/N,
                     gamma*I,
                     beta*I*S/N-gamma*I-mu*I,
                     mu*I])
    
#S SIRD model solutions
@njit(fastmath=True)
def SIRD_sol(t, params, y0):
    
    y0[0] = params[1] - (y0[1] + y0[2] + y0[3])

    sol = rk4(SIRD, y0, t, params)
    
    I_tot = np.sum(sol[:,1:], axis = 1)
    D = sol[:,3]
    return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))


# SEIRD model differential equation
@njit(fastmath=True)
def SEIRD(y, t, params):
    
    S, E, R, I, D = y
    beta, N, gamma, mu, c, Pex = params
    
    return np.array([-beta*(1-Pex)*I*S/N - beta*Pex*E*S/N,
                     beta*(1-Pex)*I*S/N + beta*Pex*E*S/N - c*E,
                     gamma*I,
                     c*E-gamma*I-mu*I,
                     mu*I])

#S SEIRD model solutions
@njit(fastmath=True)
def SEIRD_sol(t, params, y0):
    
    y0[0] = params[1] - (y0[1] + y0[2] + y0[3] + y0[4])
    
    sol = rk4(SEIRD, y0, t, params)
    
    I_tot = np.sum(sol[:,2:], axis = 1)
    D = sol[:,4]
    return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))


# SEIHRD model differential equation
@njit(fastmath=True)
def SEIHRD(y, t, params):
    
    S, E, H, R, I, D = y
    beta, N, gamma, mu, c, Pex, Ph, th, gammah, muh = params
    
    return np.array([-beta*(1-Pex)*I*S/N - beta*Pex*E*S/N,
                     beta*(1-Pex)*I*S/N + beta*Pex*E*S/N - c*E,
                     (Ph/th)*I - gammah*H - muh*H,
                     gamma*I + gammah*H,
                     c*E - (1-Ph)*gamma*I - (1-Ph)*mu*I - (Ph/th)*I,
                     mu*I + muh*H])

#S SEIHRD model solutions
@njit(fastmath=True)
def SEIHRD_sol(t, params, y0):
    
    y0[0] = params[1] - (y0[1] + y0[2] + y0[3] + y0[4] + y0[5])
    
    sol = rk4(SEIHRD, y0, t, params)
    
    I_tot = np.sum(sol[:,2:], axis = 1)
    D = sol[:,5]
    return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))


# SEIARD model differential equation
@njit(fastmath=True)
def SEIARD(y, t, params):
    
    S, E, A, R, I, D = y
    beta, N, gamma, mu, c, Pex, Pa, gammaa = params
    
    return np.array([-beta*(1-Pex)*I*S/N - beta*Pex*E*S/N,
                     beta*(1-Pex)*I*S/N + beta*Pex*E*S/N - c*E,
                     Pa*c*E - gammaa*A,
                     gamma*I + gammaa*A,
                     (1-Pa)*c*E - gamma*I - mu*I,
                     mu*I])

#S SEIARD model solutions
@njit(fastmath=True)
def SEIARD_sol(t, params, y0):
    
    y0[0] = params[1] - (y0[1] + y0[2] + y0[3] + y0[4] + y0[5])
    
    sol = rk4(SEIARD, y0, t, params)
    
    I_tot = np.sum(sol[:,3:], axis = 1)
    D = sol[:,5]
    return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))


# SEITRD model differential equation
@njit(fastmath=True)
def SEITRD(y, t, params):
    
    S, E, I, R, D, Rt, T, Dt = y
    beta, N, gamma, mu, c, Pex, tt, Pt, gammat, mut = params
    
    return np.array([-beta*(1-Pex)*I*S/N - beta*Pex*E*S/N,
                     beta*(1-Pex)*I*S/N + beta*Pex*E*S/N - c*E,
                     c*E - gamma*I - mu*I - (Pt/tt)*I,
                     gamma*I,
                     mu*I,
                     gammat*T,
                     (Pt/tt)*I - gammat*T - mut*T,
                     mut*T,])

#S SEITRD model solutions
@njit(fastmath=True)
def SEITRD_sol(t, params, y0):
    
    y0[0] = params[1] - (y0[1] + y0[2] + y0[3] + y0[4] + y0[5] + y0[6] + y0[7])
    
    sol = rk4(SEITRD, y0, t, params)
    
    I_tot = np.sum(sol[:,5:], axis = 1)
    D = sol[:,7]
    return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))