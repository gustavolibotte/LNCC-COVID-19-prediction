#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:43:23 2020

@author: joaop
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
    
class SIRD:
    """
    SIRD epidemic model
    """
    
    # SIRD parameters
    name = "SIRD"
    ncomp = len(name)
    params = [r"$\beta$", r"$N$", r"$\gamma$", r"$\mu$"]
    nparams = len(params)
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(y, t, params):
        
        S, R, I, D = y
        beta, N, gamma, mu = params
        
        return np.array([-beta*I*S/N,
                         gamma*I,
                         beta*I*S/N-gamma*I-mu*I,
                         mu*I])
    
    # SIRD equations solution
    @classmethod
    def solution(cls, t, params, y0):
    
        y0[0] = params[1] - (y0[1] + y0[2] + y0[3])
        
        sol = rk4(cls.model, y0, t, params)

        return sol
    
    # SIRD model total infected and dead output
    @classmethod
    def infected_dead(cls, t, params, y0):
        
        y0[0] = params[1] - (y0[1] + y0[2] + y0[3])
        
        sol = rk4(cls.model, y0, t, params)
        
        I_tot = np.sum(sol[:,1:], axis = 1)
        D = sol[:,3]
        return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))
    
class SEIRD:
    """
    SEIRD epidemic model
    """
    
    # SEIRD parameters
    name = "SEIRD"
    ncomp = len(name)
    params = [r"$\beta$", r"$N$", r"$\gamma$", r"$\mu$", r"$c$", r"$\kappa$"]
    nparams = len(params)
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(y, t, params):
        
        S, E, R, I, D = y
        beta, N, gamma, mu, c, Pex = params
        
        return np.array([-beta*(1-Pex)*I*S/N - beta*Pex*E*S/N,
                         beta*(1-Pex)*I*S/N + beta*Pex*E*S/N - c*E,
                         gamma*I,
                         c*E-gamma*I-mu*I,
                         mu*I])
    
    # SEIRD equations solution
    @classmethod
    def solution(cls, t, params, y0):
    
        y0[0] = params[1] - (y0[1] + y0[2] + y0[3] + y0[4])
        
        sol = rk4(cls.model, y0, t, params)
        
        return sol
    
    # SEIRD model total infected and dead output
    @classmethod
    def infected_dead(cls, t, params, y0):
    
        y0[0] = params[1] - (y0[1] + y0[2] + y0[3] + y0[4])
        
        sol = rk4(cls.model, y0, t, params)
        
        I_tot = np.sum(sol[:,2:], axis = 1)
        D = sol[:,4]
        return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))
    
class SEIHRD:
    """
    SEIHRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIHRD parameters
    name = "SEIHRD"
    ncomp = len(name)
    params = [r"$\beta$", r"$N$", r"$\gamma$", r"$\mu$", r"$c$", r"$\kappa$", 
              r"$P_{h}$", r"\tau_{h}", r"$\gamma_{h}$", r"$\mu_{h}$"]
    nparams = len(params)
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(y, t, params):
    
        S, E, H, R, I, D = y
        beta, N, gamma, mu, c, Pex, Ph, th, gammah, muh = params
        
        return np.array([-beta*(1-Pex)*I*S/N - beta*Pex*E*S/N,
                         beta*(1-Pex)*I*S/N + beta*Pex*E*S/N - c*E,
                         (Ph/th)*I - gammah*H - muh*H,
                         gamma*I + gammah*H,
                         c*E - (1-Ph)*gamma*I - (1-Ph)*mu*I - (Ph/th)*I,
                         mu*I + muh*H])
    
    # SEIHRD equations solution
    @classmethod
    def solution(cls, t, params, y0):
    
        y0[0] = params[1] - (y0[1] + y0[2] + y0[3] + y0[4] + y0[5])
        
        sol = rk4(cls.model, y0, t, params)

        return sol
    
    # SEIHRD model total infected and dead output
    @classmethod
    def infected_dead(cls, t, params, y0):
    
        y0[0] = params[1] - (y0[1] + y0[2] + y0[3] + y0[4] + y0[5])
        
        sol = rk4(cls.model, y0, t, params)
        
        I_tot = np.sum(sol[:,2:], axis = 1)
        D = sol[:,5]
        return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))
    
class SEIARD:
    """
    SEIARD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIHRD parameters
    name = "SEIARD"
    ncomp = len(name)
    params = [r"$\beta_{I}$", r"$\beta_{A}$", r"$N$", r"$\gamma_{I}$", r"$\mu$",
              r"$c$", r"$\alpha$", r"$\gamma_{A}$"]
    nparams = len(params)
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(y, t, params):
    
        S, E, A, R, I, D = y
        betaI, betaA, N, gamma, mu, c, Pa, gammaa = params
        
        return np.array([-betaI*I*S/N - betaA*E*S/N,
                         betaI*I*S/N + betaA*E*S/N - c*E,
                         Pa*c*E - gammaa*A,
                         gamma*I + gammaa*A,
                         (1-Pa)*c*E - gamma*I - mu*I,
                         mu*I])
    
    # SEIARD equations solution
    @classmethod
    def solution(cls, t, params, y0):
    
        y0[0] = params[1] - (y0[1] + y0[2] + y0[3] + y0[4] + y0[5])
    
        sol = rk4(cls.model, y0, t, params)

        return sol
    
    # SEIARD model total infected and dead output
    @classmethod
    def infected_dead(cls, t, params, y0):
    
        y0[0] = params[1] - (y0[1] + y0[2] + y0[3] + y0[4] + y0[5])
    
        sol = rk4(cls.model, y0, t, params)
        
        I_tot = np.sum(sol[:,3:], axis = 1)
        D = sol[:,5]
        return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))
    
class SEITRD:
    """
    SEITRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEITRD parameters
    name = "SEITRD"
    ncomp = len(name)
    params = [r"$\beta$", r"$N$", r"$\gamma_{I}$", r"$\mu$", r"$c$", r"$\kappa$", 
              r"$\tau_{t}$", r"$P_{t}$", r"$\gamma_{t}$", r"\mu_{t}"]
    nparams = len(params)
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(y, t, params):
    
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
    
    # SEITRD equations solution
    @classmethod
    def solution(cls, t, params, y0):
    
        y0[0] = params[1] - (y0[1] + y0[2] + y0[3] + y0[4] + y0[5] + y0[6] + y0[7])
    
        sol = rk4(cls.model, y0, t, params)

        return sol
    
    # SEITRD model total infected and dead output
    @classmethod
    def infected_dead(cls, t, params, y0):
    
        y0[0] = params[1] - (y0[1] + y0[2] + y0[3] + y0[4] + y0[5] + y0[6] + y0[7])
    
        sol = rk4(cls.model, y0, t, params)
        
        I_tot = np.sum(sol[:,5:], axis = 1)
        D = sol[:,7]
        return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))