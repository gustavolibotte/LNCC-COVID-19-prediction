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
    
    t_ = np.arange(t[0], t[-1]+h, h)
    n = len(t_)
    y_ = np.zeros((n, len(y0)))
    y_[0] = y0
    
    for i in range(n-1):
        
        k1 = f(t_[i], y_[i], args)
        k2 = f(t_[i] + h / 2., y_[i] + k1 * h / 2., args)
        k3 = f(t_[i] + h / 2., y_[i] + k2 * h / 2., args)
        k4 = f(t_[i] + h, y_[i] + k3 * h, args)
        y_[i+1] = y_[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    
    y = np.zeros((len(t), len(y0)))
    
    for i in range(len(y0)):
        
        y[:,i] = np.interp(t, t_, y_[:,i]) # Interpolate solution to wished time points
    
    return y

##########################################################################

class SIR:
    """
    SIR epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    name = "SIR"
    plot_name = "SIR"
    ncomp = len(name)
    params = [r"$\beta$", r"$N$", r"$\gamma$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    prior_func = np.array(["uniform",
                           "uniform",
                           "uniform"])
    prior_args = np.array([(0., 1.),
                           (0., 1.),
                           (0., 1.)])
    prior_bounds = np.array([(0., 1.),
                             (0., 1.),
                             (0., 1.)])
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
        S, I, R = y
        beta, N, gamma = params
        
        return np.array([-beta*I*S,
                beta*I*S-gamma*I,
                gamma*I])
    
    # SIRD equations solution
    @staticmethod
    @njit(fastmath=True)
    def solution(t, params, y0):
    
        def model(t, y, params):
            
            S, I, R = y
            beta, N, gamma = params
            
            return np.array([-beta*I*S/N,
                             beta*I*S/N-gamma*I,
                             gamma*I])    
        
        y0_ = np.copy(y0)
        
        y0_[0] = params[1] - (y0[1] + y0[2])
        
        sol = rk4(model, y0_, t, params)

        return sol.T
    
    # SIRD model total infected and dead output
    @staticmethod
    @njit(fastmath=True)
    def infected_dead(t, params, y0):
        
        def model(t, y, params):
            
            S, I, R = y
            beta, N, gamma = params
            
            return np.array([-beta*I*S/N,
                             beta*I*S/N-gamma*I,
                             gamma*I])
        y0_ = np.copy(y0)
        
        y0_[0] = params[1] - (y0[1] + y0[2])
        
        sol = rk4(model, y0_, t, params)
        # print(y0_, params)
        I_tot = np.sum(sol[:,1:], axis=1)
        D = sol[:,2]
        
        return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params

class SIRD:
    """
    SIRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SIRD parameters
    name = "SIRD"
    ncomp = len(name)
    params = [r"$\beta$", r"$N$", r"$\gamma$", r"$\mu$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
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
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params
        
class SEIRD:
    """
    SEIRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIRD parameters
    name = "SEIRD"
    plot_name = "SEIRD"
    ncomp = len(name)
    params = [r"$\beta_{I}$", r"$\beta_{E}$", r"$N$", r"$P_{IFR}$", r"$t_{d}$", r"$t_{r}$", r"$t_{c}$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    prior_func = ["uniform",
                  "uniform",
                  "uniform",
                  "uniform",
                  "uniform",
                  "uniform",
                  "uniform"]
    prior_args = [(0., 1.),
                  (0., 1.),
                  (0., 1.),
                  (0., 1.),
                  (0., 30.),
                  (0., 30.),
                  (0., 30.),]
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
        S, E, R, I, D = y
        beta_I, beta_E, N, p_ifr, t_d, t_r, t_c = params
        gamma = (1-p_ifr)/t_r
        mu = p_ifr/t_d
        
        c = 1/t_c
        
        return np.array([-beta_I*I*S/N - beta_E*E*S/N,
                         beta_I*I*S/N + beta_E*E*S/N - c*E,
                         gamma*I,
                         c*E-gamma*I-mu*I,
                         mu*I])
    
    # SEIRD equations solution
    @staticmethod
    @njit(fastmath=True)
    def solution(t, params, y0):
        
        def model(t, y, params):
        
            S, E, R, I, D = y
            beta_I, beta_E, N, p_ifr, t_d, t_r, t_c = params
            gamma = (1-p_ifr)/t_r
            mu = p_ifr/t_d
            
            c = 1/t_c
            
            return np.array([-beta_I*I*S/N - beta_E*E*S/N,
                             beta_I*I*S/N + beta_E*E*S/N - c*E,
                             gamma*I,
                             c*E-gamma*I-mu*I,
                             mu*I])
        
        y0_ = np.copy(y0)
        
        y0_[0] = params[2] - (y0_[1] + y0_[2] + y0_[3] + y0_[4])
        
        sol = rk4(model, y0_, t, params)
        
        return sol
    
    # SEIRD model total infected and dead output
    @staticmethod
    @njit(fastmath=True)
    def infected_dead(t, params, y0):
        
        def model(t, y, params):
        
            S, E, R, I, D = y
            beta_I, beta_E, N, p_ifr, t_d, t_r, t_c = params
            gamma = (1-p_ifr)/t_r
            mu = p_ifr/t_d
            
            c = 1/t_c
            
            return np.array([-beta_I*I*S/N - beta_E*E*S/N,
                             beta_I*I*S/N + beta_E*E*S/N - c*E,
                             gamma*I,
                             c*E-gamma*I-mu*I,
                             mu*I])
        
        y0_ = np.copy(y0)
        
        y0_[0] = params[2] - (y0_[1] + y0_[2] + y0_[3] + y0_[4])
        
        sol = rk4(model, y0_, t, params)
        
        I_tot = np.sum(sol[:,2:], axis = 1)
        D = sol[:,4]
        return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params
        
class SEIRD2:
    """
    SEIRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIRD parameters
    name = "SEIRD"
    plot_name = "SEIRD"
    ncomp = len(name)
    params = [r"$\beta_{I}$", r"$\beta_{E}$", r"$N$", r"$P_{IFR}$", r"$t_{d}$", 
              r"$t_{r}$", r"$t_{c}$", r"RF_{0}", r"$E_{0}$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    prior_func = np.array(["uniform",
                           "uniform",
                           "uniform",
                           "uniform",
                           "uniform",
                           "uniform",
                           "uniform",
                           "uniform",
                           "uniform"])
    prior_args = np.array([(0., 1.),
                           (0., 1.),
                           (0., 1.),
                           (0., 1.),
                           (0., 30.),
                           (0., 30.),
                           (0., 30.),
                           (0., 1.),
                           (0., 2.)])
    prior_bounds = np.array([(0., 1.),
                             (0., 1.),
                             (0., 1.),
                             (0., 1.),
                             (0., 30.),
                             (0., 30.),
                             (0., 30.),
                             (0., 1.),
                             (0., 2.)])
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
        S, E, R, I, D = y
        beta_I, beta_E, N, p_ifr, t_d, t_r, t_c = params
        gamma = (1-p_ifr)/t_r
        mu = p_ifr/t_d
        
        c = 1/t_c
        
        return np.array([-beta_I*I*S/N - beta_E*E*S/N,
                         beta_I*I*S/N + beta_E*E*S/N - c*E,
                         gamma*I,
                         c*E-gamma*I-mu*I,
                         mu*I])
    
    # SEIRD equations solution
    @staticmethod
    @njit(fastmath=True)
    def solution(t, params, y0):
        
        def model(t, y, params):
        
            S, E, R, I, D = y
            beta_I, beta_E, N, p_ifr, t_d, t_r, t_c, RF0, E0 = params
            gamma = (1-p_ifr)/t_r
            mu = p_ifr/t_d
            
            c = 1/t_c
            
            return np.array([-beta_I*I*S/N - beta_E*E*S/N,
                             beta_I*I*S/N + beta_E*E*S/N - c*E,
                             gamma*I,
                             c*E-gamma*I-mu*I,
                             mu*I])
        
        y0_ = np.copy(y0)
        
        RF0, E0 = params[-2:]
        
        y0_[1] = E0*y0_[-2]
        I0 = y0_[-2]
        y0_[-3] = I0*RF0
        y0_[-2] = I0*(1-RF0)        
        y0_[0] = params[2] - (y0_[1] + y0_[2] + y0_[3] + y0_[4])
        
        sol = rk4(model, y0_, t, params)
        
        return sol
    
    # SEIRD model total infected and dead output
    @staticmethod
    @njit(fastmath=True)
    def infected_dead(t, params, y0):
        
        def model(t, y, params):
        
            S, E, R, I, D = y
            beta_I, beta_E, N, p_ifr, t_d, t_r, t_c, RF0, E0 = params
            gamma = (1-p_ifr)/t_r
            mu = p_ifr/t_d
            
            c = 1/t_c
            
            return np.array([-beta_I*I*S/N - beta_E*E*S/N,
                             beta_I*I*S/N + beta_E*E*S/N - c*E,
                             gamma*I,
                             c*E-gamma*I-mu*I,
                             mu*I])
        
        y0_ = np.copy(y0)
        
        RF0, E0 = params[-2:]
        
        y0_[1] = E0*y0_[-2]
        I0 = y0_[-2]
        y0_[-3] = I0*RF0
        y0_[-2] = I0*(1-RF0)        
        y0_[0] = params[2] - (y0_[1] + y0_[2] + y0_[3] + y0_[4])
        
        sol = rk4(model, y0_, t, params)
        
        I_tot = np.sum(sol[:,2:], axis = 1)
        D = sol[:,4]
        return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params

class SEIRD_bias:
    """
    SEIRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIRD parameters
    name = "SEIRD"
    plot_name = "SEIRD: Biased $t_{r}$, $t_{d}$, $t_{c}$"
    ncomp = len(name)
    params = [r"$\beta$", r"$N$", r"$P_{IFR}$", r"$t_{d}$", r"$t_{r}$", r"$t_{c}$", r"$\kappa$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    priors = [[np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.lognormal, 2.84, 0.58],
              [np.random.gamma, 6.68, 1/0.33],
              [np.random.lognormal, 1.57, 0.65],
              [np.random.uniform, 0, 1]]
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
        S, E, R, I, D = y
        beta, N, p_ifr, t_d, t_r, t_c, Pex = params
        gamma = (1-p_ifr)/t_r
        mu = p_ifr/t_d
        
        c = 1/t_c
        
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
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params
        
class SEIRD3_bias:
    """
    SEIRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIRD parameters
    name = "SEIRD"
    plot_name = "SEIRD: Biased $t_{r}$, $t_{d}$, $t_{c}$"
    ncomp = len(name)
    params = [r"$\beta_{I}$", r"$\beta_{E}$", r"$N$", r"$P_{IFR}$", r"$t_{d}$", r"$t_{r}$", r"$t_{c}$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    priors = [[np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.lognormal, 2.84, 0.58],
              [np.random.gamma, 6.68, 1/0.33],
              [np.random.lognormal, 1.57, 0.65]]
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
        S, E, R, I, D = y
        beta_I, beta_E, N, p_ifr, t_d, t_r, t_c = params
        gamma = (1-p_ifr)/t_r
        mu = p_ifr/t_d
        
        c = 1/t_c
        
        return np.array([-beta_I*I*S/N - beta_E*E*S/N,
                         beta_I*I*S/N + beta_E*E*S/N - c*E,
                         gamma*I,
                         c*E-gamma*I-mu*I,
                         mu*I])
    
    # SEIRD equations solution
    @classmethod
    def solution(cls, t, params, y0):
    
        y0[0] = params[2] - (y0[1] + y0[2] + y0[3] + y0[4])
        
        sol = rk4(cls.model, y0, t, params)
        
        return sol
    
    # SEIRD model total infected and dead output
    @classmethod
    def infected_dead(cls, t, params, y0):
        
        y0[0] = params[2] - (y0[1] + y0[2] + y0[3] + y0[4])
        
        sol = rk4(cls.model, y0, t, params)
        
        I_tot = np.sum(sol[:,2:], axis = 1)
        D = sol[:,4]
        return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params

class SEIRD2_bias:
    """
    SEIRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIRD parameters
    name = "SEIRD"
    plot_name = "SEIRD: Biased $t_{r}$, $t_{d}$, $t_{c}$"
    ncomp = len(name)
    params = [r"$\beta$", r"$N$", r"$t_{d}$", r"$t_{r}$", r"$t_{c}$", r"$\kappa$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    priors = [[np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.lognormal, 2.84, 0.58],
              [np.random.gamma, 6.68, 1/0.33],
              [np.random.lognormal, 1.57, 0.65],
              [np.random.uniform, 0, 1]]
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
        S, E, R, I, D = y
        beta, N, t_d, t_r, t_c, Pex = params
        gamma = 1/t_r
        mu = 1/t_d
        
        c = 1/t_c
        
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
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params

class SEIRD_bias_tr:
    """
    SEIRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIRD parameters
    name = "SEIRD"
    plot_name = "SEIRD: Biased $t_{r}$"
    ncomp = len(name)
    params = [r"$\beta$", r"$N$", r"$P_{IFR}$", r"$t_{d}$", r"$t_{r}$", r"$t_{c}$", r"$\kappa$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    priors = [[np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 0, 30],
              [np.random.gamma, 6.68, 1/0.33],
              [np.random.uniform, 0, 30],
              [np.random.uniform, 0, 1]]
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
        S, E, R, I, D = y
        beta, N, p_ifr, t_d, t_r, t_c, Pex = params
        gamma = (1-p_ifr)/t_r
        mu = p_ifr/t_d
        
        c = 1/t_c
        
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
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params

class SEIRD3_bias_tr:
    """
    SEIRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIRD parameters
    name = "SEIRD"
    plot_name = "SEIRD: Biased $t_{r}$"
    ncomp = len(name)
    params = [r"$\beta_{I}$", r"$\beta_{E}$", r"$N$", r"$P_{IFR}$", r"$t_{d}$", r"$t_{r}$", r"$t_{c}$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    priors = [[np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 0, 30],
              [np.random.gamma, 6.68, 1/0.33],
              [np.random.uniform, 0, 30]]
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
        S, E, R, I, D = y
        beta_I, beta_E, N, p_ifr, t_d, t_r, t_c = params
        gamma = (1-p_ifr)/t_r
        mu = p_ifr/t_d
        
        c = 1/t_c
        
        return np.array([-beta_I*I*S/N - beta_E*E*S/N,
                         beta_I*I*S/N + beta_E*E*S/N - c*E,
                         gamma*I,
                         c*E-gamma*I-mu*I,
                         mu*I])
    
    # SEIRD equations solution
    @classmethod
    def solution(cls, t, params, y0):
    
        y0[0] = params[2] - (y0[1] + y0[2] + y0[3] + y0[4])
        
        sol = rk4(cls.model, y0, t, params)
        
        return sol
    
    # SEIRD model total infected and dead output
    @classmethod
    def infected_dead(cls, t, params, y0):
        
        y0[0] = params[2] - (y0[1] + y0[2] + y0[3] + y0[4])
        
        sol = rk4(cls.model, y0, t, params)
        
        I_tot = np.sum(sol[:,2:], axis = 1)
        D = sol[:,4]
        return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params

class SEIRD2_bias_tr:
    """
    SEIRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIRD parameters
    name = "SEIRD"
    plot_name = "SEIRD: Biased $t_{r}$"
    ncomp = len(name)
    params = [r"$\beta$", r"$N$", r"$t_{d}$", r"$t_{r}$", r"$t_{c}$", r"$\kappa$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    priors = [[np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 0, 30],
              [np.random.gamma, 6.68, 1/0.33],
              [np.random.uniform, 0, 30],
              [np.random.uniform, 0, 1]]
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
        S, E, R, I, D = y
        beta, N, t_d, t_r, t_c, Pex = params
        gamma = 1/t_r
        mu = 1/t_d
        
        c = 1/t_c
        
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
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params

class SEIRD_fixed_tc_td_mean:
    """
    SEIRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIRD parameters
    name = "SEIRD"
    plot_name = "SEIRD: Fixed $t_{c}$, $t_{d}$ on mean"
    ncomp = len(name)
    params = [r"$\beta$", r"$N$", r"$P_{IFR}$", r"$t_{d}$", r"$t_{r}$", r"$t_{c}$", r"$\kappa$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    priors = [[np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 20.2, 20.2],
              [np.random.uniform, 0, 30],
              [np.random.uniform, 5.93, 5.93],
              [np.random.uniform, 0, 1]]
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
        S, E, R, I, D = y
        beta, N, p_ifr, t_d, t_r, t_c, Pex = params
        gamma = (1-p_ifr)/t_r
        mu = p_ifr/t_d
        
        c = 1/t_c
        
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
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params

class SEIRD3_fixed_tc_td_mean:
    """
    SEIRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIRD parameters
    name = "SEIRD"
    plot_name = "SEIRD: Fixed $t_{c}$, $t_{d}$ on mean"
    ncomp = len(name)
    params = [r"$\beta_{I}$", r"$\beta_{E}$", r"$N$", r"$P_{IFR}$", r"$t_{d}$", r"$t_{r}$", r"$t_{c}$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    priors = [[np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 20.2, 20.2],
              [np.random.uniform, 0, 30],
              [np.random.uniform, 5.93, 5.93]]
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
        S, E, R, I, D = y
        beta_I, beta_E, N, p_ifr, t_d, t_r, t_c = params
        gamma = (1-p_ifr)/t_r
        mu = p_ifr/t_d
        
        c = 1/t_c
        
        return np.array([-beta_I*I*S/N - beta_E*E*S/N,
                         beta_I*I*S/N + beta_E*E*S/N - c*E,
                         gamma*I,
                         c*E-gamma*I-mu*I,
                         mu*I])
    
    # SEIRD equations solution
    @classmethod
    def solution(cls, t, params, y0):
    
        y0[0] = params[2] - (y0[1] + y0[2] + y0[3] + y0[4])
        
        sol = rk4(cls.model, y0, t, params)
        
        return sol
    
    # SEIRD model total infected and dead output
    @classmethod
    def infected_dead(cls, t, params, y0):
        
        y0[0] = params[2] - (y0[1] + y0[2] + y0[3] + y0[4])
        
        sol = rk4(cls.model, y0, t, params)
        
        I_tot = np.sum(sol[:,2:], axis = 1)
        D = sol[:,4]
        return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params

class SEIRD2_fixed_tc_td_mean:
    """
    SEIRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIRD parameters
    name = "SEIRD"
    plot_name = "SEIRD: Fixed $t_{c}$, $t_{d}$ on mean"
    ncomp = len(name)
    params = [r"$\beta$", r"$N$", r"$t_{d}$", r"$t_{r}$", r"$t_{c}$", r"$\kappa$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    priors = [[np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 20.2, 20.2],
              [np.random.uniform, 0, 30],
              [np.random.uniform, 5.93, 5.93],
              [np.random.uniform, 0, 1]]
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
        S, E, R, I, D = y
        beta, N, t_d, t_r, t_c, Pex = params
        gamma = 1/t_r
        mu = 1/t_d
        
        c = 1/t_c
        
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
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params

class SEIRD_bias_tr_fixed_tc_td_mean:
    """
    SEIRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIRD parameters
    name = "SEIRD"
    plot_name = "SEIRD: Biased $t_{r}$, Fixed $t_{c}$, $t_{d}$ on mean"
    ncomp = len(name)
    params = [r"$\beta$", r"$N$", r"$P_{IFR}$", r"$t_{d}$", r"$t_{r}$", r"$t_{c}$", r"$\kappa$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    priors = [[np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 20.2, 20.2],
              [np.random.gamma, 6.68, 1/0.33],
              [np.random.uniform, 5.93, 5.93],
              [np.random.uniform, 0, 1]]
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
        S, E, R, I, D = y
        beta, N, p_ifr, t_d, t_r, t_c, Pex = params
        gamma = (1-p_ifr)/t_r
        mu = p_ifr/t_d
        
        c = 1/t_c
        
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
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params
        
class SEIRD3_bias_tr_fixed_tc_td_mean:
    """
    SEIRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIRD parameters
    name = "SEIRD"
    plot_name = "SEIRD: Biased $t_{r}$, Fixed $t_{c}$, $t_{d}$ on mean"
    ncomp = len(name)
    params = [r"$\beta_{I}$", r"$\beta_{E}$", r"$N$", r"$P_{IFR}$", r"$t_{d}$", r"$t_{r}$", r"$t_{c}$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    priors = [[np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 20.2, 20.2],
              [np.random.gamma, 6.68, 1/0.33],
              [np.random.uniform, 5.93, 5.93]]
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
        S, E, R, I, D = y
        beta_I, beta_E, N, p_ifr, t_d, t_r, t_c = params
        gamma = (1-p_ifr)/t_r
        mu = p_ifr/t_d
        
        c = 1/t_c
        
        return np.array([-beta_I*I*S/N - beta_E*E*S/N,
                         beta_I*I*S/N + beta_E*E*S/N - c*E,
                         gamma*I,
                         c*E-gamma*I-mu*I,
                         mu*I])
    
    # SEIRD equations solution
    @classmethod
    def solution(cls, t, params, y0):
    
        y0[0] = params[2] - (y0[1] + y0[2] + y0[3] + y0[4])
        
        sol = rk4(cls.model, y0, t, params)
        
        return sol
    
    # SEIRD model total infected and dead output
    @classmethod
    def infected_dead(cls, t, params, y0):
        
        y0[0] = params[2] - (y0[1] + y0[2] + y0[3] + y0[4])
        
        sol = rk4(cls.model, y0, t, params)
        
        I_tot = np.sum(sol[:,2:], axis = 1)
        D = sol[:,4]
        return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params

class SEIRD2_bias_tr_fixed_tc_td_mean:
    """
    SEIRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIRD parameters
    name = "SEIRD"
    plot_name = "SEIRD: Biased $t_{r}$, Fixed $t_{c}$, $t_{d}$ on mean"
    ncomp = len(name)
    params = [r"$\beta$", r"$N$", r"$t_{d}$", r"$t_{r}$", r"$t_{c}$", r"$\kappa$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    priors = [[np.random.uniform, 0, 1],
              [np.random.uniform, 0, 1],
              [np.random.uniform, 20.2, 20.2],
              [np.random.gamma, 6.68, 1/0.33],
              [np.random.uniform, 5.93, 5.93],
              [np.random.uniform, 0, 1]]
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
        
        S, E, R, I, D = y
        beta, N, t_d, t_r, t_c, Pex = params
        gamma = 1/t_r
        mu = 1/t_d
        
        c = 1/t_c
        
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
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params

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
    post = np.empty(0)
    best_params = np.empty(0)
    # priors = ["uniform",
    #           "uniform",
    #           [np.random.lognormal, 1, 1],
    #           [np.random.lognormal, 1, 1],
    #           [np.random.lognormal, 1, 1],
    #           "uniform",
    #           "uniform",
    #           [np.random.lognormal, 1, 1],
    #           [np.random.lognormal, 1, 1],
    #           [np.random.lognormal, 1, 1],]
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
    
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
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params
    
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
    post = np.empty(0)
    best_params = np.empty(0)
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
    
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
    
        y0[0] = params[2] - (y0[1] + y0[2] + y0[3] + y0[4] + y0[5])
    
        sol = rk4(cls.model, y0, t, params)

        return sol
    
    # SEIARD model total infected and dead output
    @classmethod
    def infected_dead(cls, t, params, y0):
    
        y0[0] = params[2] - (y0[1] + y0[2] + y0[3] + y0[4] + y0[5])
    
        sol = rk4(cls.model, y0, t, params)
        
        I_tot = np.sum(sol[:,3:], axis=1)
        D = sol[:,5]
        return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params
    
class SEIQRD:
    """
    SEITRD epidemic model
    """
    
    def __init__(self, dat):
        
        self.dat = dat
    
    # SEIQRD parameters
    name = "SEIQRD"
    ncomp = 8
    params = [r"$\beta$", r"$N$", r"$\gamma$", r"$\mu$", r"$c$", r"$\sigma$",
              r"$q_{1}$", r"$q_{2}$", r"$q_{3}$"]
    nparams = len(params)
    post = np.empty(0)
    best_params = np.empty(0)
    
    # Differential equations model
    @staticmethod
    @njit(fastmath=True)
    def model(t, y, params):
    
        S, E, Sq, Eq, R, Iq, I, D = y
        beta, N, gamma, mu, c, sigma, q1, q2, q3 = params
        
        return np.array([-beta*I*S/N - sigma*E*S/N - q1*S, #S
                         beta*I*S/N + sigma*E*S/N - c*E - q2*E, #E
                         q1*S, #Sq
                         q2*E - c*Eq, #Eq
                         gamma*Iq, #R
                         q3*I + c*Eq - gamma*Iq - mu*Iq, #Iq
                         c*E - q3*I, #I
                         mu*Iq]) #D
    
    # SEIQRD equations solution
    @classmethod
    def solution(cls, t, params, y0):
    
        y0[0] = params[1] - (y0[1] + y0[2] + y0[3] + y0[4] + y0[5] + y0[6] + y0[7])
    
        sol = rk4(cls.model, y0, t, params)

        return sol
    
    # SEIQRD model total infected and dead output
    @classmethod
    # @njit(fastmath=True)
    def infected_dead(cls, t, params, y0):
    
        y0[0] = params[1] - (y0[1] + y0[2] + y0[3] + y0[4] + y0[5] + y0[6] + y0[7])
    
        sol = rk4(cls.model, y0, t, params)
        
        I_tot = np.sum(sol[:,4:], axis = 1)
        D = sol[:,7]
        return np.concatenate((I_tot, D)).reshape((2, len(I_tot)))
    
    # Save posterior
    @classmethod
    def set_post(cls, post):
        
        cls.post = post
    
    # Save best parameters
    @classmethod
    def set_best_params(cls, best_params):
        
        cls.best_params = best_params
    
