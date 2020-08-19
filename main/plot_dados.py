#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:40:32 2020

@author: igor
"""

import matplotlib.pyplot as plt
import numpy as np
from epidemic_models import *



class PlotData:
    
    
    """
    We where discussing if it would be better to create a function to each one of 
    the models that we're going to study. Let's decide the best way to write
    the functions after we have used the data dataframe and noticed the necessary 
    changes to the code for every model.
    """
    @staticmethod
    def PlotState(f, data, params, fit_len):
        """
        Function dedicated to plot the date for each state.
        Here I assumed that the dataframes that we're going to use are going
        to have the caracteristcs given in the data_loading.py file.       
        """
        confirmed = data["confirmed"]
        dead = data["dead"]
        day = data["day"]
        
        y0 = np.zeros(3)
        y0[1] = data["confirmed"][0]
        y0[2] = data["dead"][0]
        
        """
        Defining some variables and the initial conditions that we're going to
        use in the runge-kutta method.
        """
        
        if fit_len > len(day):
            t = np.arange(0, fit_len-1, fit_len)
            plt.scatter(day, confirmed, marker = 'o', c = 'blue', label = 'Infectados')
            plt.scatter(day, dead, marker = 'D', c = 'red', label = 'Mortos')
            plt.plot(t, f(t, params, y0)[0], c = 'blue', label = 'Infectados')
            plt.plot(t, f(t, params, y0)[1], c = 'red', label = 'Mortos')
        else:
            plt.scatter(day[:fit_len],confirmed[:fit_len],marker = 'o', c = 'blue', label = 'Infectados')
            plt.scatter(day[:fit_len], dead[:fit_len], marker = 'D', c = 'red', label = 'Mortos')
            plt.plot(day[:fit_len], f(day[:fit_len], params. y0)[0], c = 'blue', label = 'Infectados')
            plt.plot(day[:fit_len], f(day[:fit_len], params. y0)[1], c = 'red', label = 'Mortos')

        """
        Not sure if returning 0 is necessary.
        """
        return 0
    
    @staticmethod
    def PlotCity(f, data, params, fit_len):
        
        confirmed = data["confirmed"]
        dead = data["dead"]
        day = data["day"]
        
        y0 = np.zeros(3)
        y0[1] = data["confirmed"][0]
        y0[2] = data["dead"][0]
        
        if fit_len > len(day):
            t = np.arange(0, fit_len-1, fit_len)
            plt.scatter(day, confirmed, marker = 'o', c = 'blue', label = 'Infectados')
            plt.scatter(day, dead, marker = 'D', c = 'red', label = 'Mortos')
            plt.plot(t, f(t, params, y0)[0], c = 'blue', label = 'Infectados')
            plt.plot(t, f(t, params, y0)[1], c = 'red', label = 'Mortos')
            plt.xlabel('Dias')
            plt.legend()

        else:
            plt.plot(day[:fit_len], confirmed[:fit_len], marker = 'o', c = 'blue', label = 'Infectados')
            plt.scatter(day[:fit_len], dead[:fit_len], marker = 'D', c = 'red', label = 'Mortos')
            plt.plot(day[:fit_len], f(day[:fit_len], params. y0)[0], c = 'blue', label = 'Infectados')
            plt.plot(day[:fit_len], f(day[:fit_len], params. y0)[1], c = 'red', label = 'Mortos')
            plt.xlabel('Dias')
            plt.legend()
        return 0
    
    @staticmethod 
    def PlotHist(post):
        
        """
        In the future, it would be ideal to substitute the post_names
        variable by something more general to any given model that we
        decide to use.
        """        
        
        post_names = ['Beta', 'N', 'Gamma']
        lincol = [2,2]
        
        while lincol[0]*lincol[1] < len(post):
            a = lincol.index(min(lincol))
            lincol[a] += 1
            
        plt.figure(1)
        for i in range(len(post)):
            plt.subplot(lincol[0],lincol[1],i)
            plt.hist(post[i],len(post[i]), label = post_names[i])
            plt.legend()
            
        return 0
            
            
        
        
    
    
    
            
        
    