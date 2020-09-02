#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:40:32 2020

@author: igor
"""

import matplotlib.pyplot as plt
import numpy as np
from epidemic_models import *

import epidemic_models_classes as epi_mod
"""
Importing the epidemic models as classes.
"""



class PlotData:
    
    
    @staticmethod
    def Plot(f, data, params_post, fit_len, location):
        """
        Function dedicated to plot the date for each state.
        Here I assumed that the dataframes that we're going to use are going
        to have the caracteristcs given in the data_loading.py file. 
        """
        
        """
        f is the name of the model to be plotted; params_post is the posterior
        data, which i have assumed to be a dataframe with headers named as 
        the model corresponding to it; fit_len is the len of the x axis (days);
        location is a strig containing the location from where the data is from
        """
        
        model = epi_mod.f
        name = model.name
        params_names = model.params
        post = params_post["f"] # params_post[np.where(params_post[:,-1] == "f")]gy
        
        confirmed = data["confirmed"]
        dead = data["dead"]
        day = data["day"]
        
        y0 = np.zeros(model.ncomp)
        y0[1] = data["confirmed"][0]
        y0[2] = data["dead"][0]
        
        """
        Defining some variables and the initial conditions that we're going to
        use in the runge-kutta method.
        """
        
        if fit_len > len(day):
            t = np.arange(0, fit_len)
            plt.title("Modelo %s" %name + ": %s" %location[-1])
            plt.scatter(day, confirmed, marker = 'o', c = 'blue', label = 'Infectados')
            plt.scatter(day, dead, marker = 'D', c = 'red', label = 'Mortos')
            plt.plot(t, model(t, params, y0)[0], c = 'blue', label = 'Infectados')
            plt.plot(t, model(t, params, y0)[1], c = 'red', label = 'Mortos')
            plt.xlabel('Dias')
            plt.legend()
        else:
            plt.title("Modelo %s" %name + ": %s" %location[-1])
            plt.scatter(day[:fit_len],confirmed[:fit_len],marker = 'o', c = 'blue', label = 'Infectados')
            plt.scatter(day[:fit_len], dead[:fit_len], marker = 'D', c = 'red', label = 'Mortos')
            plt.plot(day[:fit_len], model(day[:fit_len], params, y0)[0], c = 'blue', label = 'Infectados')
            plt.plot(day[:fit_len], model(day[:fit_len], params, y0)[1], c = 'red', label = 'Mortos')
            plt.xlabel("Dias")
            plt.legend()

        """
        Not sure if returning 0 is necessary.
        """
        return 0
    
    @staticmethod 
    def PlotHist(f, params_post, location):
        
        """
        All the histograms are being plotted in the same image, but i think 
        it would be better to have separate images for each of the histograms.
        However, this would result in many imagens in the case of fitting 
        all the models at once.
        """        
        
        model = epi_mod.f
        name = model.name
        params_names = mode.params
        post = params_post["f"]
        lincol = [2,2]
        
        while lincol[0]*lincol[1] < len(post):
            a = lincol.index(min(lincol))
            lincol[a] += 1
            
        plt.figure(1)
        for i in range(len(post)):
            plt.subplot(lincol[0],lincol[1],i)
            plt.hist(post[i],len(post[i]), label = params_names[i])
            plt.legend()
            
        return 0
    
    """
    I think it can be interesting to create a function to plot more than one
    model in the same image, since it would make possible to compare them.
    """