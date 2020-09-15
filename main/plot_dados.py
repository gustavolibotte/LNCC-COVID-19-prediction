#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:40:32 2020

@author: igor
"""

import matplotlib.pyplot as plt
import numpy as np
import epidemic_model_classes as epi_mod

class PlotData:
    
    @staticmethod
    def Plot(f, data, fit_len, location, diff_img, path):
        
        model = np.zeros(len(f), dtype="object")
        name = np.zeros(len(f), dtype="object")
        params_names = np.zeros(len(f), dtype="object")
        params = np.zeros(len(f), dtype="object")
        #post = np.zeros(len(f))
        
        for i in range(len(f)):
            model[i] = getattr(epi_mod,f[i])
            name[i] = model[i].name
            params_names[i] = model[i].params
            params[i] = model[i].best_params
            #post[i] = model[i].post
            
        confirmed = np.array(data["confirmed"])
        dead = np.array(data["dead"])
        day = np.array(data["day"])
        
        if diff_img == True:
            for i in range(len(f)):
                
                y0 = np.zeros(model[i].ncomp)
                y0[-2] = data["confirmed"][0]
                y0[-1] = data["dead"][0]
                
                if fit_len > len(day):
                    t = np.arange(0, fit_len)
                    plt.figure(figsize=(15,10))
                    plt.title("%s model" %name[i] + ": %s" % (location))
                    plt.scatter(day, confirmed, facecolors = "none", edgecolors = "green", label = 'Infected')
                    plt.scatter(day, dead, facecolors = "none", edgecolors = "red", label = 'Dead')
                    plt.plot(t, model[i].infected_dead(t, params[i], y0)[0], c = "green", label = '%s : Infected' % (name[i]))
                    plt.plot(t, model[i].infected_dead(t, params[i], y0)[1], c = "red", label = '%s : Dead' % (name[i]))
                    plt.xlabel('Days')
                    plt.legend()
                    plt.savefig(path+r"%s/%s_fit.png" % (name[i], name[i]))
                else:
                    plt.figure(figsize=(15,10))
                    plt.title("%s model" %name[i] + ": %s" % (location))
                    plt.scatter(day[:fit_len],confirmed[:fit_len], facecolors = "none", edgecolors = "green", label = 'Infected')
                    plt.scatter(day[:fit_len], dead[:fit_len], facecolors = "none", edgecolors = 'red', label = 'Dead')
                    plt.plot(day[:fit_len], model[i].infected_dead(day[:fit_len], params[i], y0)[0], c = 'green', label = '%s : Infected' % (name[i]))
                    plt.plot(day[:fit_len], model[i].infected_dead(day[:fit_len], params[i], y0)[1], c = 'red', label = '%s : Dead' % (name[i]))
                    plt.xlabel("Days")
                    plt.legend()
                    plt.savefig(path+r"/%s/%s_fit.png" % (name[i], name[i]))
                
                # Comando para salvar aqui
                
        elif diff_img == False:
            lstyles = np.array(["solid", "dotted", "dashed", "dashdot", (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 1, 1, 1))])
            if fit_len > len(day):
                t = np.arange(0, fit_len)
                plt.figure(figsize=(15,10))
                plt.title("%s model" % (name[i]) + ": %s" % (location[-1]))
                plt.scatter(day, confirmed, facecolors = "none", edgecolors = 'green', label = 'Infected')
                plt.scatter(day, dead, facecolors = "none", edgecolors = 'red', label = 'Dead')
                for i in range(len(f)):
                    y0 = np.zeros(model.ncomp)
                    y0[-2] = data["confirmed"][0]
                    y0[-1] = data["dead"][0]
                    plt.plot(t, model[i].infected_dead(t, *params[i], y0)[0], linestyle=lstyles[1], lw=3, c = 'green', label = '%s : Infected' %name[i])
                    plt.plot(t, model[i].infected_dead(t, *params[i], y0)[1], linestyle=lstyles[1], lw=3, c = 'red', label = '%s : Dead' %name[i])
                plt.xlabel('Dias')
                plt.legend()
            else:
                plt.figure(figsize=(15,10))
                plt.title("%s model" %name[i] + ": %s" %location[-1])
                plt.scatter(day[:fit_len],confirmed[:fit_len], facecolors = "none", edgecolors = 'green', label = 'Infected')
                plt.scatter(day[:fit_len], dead[:fit_len], facecolors = "none", edgecolors = 'red', label = 'Dead')
                for i in range(len(f)):
                    y0 = np.zeros(model[i].ncomp)
                    y0[-2] = data["confirmed"][0]
                    y0[-1] = data["dead"][0]
                    plt.plot(day[:fit_len], model[i].infected_dead(day[:fit_len], params[i], y0)[0], linestyle=lstyles[1], lw=3, c = 'green', label = '%s : Infected' %name[i])
                    plt.plot(day[:fit_len], model[i].infected_dead(day[:fit_len], params[i], y0)[1], linestyle=lstyles[1], lw=3, c = 'red', label = '%s : Dead' %name[i])
                plt.xlabel("Dias")
                plt.legend()
                    
            # Comando para salvar aqui
        return 0
    
    @staticmethod
    def PlotHist(f, location):
        
        model = np.zeros(len(f))
        name = np.zeros(len(f))
        params_names = np.zeros(len(f))
        post = np.zeros(len(f))
        
        for i in range(len(f)):
            model[i] = epi_mod.f[i]
            name[i] = model[i].name
            params_names[i] = model[i].params
            post[i] = model[i].post
        
        for i in range(len(f)):
            for j in range(len(post[i])):
                plt.title("%s model:" %name[i] + "%s" %params_names[i][j])
                plt.hist(post[i][j], len(post[i][j]), label = params_names[i][j])
                plt.legend()
                # Comando para salvar aqui
        return 0      