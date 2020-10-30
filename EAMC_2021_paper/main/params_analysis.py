#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:41:22 2020

@author: joaop
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import epidemic_model_classes as epi_mod
from scipy.integrate import simps

models = ["SEIRD3",
"SEIRD3_bias_tr",
"SEIRD3_bias_tr_fixed_tc_td_mean",
"SEIRD3_fixed_tc_td_mean",
"SEIRD3_bias"]
locations = ["China"]

def best_layout(n, axis=0):
    
    row = 1
    col = 1
    
    while (row*col < n):
        
        if (row == col):
            
            row += 1
            
        else:
            
            col += 1
    
    if (axis == 0):
    
        return row, col
    
    else:
        
        return col, row
    
def cross(x, f, lim):
    
    c = []
    
    for i in range(len(f)-1):
        
        if (f[i] < lim < f[i+1] or f[i] > lim > f[i+1]):
            
            c.append(np.sum(x[i:i+2])/2)
            
    return c

def hdr(x, pdf, dens, res):
    
    h = np.linspace(np.max(pdf), 0, res)
    
    pdf /= simps(pdf, x)
    
    for i in range(len(h)):
        
        intervals = cross(x, pdf, h[i])
        
        if (len(intervals) == 1):
            
            if (pdf[-1] > pdf[0]):
            
                intervals.append(x[-1])
            
            else:
                
                intervals = [x[0], intervals[0]]
        
        if (len(intervals) == 0):
            
            intervals = [x[0], x[-1]]
        
        if (len(intervals) > 0 and len(intervals) % 2 == 0):
        
            x_ = np.copy(x)
            
            for j in range(len(intervals)):
                
                x_ = np.concatenate((x_[np.where(x_ < intervals[j])[0]], [intervals[j]], 
                                    x_[np.where(x_ > intervals[j])[0]]))
                
            f = np.interp(x_, x, pdf)
            
            area = 0
            
            for j in range(0, len(intervals), 2):
                
                d0 = np.argwhere(x_ == intervals[j])[0][0]
                d1 = np.argwhere(x_ == intervals[j+1])[0][0]
                
                area += simps(f[d0:d1+1], x_[d0:d1+1])
            
            if (area >= dens):
                
                lim = h[i]
                break
            
    return lim, intervals

paths = ["/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-24_09-51-16",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-24_10-12-45",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-24_10-31-12",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-24_10-49-39",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-24_11-07-56",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-24_11-26-19",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-24_11-44-41",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-24_12-03-05",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-26_03-41-03",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-26_04-00-34",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-26_04-20-27",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-26_04-40-48",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-26_05-01-09",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-26_05-21-28",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-26_05-41-50",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-26_06-02-18",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-26_06-22-41",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-26_06-43-02",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-26_11-31-44",
"/home/joaop/Documents/CLICKCOVID_LNCC/git/joaopedro/LNCC-COVID-19-prediction/logs/sdumont/China_20_bias_fixedandnot_beta_I_E/log2020-10-26_11-52-10"]

p_fold = []
rmsd_fold = []

colors = ["blue", "red"]

for path in paths:
    
    os.chdir(path)
    
    if ("eps.txt" in os.listdir()):
        
        os.remove("eps.txt")
    
    for i in range(1, 21):
    
        os.system(r"grep eps Posterior%i/*/*/*log* >> eps.txt" % (i))
    
    eps = []
    
    for i in range(len(paths)):
        
        f = open(r"eps.txt")
        a = f.read().split("\n")[:-1]
        for k in range(len(a)):
            a[k] = float(a[k].split(" ")[-1])
        a = np.array(a).reshape((int(len(a)/len(models)), len(models)))
        
        eps.append(a)
                
    eps = np.array(eps)

plt.figure(figsize=(15, 10))
for i in range(len(models)):
    
    model = getattr(epi_mod, models[i])
    plt.plot(range(1, 21), np.mean(eps, axis=0)[:,i], "-o", lw=3, markersize=10, label=model.plot_name)
    plt.fill_between(range(1, 21), np.mean(eps, axis=0)[:,i]-np.std(eps, axis=0)[:,i],
                      np.mean(eps, axis=0)[:,i]+np.std(eps, axis=0)[:,i])
plt.xlabel("ABCSMC Epochs")
plt.ylabel("Tolerance")
plt.legend()
plt.savefig("eps.png", format="png", dpi=300, bbox_inches="tight")
plt.show()

for path in paths:

    os.chdir(path)
    
    if ("params.txt" in os.listdir()):
        
        os.remove("params.txt")
    
    p = []
    
    # ci = []
    
    for location in locations:
        for model in models:
            for i in range(1, 21):
    
                p.append(np.genfromtxt(r"Posterior%i/%s/%s/best_params.txt" % (i, location, model)))
                
                # post = np.genfromtxt(r"Posterior%i/%s/%s/post.txt" % (i, location, model))
                
                # for j in range(len(post[0])-1):
                    
                #     h, b = np.histogram(post[:,j], 30, density=True)
                #     b = (b[1:] + b[:-1])/2
                    
                #     ci.append(hdr(b, h, 0.95, 1000)[1])
    
    p = np.concatenate(p).reshape((len(p), len(p[0])))
    # ci = np.array(ci)
    # ci = ci.reshape((len(post[0]))-1, 2, 10, len(models))
    
    np.savetxt(r"params.txt", p)
    
    p_sets = []
    
    n = int(len(p)/len(models))

    for i in range(len(models)):
    
        p_sets.append(p[i*n:(i+1)*n])
    
    plt.rcParams.update({'font.size': 26})
    fig = plt.figure(figsize=(30, 20))
    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    
    rows, cols = best_layout(len(p_sets[0][0]))
    
    for i in range(len(p_sets[0][0])):
        
        plt.subplot(rows, cols,i+1)
        for j in range(len(p_sets)):
            
            model = getattr(epi_mod, models[j])
            
            plt.plot(range(1, len(p_sets[0])+1), p_sets[j][:,i], "-o", lw=3, label=model.plot_name)
            plt.ylabel(model.params[i])
            plt.ticklabel_format(axis="y", scilimits=[-3, 3])
            
            if (i <= rows*(cols-1)):
                
                plt.tick_params(labelbottom=False)
            
            else:
            
                plt.xlabel("ABC SMC Epochs")
                
            # if (i == cols-1):
                
            #     plt.legend()
                
            if (i+1 == len(p_sets[0][0])):
            
                plt.legend(bbox_to_anchor=(1.1,0.9))
    
    plt.savefig(r"params_evol.png", format="png", dpi=300, bbox_inches="tight")
    plt.show()
    
    ##############################################################################
    
    if ("RMSD.txt" in os.listdir()):
        
        os.remove("RMSD.txt")
    
    os.system(r"grep RMSD log_geral%s.txt >> RMSD.txt" % (path.split("/")[-1][3:]))
    time.sleep(1)
    
    rmsd_file = open("RMSD.txt")
    rmsd = rmsd_file.read().split("\n")[:-2]
    rmsd_file.close()
    
    for i in range(len(rmsd)):
        
        rmsd[i] = np.float64(rmsd[i].split(": ")[-1])
        
    rmsd = np.array(rmsd)
    
    rmsd_sets = []
    
    for i in range(len(models)):
    
        rmsd_sets.append(rmsd[i*n:(i+1)*n])
        
    plt.figure(figsize=(15, 10))
    for i in range(len(rmsd_sets)):
        
        model = getattr(epi_mod, models[i])
        plt.plot(rmsd_sets[i], "-o", lw=3, label=model.plot_name)
    
    plt.xlabel("ABC SMC Epochs")
    plt.ylabel("RMSD")
    plt.title(locations[0] + " Data Fit RMSD")
    plt.legend()
    plt.savefig(r"RMSD.png", format="png", dpi=300, bbox_inches="tight")
    plt.show()
    
    p_fold.append(p_sets)
    rmsd_fold.append(rmsd_sets)

##############################################################################

p_fold = np.array(p_fold)
rmsd_fold = np.array(rmsd_fold)

colors = ["blue", "red"]

plt.figure(figsize=(15, 10))
for i in range(rmsd_fold.shape[1]):
    
    model = getattr(epi_mod, models[i])
    
    plt.plot(range(1, rmsd_fold.shape[-1]+1), np.mean(rmsd_fold, axis=0)[i], "-o", 
             lw=3, markersize=10, label=model.plot_name)
    plt.fill_between(range(1, rmsd_fold.shape[-1]+1), np.mean(rmsd_fold, axis=0)[i]-np.std(rmsd_fold, axis=0)[i], 
                     np.mean(rmsd_fold, axis=0)[i]+np.std(rmsd_fold, axis=0)[i], alpha=0.2)
plt.xlim(1, rmsd_fold.shape[-1])
plt.ylabel("RMSD")
plt.legend(fontsize=16)
plt.savefig("rmsd_fold.png", fomrat="png", dpi=300, bbox_inches="tight")
plt.show()

##############################################################################

fig = plt.figure(figsize=(30, 20))
plt.subplots_adjust(wspace=0.2, hspace=0.1)

rows, cols = best_layout(len(p_sets[0][0]))

for i in range(len(p_sets[0][0])):
    
    plt.subplot(rows, cols,i+1)
    for j in range(len(p_sets)):
        
        model = getattr(epi_mod, models[j])
        
        plt.plot(range(1, p_fold.shape[-2]+1), np.mean(p_fold, axis=0)[j][:,i], "-o", 
                 lw=3, markersize=10, label=model.plot_name)
        plt.fill_between(range(1, p_fold.shape[-2]+1), np.mean(p_fold, axis=0)[j][:,i]-np.std(p_fold, axis=0)[j][:,i],
                         np.mean(p_fold, axis=0)[j][:,i]+np.std(p_fold, axis=0)[j][:,i], alpha=0.2)
        plt.ylabel(model.params[i])
        plt.ticklabel_format(axis="y", scilimits=[-3, 3])
        
        # if (i <= rows*(cols-1)):
            
        #     plt.tick_params(labelbottom=False)
        
        # else:
        
        #     plt.xlabel("ABC SMC Epochs")
            
        if (i+1 == len(p_sets[0][0])):
            
            plt.legend(bbox_to_anchor=(1.1,0.9))
plt.savefig("p-fold_evol.png", format="png", dpi=300, bbox_inches="tight")
plt.show()