#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:42:48 2020

@author: joaop
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpi4py import MPI
import time
from ABC_backend import *
from epidemic_models import *
from data_loading import LoadData

plt.rcParams.update({'font.size': 22})

##########################################################################

# MPI communications
comm = MPI.COMM_WORLD
root = 0 # Master core
rank = comm.rank # Number of actual core
size = comm.size # Number of used cores

##########################################################################

# Import data
# data = pd.read_csv(r"covid_19_clean_complete.csv")
# germany = data[data["Country/Region"] == "Germany"] # Germany data

# # Data organization
# start = np.where(germany.Active!=0)[0][0] # First day with more than zero infected people
# x = np.linspace(start, len(germany), len(germany[start:])) # Days from start as natural numbers
# y = np.concatenate((germany.Active[start:], germany.Recovered[start:])).reshape(2, len(germany[start:])) # Germany's Active and Recovered data

######################################################################################
# Uncomment to run an example of loading data (do not forget to uncomment the import) #
# TODO: use this data after inserting the new models                                  #
######################################################################################
df_brazil_state_cases = LoadData.getBrazilDataFrame(5, True)
# print(df_brazil_state_cases)
df_state_cases = LoadData.getBrazilStateDataFrame(df_brazil_state_cases, "DF")
y = df_state_cases[["confirmed", "dead"]]
y = np.array(y).T
# print(rj_state_cases)
# rj_state_cities_cases = LoadData.getBrazilStateCityDataFrame("RJ", True)
# print(rj_state_cities_cases)
# petropolis_cities_cases = LoadData.getBrazilCityDataFrame(rj_state_cities_cases, "Petrópolis/RJ")
# print(petropolis_cities_cases)
######################################################################################

#########################################################################

x = np.array(df_state_cases.day)

# Initial conditions (SIRD)
y0 = np.zeros(6)
y0[-2:] = df_state_cases.loc[0,["confirmed", "dead"]]

# Ranges for initial uniform parameter priors (beta, N, gamma, mu)
priors = np.array([[0,1], [3e4, 3e6], [0, 1], [0,1], [0, 1], [0,1], [0, 1], [0,20], [0, 1], [0,1]], dtype=np.float64)

n = 100000 # Number of samples
repeat = 1 # Number of posteriors to be calculated
eps = 1000000 # Tolerance

# First run for numba pre compilation
rejABC(SEIHRD_sol, priors, x, y, y0, eps, n_sample=100)

##########################################################################

t_tot = 0 # Counting total execution time 

# First posterior calculation
t = time.time()
post_ = rejABC(SEIHRD_sol, priors, x, y, y0, eps, n_sample=np.int(n/size)) # Posterior calculation

post = comm.gather(post_, root) # Gathering data from all cores to master core
t = time.time() - t
t_tot += t # Add posterior calculation time to total execution time

##########################################################################

# First posterior analysis running on master core
if (rank == root):
    
    # Info
    print("\n#####################################################################\n")
    print("Number of Iterations: %i" % (n))
    print("Number of Posteriors: %i" % (repeat))
    print("\n#####################################################################\n")
    print("Posterior No. 1")
    print("Execution Time: %.3f s" % (t))
    print("eps = %.2f" % (eps))
    print("Priors' Ranges:")
    for i in range(len(priors)):
        print("\t %f <-> %f" % tuple(priors[i]))
    
    post = np.concatenate(post) # Join results from different cores in a numpy array
    post = post[np.where(post.astype(np.str)[:,-1]!="nan")]
    
    #Plot posterior distributions
    plt.figure(figsize=(15, 12))
    plt.suptitle("Posterior Distributions", fontsize=40)
    for i in range(0, len(post[0])-1):
        plt.subplot(2,5,i+1)
        plt.hist(post[:,i], bins=20, density=True)
        plt.title("Model Parameter %i" % (i+1), fontsize=26)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.savefig(r"posterior.png", format="png", dpi=300, bbox_to_inches=None)
    
    p = post[np.where(post[:,-1] == np.min(post[:,-1]))[0][0]] # Parameter as average of posterior weighted by model-data distance
    p_std = np.std(post[:,:-1], axis=0) # Parameter error as standard deviation of posterior

    print("\nEstimated parameters (av +/- std):")
    for i in range(len(p)-1):
        print("\t %f +/- %f" % (p[i], p_std[i]))
    
    params = np.concatenate((p[:-1],p_std)).reshape((2, len(p[:-1]))) # Join parameters and errors in a numpy array
    
    eps = np.mean(post[:,-1]) # Get average model-data distance associated to parameters that joined the posterior

# Due to MPI use, need to initiate variables to be broadcast
else:
    
    params = None

params = comm.bcast(params, root) # Share posterior analysis results with other cores

##########################################################################

# Same procedure for the calculation of following posteriors
for i in range(repeat-1):
    
    # Get last calculated parameters and errors
    p = params[0]
    p_std = params[1]
    
    # Define new priors
    for j in range(len(priors)):
        
        priors[j] = [np.max([0,p[j]-p_std[j]]), p[j]+p_std[j]]
        
    t = time.time()
    post_ = rejABC(SEIHRD_sol, priors, x, y, y0, eps, n_sample=np.int(n/size))
    
    post = comm.gather(post_, root)
    t = time.time() - t
    t_tot += t
    
    if (rank == root):
    
        print("\n#####################################################################\n")
        print("Posterior No. %i" % (i+2))
        print("Execution Time: %.3f s" % (t))
        print("eps = %.2f" % (eps))
        print("Priors' Ranges:")
        for j in range(len(priors)):
            print("\t %f <-> %f" % tuple(priors[j]))
        
        post = np.concatenate(post)
        
        p = np.average(post[:,:-1], axis=0, weights=1/post[:,-1])
        p_std = np.std(post[:,:-1], axis=0)
    
        print("\nEstimated parameters (av +/- std):")
        for j in range(len(p)):
            print("\t %f +/- %f" % (p[j], p_std[j]))
        
        params = np.concatenate((p,p_std)).reshape((2, len(p)))
        
        eps = np.mean(post[:,-1])
    
    else:
        
        params = None
    
    params = comm.bcast(params, root)

##########################################################################

# Plotting final results
if (rank == root):

    p = params[0]
    p_std = params[1]
        
    print("\nTotal time on ABC: %.3f" %(t_tot))
    plt.figure(figsize=(15, 10))
    plt.scatter(x, y[0], facecolors="none", edgecolors="red",  label="Infected Data")
    plt.scatter(x, y[1], facecolors="none", edgecolors="green", label="Recovered Data")
    plt.plot(x, SEIHRD_sol(x, p, y0)[0], lw=3, color="red", label="Infected Fit")
    plt.plot(x, SEIHRD_sol(x, p, y0)[1], lw=3, color="green", label="Recovered Fit")
    plt.xlabel("Days since first infection", fontsize=26)
    plt.legend()
    plt.savefig(r"model_fit.png", format="png", dpi=300, bbox_to_inches=None)