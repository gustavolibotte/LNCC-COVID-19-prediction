#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:42:48 2020

@author: joaop
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt # plotting
import pandas as pd # data processing
from mpi4py import MPI # module for MPI parallelization
import time # time module for counting execution time
import datetime # date and time for logs
import os

from ABC_backend import *
    # sort: function to sort random numbers according to a given numerical histogram
    # rejABC: Rejection ABC implementation

from epidemic_models import *
    # rk4: 4th order Runge-Kutta for differential equation numerical integration
    # SIR: SIR differential equations model
    # SIR_sol: solution to the SIR model

import epidemic_model_classes as epi_mod
from data_loading import LoadData 
from proj_consts import ProjectConsts

plt.rcParams.update({'font.size': 22})

##########################################################################

# MPI communications
comm = MPI.COMM_WORLD
root = 0 # Master core
rank = comm.rank # Number of actual core
size = comm.size # Number of used cores

# Rejection ABC parameters
n = 100 # Number of samples
repeat = 1 # Number of posteriors to be calculated
# eps = 10000000 # Tolerance

#######################################################################################
# Uncomment to run an example of loading data (do not forget to uncomment the import) #
# TODO: use this data after inserting the new models                                  #
#######################################################################################
# df_brazil_state_cases = LoadData.getBrazilDataFrame(5, True)
# print(df_brazil_state_cases)
# rj_state_cases = LoadData.getBrazilStateDataFrame(df_brazil_state_cases, "RJ")
# print(rj_state_cases)
# rj_state_cities_cases = LoadData.getBrazilStateCityDataFrame("RJ", True)
# print(rj_state_cities_cases)
# petropolis_cities_cases = LoadData.getBrazilCityDataFrame(rj_state_cities_cases, "Petrópolis/RJ")
# print(petropolis_cities_cases)
#######################################################################################

df_countries = pd.read_csv(r"/home/joaop/Documents/CLICKCOVID_LNCC/covid_19_clean_complete.csv")

# States' populations
pop_state_dat = open(f"{ProjectConsts.DATA_PATH}/population_by_country_2020.csv", "r")
pop_state_dat.readline()
pop_state_dat = pop_state_dat.read().split("\n")

pop_state = {}
for i in range(len(pop_state_dat)-1):
    pop_state_dat[i] = pop_state_dat[i].split(",")
    pop_state[pop_state_dat[i][0]] = int(pop_state_dat[i][1])

# Execution date and time
if (rank == root):
    
    datetime_now = str(datetime.datetime.now()).split(".")[0].replace(" ", "_").replace(":","-")

else:
    
    datetime_now = None
    
datetime_now = comm.bcast(datetime_now, root)

# Locations
locations = open("locations_countriesIN.txt", "r").read().split("\n")[:-1]

# Models
models = open("modelsIN.txt", "r").read().split("\n")[:-1]

if (os.path.exists("../logs/") == False):
    
    os.mkdir("../logs/")

if (rank == root):
    
    os.mkdir("../logs/log"+datetime_now)
    
    winner = 0
    aic_win = np.finfo(np.float64).max
    
    log_geral = open("../logs/log"+datetime_now+"/log_geral"+datetime_now+".txt", "w")
    log_geral.write(datetime_now + "\n\n")
    log_geral.write("Data Source: COVID-19 Dataset (on Kaggle)\n")
    log_geral.write("https://www.kaggle.com/imdevskp/corona-virus-report\n\n")
    log_geral.write("Rejection ABC fitting of epidemic curves\n")
    log_geral.write("\nModels: "+", ".join(models))
    log_geral.write("\nLocations: "+", ".join(locations))
    log_geral.write("\n\n#####################################################################\n\n")
    log_geral.write("Number of Iterations: %i\n" % (n))
    log_geral.write("Number of Posteriors: %i\n" % (repeat))
    log_geral.write("\n\n############################   RESULTS   ############################\n\n")
    
    wait_var = 0

else:
    
    wait_var = 0
    
wait_var = comm.gather(wait_var, root)

for i in range(len(locations)):
    
    if (rank == root):
        
        os.mkdir("../logs/log"+datetime_now+"/"+locations[i])
        log_geral.write("Location: "+locations[i]+"\n\n")
        
        wait_var = 0

    else:
        
        wait_var = 0
        
    wait_var = comm.gather(wait_var, root)
    
    for j in range(len(models)):
        
        filepath = "../logs/log"+datetime_now+"/"+locations[i]+"/"+models[j]
        
        if (rank == root):
        
            os.mkdir(filepath)
            wait_var = 0

        else:
            
            wait_var = 0
            
        wait_var = comm.gather(wait_var, root)
        
        # Get data
        data = df_countries[df_countries["Country/Region"] == "Germany"][["Confirmed", "Deaths"]]
        start = np.where(data["Confirmed"] >= 5)[0][0]
        
        # Choose model
        model = getattr(epi_mod, models[j])
        
        # Initial conditions
        x = np.arange(0, len(data[start:]))
        y = np.array(data[start:]).T
        y0 = np.zeros(model.ncomp, dtype=np.float64)
        y0[-2:] = y[:,0]
        
        # Ranges for initial uniform parameter priors (beta, N, gamma)
        priors = []
        for k in range(model.nparams):
            
            if (model.params[k] == r"$N$"):
                
                priors.append([int(pop_state[locations[i]]/100), int(pop_state[locations[i]]/2)])
            
            elif (r"tau" in model.params[k]):
                
                priors.append([1, 30])
            
            else:
                
                priors.append([0, 1])
        
        priors = np.array(priors, dtype=np.float64)
        
        # First run for numba pre compilation
        eps = np.max(y)/20
        rejABC(model.infected_dead, priors, x, y, y0, eps, n_sample=10)
        
        ##########################################################################
        
        t_tot = 0 # Counting total execution time 
        
        # First posterior calculation
        t = time.time()
        post_ = rejABC(model.infected_dead, priors, x, y, y0, eps, n_sample=np.int(n/size))[:np.int(1e5/size)] # Posterior calculation
        
        post = comm.gather(post_, root) # Gathering data from all cores to master core
        t = time.time() - t
        t_tot += t # Add posterior calculation time to total execution time
        
        ##########################################################################
        
        # First posterior analysis running on master core
        if (rank == root):
            
            log = open(filepath+"/%s_log.out" % (model.name), "w")
            
            #Info
            log.write(datetime_now + "\n\n")
            log.write("Data Source: Número de casos confirmados de COVID-19 no Brasil (on GitHub)\n")
            log.write("https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv\n\n")
            log.write("Rejection ABC fitting of epidemic curves\n\n")
            log.write("Model: %s\n" % (model.name))
            log.write("Parameters: " + ", ".join(model.params))
            log.write("\n#####################################################################\n\n")
            log.write("Number of Iterations: %i\n" % (n))
            log.write("Number of Posteriors: %i\n" % (repeat))
            log.write("\n#####################################################################\n\n")
            log.write("Posterior No. 1\n")
            log.write("Execution Time: %.3f s\n" % (t))
            log.write("Tolerance: eps = %.2f\n" % (eps))
            log.write("\nPriors' Ranges:\n\n")
            for k in range(len(priors)):
                log.write("\t %s" % (model.params[k]) + ": %f <-> %f\n" % tuple(priors[k]))
            
            post = np.concatenate(post) # Join results from different cores in a numpy array
            
            # Plot posterior distributions
            # plt.suptitle("Posterior Distributions", fontsize=40)
            for k in range(0, len(post[0])-1):
                plt.figure(figsize=(15, 12))
                plt.hist(post[:,k], bins=20, density=True)
                plt.title("Parameter %s posterior distribution" % (model.params[k]), fontsize=26)
                plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
                plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                plt.savefig(filepath+r"/posterior_%s.png" % (model.params[k].replace("$","").replace("\\","")), format="png", dpi=300, bbox_to_inches=None)
                plt.close()
            
            # p = np.average(post[:,:-1], axis=0, weights=1/post[:,-1]) # Parameter as average of posterior weighted by model-data distance
            p = post[np.where(post[:,-1] == np.min(post[:,-1]))[0][0]][:-1]
            p_std = np.std(post[:,:-1], axis=0) # Parameter error as standard deviation of posterior
            
            best_params = open(filepath+"/best_params.txt", "w")
            best_params.write(" ".join([str(p[i]) for i in range(len(p))]))
            best_params.close()
            
            log.write("\nEstimated parameters (av +/- std):\n\n")
            for k in range(len(p)):
                log.write("\t %s" % (model.params[k]) + ": %f +/- %f\n" % (p[k], p_std[k]))
                
            log.write("\nPosterior distribution on file posterior_'param'.png\n")
            
            params = np.concatenate((p,p_std)).reshape((2, len(p))) # Join parameters and errors in a numpy array
            
            eps = np.mean(post[:,-1]) # Get average model-data distance associated to parameters that joined the posterior
        
        # Due to MPI use, need to initiate variables to be broadcast
        else:
            
            params = None
        
        params = comm.bcast(params, root) # Share posterior analysis results with other cores
        
        ##########################################################################
        
        # Same procedure for the calculation of following posteriors
        # for i in range(repeat-1):
            
        #     # Get last calculated parameters and errors
        #     p = params[0]
        #     p_std = params[1]
            
        #     # Define new priors
        #     for j in range(len(priors)):
                
        #         priors[j] = [np.max([0,p[j]-p_std[j]]), p[j]+p_std[j]]
                
        #     t = time.time()
        #     post_ = rejABC(SEIHRD_sol, priors, x, y, y0, eps, n_sample=np.int(n/size))
            
        #     post = comm.gather(post_, root)
        #     t = time.time() - t
        #     t_tot += t
            
        #     if (rank == root):
            
        #         log.write("\n#####################################################################\n\n")
        #         log.write("Posterior No. %i\n" % (i+2))
        #         log.write("Execution Time: %.3f s\n" % (t))
        #         log.write("Tolerance: eps = %.2f\n" % (eps))
        #         log.write("\nPriors' Ranges:\n\n")
        #         for j in range(len(priors)):
        #             log.write("\t %f <-> %f\n" % tuple(priors[j]))
                
        #         post = np.concatenate(post)
                
        #         p = np.average(post[:,:-1], axis=0, weights=1/post[:,-1])
        #         p_std = np.std(post[:,:-1], axis=0)
            
        #         log.write("\nEstimated parameters (av +/- std):\n\n")
        #         for j in range(len(p)):
        #             log.write("\t %f +/- %f\n" % (p[j], p_std[j]))
                
        #         params = np.concatenate((p,p_std)).reshape((2, len(p)))
                
        #         eps = np.mean(post[:,-1])
            
        #     else:
                
        #         params = None
            
        #     params = comm.bcast(params, root)
        
        ##########################################################################
        
        # Plotting final results
        if (rank == root):
        
            p = params[0]
            p_std = params[1]
            
            L = np.sqrt(np.sum((y-model.infected_dead(x, p, y0))**2))/len(x)
            aic = AIC(model.nparams, L)
            
            if (aic < aic_win):
                
                aic_win = aic
                winner = j
            
            log_geral.write(model.name+":\n")
            log_geral.write("\n\tRMSD: %f" % (L))
            log_geral.write("\n\tAIC: %f\n" % (aic))
            
            log.write("\nRMSD: %f" % (L))
            log.write("\nAIC: %f\n" % (AIC(model.nparams, L)))
            log.write("\n#####################################################################\n\n")    
            log.write("Total time on ABC: %.3f s\n" %(t_tot))
            log.write("\nFit on file model_fit.png")
            log.close()
            
            plt.figure(figsize=(15, 10))
            plt.scatter(x, y[0], facecolors="none", edgecolors="red",  label="Infected Data")
            plt.scatter(x, y[1], facecolors="none", edgecolors="green", label="Dead Data")
            plt.plot(x, model.infected_dead(x, p, y0)[0], lw=3, color="red", label="Infected Fit")
            plt.plot(x, model.infected_dead(x, p, y0)[1], lw=3, color="green", label="Dead Fit")
            plt.title("%s: %s Fit" % (locations[i], model.name))
            plt.xlabel("Days", fontsize=26)
            plt.legend()
            plt.savefig(filepath+r"/%s_fit.png" % (model.name), format="png", dpi=300, bbox_to_inches=None)
            plt.close()
            
    if (rank == root):
        
        log_geral.write("\nSmallest AIC: "+models[winner])
        log_geral.write("\n\n#####################################################################\n\n")
            
if (rank == root):
    
    log_geral.close()