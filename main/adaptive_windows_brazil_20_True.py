#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 18:17:55 2021

@author: joao-valeriano
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt # plotting
import pandas as pd # data processing
from mpi4py import MPI # module for MPI parallelization
import time # time module for counting execution time
import datetime # date and time for logs
import os
import sys

from ABC_backend_numba import *
    # sort: function to sort random numbers according to a given numerical histogram
    # rejABC: Rejection ABC implementation

import epidemic_model_classes_numba as epi_mod
from data_loading import LoadData
from proj_consts import ProjectConsts

plt.rcParams.update({'font.size': 22})

##########################################################################

# MPI communications
comm = MPI.COMM_WORLD
root = 0 # Master core
rank = comm.rank # Number of actual core
size = comm.size # Number of used cores

np.random.seed

# Rejection ABC parameters
n = 1000 # Number of samples
n_max = 10000
n_tosave = 100
repeat = 3 # Number of posteriors to be calculated
noise_scale = 2.
past_window_post = repeat
past_run_last_window_post = repeat
max_trials = 10000000000000000000000000000
n_bins = 20
# eps = 10000000 # Tolerance
#data_length = 200
#t = np.linspace(1, data_length, data_length)
day_step = 5
day_set_size = 20
window_size = day_set_size
min_window_size = 10
max_window_size = 50
val_set_size = 10
day_start = 0
past_run_filepath = ""
post_perc_reduction = 100
post_perc_tol_reduction = 50
use_last_post = True

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

country_data = pd.read_csv("../data/owid-covid-data.csv")

# Execution date and time
if (rank == root):
    
    datetime_now = str(datetime.datetime.now()).split(".")[0].replace(" ", "_").replace(":","-")

else:
    
    datetime_now = None
    
datetime_now = comm.bcast(datetime_now, root)

# Locations
locations = open("brazilIN.txt", "r").read().split("\n")[:-1]

# Models
models = open("modelsIN2.txt", "r").read().split("\n")[:-1]

# If we don't want to divide the data
if (day_set_size == 0):
        
        day_set_size = data_length
        day_step = 0
        val_set_size = 0

if (os.path.exists("../logs/") == False):
    
    os.mkdir("../logs/")

log_folder = "../logs/log"+"_".join([datetime_now, "adaptive", locations[0], str(n), "samples", models[0], str(repeat), 
                                     "posts", str(day_set_size), "day-window", "past", str(use_last_post)])

if (rank == root):
    
    os.mkdir(log_folder)
    for i in range(1, repeat+1):
        os.mkdir(log_folder+"/Posterior"+str(i))
        
        for j in range(len(locations)):
            os.mkdir(log_folder+"/Posterior"+str(i)+"/"+locations[j])
            
            data = country_data[country_data["location"]==locations[j]]
    
            n_sets = int((len(data)-day_set_size-val_set_size) // day_step)

            for k in range(len(models)):

                os.mkdir(log_folder+"/Posterior"+str(i)+"/"+locations[j]+"/"+models[k])
                
                for l in range(n_sets):
                    
                    if (l*day_step+day_set_size >= day_start):
                    
                        os.mkdir(log_folder+"/Posterior"+str(i)+"/"+locations[j]+"/"+models[k]+"/"+"%i_days"%(l*day_step+day_set_size))

    aic_winner = 0
    aic_win = np.finfo(np.float64).max
    rmsd_winner = 0
    rmsd_win = np.finfo(np.float64).max
    aic_winner_val = 0
    aic_win_val = np.finfo(np.float64).max
    rmsd_winner_val = 0
    rmsd_win_val = np.finfo(np.float64).max
    aic_winner_total = 0
    aic_win_total = np.finfo(np.float64).max
    rmsd_winner_total = 0
    rmsd_win_total = np.finfo(np.float64).max
    
    log_geral = open(log_folder+"/log_geral"+datetime_now+".txt", "w")
    log_geral.write(datetime_now + "\n\n")
    log_geral.write("Data Source: Número de casos confirmados de COVID-19 no Brasil (on GitHub)\n")
    log_geral.write("https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv\n\n")
    log_geral.write("Rejection ABC fitting of epidemic curves\n")
    log_geral.write("\nModels: "+", ".join(models))
    log_geral.write("\nLocations: "+", ".join(locations))
    log_geral.write("\n\n#####################################################################\n\n")
    log_geral.write("Number of Samples: %i\n" % (n))
    log_geral.write("Number of Posteriors: %i\n" % (repeat))
    log_geral.write("Training window size: %i\n" % (day_set_size))
    log_geral.write("Validation window size: %i\n" % (val_set_size))
    log_geral.write("\n############################   RESULTS   ############################\n\n")
    
    wait_var = 0

else:
    
    wait_var = 0
    n_sets = 0
    
wait_var = comm.gather(wait_var, root)
n_sets = comm.bcast(n_sets, root)

days_folders = os.listdir(log_folder+"/Posterior1/"+locations[0]+"/"+models[0])
days_folders = [int(folder.split("_")[0]) for folder in days_folders]
days_folders.sort()
days_folders = [str(folder)+"_days" for folder in days_folders]

for i in range(len(locations)):
    
    if (rank == root):
        
        log_geral.write("Location: "+locations[i]+"\n\n")
        
        wait_var = 0

    else:
        
        wait_var = 0
        
    wait_var = comm.gather(wait_var, root)
    
    for j in range(len(models)):
        
        # Get data
        data = country_data[country_data["location"]==locations[j]]
        country_pop = data.population.iloc[0]
        data = data[["total_cases", "total_deaths"]]
        data = data.fillna(0)
        data.columns = ["confirmed", "dead"]
        data["day"] = np.arange(1, len(data)+1, 1.)
        data = data.reset_index(drop=True)
        
        # Choose model
        model = getattr(epi_mod, models[j])
        
        if (rank == root):
            
            log_geral.write("#####################################################################\n\n")
            log_geral.write("Model: "+model.plot_name+"\n")
            log_geral.write("Priors:\n")
            for m in range(len(model.prior_func)):
                log_geral.write("\t" + model.prior_func[m].capitalize() + ": " + ", ".join([str(arg) for arg in model.prior_args[m]])+"\n")
            log_geral.write("\n")
            wait_var = 0
    
        else:
            
            wait_var = 0
            
        wait_var = comm.gather(wait_var, root)
        
        x_total = np.array(data.day, dtype=np.float64)
        y_total = np.array(data[["confirmed", "dead"]].T, dtype=np.float64)
        y0_total = np.zeros(model.ncomp, dtype=np.float64)
        y0_total[-2:] = data.loc[0,["confirmed", "dead"]]
        
        days_sets = []
        
        if (day_step == 0):
            
            days_sets.append(x_total)
        
        else:
            
            for k in range((len(x_total)-day_set_size-val_set_size)//day_step):
                days_sets.append(x_total[day_step*k:day_step*k+day_set_size])
        
        for k in range(len(days_sets)):
            if (days_sets[k][-1] >= day_start+x_total[0]-1):                
                start_idx = k
                break
        
        rmsd_list = np.zeros(len(days_sets))
        
        days_sets0 = [i for i in days_sets]
        
        for days_idx in range(start_idx, len(days_sets)):
            
            if rank == root:
                print(f"Window {days_idx}")
                print("Posterior 1")
            
            filepath =  log_folder+"/Posterior1"+"/"+locations[i]+"/"+models[j]+"/"+"%i_days"%(days_idx*day_step+day_set_size)
        
            # Initial conditions
            if days_idx >= 2:
                if rmsd_list[days_idx-1] < rmsd_list[days_idx-2] and window_size < max_window_size:
                    window_size += day_step
                    if rank == root:
                        print(window_size, min_window_size, max_window_size)
                        print(f"Aumenta janela: {window_size}")
                elif rmsd_list[days_idx-1] > rmsd_list[days_idx-2] and window_size > min_window_size:
                    window_size -= day_step
                    if rank == root:
                        print(window_size, min_window_size, max_window_size)
                        print(f"Diminui janela: {window_size}")
                else:
                    if rank == root:
                        print(window_size, min_window_size, max_window_size)
                        print(f"Feijoada: {window_size}")
                    
                days_sets[days_idx] = np.unique(np.concatenate(days_sets0[:days_idx+1]))[-window_size:]
                
                if rank == root:
                    print(f"Passed length: {len(days_sets[days_idx])}")
                
            # if past_run_filepath == "" or days_idx >= 2:
            #     if days_idx >= 2:
            #         if rmsd_list[days_idx-1] < rmsd_list[days_idx-2] and window_size < max_window_size:
            #             window_size += day_step
            #             days_sets[days_idx] = np.unique(np.concatenate(days_sets0[:days_idx+1]))[-window_size:]
            #             print("Aumenta janela")
            #         elif rmsd_list[days_idx-1] > rmsd_list[days_idx-2] and window_size > min_window_size:
            #             window_size -= day_step
            #             days_sets[days_idx] = np.unique(np.concatenate(days_sets0[:days_idx+1]))[-window_size:]
            #             print("Diminui janela")
            #         else:
            #             print("Feijoada")
            # else:
            #     past_post_file = f"Posterior{past_window_post}".join(filepath.split("Posterior1"))
            #     past_post_file = "/".join(past_post_file.split("/")[:-1])+"/"+days_folders[days_folder-1]
                
            
            x = days_sets[days_idx]
            x_to_end = x_total[np.where(x_total == days_sets[days_idx][0])[0][0]:]
            x_val = (days_sets[days_idx]+val_set_size)[-val_set_size:]
            x_dat_val = np.concatenate((x, x_val))
            y = np.array(data[["confirmed", "dead"]].T, dtype=np.float64)[:,days_sets[days_idx].astype(np.int)-int(days_sets[0][0])]
            y_to_end = np.array(data[["confirmed", "dead"]].T, dtype=np.float64)[:,(days_sets[days_idx][0].astype(np.int)-int(days_sets[0][0])):]
            y_val = np.array(data[["confirmed", "dead"]].T, dtype=np.float64)[:,(days_sets[days_idx]+val_set_size)[-val_set_size:].astype(np.int)-int(days_sets[0][0])]
            y_dat_val = np.concatenate((y, y_val), axis=1)
            
            np.savetxt(filepath+"/data.txt", np.concatenate((x_dat_val.reshape(1,-1), y_dat_val)).T)
            
            y0 = np.zeros(model.ncomp, dtype=np.float64)
            y0[-1] = data.loc[int(days_sets[days_idx][0]-days_sets[0][0]),["dead"]]
            y0[-2] = data.loc[int(days_sets[days_idx][0]-days_sets[0][0]),["confirmed"]]-y0[-1]
            
            np.savetxt(filepath+"/y0.txt", y0)
            
            # Ranges for initial uniform parameter priors (beta, N, gamma)
            for k in range(model.nparams):
                
                if (model.params[k] == r"$N$"):
                    
                    model.prior_args[k] = (country_pop/1000, country_pop/5)
                    model.prior_bounds[k] = (country_pop/1000, country_pop/5)
                    
            # First run for numba pre compilation
            # eps = 0.05#np.max(y)
            weights = np.ones(2)
            # rejABC(model.infected_dead, weights, model.prior_func, model.prior_args, x, y, y0, eps, 10, n_max/size)
            
            if ((days_folders.index(filepath.split("/")[-1]) == 0 and past_run_filepath == "") or use_last_post == False):
            
                samples_ = gen_samples(model.infected_dead, weights, model.prior_func, model.prior_args, x, y, y0, n_max//size)
                
                samples = comm.gather(samples_, root)

                if (rank == root):
                    
                    samples = np.concatenate(samples)
                    eps = np.percentile(samples[:,-1], 100*n/n_max)
                    wait_var = 0
                    # print("gen samples OK!")
                
                else:
                    
                    eps = 1e6
                    wait_var = 0
                    
                wait_var = comm.gather(wait_var, root)
                
                eps = comm.bcast(eps, root)
                # eps0 = eps
                # print("eps =", eps)
                
                ##########################################################################
                
                t_tot = 0 # Counting total execution time 
                
                # First posterior calculation
                t = time.time()
                weights_str = "[1,1]"#"np.sum(y0[-3:])/max(1,y0[-1])]"
                weights = np.array(eval(weights_str), dtype=np.float64)
                post_, trials_ = rejABC(model.infected_dead, weights, model.prior_func, model.prior_args, x, y, y0, eps, n//size) # Posterior calculation
                
                post = comm.gather(post_, root) # Gathering data from all cores to master core
                trials = comm.gather(trials_, root)
                t = time.time() - t
                t_tot += t # Add posterior calculation time to total execution time
            
            else:
                
                days_folder = days_folders.index(filepath.split("/")[-1])
                
                if past_run_filepath == "" or days_folder != 0:
                    past_post_file = f"Posterior{past_window_post}".join(filepath.split("Posterior1"))
                    past_post_file = "/".join(past_post_file.split("/")[:-1])+"/"+days_folders[days_folder-1]
                else:
                    past_post_file = past_run_filepath+f"Posterior{past_run_last_window_post}"+filepath.split("Posterior1")[-1]
                    past_post_file = "/".join(past_post_file.split("/")[:-1])+"/"+str(int(filepath.split("/")[-1][:-5])-day_step)+"_days"
                    
                past_post = np.genfromtxt(past_post_file+"/post.txt")
                past_post_weights = np.genfromtxt(past_post_file+"/post_weights.txt")
                
                p_std = np.std(past_post[:,:-1], axis=0)
                
                samples_ = gen_samples_from_dist(model.infected_dead, weights, past_post, past_post_weights, model.prior_bounds, x, y, y0, n_max//size, None, 10.)
                
                samples = comm.gather(samples_, root)

                if (rank == root):
                    
                    samples = np.concatenate(samples)
                    eps = np.percentile(samples[:,-1], 100*n/n_max)
                    wait_var = 0
                    # print("gen samples OK!")
                
                else:
                    
                    eps = 1e6
                    wait_var = 0
                    
                wait_var = comm.gather(wait_var, root)
                
                eps = comm.bcast(eps, root)
                # eps0 = eps
                
                t_tot = 0 # Counting total execution time 
                
                # First posterior calculation
                t = time.time()
                weights_str = "[1,1]"#"np.sum(y0[-3:])/max(1,y0[-1])]"
                weights = np.array(eval(weights_str), dtype=np.float64)
                post_, post_weights_, trials_ = smcABC(model.infected_dead, weights, past_post, past_post_weights, model.prior_func, 
                                                       model.prior_args, model.prior_bounds, x, y, y0, eps, len(past_post)//size, None, noise_scale, 1e6)
                
                post = comm.gather(post_, root) # Gathering data from all cores to master core
                post_weights = comm.gather(post_weights_, root)
                trials = comm.gather(trials_, root)
                t = time.time() - t
                t_tot += t # Add posterior calculation time to total execution time
            
            ##########################################################################
            
            # max_trials_reached = False
            # for sub_post in post:
            #     if len(sub_post) == 1:
            #         max_trials_reached = True
                    
            # if rank  == root:
            #     if max_trials_reached:
            #         log = open(filepath+"/%s_log.out" % (model.name), "w")
            #         log.write(datetime_now + "\n\n")
            #         log.write("Data Source: Número de casos confirmados de COVID-19 no Brasil (on GitHub)\n")
            #         log.write("https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv\n\n")
            #         log.write("Rejection ABC fitting of epidemic curves\n\n")
            #         log.write("Model: %s\n" % (model.name))
            #         log.write("Parameters: " + ", ".join(model.params))
            #         log.write("\n#####################################################################\n\n")
            #         log.write("Number of Samples: %i\n" % (len(post)))
            #         log.write("Number of Posteriors: %i\n" % (repeat))
            #         log.write("RMSD Weights: %s\n" % (weights_str))
            #         log.write("Training window size: %i\n" % (len(x)))
            #         log.write("Validation window size: %i\n" % (len(x_val)))
            #         log.write("\n#####################################################################\n\n")
            #         log.write("Posterior No. 1\n")
            #         log.write("Execution Time: %.3f s\n" % (t))
            #         log.write("Tolerance: eps = %.2f\n" % (eps))
                    
            #         log.write(f"\nReached maximum number of trials per sample: {max_trials}")
            #         log_geral.write(f"\nReached maximum number of trials per sample: {max_trials}")
                    
            #         log.close()
            #         log_geral.close()
            
            # if max_trials_reached:
            #     past_window_post = int(filepath.split("/")[3][9:])-1
            
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
                log.write("Number of Samples: %i\n" % (len(post)))
                log.write("Number of Posteriors: %i\n" % (repeat))
                log.write("RMSD Weights: %s\n" % (weights_str))
                log.write("Training window size: %i\n" % (len(x)))
                log.write("Validation window size: %i\n" % (len(x_val)))
                log.write("\n#####################################################################\n\n")
                log.write("Posterior No. 1\n")
                log.write("Execution Time: %.3f s\n" % (t))
                log.write("Tolerance: eps = %f\n" % (eps))
                log.write("\nPriors' Ranges:\n\n")
                for k in range(len(model.prior_args)):
                    log.write("\t %s" % (model.params[k]) + ": %f <-> %f\n" % tuple(model.prior_args[k]))
                
                post = np.concatenate(post) # Join results from different cores in a numpy array
                
                if (days_folders.index(filepath.split("/")[-1]) == 0 or use_last_post == False):
                    
                    post_weights = np.ones(len(post))/len(post)
                    
                else:
                    
                    post_weights = np.concatenate(post_weights)
                    post_weights /= np.sum(post_weights)
                
                trials = np.sum(trials)
                
                # Plot posterior distributions
                # plt.suptitle("Posterior Distributions", fontsize=40)
                # for k in range(0, len(post[0])-1):
                #     plt.figure(figsize=(15, 12))
                #     plt.hist(post[:,k], bins=20, density=True)
                #     plt.ylim(0, np.max(np.histogram(post[:,k], 20, density=True)[0])*1.1)
                #     plt.title("Parameter %s posterior distribution" % (model.params[k]), fontsize=26)
                #     plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
                #     plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                #     plt.savefig(filepath+r"/posterior_%s.png" % (model.params[k].replace("$","").replace("\\","")), format="png", dpi=300, bbox_inches=None)
                #     plt.close()
                
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
                
                L = distance(y, model.infected_dead(x, p, y0), np.ones(2))
                np.savetxt(filepath+"/fit_error.txt", (y-model.infected_dead(x, p, y0)).T)
                
                L_weight = distance(y, model.infected_dead(x, p, y0), weights)
                rmsd_list[days_idx] = L_weight
                print(f"Posterior 1: RMSD: {L_weight}")
                np.savetxt(filepath+"/rmsd_list.txt", rmsd_list)
                
                L_val = distance(y_dat_val[:,-val_set_size:], model.infected_dead(x_dat_val, p, y0)[:,-val_set_size:], np.ones(2))
                np.savetxt(filepath+"/val_error.txt", (y_dat_val[:,-val_set_size:]-model.infected_dead(x_dat_val, p, y0)[:,-val_set_size:]).T)
                
                L_total = distance(y_total, model.infected_dead(x_total, p, y0_total), np.ones(2))
                aic = AIC(model.nparams, L)
                aic_val = AIC(model.nparams, L_val)
                aic_total = AIC(model.nparams, L_total)
                
                if (aic < aic_win):
                    
                    aic_win = aic
                    aic_winner = j
                    
                if (aic_val < aic_win_val):
                    
                    aic_win_val = aic_val
                    aic_winner_val = j
                    
                if (aic_total < aic_win_total):
                    
                    aic_win_total = aic_total
                    aic_winner_total = j
                    
                if (L < rmsd_win):
                    
                    rmsd_win = L
                    rmsd_winner = j
                    
                if (L_weight < rmsd_win):
                    
                    rmsd_win = L_weight
                    rmsd_winner = j
                    
                if (L_val < rmsd_win_val):
                    
                    rmsd_win_val = L_val
                    rmsd_winner_val = j
                    
                if (L_total <  rmsd_win_total):
                    
                    rmsd_win_total = L_total
                    rmsd_winner_total = j
                
                log_geral.write("Posterior 1:")
                log_geral.write("\n\tRMSD: %f" % (L))
                log_geral.write("\n\tWeighted RMSD: %f" % (L_weight))
                log_geral.write("\n\tRMSD for validation data: %f" % (L_val))
                log_geral.write("\n\tRMSD for total data: %f" % (L_total))
                log_geral.write("\n\tAIC: %f" % (AIC(model.nparams, L)))
                log_geral.write("\n\tAIC for validation data: %f" % (AIC(model.nparams, L_val)))
                log_geral.write("\n\tAIC for total data: %f\n\n" % (AIC(model.nparams, L_total)))
                
                log.write("\nRMSD: %f" % (L))
                log.write("\nWeighted RMSD: %f" % (L_weight))
                log.write("\nRMSD for validation data: %f" % (L_val))
                log.write("\nRMSD for total data: %f" % (L_total))
                log.write("\nAIC: %f" % (AIC(model.nparams, L)))
                log.write("\nAIC for validation data: %f" % (AIC(model.nparams, L_val)))
                log.write("\nAIC for total data: %f\n" % (AIC(model.nparams, L_total)))
                log.write("\n#####################################################################\n\n")
                log.write("Total time on ABC: %.3f s\n" %(t_tot))
                log.write("Number of trials: %i\n" % trials)
                log.write("\nFit on file model_fit.png")
                log.close()
                
                # if (len(x) < len(x_total)):

                #     plt.figure(figsize=(15, 10))
                #     plt.scatter(x_dat_val, y_dat_val[0], facecolors="none", edgecolors="red",  label="Fit Cumulative Infected Data")
                #     plt.scatter(x_dat_val, y_dat_val[1], facecolors="none", edgecolors="green", label="Fit Cumulative Dead Data")
                #     plt.plot(x_dat_val, model.infected_dead(x_dat_val, p, y0)[0], lw=3, color="red", label="Cumulative Infected Fit")
                #     plt.plot(x_dat_val, model.infected_dead(x_dat_val, p, y0)[1], lw=3, color="green", label="Cumulative Dead Fit")
                #     plt.title("%s: %s Fit" % (locations[i], model.name))
                #     plt.vlines(x[-1], 0, np.max(y_dat_val), color="black", linestyles="dashed")
                #     plt.xlabel("Days", fontsize=26)
                #     plt.legend()
                #     plt.savefig(filepath+r"/%s_val_fit.png" % (model.name), format="png", dpi=300, bbox_inches=None)
                #     plt.close()
                    
                #     plt.figure(figsize=(15, 10))
                #     plt.scatter(x_dat_val, y_dat_val[0], facecolors="none", edgecolors="red",  label="Fit Cumulative Infected Data")
                #     plt.plot(x_dat_val, model.infected_dead(x_dat_val, p, y0)[0], lw=3, color="red", label="Cumulative Infected Fit")
                #     plt.title("%s: %s Fit" % (locations[i], model.name))
                #     plt.vlines(x[-1], np.min(y_dat_val[0]), np.max(y_dat_val[0]), color="black", linestyles="dashed")
                #     plt.xlabel("Days", fontsize=26)
                #     plt.legend()
                #     plt.savefig(filepath+r"/%s_confirmed_val_fit.png" % (model.name), format="png", dpi=300, bbox_inches=None)
                #     plt.close()
                    
                #     plt.figure(figsize=(15, 10))
                #     plt.scatter(x_dat_val, y_dat_val[1], facecolors="none", edgecolors="green", label="Fit Cumulative Dead Data")
                #     plt.plot(x_dat_val, model.infected_dead(x_dat_val, p, y0)[1], lw=3, color="green", label="Cumulative Dead Fit")
                #     plt.title("%s: %s Fit" % (locations[i], model.name))
                #     plt.vlines(x[-1], np.min(y_dat_val[1]), np.max(y_dat_val[1]), color="black", linestyles="dashed")
                #     plt.xlabel("Days", fontsize=26)
                #     plt.legend()
                #     plt.savefig(filepath+r"/%s_dead_val_fit.png" % (model.name), format="png", dpi=300, bbox_inches=None)
                #     plt.close()

                    # plt.figure(figsize=(15, 10))
                    # plt.scatter(x_to_end, y_to_end[0], facecolors="none", edgecolors="red",  label="Fit Cumulative Infected Data")
                    # plt.scatter(x_to_end, y_to_end[1], facecolors="none", edgecolors="green", label="Fit Cumulative Dead Data")
                    # plt.plot(x_to_end, model.infected_dead(x_to_end, p, y0)[0], lw=3, color="red", label="Cumulative Infected Fit")
                    # plt.plot(x_to_end, model.infected_dead(x_to_end, p, y0)[1], lw=3, color="green", label="Cumulative Dead Fit")
                    # plt.vlines(x_to_end[len(x)-1], 0, np.max(y_total), color="black", linestyles="dashed")
                    # plt.title("%s: %s Fit" % (locations[i], model.name))
                    # plt.xlabel("Days", fontsize=26)
                    # plt.legend()
                    # plt.savefig(filepath+r"/%s_total_fit.png" % (model.name), format="png", dpi=300, bbox_inches=None)
                    # plt.close()
                    
                    # plt.figure(figsize=(15, 10))
                    # plt.scatter(x_to_end, y_to_end[0], facecolors="none", edgecolors="red",  label="Fit Cumulative Infected Data")
                    # plt.plot(x_to_end, model.infected_dead(x_to_end, p, y0)[0], lw=3, color="red", label="Cumulative Infected Fit")
                    # plt.vlines(x_to_end[len(x)-1], np.min(y_total[0]), np.max(y_total[0]), color="black", linestyles="dashed")
                    # plt.title("%s: %s Fit" % (locations[i], model.name))
                    # plt.xlabel("Days", fontsize=26)
                    # plt.legend()
                    # plt.savefig(filepath+r"/%s_confirmed_total_fit.png" % (model.name), format="png", dpi=300, bbox_inches=None)
                    # plt.close()
                    
                    # plt.figure(figsize=(15, 10))
                    # plt.scatter(x_to_end, y_to_end[1], facecolors="none", edgecolors="green", label="Fit Cumulative Dead Data")
                    # plt.plot(x_to_end, model.infected_dead(x_to_end, p, y0)[1], lw=3, color="green", label="Cumulative Dead Fit")
                    # plt.vlines(x_to_end[len(x)-1], np.min(y_total[1]), np.max(y_total[1]), color="black", linestyles="dashed")
                    # plt.title("%s: %s Fit" % (locations[i], model.name))
                    # plt.xlabel("Days", fontsize=26)
                    # plt.legend()
                    # plt.savefig(filepath+r"/%s_dead_total_fit.png" % (model.name), format="png", dpi=300, bbox_inches=None)
                    # plt.close()
                    
                # else:
                    
                #     plt.figure(figsize=(15, 10))
                #     plt.scatter(x, y[0], facecolors="none", edgecolors="red",  label="Fit Cumulative Infected Data")
                #     plt.scatter(x, y[1], facecolors="none", edgecolors="green", label="Fit Cumulative Dead Data")
                #     plt.plot(x, model.infected_dead(x, p, y0_total)[0], lw=3, color="red", label="Cumulative Infected Fit")
                #     plt.plot(x, model.infected_dead(x, p, y0_total)[1], lw=3, color="green", label="Cumulative Dead Fit")
                #     plt.title("%s: %s Fit" % (locations[i], model.name))
                #     plt.xlabel("Days", fontsize=26)
                #     plt.legend()
                #     plt.savefig(filepath+r"/%s_fit.png" % (model.name), format="png", dpi=300, bbox_inches=None)
                #     plt.close()
                
                np.savetxt(filepath+r"/post.txt", post)
                np.savetxt(filepath+r"/post_weights.txt", post_weights)
                weight_var = 0
            
            else:
                
                post_weights = np.zeros(1)
                wait_var = 0
                
            wait_var = comm.gather(wait_var, root)
            
            post = comm.bcast(post, root) # Share posterior analysis results with other cores
            post_weights = comm.bcast(post_weights, root)

            ##########################################################################
            
            max_trials_reached = False
            
            # Same procedure for the calculation of following posteriors
            for l in range(1, repeat):
                if rank == root:
                    print(f"Posterior {l+1}")
                
                if max_trials_reached:
                    if rank == root:
                        past_filepath = filepath.split("/")
                        past_filepath[3] = past_filepath[3][:9]+str(l-1)
                        past_filepath = "/".join(past_filepath)
                        for posterior in range(l, repeat+1):
                            repeat_filepath = filepath.split("/")
                            repeat_filepath[3] = repeat_filepath[3][:9]+str(posterior)
                            repeat_filepath = "/".join(repeat_filepath)
                            os.system(f"rsync -av --quiet {past_filepath}/* {repeat_filepath}")
                        wait_var = 0
                    else:
                        wait_var = 0
                    wait_var = comm.bcast(wait_var, root)
                    
                    continue
                
                filepath =  log_folder+"/Posterior"+str(l+1)+"/"+locations[i]+"/"+models[j]+"/"+"%i_days"%(days_idx*day_step+day_set_size)

                p = post[np.where(post[:,-1] == np.min(post[:,-1]))[0][0]][:-1]
                p_std = np.std(post[:,:-1], axis=0) # Parameter error as standard deviation of posterior

                eps = np.percentile(post[:,-1], post_perc_tol_reduction)
                
                t = time.time()
                post_, post_weights_, trials_ = smcABC(model.infected_dead, weights, post, post_weights, model.prior_func, 
                                                       model.prior_args, model.prior_bounds, x, y, y0, eps, len(post)//size, None, noise_scale, max_trials)
                
                post = comm.gather(post_, root)
                post_weights = comm.gather(post_weights_, root)
                trials = comm.gather(trials_, root)
                t = time.time() - t
                t_tot += t
                
                max_trials_reached = False
                
                if rank  == root:
                    for sub_post in post:
                        if len(sub_post) == 1:
                            max_trials_reached = True
                    
                    # if max_trials_reached:
                    #     log = open(filepath+"/%s_log.out" % (model.name), "w")
                    #     log.write(datetime_now + "\n\n")
                    #     log.write("Data Source: Número de casos confirmados de COVID-19 no Brasil (on GitHub)\n")
                    #     log.write("https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv\n\n")
                    #     log.write("Rejection ABC fitting of epidemic curves\n\n")
                    #     log.write("Model: %s\n" % (model.name))
                    #     log.write("Parameters: " + ", ".join(model.params))
                    #     log.write("\n#####################################################################\n\n")
                    #     log.write("Number of Samples: %i\n" % (len(post)))
                    #     log.write("Number of Posteriors: %i\n" % (repeat))
                    #     log.write("RMSD Weights: %s\n" % (weights_str))
                    #     log.write("Training window size: %i\n" % (len(x)))
                    #     log.write("Validation window size: %i\n" % (len(x_val)))
                    #     log.write("\n#####################################################################\n\n")
                    #     log.write("Posterior No. 1\n")
                    #     log.write("Execution Time: %.3f s\n" % (t))
                    #     log.write("Tolerance: eps = %.2f\n" % (eps))
                        
                    #     log.write(f"\nReached maximum number of trials per sample: {max_trials}")
                    #     log_geral.write(f"\nReached maximum number of trials per sample: {max_trials}")
                        
                    #     log.close()
                
                max_trials_reached = comm.bcast(max_trials_reached)
                
                if max_trials_reached:
                    past_window_post = int(filepath.split("/")[3][9:])-1
                    continue
                
                # First posterior analysis running on master core
                if (rank == root):
                    
                    log = open(filepath+"/%s_log.out" % (model.name), "w")
                    
                    post = np.concatenate(post) # Join results from different cores in a numpy array
                    post_weights = np.concatenate(post_weights)
                    post_weights /= np.sum(post_weights)
                    trials = np.sum(trials)

                    #Info
                    log.write(datetime_now + "\n\n")
                    log.write("Data Source: Número de casos confirmados de COVID-19 no Brasil (on GitHub)\n")
                    log.write("https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv\n\n")
                    log.write("Rejection ABC fitting of epidemic curves\n\n")
                    log.write("Model: %s\n" % (model.name))
                    log.write("Parameters: " + ", ".join(model.params))
                    log.write("\n#####################################################################\n\n")
                    log.write("Number of Samples: %i\n" % (len(post)))
                    log.write("Number of Posteriors: %i\n" % (repeat))
                    log.write("RMSD Weights: %s\n" % (weights_str))
                    log.write("Training window size: %i\n" % (len(x)))
                    log.write("Validation window size: %i\n" % (len(x_val)))
                    log.write("\n#####################################################################\n\n")
                    log.write("Posterior No. %i\n" % (l+1))
                    log.write("Execution Time: %.3f s\n" % (t))
                    log.write("Tolerance: eps = %f\n" % (eps))
                    log.write("\nPriors' Ranges:\n\n")
                    for k in range(len(post[0])-1):
                        log.write("\t %s" % (model.params[k]) + ": %f <-> %f\n" % (np.min(post[:,k]), np.max(post[:,k])))
                    
                    # Plot posterior distributions
                    # plt.suptitle("Posterior Distributions", fontsize=40)
                    # for k in range(0, len(post[0])-1):
                    #     plt.figure(figsize=(15, 12))
                    #     plt.hist(post[:,k], bins=20, density=True)
                    #     plt.ylim(0, np.max(np.histogram(post[:,k], 20, density=True)[0])*1.1)
                    #     plt.title("Parameter %s posterior distribution" % (model.params[k]), fontsize=26)
                    #     plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
                    #     plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                    #     plt.savefig(filepath+r"/posterior_%s.png" % (model.params[k].replace("$","").replace("\\","")), format="png", dpi=300, bbox_inches=None)
                    #     plt.close()
                    
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
                    
                    L = distance(y, model.infected_dead(x, p, y0), np.ones(2))
                    np.savetxt(filepath+"/fit_error.txt", (y-model.infected_dead(x, p, y0)).T)
                    
                    L_weight = distance(y, model.infected_dead(x, p, y0), weights)
                    rmsd_list[days_idx] = L_weight
                    print(f"Posterior {l+1}: RMSD: {L_weight}")
                    np.savetxt(filepath+"/rmsd_list.txt", rmsd_list)
                    
                    L_val = distance(y_dat_val[:,-val_set_size:], model.infected_dead(x_dat_val, p, y0)[:,-val_set_size:], np.ones(2))
                    np.savetxt(filepath+"/val_error.txt", (y_dat_val[:,-val_set_size:]-model.infected_dead(x_dat_val, p, y0)[:,-val_set_size:]).T)
                    
                    L_total = distance(y_total, model.infected_dead(x_total, p, y0_total), np.ones(2))
                    aic = AIC(model.nparams, L)
                    aic_val = AIC(model.nparams, L_val)
                    aic_total = AIC(model.nparams, L_total)
                    
                    if (aic < aic_win):
                        
                        aic_win = aic
                        aic_winner = j
                        
                    if (aic_val < aic_win_val):
                        
                        aic_win_val = aic_val
                        aic_winner_val = j
                        
                    if (aic_total < aic_win_total):
                        
                        aic_win_total = aic_total
                        aic_winner_total = j
                        
                    if (L < rmsd_win):
                        
                        rmsd_win = L
                        rmsd_winner = j
                        
                    if (L_weight < rmsd_win):
                        
                        rmsd_win = L_weight
                        rmsd_winner = j
                        
                    if (L_val < rmsd_win_val):
                        
                        rmsd_win_val = L_val
                        rmsd_winner_val = j
                        
                    if (L_total <  rmsd_win_total):
                        
                        rmsd_win_total = L_total
                        rmsd_winner_total = j
                    
                    log_geral.write("Posterior 1:")
                    log_geral.write("\n\tRMSD: %f" % (L))
                    log_geral.write("\n\tWeighted RMSD: %f" % (L_weight))
                    log_geral.write("\n\tRMSD for validation data: %f" % (L_val))
                    log_geral.write("\n\tRMSD for total data: %f" % (L_total))
                    log_geral.write("\n\tAIC: %f" % (AIC(model.nparams, L)))
                    log_geral.write("\n\tAIC for validation data: %f" % (AIC(model.nparams, L_val)))
                    log_geral.write("\n\tAIC for total data: %f\n\n" % (AIC(model.nparams, L_total)))
                    
                    log.write("\nRMSD: %f" % (L))
                    log.write("\nWeighted RMSD: %f" % (L_weight))
                    log.write("\nRMSD for validation data: %f" % (L_val))
                    log.write("\nRMSD for total data: %f" % (L_total))
                    log.write("\nAIC: %f" % (AIC(model.nparams, L)))
                    log.write("\nAIC for validation data: %f" % (AIC(model.nparams, L_val)))
                    log.write("\nAIC for total data: %f\n" % (AIC(model.nparams, L_total)))
                    log.write("\n#####################################################################\n\n")    
                    log.write("Total time on ABC: %.3f s\n" %(t_tot))
                    log.write("Number of trials: %i\n" % trials)
                    log.write("\nFit on file model_fit.png")
                    log.close()
                    
                    # if (len(x) < len(x_total)):

                    #     plt.figure(figsize=(15, 10))
                    #     plt.scatter(x_dat_val, y_dat_val[0], facecolors="none", edgecolors="red",  label="Fit Cumulative Infected Data")
                    #     plt.scatter(x_dat_val, y_dat_val[1], facecolors="none", edgecolors="green", label="Fit Cumulative Dead Data")
                    #     plt.plot(x_dat_val, model.infected_dead(x_dat_val, p, y0)[0], lw=3, color="red", label="Cumulative Infected Fit")
                    #     plt.plot(x_dat_val, model.infected_dead(x_dat_val, p, y0)[1], lw=3, color="green", label="Cumulative Dead Fit")
                    #     plt.title("%s: %s Fit" % (locations[i], model.name))
                    #     plt.vlines(x[-1], 0, np.max(y_dat_val), color="black", linestyles="dashed")
                    #     plt.xlabel("Days", fontsize=26)
                    #     plt.legend()
                    #     plt.savefig(filepath+r"/%s_val_fit.png" % (model.name), format="png", dpi=300, bbox_inches=None)
                    #     plt.close()
                        
                    #     plt.figure(figsize=(15, 10))
                    #     plt.scatter(x_dat_val, y_dat_val[0], facecolors="none", edgecolors="red",  label="Fit Cumulative Infected Data")
                    #     plt.plot(x_dat_val, model.infected_dead(x_dat_val, p, y0)[0], lw=3, color="red", label="Cumulative Infected Fit")
                    #     plt.title("%s: %s Fit" % (locations[i], model.name))
                    #     plt.vlines(x[-1], np.min(y_dat_val[0]), np.max(y_dat_val[0]), color="black", linestyles="dashed")
                    #     plt.xlabel("Days", fontsize=26)
                    #     plt.legend()
                    #     plt.savefig(filepath+r"/%s_confirmed_val_fit.png" % (model.name), format="png", dpi=300, bbox_inches=None)
                    #     plt.close()
                        
                    #     plt.figure(figsize=(15, 10))
                    #     plt.scatter(x_dat_val, y_dat_val[1], facecolors="none", edgecolors="green", label="Fit Cumulative Dead Data")
                    #     plt.plot(x_dat_val, model.infected_dead(x_dat_val, p, y0)[1], lw=3, color="green", label="Cumulative Dead Fit")
                    #     plt.title("%s: %s Fit" % (locations[i], model.name))
                    #     plt.vlines(x[-1], np.min(y_dat_val[1]), np.max(y_dat_val[1]), color="black", linestyles="dashed")
                    #     plt.xlabel("Days", fontsize=26)
                    #     plt.legend()
                    #     plt.savefig(filepath+r"/%s_dead_val_fit.png" % (model.name), format="png", dpi=300, bbox_inches=None)
                    #     plt.close()
    
                        # plt.figure(figsize=(15, 10))
                        # plt.scatter(x_total, y_total[0], facecolors="none", edgecolors="red",  label="Fit Cumulative Infected Data")
                        # plt.scatter(x_total, y_total[1], facecolors="none", edgecolors="green", label="Fit Cumulative Dead Data")
                        # plt.plot(x_total, model.infected_dead(x_total, p, y0_total)[0], lw=3, color="red", label="Cumulative Infected Fit")
                        # plt.plot(x_total, model.infected_dead(x_total, p, y0_total)[1], lw=3, color="green", label="Cumulative Dead Fit")
                        # plt.vlines(x_total[len(x)-1], 0, np.max(y_total), color="black", linestyles="dashed")
                        # plt.title("%s: %s Fit" % (locations[i], model.name))
                        # plt.xlabel("Days", fontsize=26)
                        # plt.legend()
                        # plt.savefig(filepath+r"/%s_total_fit.png" % (model.name), format="png", dpi=300, bbox_inches=None)
                        # plt.close()
                        
                        # plt.figure(figsize=(15, 10))
                        # plt.scatter(x_total, y_total[0], facecolors="none", edgecolors="red",  label="Fit Cumulative Infected Data")
                        # plt.plot(x_total, model.infected_dead(x_total, p, y0_total)[0], lw=3, color="red", label="Cumulative Infected Fit")
                        # plt.vlines(x_total[len(x)-1], np.min(y_total[0]), np.max(y_total[0]), color="black", linestyles="dashed")
                        # plt.title("%s: %s Fit" % (locations[i], model.name))
                        # plt.xlabel("Days", fontsize=26)
                        # plt.legend()
                        # plt.savefig(filepath+r"/%s_confirmed_total_fit.png" % (model.name), format="png", dpi=300, bbox_inches=None)
                        # plt.close()
                        
                        # plt.figure(figsize=(15, 10))
                        # plt.scatter(x_total, y_total[1], facecolors="none", edgecolors="green", label="Fit Cumulative Dead Data")
                        # plt.plot(x_total, model.infected_dead(x_total, p, y0_total)[1], lw=3, color="green", label="Cumulative Dead Fit")
                        # plt.vlines(x_total[len(x)-1], np.min(y_total[1]), np.max(y_total[1]), color="black", linestyles="dashed")
                        # plt.title("%s: %s Fit" % (locations[i], model.name))
                        # plt.xlabel("Days", fontsize=26)
                        # plt.legend()
                        # plt.savefig(filepath+r"/%s_dead_total_fit.png" % (model.name), format="png", dpi=300, bbox_inches=None)
                        # plt.close()
                        
                    # else:
                        
                    #     plt.figure(figsize=(15, 10))
                    #     plt.scatter(x, y[0], facecolors="none", edgecolors="red",  label="Fit Cumulative Infected Data")
                    #     plt.scatter(x, y[1], facecolors="none", edgecolors="green", label="Fit Cumulative Dead Data")
                    #     plt.plot(x, model.infected_dead(x, p, y0_total)[0], lw=3, color="red", label="Cumulative Infected Fit")
                    #     plt.plot(x, model.infected_dead(x, p, y0_total)[1], lw=3, color="green", label="Cumulative Dead Fit")
                    #     plt.title("%s: %s Fit" % (locations[i], model.name))
                    #     plt.xlabel("Days", fontsize=26)
                    #     plt.legend()
                    #     plt.savefig(filepath+r"/%s_fit.png" % (model.name), format="png", dpi=300, bbox_inches=None)
                    #     plt.close()

                    np.savetxt(filepath+r"/post.txt", post)
                    np.savetxt(filepath+r"/post_weights.txt", post_weights)
                
                else:
                    wait_var == 0
                    
                wait_var = comm.bcast(wait_var, root)
                    
                post = comm.bcast(post, root) # Share full posterior with other cores
                post_weights = comm.bcast(post_weights, root)
                rmsd_list = comm.bcast(rmsd_list, root)
                
        if (rank == root):
            
            log_geral.write("\nSmallest RMSD: "+models[rmsd_winner])
            log_geral.write("\nSmallest AIC: "+models[aic_winner])
            log_geral.write("\n\n#####################################################################\n\n")
            
if (rank == root):
    
    log_geral.close()
