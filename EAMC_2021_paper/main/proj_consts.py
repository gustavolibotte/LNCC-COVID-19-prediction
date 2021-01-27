#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:03:27 2020

@author: gustavolibotte
"""

class ProjectConsts:
    """
    Container of constants
    """
    
    # %% Values
    BRAZIL_POPULATION = float(210147125)  # taken from IBGE 2019
    RJ_STATE_POPULATION = float(17264943) # taken from IBGE 2019
    
    # %% Paths
    DATA_PATH = "../data"
    
    # %% URLs
    CASES_BRAZIL_STATES_URL = "https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv"
    CASES_BRAZIL_CITIES_URL = "https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv"
