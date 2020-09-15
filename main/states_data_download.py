#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:15:20 2020

@author: joaop
"""

# Download data previous to running parallel program

from data_loading import LoadData

data_path = open("data_path.txt", "w")

df, path = LoadData.getBrazilDataFrame(5, True)

data_path.write(path)
data_path.close()