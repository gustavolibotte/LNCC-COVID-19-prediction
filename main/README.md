# Code

**ABC_backend_numba.py** includes the implementation of the ABC-SMC algorithm.

**epidemic_model_classes_numba.py** includes different epidemic models to be used. In the presented work, we use the one named SEIRD2 in the code.

Inside the folder of each country, there are files for running the the sequential time-window learning inference with ABC-SMC, with adaptive window sizes, starting from different initial sizes. Files ending in **False** do not implement the "learning", and use a flat prior at the beginning of each new window. Files ending in **True** do implement it, making use of the posterior distribution of the past window as the new prior.

The **Analysis** folder inclues Jupyter notebooks for analyzing data of each country and generating all the figures present in the paper.