#!/bin/bash
#SBATCH --nodes=4                      #Numero de Nós
#SBATCH --ntasks-per-node=24           #Numero de tarefas por Nó
#SBATCH --ntasks=96                    #Numero total de tarefas MPI
#SBATCH -p cpu_dev                     #Fila (partition) a ser utilizada
#SBATCH -J teste                       #Nome job
#SBATCH --exclusive                    #Utilização exclusiva dos nós durante a execução do job

module load sequana/current
module load anaconda3/2020.07_sequana
module load openmpi/gnu/4.0.1_sequana

python ../main/states_data_download.py
mpiexec -n 96 python ../main/ABC_exec_fit_data.py
