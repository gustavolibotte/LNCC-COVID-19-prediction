#!/bin/bash
#SBATCH --nodes=1                      #Numero de Nós
#SBATCH --ntasks-per-node=1            #Numero de tarefas por Nó
#SBATCH --ntasks=1                     #Numero total de tarefas MPI
#SBATCH --cpus-per-task=1              #Numero de threads
#SBATCH -p cpu_dev                     #Fila (partition) a ser utilizada
#SBATCH -J teste                       #Nome job
#SBATCH --exclusive                    #Utilização exclusiva dos nós durante a execução do job

python states_data_download.py
mpiexec -n 1 python ABC_exec.py
