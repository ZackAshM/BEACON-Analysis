#!/bin/bash
#SBATCH --nodes=1                #Node Count
#SBATCH --ntasks=1               #Number of Tasks per Node
#SBATCH --cpus-per-task=1        #Cpu-Cores per task
#SBATCH --mem-per-cpu=3G         #Memory per core
#SBATCH --time=72:00:00          #Max Time Needed (HH:MM:SS)

python3 singleRunProcessor.py $1 $2
