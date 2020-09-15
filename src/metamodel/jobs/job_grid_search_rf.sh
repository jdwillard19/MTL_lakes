#!/bin/bash -l
#PBS -l walltime=23:59:00,nodes=1:ppn=24:gpus=2,mem=16gb 
#PBS -m abe 
#PBS -N grid_search_rf_012720
#PBS -o grid_search_rf_012720.stdout 
#PBS -q k40

source takeme.sh
source activate pytorch4
python grid_search_rf.py
