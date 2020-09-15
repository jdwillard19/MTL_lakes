#!/bin/bash -l
#PBS -l walltime=23:59:00,nodes=1:ppn=24:gpus=2,mem=16gb 
#PBS -m abe 
#PBS -N run_all_sources 
#PBS -o run_all_sources.stdout 
#PBS -q k40

source takeme.sh
python model_selection_pgml_pball.py
