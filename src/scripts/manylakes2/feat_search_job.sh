#!/bin/bash -l
#PBS -l walltime=23:59:00,nodes=1:ppn=24:gpus=2,mem=16gb 
#PBS -m abe 
#PBS -N grid_search_glm_090820 
#PBS -o grid_search_glm_090820.stdout 
#PBS -q k40

cd ./research/lake_modeling/src/scripts/manylakes2/
source activate pytorch_new3
python grid_search_glm.py
