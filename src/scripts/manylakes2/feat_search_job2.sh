#!/bin/bash -l
#PBS -l walltime=23:59:00,nodes=1:ppn=24:gpus=2,mem=16gb 
#PBS -m abe 
#PBS -N feat_search_glm_090620_v
#PBS -o feat_search_glm_090620_v.stdout 
#PBS -q v100

cd ./research/lake_modeling/src/metalearning
source activate pytorch_new3
python glm_recursiveFeatureElimination_pball.py
