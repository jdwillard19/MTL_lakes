#!/bin/bash -l
#PBS -l walltime=23:59:00,nodes=1:ppn=24:gpus=2,mem=16gb 
#PBS -m abe 
#PBS -N 120020367_pgml_source 
#PBS -o 120020367_pgml_source.stdout 
#PBS -q k40 

source takeme_source.sh

source activate mtl_env
python train_source_model.py 120020367