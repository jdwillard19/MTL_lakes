#!/bin/bash -l
#PBS -l walltime=23:59:00,nodes=1:ppn=24:gpus=2,mem=16gb 
#PBS -m abe 
#PBS -N E940A362-4076-4895-A23F-1B8CCC905DEE_pgml_sparse 
#PBS -o E940A362-4076-4895-A23F-1B8CCC905DEE_pgml_sparse.stdout 
#PBS -q k40 

source takeme_source.sh

source activate mtl_env
python train_PGDL_custom_sparse.py {E940A362-4076-4895-A23F-1B8CCC905DEE}