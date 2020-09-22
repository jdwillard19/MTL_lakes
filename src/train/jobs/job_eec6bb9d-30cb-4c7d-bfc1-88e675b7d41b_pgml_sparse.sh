#!/bin/bash -l
#PBS -l walltime=23:59:00,nodes=1:ppn=24:gpus=2,mem=16gb 
#PBS -m abe 
#PBS -N eec6bb9d-30cb-4c7d-bfc1-88e675b7d41b_pgml_sparse 
#PBS -o eec6bb9d-30cb-4c7d-bfc1-88e675b7d41b_pgml_sparse.stdout 
#PBS -q k40 

source takeme_source.sh

source activate mtl_env
python train_PGDL_custom_sparse.py eec6bb9d-30cb-4c7d-bfc1-88e675b7d41b