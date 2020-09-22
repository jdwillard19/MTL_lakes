#!/bin/bash -l
#PBS -l walltime=23:59:00,nodes=1:ppn=24:gpus=2,mem=16gb 
#PBS -m abe 
#PBS -N 0b5e30c9-8c3f-401b-9931-91ac59b2662a_pgml_sparse 
#PBS -o 0b5e30c9-8c3f-401b-9931-91ac59b2662a_pgml_sparse.stdout 
#PBS -q k40 

source takeme_source.sh

source activate mtl_env
python train_PGDL_custom_sparse.py 0b5e30c9-8c3f-401b-9931-91ac59b2662a