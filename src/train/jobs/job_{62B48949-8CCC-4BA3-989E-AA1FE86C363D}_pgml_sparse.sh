#!/bin/bash -l
#PBS -l walltime=23:59:00,nodes=1:ppn=24:gpus=2,mem=16gb 
#PBS -m abe 
#PBS -N 62B48949-8CCC-4BA3-989E-AA1FE86C363D_pgml_sparse 
#PBS -o 62B48949-8CCC-4BA3-989E-AA1FE86C363D_pgml_sparse.stdout 
#PBS -q k40 

source takeme_source.sh

source activate mtl_env
python train_PGDL_custom_sparse.py {62B48949-8CCC-4BA3-989E-AA1FE86C363D}