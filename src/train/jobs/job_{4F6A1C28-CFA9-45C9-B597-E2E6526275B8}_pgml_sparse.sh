#!/bin/bash -l
#PBS -l walltime=23:59:00,nodes=1:ppn=24:gpus=2,mem=16gb 
#PBS -m abe 
#PBS -N 4F6A1C28-CFA9-45C9-B597-E2E6526275B8_pgml_sparse 
#PBS -o 4F6A1C28-CFA9-45C9-B597-E2E6526275B8_pgml_sparse.stdout 
#PBS -q k40 

source takeme_source.sh

source activate mtl_env
python train_PGDL_custom_sparse.py {4F6A1C28-CFA9-45C9-B597-E2E6526275B8}