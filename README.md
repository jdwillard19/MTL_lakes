Meta Transfer Learning application for water temperature prediction of unmonitored lakes. 

==============================

This code uses data from 2233 total lakes in the Midwestern United States to demonstrate a method for predicting water temperature at depth in situations where no temperature data is available. Method descriptions can be found in "<preprint link here>"



Steps to run MTL pipeline
------------

1. Install necessary dependencies from yml file (Anaconda must be installed for this), and activate conda environment
`conda env create -f mtl_env.yml`
`conda activate mtl_env`

2. Pull raw data from Sciencebase
`Rscript pull_data.r`


3. Process zipped data (code in src/data/), format for preprocessing with the following two scripts
`python process_zip_data.py`
`python process_meteo_obs.py`

4. Run main preprocessing script (code in src/data) and also grab morphometry from config files
`python preprocess.py`
`python preprocess_morphometry.py`

5. Create metadata files (code in src/metadata), must run in order
`python calculateMetadata.py`
`python createMetadataDiffs.py`
`python createMetadataDiffsPB.py`

6. Train source models (formatted for running on HPC, may have to customize to different HPC cluster if not on UMN MSI) (code in src/train/)
`python job_create_source_models.py` (create HPC job files (n=145))
`qsub /jobs/qsub_script_pgml_source.sh` (mass submit script created in previous command)

7. Run feature selection scripts for metamodeling (code in src/metamodel/). Record results for use in Step 8
`python pbmtl_feature_selection.py`
`python pgmtl_feature_selection.py`

8. Run hyperparameter optimization for metamodel build on features found in Step 7 (must manually past in code, directions in comments of below scripts). Record optimal hyperparameters for use in Step 9
`python pbmtl_hyperparameter_search.py`
`python pgmtl_hyperparameter_search.py`

9.Train the metamodel using the features found in Step 7 and hyperparameters found in Step 8 (must manually past in code, directions in comments of below scripts) (code in src/metamodel/)
`python pbmtl_train_metamodel.py`
`python pgmtl_train_metamodel.py`

10. Find the optimal number "k" of lakes to ensemble for PG-MTL9, using features found in Step 7 (must manually past in code, directions in comments of below scripts). Can be run simultaneously as Step 9. (code in src/metamodel/)
`python findEnsembleK.py`

11. Using hyperparameters found in Step 8, Experiments 1 and 3 can now be performed using the following scripts (code in src/evaluate/)
`python predict_pb-mtl.py` - process-based model selection, single source
`python predict_pb-mtl145.py` -  process-based model selection, all possible  sources
`python predict pb_mtl_extended` - process-based model selection for expanded test set of 1882 lakes
`python predict_pg-mtl.py` - PGDL model selection, single source
`python predict_pg-mtl9.py` - PGDL model selection, ensembled sources (specify k in file)
`python predict_pg-mtl145.py` - PGDL run all PGDL sources on targets


12. For Experiment 2, build additional PGDL models with different levels of sparsify (formatted for running on HPC, may have to customize to different HPC cluster if not on UMN MSI) (code in src/train/))
`python job_create_pgml_sparse.py`
`qsub /jobs/qsub_script_pgml_sparse.sh` (mass submit script created in previous command)

13. Compile results for completed jobs in Step 12 (code in src/evaluate/), for use in Experiment 2
`python runCustomSparseModels.py`











Project Organization (note: \[lake_id*\] can be any number of lake site ids)
------------

    ├── LICENSE
    ├── README.md           <- The top-level README for developers using this project.
    ├── data/               <- Primary data storage directory
        |
        ├── processed/      <- The final, canonical data sets for modeling.
        └── raw/            <- The original, immutable data dump.
            └── sb_mtl_data_release <- contains all files downloaded from USGS data release
    |
    ├── models/             <- Trained and serialized models, model predictions, or model summaries
        |
        └── [lake_id*]/    <- Models specific to individual lake
    │
    ├── results/            <- Experiment results and results used to train metamodels 
        |
        ├── [lake_id*]/    <- Results specific to individual lake
        ├── glm_transfer/    <- results of GLM parameter transfer between lakes
        └── transfer_learning <- results of PGDL transfers between lakes
            |
            └── target_[lake_id*] <- results of transfering to particular target lake
    │
    ├── src                <- Source code for use in this project.
        │
        ├── data           <- Scripts to download, preprocess, or operate on data
            |
            ├── pull_data.r <- Pull raw data from Sciencebase
            ├── process_zip_data.py <- process zipped Sciencebase data
            ├── process_meteo_obs.py <- create lake-specific meteo and obs files
            ├── preprocess.py <- main preprocessing script
            └── preprocess_morphometry.py <- parse lake geometries for modeling
    
        ├── train           <- Scripts to train PGDL models
            |
            ├── job_create_source_models.py - create HPC jobs for source PGDL models
            ├── train_source_model.py - trains a source PGDL for a given lake
            ├── job_create_pgml_sparse.py - create HPC jobs for PGDL w/sparse data
            └── train_PGDL_custom_sparse.py - trains a source PGDL for a given lake on varied sparse data
        | 
        ├── metadata         <- Scripts to create metadata for metamodel 
            |
            ├── runSourceModelsOnAllSources.py - create performance metadata
            ├── calculateMetadata.py - record metadata for each lake
            ├── createMetadataDiffs.py - record metadata differences between lakes
            └── createMetadataDiffsPB.py - record metadata differences between lakes for PB-MTL training
    models to make
        |
        ├── metamodel         <- Scripts to build metamodel  
            |
            ├── pbmtl_feature_selection.py - feature selection for pb-mtl
            ├── pgmtl_feature_selection.py - feature selection for pg-mtl
            ├── pbmtl_hyperparameter_search.py - find params for pb-mtl
            ├── pgmtl_hyperparameter_search.py - find params for pg-mtl
            ├── pbmtl_train_metamodel.py - train pb-mtl metamodel
            └── pgmtl_train_metamodel.py - train pg-mtl metamodel
        |
        ├── evaluate         <- Scripts to evaluate operations framework performance
            |
            ├── runCustomSparseModels.py      <- evaluation of sparse models
            ├── predict_pb-mtl.py             <- evaluate single source PB-MTL
            ├── predict_pb-mtl145.py          <- evaluate all sources PB-MTL
            ├── predict_pb-mtl_expanded.py    <- evaluate PB-MTL expanded test set
            ├── predict_pg-mtl.py             <- evaluate single source PG-MTL
            ├── predict_pg-mtl9.py            <- evaluate ensemble source PB-MTL  
            └── predict_pg-mtl9_expanded.py   <- evaluate ensemble source PB-MTL on expanded test set
    
        |
        └──  models         <- Scripts to support operations on models 
            |
            └── pytorch_model_operations <- helper functions for model ops 
         



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
