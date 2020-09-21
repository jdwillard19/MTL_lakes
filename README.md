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

10. Using hyperparameters found in Step 8, Experiments 1 and 3 can now be performed using the following scripts (code in src/evaluate/)
`python predict_pb-mtl.py`


11. For Experiment 2, build additional PGDL models with different levels of sparsify (formatted for running on HPC, may have to customize to different HPC cluster if not on UMN MSI) (code in src/train/))
`python job_create_`











Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    |       └──sb_mtl_data_release <- contains all files downloaded from USGS data release
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    │
    ├── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
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
        ├── metamodel         <- Scripts to build metamodel  
    |   |   ├── pbmtl_feature_selection.py - feature selection for pb-mtl
    |   |   ├── pgmtl_feature_selection.py - feature selection for pg-mtl
    |   |   ├── pbmtl_hyperparameter_search.py - find params for pb-mtl
    |   |   ├── pgmtl_hyperparameter_search.py - find params for pg-mtl
    |   |   ├── pbmtl_train_metamodel.py - train pb-mtl metamodel
    |   |   └── pgmtl_train_metamodel.py - train pg-mtl metamodel
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    |           └──ec_experiment.py <- initial energy conservation experiment
    │   │   └── predict_model.py
    │   │
    └──


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
