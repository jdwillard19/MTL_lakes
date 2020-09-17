Meta Transfer Learning application for water temperature prediction of unmonitored lakes
==============================




Steps to run MTL pipeline
------------

1. Install necessary dependencies from yml file (Anaconda must be installed for this), and activate conda environment
`conda env create -f mtl_env.yml`
`conda activate mtl_env`

2. Pull raw data from Sciencebase
`Rscript pull_data.r`


3. Process zipped data, format for preprocessing with the following two scripts
`cd src/data`
`python process_zip_data.py`
`python process_meteo_obs.py`

4. Run main preprocessing script (still in src/data/ folder) and also grab morphometry from config files
`python preprocess.py`
`python preprocess_morphometry.py`







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
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download, preprocess, or operate on data
            |
            ├── pull_data.r <- Pull raw data from Sciencebase
            ├── process_zip_data.py <- process zipped Sciencebase data
            ├── process_meteo_obs.py <- create lake-specific meteo and obs files
            ├── preprocess.py <- main preprocessing script
            └── preprocess_morphometry.py <- parse lake geometries for modeling
    
    │   │
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
