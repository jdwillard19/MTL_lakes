lake_monitoring_project
==============================

physics-guided neural network for lake temperature and related methods

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    |       └──mendota.mat <- Lake mendota matlab dataset for observed data
    |       └──mendota_sampled.mat <- lake mendota sampled output from GLM
    │   └── raw            <- The original, immutable data dump.
    |       └──Mendota_hypsography.csv <- depth area information for lake |                                     mendota
    │    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc. |                         (currently unused)
    │   └── figures        <- Generated graphics and figures to be used in reporting(currently unused)
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download, preprocess, or operate on data
    │   │   └── preprocess.py
    │   │   └── make_dataset.py
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    |           └──ec_experiment.py <- initial energy conservation experiment
    │   │   └── predict_model.py
    │   │
    │   └──scripts
    |       |
    |       └──experiments <- code for experiments
    |           |
    |           └──ec_experiment.py <- initial energy conservation experiment
    |       └──exploratory <- sandbox where I try out new things
    |       └──one-off     <- scripts for persistant data operations
    |           |
    |           └──createDepthAreaData.py <- adds hypsography data to dataset
    |           └──createDepthAreaPercentData.py <- addspercentage hypsography data
    |           └──interpolateXcDoyDepthDatenum.py <- interpolates data missing at lower depths so we have even sequences per depth
    |           └──RNN_preprocess.py <- necessary to run any of the data operations to build any sequence datasets for RNN
    |           └──unstandardizeXcDoy.py <- reverts features to pre-standardization values and adds to dataset


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
