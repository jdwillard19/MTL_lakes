import pandas as pd
import numpy as np
import pdb
import sys
sys.path.append('../../data')
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import re
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../results/glm_transfer/glm_meta_train_data.feather")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)

###########################################################################################3
# (Sept 2020 - Jared) - metamodel hyperparameter tuning using features found in "pbmtl_feature_selection.py"
######################################################################################

#########################################################################################
#paste features found in "pbmtl_feature_selection.py" here
feats = ['n_obs_su', 'obs_temp_mean', 'obs_temp_skew', 'obs_temp_kurt',
       'ad_glm_strat_perc', 'obs_temp_mean_airdif', 'dif_max_depth',
       'dif_surface_area', 'dif_sw_mean_au', 'dif_ws_mean_au',
       'dif_lathrop_strat', 'dif_glm_strat_perc', 'perc_dif_max_depth',
       'perc_dif_sqrt_surface_area']
###################################################################################

#training data
X = pd.DataFrame(train_df[feats])
y = np.ravel(pd.DataFrame(train_df['rmse']))

#cv params
nfolds = 24

def gb_param_selection(X, y, nfolds):
    ests = np.arange(1000,6000,500)
    lrs = [.05,.1]
    param_grid = {'n_estimators': ests, 'learning_rate' : lrs}
    grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

res = gb_param_selection(X, y, nfolds)

print(res)


