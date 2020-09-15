import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pdb
import sys
sys.path.append('../../data')
from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune2, parseMatricesFromSeqs
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal_
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import re
# build models to predict which models to use for transfer learning
glm_all_f = pd.read_csv("../../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../../results/glm_transfer/glm_meta_train_data.feather")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)

# print(train_lakes.shape[0], " training lakes")


############################################################################
#use CV to find which to drop
################################################################################

feats = train_df.columns[81:-1]

feats = ['n_obs', 'obs_temp_mean', 'obs_temp_skew', 'obs_temp_kurt',
       'obs_temp_mean_airdif', 'dif_max_depth',
       'dif_surface_area', 'dif_sw_mean_au', 'dif_ws_mean_au',
       'dif_lathrop_strat', 'dif_glm_strat_perc', 'ad_glm_strat_perc','perc_dif_max_depth',
       'perc_dif_surface_area']

#training data
X = pd.DataFrame(train_df[feats])
y = np.ravel(pd.DataFrame(train_df['rmse']))

def gb_param_selection(X, y, nfolds):
    ests = np.arange(1000,6000,100)
    lrs = [.05,.1]
    param_grid = {'n_estimators': ests, 'learning_rate' : lrs}
    grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

res = gb_param_selection(X, y, 24)
print("DONE\nDONE\nDONE")

print(res)


