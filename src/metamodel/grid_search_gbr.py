import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pdb
import sys
sys.path.append('../data')
from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune2, parseMatricesFromSeqs
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal_
from sklearn.ensemble import GradientBoostingRegressor
from similarity_calc import getSimilarLakes
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from metric_learn import MLKR
from sklearn.model_selection import GridSearchCV

#get all lake names, divide into train and test
metadata = pd.read_feather("../../metadata/lake_metadata_wNew2.feather")
lakenames = np.array([str(i) for i in metadata.iloc[:,0].values])# to loop through all lakes
metadata.set_index("nhd_id", inplace=True)
metadata.columns = [c.replace(' ', '_') for c in metadata.columns]

trn_tst_f = pd.read_feather("../../data/processed/lake_splits/trainTestNewLakes.feather")

train_lakes = trn_tst_f[trn_tst_f['isTrain']]['id'].values
test_lakes = trn_tst_f[~trn_tst_f['isTrain']]['id'].values
to_del = ['120018008', '120020307', '120020636', '32671150', '58125241', '120020800', '91598525']
train_lakes = np.setdiff1d(train_lakes, to_del)
# train_lakes = lakenames
n_train_lakes = len(train_lakes)
#model params
seq_length = 350 #how long of sequences to use in model
begin_loss_ind = 175#index in sequence where we begin to calculate error or predict
n_features = 8  #number of physical drivers
win_shift = 175 #how much to slide the window on training set each time

feats = ['n_obs', 'n_obs_sp', 'obs_depth_frac', 'obs_temp_std',
       'obs_temp_mean_airdif', 'dif_SDF', 'dif_max_depth', 'dif_surface_area',
       'dif_rain_mean_sp', 'dif_rain_mean_su', 'dif_lathrop_strat',
       'dif_glm_strat_perc', 'perc_dif_max_depth',
       'perc_dif_sqrt_surface_area']
#compile training data from all training lakes
train_df = pd.read_feather("../../results/transfer_learning/glm/train_rmses.feather")
# X = pd.DataFrame(train_df[rmse_feats])
# y = pd.DataFrame(train_df['rmse'])
#ENABLE FOR SHORT DEBUG RUN
# rfecv = RFECV(estimator=est, cv=24, step=4, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

#declare model
# model = RandomForestRegressor(max_depth=None,random_state=1, n_estimators=10000, max_features=3, n_jobs=-1)
# param = [.09,.11,.08,.12,.07,.13,.06,.14,.05,.15,.04,.16,.03,.17,.02,.18,.01,.19,.20,.21,.22]
# param = np.arange(1,50)
# param = np.arange(.05,.20, .01)
# # param = np.arange(500,10000, 500)
X_trn = pd.DataFrame(train_df[feats])
y_trn = np.ravel(pd.DataFrame(train_df['rmse']))
# model = GradientBoostingRegressor(n_estimators=4000, learning_rate=.08, max_depth=3)
# model = RandomForestRegressor(n_estimators=4000, learning_rate=.08, max_depth=3)
def gb_param_selection(X, y, nfolds):
    ests = np.arange(250,8000,500)
    lrs = [.1]
    param_grid = {'n_estimators': ests, 'learning_rate' : lrs}
    grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    grid_search.best_params_
    print("BEST SCORE: ", grid_search.best_score_)
    return grid_search.best_params_



print(gb_param_selection(X_trn, y_trn, 10))

# for i, p in enumerate(param):

# 	model = GradientBoostingRegressor(n_estimators=3500, learning_rate=p, max_depth=3)
# # model = MLKR(
# 	cvscore = cross_val_score(model, X_trn, np.ravel(y_trn), cv=n_train_lakes, n_jobs=-1).mean()
# 	# print("cv: ", cvscore)
# 	print("param=", p, "->CVscore: ",cvscore)

