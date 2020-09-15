import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut
import pdb
import sys
sys.path.append('../data')
from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune2, parseMatricesFromSeqs
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal_
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from similarity_calc import getSimilarLakes
from sklearn.feature_selection import RFECV
import math
from random import random
from copy import deepcopy

err_cutoff = 0

#get all lake names, divide into train and test
metadata = pd.read_feather("../../metadata/lake_metadata_wStats2.feather")
lakenames = np.array([str(i) for i in metadata.iloc[:,0].values])# to loop through all lakes
metadata.set_index("nhd_id", inplace=True)
metadata.columns = [c.replace(' ', '_') for c in metadata.columns]
metadata = metadata.drop(['glm_uncal_rmse', 'canopy'], axis=1)
trn_tst_f = pd.read_feather("../../data/processed/lake_splits/trn_tst2.feather")

train_lakes = trn_tst_f[trn_tst_f['isTrain2']]['id'].values
test_lakes = trn_tst_f[~trn_tst_f['isTrain2']]['id'].values
to_del = ['120018008', '120020307', '120020636', '32671150', '58125241', '120020800', '91598525']
train_lakes = np.setdiff1d(train_lakes, to_del)
test_lakes = np.setdiff1d(test_lakes, to_del)

n_test_lakes = len(train_lakes)

# print(train_lakes.shape[0], " training lakes")


############################################################################
#use CV to find which to drop
################################################################################

metadata.drop('fullname', axis=1, inplace=True)
n_feat = metadata.columns.shape[0]
test_lakes = train_lakes

test_lake_csv = []
n_lakes = test_lakes.shape[0]
k_arr = np.arange(1,11)
# cutoff_arr = np.arange(1,12)
# cutoff_arr = np.array([5])
csv = []

err_array = np.empty((n_lakes, k_arr.shape[0]))
err_array[:] = np.nan
errs = []
#get metadata columns
temp_meta = pd.read_csv("../../metadata/diff/target_13293262_wStats_wSeason_wPercDiffDepth.csv")
temp_meta.set_index('nhd_id', inplace=True)
feat_cand_arr = temp_meta.columns
n_labels = feat_cand_arr.shape[0]
# curr_feats = []
feats = feat_cand_arr[76:]
# ranks = [1, 5, 5, 1, 4, 6, 1, 4, 1, 3, 3, 13, 5, 19, 7, 17, 1, 1, 2, 16, 22,  1, 1, 7 \
# , 12, 15, 15, 23, 20, 17, 22, 16, 18, 14, 22, 21, 17, 8, 9, 23, 10, 16, 19 , 5, 11, 3, 10, 1 \
# , 16, 21, 14, 8, 4, 17, 8, 18, 11, 21, 21, 12, 10, 1, 20, 3, 14, 4, 9, 9,9,  6, 13, 13 \
# , 11, 8, 6, 6, 7, 23, 19, 13, 14, 10, 15, 23, 19, 15, 22, 12, 18, 11, 18, 12, 7, 20, 20, 1 , 1, \
#  2, 1, 2, 2, 1]
# feats = ['n_obs', 'n_obs_sp', 'obs_depth_frac', 'obs_temp_std',
#        'obs_temp_mean_airdif', 'dif_SDF', 'dif_max_depth', 'dif_surface_area',
#        'dif_rain_mean_sp', 'dif_rain_mean_su', 'dif_lathrop_strat',
#        'dif_glm_strat_perc', 'perc_dif_max_depth',
#        'perc_dif_sqrt_surface_area']
pdb.set_trace()

print("feats ", feats)


train_df = pd.read_feather("../../results/transfer_learning/glm/train_rmses.feather")
est = GradientBoostingRegressor(n_estimators=2000)
X = pd.DataFrame(train_df[feats])
y = pd.DataFrame(train_df['rmse'])
#ENABLE FOR SHORT DEBUG RUN
rfecv = RFECV(estimator=est, cv=24, step=4, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
print("ranking: ", rfecv.ranking_)

print("scores: ", rfecv.grid_scores_)
pdb.set_trace()




# writeF.close()
# print("mean test RMSE: ",rmse_per_testlake.mean())
# 


# df['rmse_pred'] = y_pred
# df = df.sort_values(by=['rmse_pred'])
# print(df)
#assess performance of the model


# scores = []
# kfold = KFold(n_splits=10, shuffle=True, random_state=42)
# for i, (train, test) in enumerate(kfold.split(X, y)):
# 	model.fit(X.iloc[train,:], y.iloc[train,:])
# 	score = model.score(X.iloc[test,:], y.iloc[test,:])
# 	scores.append(score)