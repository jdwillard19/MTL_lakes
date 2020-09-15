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
from sklearn.feature_selection import RFECV
import math
from random import random
from copy import deepcopy
import re


# print(train_lakes.shape[0], " training lakes")

glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../results/transfer_learning/glm/train_rmses_pball.feather")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)
feats = train_df.columns[80:-1]
# ranks = np.array([ 2,  4,  8,  1,  8, 11,  3,  1,  4,  6,  9,  7,  2, 12, 11,  8,  1,
#         2,  3, 17, 12,  1,  1, 14,  8, 21, 13, 23, 18, 24, 23, 19, 22, 15,
#        19, 21, 14,  9,  7, 22, 11, 23, 17,  5, 13, 15,  6,  4, 18, 22, 19,
#        12,  4, 24,  5, 16, 14, 20, 20, 13,  9,  3, 16,  3, 10, 10, 10, 20,
#        18,  7,  9,  1,  6,  1, 13, 12, 14, 24, 21,  5, 20, 22, 15, 24, 17,
#        15, 16, 19, 18, 23, 16, 11,  6, 17, 21,  5,  7, 10,  1,  1,  2,  1])
# # ranks = np.array([ 1,  4, 11,  1,  6,  8,  4,  1,  4,  6,  8,  7,  2, 11, 12, 10,  1,
# #         2,  3, 16, 12,  1,  1, 14,  7, 20, 13, 23, 16, 24, 23, 17, 22, 17,
# #        19, 21, 14, 10,  8, 22, 11, 23, 15,  5, 13, 15,  5,  5, 17, 22, 16,
# #        12,  3, 24,  6, 20, 14, 18, 21, 13,  7,  5, 19,  3, 10,  9, 11, 20,
# #        18,  7,  8,  1,  6,  2, 13, 12, 14, 24, 21,  3, 19, 22, 15, 24, 20,
# #        16, 19, 18, 18, 23, 15,  9,  9, 17, 21,  4,  9, 10,  1,  1,  2,  1])
ranks = np.array([1, 10, 19, 8, 4, 12, 2, 1, 4, 8, 7, 17, 10, 26, 7, 25, 2, 3, 11, 39, 39, 1, 1, 21
, 29, 46, 17, 38, 36, 47, 40, 40, 41, 42, 30, 20, 33, 14, 16, 43, 28, 44, 22, 14, 23, 38, 9, 5
, 21, 41, 42, 25, 5, 44, 13, 37, 19, 35, 28, 18, 16, 26, 37, 9, 24, 6, 23, 32, 34, 31, 11, 1
, 29, 6, 20, 36, 18, 47, 34, 15, 31, 45, 24, 46, 12, 32, 30, 33, 27, 43, 45, 15, 13, 35, 27, 1
, 1, 22, 1, 1, 3, 1])
print(feats[ranks==1])
# sys.exit()
############################################################################
#use CV to find which to drop
################################################################################

train_df = pd.DataFrame()
# ranks = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 5, 1, 5, 2, 1, 4, 1, 1, 1, 1, 1, 1 \
# , 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 4, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1 \
# , 1, 1, 1, 5, 1, 1, 1, 1, 1, 4, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
# pdb.set_trace()


for _, lake_id in enumerate(train_lakes):

	new_df = pd.DataFrame()
	# temp_df = pd.read_csv("../../../metadata/diff/target_"+lake_id+"_wStats.csv")
	# temp_df.drop(np.where(np.isin(temp_df['nhd_id'], temp_test_lakes[0]) == True)[0][0], inplace=True)

	lake_df_res = pd.read_csv("../../results/transfer_learning/target_"+lake_id+"/results_all_source_models.csv") 
	lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']

	lake_df = pd.read_feather("../../metadata/diff/target_"+lake_id+"_pball_Aug2020.feather")

	# lake_df = lake_df[np.isin(lake_df['source_id'], train_lakes)]
	lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes)]
	lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
	lake_df = pd.merge(left=lake_df, right=lake_df_res.astype('object'), left_on='site_id', right_on='source_id')
	# lake_df = pd.concat([lake_df, lake_df_res['rmse']], axis=1)
	# glm_uncal_rmse = float(metadata[metadata['nhd_id']==int(lake_id)].glm_uncal_rmse)
	# temp_df['rmse_improve'] = temp_df['rmse'] - glm_uncal_rmse
	new_df = lake_df
	train_df = pd.concat([train_df, new_df], ignore_index=True)




est = GradientBoostingRegressor(n_estimators=800, learning_rate=0.1)
X = pd.DataFrame(train_df[feats])
y = np.ravel(pd.DataFrame(train_df['rmse']))
# y = [np.log(float(x)) for x in y.values]
# y = [np.log(float(x)) for x in y.values]

rfecv = RFECV(estimator=est, cv=24, step=2, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
print("ranking: ", rfecv.ranking_)

print("scores: ", rfecv.grid_scores_)

print("ranking: ", repr(rfecv.ranking_))

print("scores: ", repr(rfecv.grid_scores_))





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