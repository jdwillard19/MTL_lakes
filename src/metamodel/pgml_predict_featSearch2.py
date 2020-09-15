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
from sklearn.linear_model import LinearRegression
import math
from random import random
from copy import deepcopy
import re
from joblib import dump, load


metadata = pd.read_feather("../../metadata/lake_metadata_2700plus.feather")
ids = pd.read_csv('../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
# print(train_lakes.shape[0], " training lakes")
data = pd.read_csv("../scripts/bean_plot_data.csv")
pgml_data = pd.read_feather("../../results/customSparseResults.feather")

glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../results/transfer_learning/glm/train_rmses_pball.feather")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
test_lakes = ids[~np.isin(ids, train_lakes)]
n_lakes = len(train_lakes)
feats = metadata.columns[2:].drop(['fullname','canopy'])


#scores:  array([ -0.68659815,  -0.72257121,  -0.74890066,  -0.83599025,
       #  -0.85464896,  -0.81669765,  -0.68492737,  -0.95000229,
       #  -0.82382909,  -0.86485528,  -0.94709046,  -1.63610654,
       #  -2.77388721,  -3.96870469,  -7.80419054, -12.52510872,
       # -13.52180269, -16.3289937 , -20.584997  , -16.30077677,
       # -28.45866809, -33.84999354, -28.56021154, -38.34161787])


df = pd.DataFrame()
y_si = data['single_rmse'].values[:145]
y_en = data['ens_rmse'].values[:145]
y_pb0 = data['glm_rmse'].values[:145]
y_pb = data['glm_t_rmse'].values[:145]
ids = data['target_id'].values[:145]
train_lakes = data['target_id'].values[:145]
test_lakes = test_lakes[~np.isin(test_lakes, train_lakes)]


# ranks = np.array([1,6,7,3,3,7,5,1,5,7,9,11,4,12,5,11,3,2,6,20,18,1,1,9\
# ,17,24,15,22,18,25,23,22,20,22,13,19,20,14,12,22,16,21,19,6,9,16,4,8\
# ,17,23,15,10,10,24,4,21,14,21,13,15,12,8,19,11,8,9,13,21,18,15,6,2\
# ,10,5,14,20,16,25,23,4,16,23,11,25,8,19,14,18,17,25,24,12,13,24,17,7\
# ,3,10,1,2,2,1])
# print(feats[ranks==1])
# sys.exit()
############################################################################
#use CV to find which to drop
################################################################################

train_df = pd.DataFrame()
# ranks = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 5, 1, 5, 2, 1, 4, 1, 1, 1, 1, 1, 1 \
# , 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 4, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1 \
# , 1, 1, 1, 5, 1, 1, 1, 1, 1, 4, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
# pdb.set_trace()
# ranks = np.array([15, 16, 16, 10, 11, 16, 18,  1, 12,  1, 11,  1,  7,  1,  9,  1,  7,  3,  4,  2,  6,  1, 14,  1,
#  10,  1, 12,  1, 13,  1,  8,  3,  5,  2,  6,  1, 15,  1, 11,  1,  8,  1,  9,  1,  7,  3,  5,  2,
#   6,  1, 13,  1, 13,  1,  7,  1, 10,  1, 12,  3,  5,  4,  4,  1, 13,  1,  9,  1,  9,  1, 10,  1,
#   8,  4,  5,  2,  6, 17, 18, 18, 17, 18, 17, 12, 15, 17, 14, 14, 16, 11, 15, 14,  8])


# feats = feats[ranks2 == 1]
ranks2 = np.array([78, 81, 79, 71, 55, 77, 90,  1, 66, 10, 50, 14, 46,  3, 65, 21, 44,
       24, 34, 29, 39,  1, 69,  9, 57, 15, 53,  5, 68, 19, 45, 23, 38, 30,
       40,  1, 73, 11, 52, 13, 56,  4, 63, 18, 43, 25, 37, 28, 42,  1, 80,
        8, 54, 16, 48,  6, 62, 20, 60, 26, 35, 32, 33,  2, 67, 12, 51, 17,
       47,  7, 59, 22, 58, 27, 36, 31, 41, 88, 84, 85, 89, 87, 86, 74, 76,
       72, 82, 83, 70, 49, 75, 61, 64])
feats = feats[ranks2 == 1]

for _, lake_id in enumerate(train_lakes):

	new_df = pd.DataFrame()
	new_df[feats] = metadata[metadata['site_id'] == lake_id][feats]
	# new_df['rmse'] = y_en[ids == lake_id]

	new_df['rmse'] = pgml_data[pgml_data['site_id'] == lake_id]['50 obs median'].values[0]
	train_df = pd.concat([train_df, new_df], ignore_index=True)




est = LinearRegression()
X = pd.DataFrame(train_df[feats])
y = pd.DataFrame(train_df['rmse'])
est.fit(X,y)
# dump(est, "./ens_pred_linear_model")
dump(est, "./pgdl_pred_linear_model")
sys.exit()

#ENABLE FOR SHORT DEBUG RUN
rfecv = RFECV(estimator=est, cv=145, step=1, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
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