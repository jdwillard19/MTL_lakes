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




df = pd.DataFrame()
y_si = data['single_rmse'].values[:145]
y_en = data['ens_rmse'].values[:145]
y_pb0 = data['glm_rmse'].values[:145]
y_pb = data['glm_t_rmse'].values[:145]
ids = data['target_id'].values[:145]
y_si_ts = data['single_rmse'].values[145:]
y_en_ts = data['ens_rmse'].values[145:]
y_pb0_ts = data['glm_rmse'].values[145:]
y_pb_ts = data['glm_t_rmse'].values[145:]
ids_ts = data['target_id'].values[145:]
train_lakes = data['target_id'].values[:145]
test_lakes = data['target_id'].values[145:]
test_lakes = test_lakes[~np.isin(test_lakes, train_lakes)]


lin_ranks_ens = np.array([15, 16, 16, 10, 11, 16, 18,  1, 12,  1, 11,  1,  7,  1,  9,  1,  7,  3,  4,  2,  6,  1, 14,  1,
 10,  1, 12,  1, 13,  1,  8,  3,  5,  2,  6,  1, 15,  1, 11,  1,  8,  1,  9,  1,  7,  3,  5,  2,
  6,  1, 13,  1, 13,  1,  7,  1, 10,  1, 12,  3,  5,  4,  4,  1, 13,  1,  9,  1,  9,  1, 10,  1,
  8,  4,  5,  2,  6, 17, 18, 18, 17, 18, 17, 12, 15, 17, 14, 14, 16, 11, 15, 14,  8])
lin_ranks_pg = np.array([78, 81, 79, 71, 55, 77, 90,  1, 66, 10, 50, 14, 46,  3, 65, 21, 44,
       24, 34, 29, 39,  1, 69,  9, 57, 15, 53,  5, 68, 19, 45, 23, 38, 30,
       40,  1, 73, 11, 52, 13, 56,  4, 63, 18, 43, 25, 37, 28, 42,  1, 80,
        8, 54, 16, 48,  6, 62, 20, 60, 26, 35, 32, 33,  2, 67, 12, 51, 17,
       47,  7, 59, 22, 58, 27, 36, 31, 41, 88, 84, 85, 89, 87, 86, 74, 76,
       72, 82, 83, 70, 49, 75, 61, 64])

rf_ranks_pg = np.array([ 1,  1,  1,  1, 11,  2,  1,  1,  4, 13,  1, 16, 26, 28, 29, 17,  1,
        1,  1, 12, 19, 21,  1, 29,  9, 20, 28, 25, 16, 15, 25,  6, 15, 18,
        3,  1, 10,  9,  5,  7,  6, 30, 19,  4,  1,  1,  1,  1,  1, 14,  1,
        1, 11,  1, 14, 22, 21, 23, 17,  8,  2, 33, 32,  3,  1, 27,  5,  1,
       22, 26, 32, 31, 27, 30,  1, 24, 13, 24,  1, 20, 12, 18,  1,  1, 10,
        1,  8,  1, 31,  7, 33,  1, 23])
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
lin_ens_feats = feats[lin_ranks_ens == 1]
lin_pg_feats = feats[lin_ranks_pg == 1]
rf_ens_feats = feats[:]
rf_pg_feats = feats[rf_ranks_pg == 1]



train_df = pd.DataFrame()
# ranks = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 5, 1, 5, 2, 1, 4, 1, 1, 1, 1, 1, 1 \
# , 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 4, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1 \
# , 1, 1, 1, 5, 1, 1, 1, 1, 1, 4, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
# pdb.set_trace()
# ranks = np.array([15, 16, 16, 10, 11, 16, 18,  1, 12,  1, 11,  1,  7,  1,  9,  1,  7,  3,  4,  2,  6,  1, 14,  1,
#  10,  1, 12,  1, 13,  1,  8,  3,  5,  2,  6,  1, 15,  1, 11,  1,  8,  1,  9,  1,  7,  3,  5,  2,
#   6,  1, 13,  1, 13,  1,  7,  1, 10,  1, 12,  3,  5,  4,  4,  1, 13,  1,  9,  1,  9,  1, 10,  1,
#   8,  4,  5,  2,  6, 17, 18, 18, 17, 18, 17, 12, 15, 17, 14, 14, 16, 11, 15, 14,  8])

for _, lake_id in enumerate(test_lakes):

	new_df = pd.DataFrame()
	# temp_df = pd.read_csv("../../../metadata/diff/target_"+lake_id+"_wStats.csv")
	# temp_df.drop(np.where(np.isin(temp_df['nhd_id'], temp_test_lakes[0]) == True)[0][0], inplace=True)
	# lake_df_res = pd.read_csv("../../results/transfer_learning/target_"+lake_id+"/resultsPGRNNbasic_pball") 
	# lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']
	new_df[feats] = metadata[metadata['site_id'] == lake_id][feats]
	new_df['ens_rmse'] = y_en_ts[ids_ts == lake_id]
	new_df['pg_rmse'] = pgml_data[pgml_data['site_id'] == lake_id]['50 obs median'].values[0]

	# lake_df = pd.read_feather("../../metadata/diff/target_"+lake_id+"_pball_update.feather")

	# # lake_df = lake_df[np.isin(lake_df['source_id'], train_lakes)]
	# lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes)]
	# lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
	# lake_df = pd.merge(left=lake_df, right=lake_df_res, left_on='site_id', right_on='source_id')
	# lake_df = pd.concat([lake_df, lake_df_res['rmse']], axis=1)
	# glm_uncal_rmse = float(metadata[metadata['nhd_id']==int(lake_id)].glm_uncal_rmse)
	# temp_df['rmse_improve'] = temp_df['rmse'] - glm_uncal_rmse
	train_df = pd.concat([train_df, new_df], ignore_index=True)

def rmse(predictions, targets): 
    return np.sqrt(((predictions - targets) ** 2).mean())
#models
lin_model_ens = load("./ens_pred_linear_model")
lin_model_pg = load("./pgdl_pred_linear_model")
rf_model_pg = load("./pg_rf_model")
rf_model_ens = load("./ens_rf_model")

#predictors
X_lin_ens = pd.DataFrame(train_df[lin_ens_feats])
X_lin_pg = pd.DataFrame(train_df[lin_pg_feats])
X_rf_ens = pd.DataFrame(train_df[rf_ens_feats])
X_rf_pg = pd.DataFrame(train_df[rf_pg_feats])

#predictions
lin_ens_pred = lin_model_ens.predict(X_lin_ens)
lin_pg_pred = lin_model_pg.predict(X_lin_pg)
rf_ens_pred = rf_model_ens.predict(X_rf_ens)
rf_pg_pred = rf_model_pg.predict(X_rf_pg)

y_ens_act = pd.DataFrame(train_df['ens_rmse']).values
y_pg_act = pd.DataFrame(train_df['pg_rmse']).values

success_ct = 0
total_ct = len(test_lakes)
for i in range(len(test_lakes)):
	if lin_ens_pred[i] < lin_pg_pred[i]:
		if y_ens_act[i] < y_pg_act[i]:
			success_ct += 1
			continue
		else:
			tmp = 0
			continue
	else:
		if y_pg_act[i] < y_ens_act[i]:
			success_ct += 1

print("success rate ", success_ct, "/", total_ct)
print("lin_ens rmse", rmse(lin_ens_pred,y_ens_act))
print("rf_ens rmse", rmse(rf_ens_pred,y_ens_act))
print("lin_pg rmse", rmse(lin_pg_pred,y_pg_act))
print("rf_pg rmse", rmse(rf_pg_pred,y_pg_act))

# est.fit(X,y)
# dump(est, "./ens_pred_linear_model")
# sys.exit()

#ENABLE FOR SHORT DEBUG RUN
# rfecv = RFECV(estimator=est, cv=145, step=1, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
# rfecv.fit(X, y)

# print("Optimal number of features : %d" % rfecv.n_features_)

# # Plot number of features VS. cross-validation scores
# print("ranking: ", rfecv.ranking_)

# print("scores: ", rfecv.grid_scores_)

# print("ranking: ", repr(rfecv.ranking_))

# print("scores: ", repr(rfecv.grid_scores_))





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