import pandas as pd
import numpy as np
import pdb
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFECV
import math
import re


###################################################################################################
# (Sept 2020 - Jared) - This script performs feature selection for PG-MTL metamodel
##############################################################################################


#load source lake list
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../results/transfer_learning/glm/train_rmses_pball.feather")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
train_lakes_wp = np.unique(glm_all_f['target_id'].values)
n_lakes = len(train_lakes)
feats = train_df.columns[80:-1]

train_df = pd.DataFrame()


#compile all the meta-features and meta-target values into one dataframe
for _, lake_id in enumerate(train_lakes):

	new_df = pd.DataFrame()

	#get performance results (metatargets), filter out target as source
	lake_df_res = pd.read_csv("../../results/transfer_learning/target_"+lake_id+"/resultsPGRNNbasic_pball",header=None,names=['source_id','rmse'])
	lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']

	#get metadata differences between target and all the sources
	lake_df = pd.read_feather("../../metadata/diffs/target_nhdhr_"+lake_id+".feather")
	lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes_wp)]
	lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
	lake_df_res['source_id2'] = ['nhdhr_'+str(x) for x in lake_df_res['source_id'].values]
	lake_df = pd.merge(left=lake_df, right=lake_df_res.astype('object'), left_on='site_id', right_on='source_id2')
	new_df = lake_df
	train_df = pd.concat([train_df, new_df], ignore_index=True)



#declare model and predictors and response
est = GradientBoostingRegressor(n_estimators=800, learning_rate=0.1)
X = pd.DataFrame(train_df[feats])
y = np.ravel(pd.DataFrame(train_df['rmse']))

#perform recursive feature elimination
rfecv = RFECV(estimator=est, cv=24, step=2, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
print("ranking: ", rfecv.ranking_)

print("scores: ", rfecv.grid_scores_)

print("ranking: ", repr(rfecv.ranking_))

print("scores: ", repr(rfecv.grid_scores_))

print("selected features\n---------------------------------------------------------------------------")
print(feats[rfecv.ranking_==1])
print("------------------------------------------------------------------------")




