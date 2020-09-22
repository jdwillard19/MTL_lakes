import pandas as pd
import numpy as np
import pdb
import sys
sys.path.append('../data')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFECV
import math
import re 



############################################################################
# (Sept 2020 - Jared) PB-MTL feature selection (RFECV)
################################################################################

#load GLM transfer 
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../results/glm_transfer/glm_meta_train_data.feather")

train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)



#candidate features
feats = train_df.columns[81:-1]

#training data
est = GradientBoostingRegressor(n_estimators=800,learning_rate=0.1)
X = pd.DataFrame(train_df[feats])
y = np.ravel(pd.DataFrame(train_df['rmse']))

#do recursive feature elimination cross validation (RFECV)
rfecv = RFECV(estimator=est, cv=24, step=2, scoring='neg_mean_squared_error', verbose=1, n_jobs=24)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)


print("ranking: ", rfecv.ranking_)

print("scores: ", rfecv.grid_scores_)


#print final features
print(feats[rfecv.ranking_ == 1])
