import pandas as pd
import numpy as np
import pdb
import sys
sys.path.append('../data')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFECV
import math
import re 

#load GLM transfer 
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../results/glm_transfer/glm_meta_train_data.feather")

train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)



############################################################################
#use CV to find which to drop
################################################################################
pdb.set_trace()
feats = train_df.columns[81:-1]


est = GradientBoostingRegressor(n_estimators=800,learning_rate=0.1)
X = pd.DataFrame(train_df[feats])
y = np.ravel(pd.DataFrame(train_df['rmse']))
#ENABLE FOR SHORT DEBUG RUN
rfecv = RFECV(estimator=est, cv=24, step=2, scoring='neg_mean_squared_error', verbose=1, n_jobs=24)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
print("ranking: ", rfecv.ranking_)

print("scores: ", rfecv.grid_scores_)



print(feats[rfecv.ranking_ == 1])

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