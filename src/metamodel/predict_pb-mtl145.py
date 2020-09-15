import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pdb
import sys
sys.path.append('../data')
from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune2, parseMatricesFromSeqs
import torch
from metadata_ops import nhd2nhdhr
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal_
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr
from joblib import dump, load
import re
# build models to predict which models to use for transfer learning



ids = pd.read_csv('../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../results/glm_transfer/glm_meta_train_data.feather")
train_site_nhd = np.array(['test_nhdhr_'+x for x in train_df['site_id'].values])
result_df = pd.read_csv("../../results/glm_transfer/RMSE_transfer_test_extended_glm.csv")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)
test_lakes = ids[~np.isin(ids, train_lakes)]
test_site_nhd = np.array(['test_nhdhr_'+x for x in test_lakes])





feats = ['n_obs', 'obs_temp_mean', 'obs_temp_skew', 'obs_temp_kurt',
       'obs_temp_mean_airdif', 'dif_max_depth',
       'dif_surface_area', 'dif_sw_mean_au', 'dif_ws_mean_au',
       'dif_lathrop_strat', 'dif_glm_strat_perc', 'ad_glm_strat_perc','perc_dif_max_depth',
       'perc_dif_surface_area']


#metamodel

train = False
if train:
	model = GradientBoostingRegressor(n_estimators=3700,learning_rate=0.05)
	X = pd.DataFrame(train_df[feats])
	y = np.ravel(pd.DataFrame(train_df['rmse']))

	# model = GradientBoostingRegressor(n_estimators=3100, learning_rate=.05)
	model.fit(X,y)
	dump(model, 'metamodel_glm_RMSE_GBR.joblib') 



model = load('metamodel_glm_RMSE_GBR.joblib') 
###############################################################

#####feature importance##############################
# forest = model
# importances = forest.feature_importances_
# print(importances)
# sys.exit()
# # std = np.std([tree.feature_importances_ for tree in forest.estimators_],
# #              axis=0)
# indices = np.argsort(importances)[::-1]

# Print the feature ranking
# print("Feature ranking:")
# X = X_trn
# feats = [rmse_feats[i] for i in indices[:14]]
# for f in range(X.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# print(feats)
# plt.bar(range(14), importances[indices][:14],
#        color="r", align="center")
# plt.xticks(range(14), feats, rotation='vertical')
# plt.xlim([-1, 14])
# # plt.show()
# plt.tight_layout()
# plt.savefig("./feat_importances_glm.png")
# sys.exit()

############################################################################
#use model to predict top GLM source model for every target lake
################################################################################

#data structs to fill
rmse_per_lake = np.empty(test_lakes.shape[0])
glm_rmse_per_lake = np.empty(test_lakes.shape[0])
meta_rmse_per_lake = np.empty(test_lakes.shape[0])
top_rmse_per_lake = np.empty((test_lakes.shape[0], 10))
rmse_per_lake[:] = np.nan
glm_rmse_per_lake[:] = np.nan
srcorr_per_lake = np.empty(test_lakes.shape[0])
srcorr_per_lake[:] = np.nan
meta_rmse_per_lake[:] = np.nan

#initialize csv struct
csv = [ 'target_id,source_id,pb-mtl_rmse,predicted_rmse , n_obs ,  obs_temp_mean ,  obs_temp_skew , obs_temp_kurt ,\
        obs_temp_mean_airdif ,  dif_max_depth ,  dif_surface_area ,\
        dif_sw_mean_au ,  dif_ws_mean_au ,  dif_lathrop_strat , dif_glm_strat_perc ,  ad_glm_strat_perc, \
        perc_dif_max_depth ,  perc_dif_surface_area' ]

for targ_ct, target_id in enumerate(test_lakes): #for each target lake
	print("target lake ", targ_ct, ":", target_id)

	lake_df = pd.read_feather("../../metadata/diff/target_"+ target_id +"_pball_Aug2020.feather")

	lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes)]

	lake_df['site_id2'] = ['nhdhr_'+x for x in lake_df['site_id'].values] 

	targ_result_df = result_df[np.isin(result_df['target_id'], 'test_nhdhr_'+target_id)] 

	lake_df = lake_df.merge(targ_result_df, left_on='site_id2', right_on='source_id')

	X = pd.DataFrame(lake_df[feats])

	y_pred = []
	top_ids = []

	y_pred = model.predict(X)
	lake_df['rmse_pred'] = y_pred

	y_act = lake_df['rmse']

	y_pred = y_pred[np.isfinite(y_act)]
	y_act = y_act[np.isfinite(y_act)]
	meta_rmse_per_lake[targ_ct] = np.median(np.sqrt(((y_pred - y_act) ** 2).mean()))
	srcorr_per_lake[targ_ct] = spearmanr(y_pred, y_act).correlation

	print("meta rmse: ", meta_rmse_per_lake[targ_ct])
	print("srcorr: ", srcorr_per_lake[targ_ct])


	lake_df.sort_values(by=['rmse_pred'], inplace=True)
	best_predicted = lake_df.iloc[0]['site_id']
	y_act_top = float(lake_df.iloc[0]['rmse'])
	print("rmse: ", y_act_top)
	rmse_per_lake[targ_ct] = y_act_top

	best_predicted_rmse = lake_df.iloc[0]['rmse_pred']
	# lake_df = lake_df[lake_df['rmse_pred'] < lowest_rmse+rel_cut]
	for i in range(145):
		csv.append(",".join(['nhdhr_'+str(target_id), 'nhdhr_'+str(lake_df.iloc[i]['site_id']), str(lake_df.iloc[i]['rmse']), str(lake_df.iloc[i]['rmse_pred'])]+ [str(x) for x in lake_df.iloc[0][feats].values]))




#write to file
with open('pbmtl_all_sources_with_predictions.csv','w') as file:
    for line in csv:
        file.write(line)
        file.write('\n')
