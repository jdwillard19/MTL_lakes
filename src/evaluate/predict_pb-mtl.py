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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from scipy.stats import spearmanr
from joblib import dump, load
import re

################################################################################3
# (Sept 2020 - Jared) - evaluate PB-MTL model by predicting best source model for each of 305 test lakes
# Features and hyperparamters must be manually specified below 
# (e.g. feats = ['dif_max_depth', ....]; n_estimators = 4000, etc)
#############################################################################################################

#file to save results to
save_file_path = '../../results/pbmtl_glm_transfer_results.csv'



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



#########################################################################################
#paste features found in "pbmtl_feature_selection.py" here
feats = ['n_obs_su', 'obs_temp_mean', 'obs_temp_skew', 'obs_temp_kurt',
       'ad_glm_strat_perc', 'obs_temp_mean_airdif', 'dif_max_depth',
       'dif_surface_area', 'dif_sw_mean_au', 'dif_ws_mean_au',
       'dif_lathrop_strat', 'dif_glm_strat_perc', 'perc_dif_max_depth',
       'perc_dif_sqrt_surface_area']
###################################################################################

#######################################################################3
#paste hyperparameters found in "pbmtl_hyperparameter_search.py" here

n_estimators = 4000
lr = .05
#####################################################################



########################
##########################
#metamodel training code
##########################
#######################


train = True #can disable if re-running but don't need to train
if train:
	print("Model training in progress...")
	model = GradientBoostingRegressor(n_estimators=4000,learning_rate=0.05)
	X = pd.DataFrame(train_df[feats])
	y = np.ravel(pd.DataFrame(train_df['rmse']))

	model.fit(X,y)
	print("model train complete")
	dump(model, 'metamodel_glm_RMSE_GBR.joblib') 



model = load('metamodel_glm_RMSE_GBR.joblib') 


importances = model.feature_importances_
print("##Feature Importances#################")
print(feats)
print(importances)
print("#####################")

########################
##########################
# framework evaluation code
##########################
#######################


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
csv = ['target_id,source_id,pb-mtl_rmse,predicted_rmse,spearman,meta_rmse']
for feat in feats:
	csv += ','+str(feat)


#for each test lake, select source lake and record results
for targ_ct, target_id in enumerate(test_lakes): #for each target lake
	print("predicting target lake ", targ_ct, ":", target_id)
	lake_df = pd.read_feather("../../metadata/diffs/target_"+ target_id +".feather")
	lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes)]
	lake_df['site_id2'] = ['nhdhr_'+x for x in lake_df['site_id'].values] 
	targ_result_df = result_df[np.isin(result_df['target_id'], 'test_nhdhr_'+target_id)] 
	pdb.set_trace()
	lake_df = lake_df.merge(targ_result_df, left_on='site_id2', right_on='source_id')

	X = pd.DataFrame(lake_df[feats])

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
	csv.append(",".join(['nhdhr'+str(target_id), 'nhdhr'+str(best_predicted), str(y_act_top), str(best_predicted_rmse),str(srcorr_per_lake[targ_ct]),str(meta_rmse_per_lake[targ_ct]) ] + [str(x) for x in lake_df.iloc[0][feats].values]))

#print medians
print("median meta test RMSE: ",np.median(meta_rmse_per_lake))
print("median spearman RMSE: ",np.median(srcorr_per_lake))
print("median test RMSE: ",np.median(rmse_per_lake))

#write results to file
with open(save_file_path,'w') as file:
    for line in csv:
        file.write(line)
        file.write('\n')


