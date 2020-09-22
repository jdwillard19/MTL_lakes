# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import KFold
# import pdb
# import sys
# sys.path.append('../data')
# from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune2, parseMatricesFromSeqs
# import torch
# from metadata_ops import nhd2nhdhr
# import torch.nn as nn
# import torch.utils.data
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.init import xavier_normal_
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.model_selection import cross_val_score
# from sklearn.neural_network import MLPRegressor
# from scipy.stats import spearmanr
# from joblib import dump, load
# import re
# # build models to predict which models to use for transfer learning

import pandas as pd
import numpy as np
import pdb
import sys
sys.path.append('../data')
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr
from joblib import dump, load
import re

################################################################################3
# (Sept 2020 - Jared) - evaluate PB-MTL model by predicting best source model for each of 1882 expanded test lakes
# Features and hyperparamters must be manually specified below 
# (e.g. feats = ['dif_max_depth', ....]; n_estimators = 4000, etc)
#############################################################################################################

#file to save results to
save_file_path = '../../results/pbmtl_glm_transfer_results_expanded.csv'

#path to load metamodel from
model_path = "../../models/metamodel_glm_RMSE_GBR.joblib"


metadata = pd.read_feather("../../metadata/lake_metadata_full.feather")
sites = pd.read_csv('../../metadata/sites_moreThan10ProfilesWithGLM_June2020Update.csv')
ids = pd.read_csv('../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../results/transfer_learning/glm/train_rmses_pball.feather")
result_df = pd.read_csv("../../results/glm_transfer/RMSE_transfer_test_extended_glm.csv")

pdb.set_trace()
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)
all_sites = metadata['site_id'].values
test_lakes = all_sites[~np.isin(all_sites, train_lakes)]
metadata.set_index('site_id', inplace=True)
test_site_nhd = np.array(['test_nhdhr_'+x for x in test_lakes])




#########################################################################################
#paste features found in "pbmtl_feature_selection.py" here
feats = ['n_obs_su', 'obs_temp_mean', 'obs_temp_skew', 'obs_temp_kurt',
       'ad_glm_strat_perc', 'obs_temp_mean_airdif', 'dif_max_depth',
       'dif_surface_area', 'dif_sw_mean_au', 'dif_ws_mean_au',
       'dif_lathrop_strat', 'dif_glm_strat_perc', 'perc_dif_max_depth',
       'perc_dif_sqrt_surface_area']
###################################################################################



#load metamodel
model = load(model_path)

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
csv = [ 'target_id,source_id,predicted_rmse , n_obs ,  obs_temp_mean ,  obs_temp_skew , obs_temp_kurt ,\
        obs_temp_mean_airdif ,  dif_max_depth ,  dif_surface_area ,\
        dif_sw_mean_au ,  dif_ws_mean_au ,  dif_lathrop_strat , dif_glm_strat_perc ,  ad_glm_strat_perc, \
        perc_dif_max_depth ,  perc_dif_surface_area' ]
# mean_per_k = np.empty((7, test_lakes.shape[0]))
# mean_per_k[:] = np.nan
for targ_ct, target_id in enumerate(test_lakes): #for each target lake
	print("target lake ", targ_ct, ":", target_id)
	lake_df = []
	# if os.path.exists("../../results/transfer_learning/target_"+target_id+"/resultsPGRNNbasic5"):
	# 	lake_df = pd.read_csv("../../results/transfer_learning/target_"+target_id+"/resultsPGRNNbasic5") #load metadata/rmse for model selection
	# else:
	# lake_df = pd.read_csv("../../metadata/diff/target_"+str(target_id)+".csv")
	# lake_df = lake_df[np.isin(lake_df['nhd_id'], train_lakes)]

	# lake_df = pd.read_csv("../../results/transfer_learning/target_"+target_id+"/resultsPGRNN_norm_all") #load metadata/rmse for model selection
	# lake_df_res = pd.read_csv("../../results/transfer_learning/target_"+target_id+"/resultsPGRNNbasic_wNew_norm2") 
	lake_df = pd.read_feather("../../metadata/diff/target_"+ target_id +"_pball_Aug2020.feather")
	# lake_df = lake_df[np.isin(lake_df['source_id'], train_lakes)]
	lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes)]
	lake_df['site_id2'] = ['nhdhr_'+x for x in lake_df['site_id'].values] 

	# lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]

	# targ_result_df = result_df[np.isin(result_df['target_id'], 'test_nhdhr_'+target_id)] 

	# lake_df = lake_df.merge(targ_result_df, left_on='site_id2', right_on='source_id')
	# lake_df = pd.concat([lake_df[['nhd_id', 'ad_lat', 'ad_SDF', 'ad_surface_area','ad_max_depth','ad_long','ad_k_d']], lake_df_res['rmse']], axis=1)
	# lake_df = pd.concat([lake_df['nhd_id'], lake_df[rmse_feats], lake_df_res['rmse']], axis=1)

	# lake_df = pd.read_csv("../../metadata/diff/target_"+target_id+".csv")#load metadata/rmse for model selection
	# lake_df = pd.read_csv("../../results/transfer_learning/target_"+target_id+"/resultsPGRNN_CV") #load metadata/rmse for model selection
	# X = pd.DataFrame(lake_df[['geo','tempdtw','surf_area','max_depth','lat','long','k_d']])
	# X = pd.DataFrame(lake_df[['lat', 'SDF', 'surf_area','max_depth','long','k_d']])
	# X = pd.DataFrame(lake_df[['ad_lat', 'ad_SDF', 'ad_surface_area','ad_max_depth','ad_long','ad_k_d']])
	X = pd.DataFrame(lake_df[feats])
	# X = pd.DataFrame(lake_df[['SDF','lat','surf_area','max_depth','long','k_d', 'canopy']])
	# y = pd.DataFrame(lake_df['rmse_improve'])

	y_pred = []
	top_ids = []

	y_pred = model.predict(X)
	lake_df['rmse_pred'] = y_pred

	# y_act = lake_df['rmse']

	# y_pred = y_pred[np.isfinite(y_act)]
	# y_act = y_act[np.isfinite(y_act)]
	# meta_rmse_per_lake[targ_ct] = np.median(np.sqrt(((y_pred - y_act) ** 2).mean()))
	# srcorr_per_lake[targ_ct] = spearmanr(y_pred, y_act).correlation

	# print("meta rmse: ", meta_rmse_per_lake[targ_ct])
	# print("srcorr: ", srcorr_per_lake[targ_ct])


	lake_df.sort_values(by=['rmse_pred'], inplace=True)
	best_predicted = lake_df.iloc[0]['site_id']
	# y_act_top = float(lake_df.iloc[0]['rmse'])
	# print("rmse: ", y_act_top)
	# rmse_per_lake[targ_ct] = y_act_top

	best_predicted_rmse = lake_df.iloc[0]['rmse_pred']
	# lake_df = lake_df[lake_df['rmse_pred'] < lowest_rmse+rel_cut]
	csv.append(",".join(['nhdhr_'+str(target_id), 'nhdhr_'+str(best_predicted), str(best_predicted_rmse)] + [str(x) for x in lake_df.iloc[0][feats].values]))


# with open("../../results/transfer_learning/rf_testset.csv",'a') as file:
# 	for line in test_lake_csv:
# 		file.write(line)
# 		file.write('\n')
print("median meta test RMSE: ",np.median(meta_rmse_per_lake))
print("median spearman RMSE: ",np.median(srcorr_per_lake))
print("median test RMSE: ",np.median(rmse_per_lake))
# print("mean test RMSE: ",rmse_per_testlake.mean())
# 

# with open('glm_transfer_pball_test_lakes_wRange.csv','w') as file:
with open('pbmtl_glm_transfer_results_Sept8_extended.csv','w') as file:
    for line in csv:
        file.write(line)
        file.write('\n')


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