import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pdb
import sys
sys.path.append('../../data')
from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune2, parseMatricesFromSeqs
import torch
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


metadata = pd.read_feather("../../../metadata/lake_metadata_2700plus.feather")
metadata.set_index('site_id', inplace=True)
ids = pd.read_csv('../../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
glm_all_f = pd.read_csv("../../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../../results/transfer_learning/glm/train_rmses_pball.feather")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)
test_lakes = ids[~np.isin(ids, train_lakes)]
k = 145
biases = []
# print(train_lakes.shape[0], " training lakes")

mat_csv = []
mat_csv.append(",".join(["target_id","source_id","rmse","source_observations","source_observation_mean_temp","dif_surface_area","percent_dif_max_depth","percent_dif_sqrt_surface_area"]))
############################################################################
#use CV to find which to drop
################################################################################

feats = train_df.columns[80:-1]
feats = ['n_obs', 'obs_temp_mean', 'dif_max_depth', 'dif_surface_area', 'perc_dif_max_depth', 'perc_dif_sqrt_surface_area']
# ranks = np.array([ 1,  5,  6,  2,  1,  7,  4,  1,  3,  7,  8, 12,  3, 11,  4, 11,  2,
#         2,  6, 18, 20,  1,  1,  9, 16, 23, 15, 23, 14, 23, 22, 21, 18, 21,
#        16, 16, 19, 12, 10, 22, 13, 21, 14,  5,  9, 13,  3,  7, 13, 19, 17,
#         9,  8, 24,  3, 20, 11, 19, 15, 17, 10,  8, 17,  6,  7,  5, 11, 20,
#        20, 12,  4,  5,  9,  8, 15, 18, 12, 24, 21,  1, 17, 23, 10, 24, 14,
#        19, 13, 18, 16, 24, 22, 14, 10, 22, 15,  6,  2,  4,  1,  1,  1,  1])


# feats = feats[ranks == 1]
train = False
model = []
upper_model = []
lower_model = []
if train:
	# #compile training data from all training lakes
	train_df = pd.DataFrame()

	for _, lake_id in enumerate(train_lakes):

		new_df = pd.DataFrame()
		# temp_df = pd.read_csv("../../../metadata/diff/target_"+lake_id+"_wStats.csv")
		# temp_df.drop(np.where(np.isin(temp_df['nhd_id'], temp_test_lakes[0]) == True)[0][0], inplace=True)

		lake_df_res = pd.read_csv("../../../results/transfer_learning/target_"+lake_id+"/resultsPGRNNbasic_pball") 
		lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']

		lake_df = pd.read_feather("../../../metadata/diff/target_"+lake_id+"_pball_update.feather")

		# lake_df = lake_df[np.isin(lake_df['source_id'], train_lakes)]
		lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes)]
		lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
		lake_df = pd.merge(left=lake_df, right=lake_df_res, left_on='site_id', right_on='source_id')
		# lake_df = pd.concat([lake_df, lake_df_res['rmse']], axis=1)
		new_df = lake_df
		train_df = pd.concat([train_df, new_df], ignore_index=True)

	# # #declare model
	# # # model = RandomForestRegressor(max_depth=None,random_state=1, n_estimators=10000, max_features=3, n_jobs=-1)
	# # # param = [.09,.11,.08,.12,.07,.13,.06,.14,.05,.15,.04,.16,.03,.17,.02,.18,.01,.19,.20,.21,.22]
	# # # param = np.arange(1,50)
	# # # param = np.arange(.05,.20, .01)
	# # # # param = np.arange(500,10000, 500)
	# # # X_trn = pd.DataFrame(train_df[['ad_lat', 'ad_SDF', 'ad_surface_area','ad_max_depth','ad_long','ad_k_d']])
	X_trn = pd.DataFrame(train_df[feats])
	# # # X_trn = pd.DataFrame(train_df[['SDF','lat','surf_area','max_depth','long','k_d', 'canopy']])
	y_trn = np.array([np.log(float(x)) for x in np.ravel(pd.DataFrame(train_df['rmse']))])
	# # model = GradientBoostingRegressor(n_estimators=30, learning_rate=.05, max_depth=3)
	# model = RandomForestRegressor(n_estimators=10000)
	# model = 
	# alpha = 0.90
	# upper_model = GradientBoostingRegressor(n_estimators=3800, learning_rate=.05, loss='quantile', alpha=alpha)
	# upper_model.fit(X_trn, y_trn)

	# lower_model = GradientBoostingRegressor(n_estimators=3800, learning_rate=.05, loss='quantile', alpha=1-alpha)
	# lower_model.fit(X_trn, y_trn)

	# med_model = GradientBoostingRegressor(n_estimators=3800, learning_rate=.05, loss='quantile', alpha=0.5)
	# med_model.fit(X_trn, y_trn)

	model = GradientBoostingRegressor(n_estimators=3800, learning_rate=.05)
	model.fit(X_trn, y_trn)

	# model = GradientBoostingRegressor(n_estimators=4000, learning_rate=.05)
	# {'learning_rate': 0.05, 'n_estimators': 3800}

	# dump(upper_model, "PGML_RMSE_GBR_pball_upper90.joblib")
	# dump(lower_model, "PGML_RMSE_GBR_pball_lower90.joblib")
	# dump(med_model, "PGML_RMSE_GBR_pball_median.joblib")
	dump(model, "PGML_RMSE_GBR_pball_logloss.joblib")
# model = load("PGML_RMSE_GBR_pball2.joblib")
model = load("PGML_RMSE_GBR_pball.joblib")
# model = load("PGML_RMSE_GBR_pball_logloss.joblib")
upper_model = load("PGML_RMSE_GBR_pball_upper90.joblib")
lower_model = load("PGML_RMSE_GBR_pball_lower90.joblib")
med_model = load("PGML_RMSE_GBR_pball_median.joblib")
# # for i, p in enumerate(param):

# # 	model = GradientBoostingRegressor(n_estimators=3500, learning_rate=p, max_depth=3)
# # # model = MLKR(
# # 	cvscore = cross_val_score(model, X_trn, np.ravel(y_trn), cv=n_train_lakes, n_jobs=-1).mean()
# # 	# print("cv: ", cvscore)
# # 	print("param=", p, "->CVscore: ",cvscore)

# # csv.append(",".join([str(p), str(cvscore)]))

# # with open("./hypertune.csv",'a') as file:
# # 	for line in csv:
# # 		file.write(line)
# # 		file.write('\n')
# # model = GradientBoostingRegressor(n_estimators=1000)
# # model = MLPRegressor(hidden_layer_sizes=(100, 1), activation='relu', solver='adam', alpha=0.0001, \
# # 					 batch_size='auto', learning_rate='constant', learning_rate_init=0.001,  \
# # 					 power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001, \
# # 					 verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, \
# # 					 early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, \
# # 					 epsilon=1e-08, n_iter_no_change=10)

# #define x and y
# # X_trn = pd.DataFrame(train_df[['geo','tempdtw','surf_area','max_depth','lat','long','k_d']])
# # X_trn = pd.DataFrame(train_df[['canopy','SDF','surf_area','max_depth','lat','long', 'k_d', 'geo', 'tempdtw']])


# #fit model
# test_lakes = np.array(['86274749', '1099136', '121839184', '70334209', '13393533', \
# 			 '13393567', '166868528', '47726570', '120019294', '{4EA76133-68CD-4DA8-A82B-0B568CD9C9B2}', '14783883'])


############################################################
# dump(model, 'gbr_123019_tuned.joblib') 
# model = load('rf_121819.joblib') 
###############################################################



	#predict rmse
# y_pred_trn = model.predict(X_trn)
# print("params", model.get_params())

# print("model score " , model.score(X_trn, y_trn))

# forest = model
# # import matplotlib.pyplot as plt
# importances = forest.feature_importances_
# std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]

# # # Print the feature ranking
# print("Feature ranking:")
# for f in range(X_trn[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# # Plot the feature importances of the forest
# # plt.figure()
# # plt.title("Feature importances")
# # plt.bar(range(X_trn.shape[1]), importances[indices],
# #        color="r", yerr=std[indices], align="center")
# # plt.xticks(range(X_trn.shape[1]), indices)
# # plt.xlim([-1, X_trn.shape[1]])
# # plt.show()
############################################################################
#use model to predict top 3 for every source target lake, then predict target
################################################################################
# test_lakes = train_lakes
rmse_per_lake = np.empty(test_lakes.shape[0])
glm_rmse_per_lake = np.empty(test_lakes.shape[0])
srcorr_per_lake = np.empty(test_lakes.shape[0])

meta_rmse_per_lake = np.empty(test_lakes.shape[0])
med_meta_rmse_per_lake = np.empty(test_lakes.shape[0])
rmse_per_lake[:] = np.nan
glm_rmse_per_lake[:] = np.nan
meta_rmse_per_lake[:] = np.nan
csv = []
csv.append('target_id,rmse,rmse_pred,rmse_pred_lower,rmse_pred_upper,rmse_pred_med,spearman,glm_rmse,site_id')
# mean_per_k = np.empty((7, test_lpakes.shape[0]))
# mean_per_k[:] = np.nan
# most_imp_ind_rmses = np.empty((9))
# most_imp_source_ids = np.empty((9))
# most_imp_total_rmse = np.nan
# most_imp_site_id = np.nan
# most_imp_diff = -1

err_per_source = np.empty((145,len(test_lakes)))
# test_lakes = np.array(['120020398'])
for targ_ct, target_id in enumerate(test_lakes): #for each target lake
	print(str(targ_ct),'/',len(test_lakes),':',target_id)
	lake_df = pd.DataFrame()
	# temp_df = pd.read_csv("../../../metadata/diff/target_"+lake_id+"_wStats.csv")
	# temp_df.drop(np.where(np.isin(temp_df['nhd_id'], temp_test_lakes[0]) == True)[0][0], inplace=True)
	lake_id = target_id
	lake_df_res = pd.read_csv("../../../results/transfer_learning/target_"+lake_id+"/resultsPGRNNbasic_pball") 
	lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']

	lake_df = pd.read_feather("../../../metadata/diff/target_"+lake_id+"_pball_update.feather")

	# lake_df = lake_df[np.isin(lake_df['source_id'], train_lakes)]
	lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes)]
	lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
	lake_df = pd.merge(left=lake_df, right=lake_df_res, left_on='site_id', right_on='source_id')
	# lake_df = pd.concat([lake_df, lake_df_res['rmse']], axis=1)
	# lake_df = pd.read_csv("../../../metadata/diff/target_"+target_id+".csv")#load metadata/rmse for model selection
	# lake_df = pd.read_csv("../../../results/transfer_learning/target_"+target_id+"/resultsPGRNN_CV") #load metadata/rmse for model selection
	# X = pd.DataFrame(lake_df[['geo','tempdtw','surf_area','max_depth','lat','long','k_d']])
	# X = pd.DataFrame(lake_df[['lat', 'SDF', 'surf_area','max_depth','long','k_d']])
	# X = pd.DataFrame(lake_df[['ad_lat', 'ad_SDF', 'ad_surface_area','ad_max_depth','ad_long','ad_k_d']])
	X = pd.DataFrame(lake_df[feats])
	# X = pd.DataFrame(lake_df[['SDF','lat','surf_area','max_depth','long','k_d', 'canopy']])
	# y = pd.DataFrame(lake_df['rmse'])

	y_pred = []
	top_ids = []

	y_pred = model.predict(X)
	y_pred_upper = upper_model.predict(X)
	y_pred_lower = lower_model.predict(X)
	y_pred_med = med_model.predict(X)
	lake_df['rmse_pred'] = y_pred
	lake_df['upper_rmse_pred'] = y_pred_upper
	lake_df['lower_rmse_pred'] = y_pred_lower
	lake_df['med_rmse_pred'] = y_pred_med
	y_act = np.array([float(x) for x in np.ravel(lake_df['rmse'].values)])
	# mat_csv.append(",".join(["nhdhr_"+target_id,]))
	# meta_rmse_per_lake[targ_ct] = np.sqrt(((y_pred - y_act) ** 2).mean()
	meta_rmse_per_lake[targ_ct] = np.median(np.sqrt((y_pred - y_act) ** 2))
	med_meta_rmse_per_lake[targ_ct] = np.median(np.sqrt((y_pred_med - y_act) ** 2))
	srcorr_per_lake[targ_ct] = spearmanr(y_pred, y_act).correlation
	#get top predicted lakes
	# if lake_df[lake_df['rmse_pred'] < hard_cut].shape[0] > 0:
	# 	lake_df = lake_df[lake_df['rmse_pred'] < hard_cut]
	# else:
	# 	print("no good lakes?")
	#rel cutoff

	lake_df.sort_values(by=['rmse_pred'], inplace=True)

	# rmse_pred_median = np.median(lake_df['rmse_pred'].values[:9])
	# rmse_pred_mean = (lake_df['rmse_pred'].values[:9]).mean()
	# rmse_pred_low = (lake_df['rmse_pred'].values[:9]).min()

	# rmse_pred_mad = np.median(np.absolute(lake_df['rmse_pred'].values - rmse_pred_median))
	# rmse_less_one_ct = np.sum(lake_df['rmse_pred'].values < 1)
	# lake_df = lake_df[lake_df['rmse_pred'] < lowest_rmse+rel_cut]

	# for j in range(lake_df.shape[0]):
	# 	mat_csv.append(",".join(["nhdhr_"+target_id,"nhdhr_"+lake_df.iloc[j]['site_id'],str(lake_df.iloc[j]['rmse'])] + [str(x) for x in lake_df.iloc[j][feats].values]))
	# continue
	top_ids = [str(j) for j in lake_df.iloc[:k]['site_id']]
	# for j in range(9):
	# 	mat_csv.append(",".join(["nhdhr_"+target_id,"nhdhr_"+lake_df.iloc[j]['site_id'],str(lake_df.iloc[j]['rmse'])] + [str(x) for x in lake_df.iloc[j][feats].values]))
	
	best_site = top_ids[0]
	# if method == 'geotemp':
	# 	top_ids = getSimilarLakes(target_id, method='geotemp', k=k, cand=7)

	# print("top source lakes, ", top_ids)
	#define target test data to use
	data_dir_target = "../../data/processed/lake_data/"+target_id+"/" 
	#target agnostic model and data params


	#output matrix
	n_lakes = len(top_ids)


	for i, source_id in enumerate(top_ids): 
		mat_csv.append(",".join(["nhdhr_"+target_id,"nhdhr_"+ source_id,str(lake_df.iloc[i]['rmse_pred'])] + [str(x) for x in lake_df.iloc[i][feats].values]))
		# output_df = pd.DataFrame(data=outputm_npy, index=[str(float(x/2)) for x in range(outputm_npy.shape[0])], columns=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
		# output_df.rename(columns={'index': 'depth'})
		# label_df = pd.DataFrame(data=labelm_npy, index=[str(float(x/2)) for x in range(labelm_npy.shape[0])], columns=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
				# label_df.rename(columns={'index': 'depth'})

				# if i == 0:
				# 	output_df.to_feather("good_outputs/target_nhdhr_"+target_id+"_top_source_nhdhr_"+source_id+"_outputs.feather")
				# else:
				# 	output_df.to_feather("good_outputs/target_nhdhr_"+target_id+"_source_nhdhr_"+source_id+"_outputs.feather")
				# label_df.to_feather("good_outputs/target_nhdhr_"+target_id+"_labels.feather")
				# mean_per_k[i, targ_ct] = mat_rmse




                #calculate energy at each timestep
	            # output_torch = torch.from_numpy(outputm_npy).float()
	            # if use_gpu:
	            #     output_torch = output_torch.cuda()
	            # energies = calculate_energy(output_torch, depth_areas, use_gpu)
	            # energies = energies.cpu().numpy()
	            # avg_mse = avg_mse.cpu().numpy()
	            # if save: 
	            #     saveFeatherFullDataWithEnergy(outputm_npy, labelm_npy, None, unique_tst_dates_target, source_id,target_id, 0)


with open('pgdtl_rmse_pball_ens145_wPredOnly.csv','w') as file:
    for line in mat_csv:
        file.write(line)
        file.write('\n')


# with open('pgml_result_logloss.csv','w') as file:
#     for line in mat_csv:
#         file.write(line)
#         file.write('\n')

# print("mean meta test RMSE: ",meta_rmse_per_lake.mean())
# print("median meta test RMSE: ",np.median(meta_rmse_per_lake))
# print("median srcorr: ",np.median(srcorr_per_lake))
# print("median meta test RMSE(med): ",np.median(med_meta_rmse_per_lake))
# print("mean test RMSE: ",rmse_per_lake.mean())
# print("median test RMSE: ",np.median(rmse_per_lake))
# # print("mean test RMSE: ",rmse_per_testlake.mean())
# # 
# # biases = np.array(biases)
# # np.save("./biases.csv", biases)

# print("Target: ",most_imp_site_id)
# print("sources: ",most_imp_source_ids)
# print("diff: ",most_imp_diff)
# print("total rmse: ",most_imp_total_rmse)
# print("ind rmses: ",most_imp_ind_rmses)
# print("ind rmses mean: ",most_imp_ind_rmses.mean())


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