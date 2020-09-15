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


metadata = pd.read_feather("../../../metadata/lake_metadata_baseJune2020.feather")
metadata.set_index('site_id', inplace=True)
ids = pd.read_csv('../../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
glm_all_f = pd.read_csv("../../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../../results/transfer_learning/glm/glm_meta_train_rmses.feather")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)
test_lakes = ids[~np.isin(ids, train_lakes)]
k = 145
biases = []
# print(train_lakes.shape[0], " training lakes")

mat_csv = []
csv = []
mat_csv.append(",".join(["target_id","source_id","transfer_rmse","predicted_rmse","source_observations",\
						 "source_obs_mean_temp_air_temp_diff","diff_max_depth",'diff_surface_area', 'diff_RH_mean_autumn', \
						 'percent_diff_max_depth', \
						 'percent_diff_surface_area','percent_diff_sqrt_surface_area']))

# csv.append(",".join(["target_id","pg-mtl9_rmse","pb0_rmse","ens_mean_predicted_rmse","ens_lowest_predicted_rmse"]))

############################################################################
#use CV to find which to drop
################################################################################

feats = train_df.columns[80:-1]
# feats = ['n_obs_sp', 'obs_temp_mean', 'obs_temp_mean_airdif', 'dif_max_depth',\
#        'dif_surface_area', 'dif_rh_mean_au', 'dif_ws_mean_au',\
#        'perc_dif_max_depth', 'perc_dif_surface_area',\
#        'perc_dif_sqrt_surface_area']
feats = ['n_obs' 'obs_temp_mean_airdif' 'dif_max_depth' 'dif_surface_area'
 'dif_rh_mean_au' 'perc_dif_max_depth' 'perc_dif_surface_area'
 'perc_dif_sqrt_surface_area']

# ranks = np.array([ 1,  5,  6,  2,  1,  7,  4,  1,  3,  7,  8, 12,  3, 11,  4, 11,  2,
#         2,  6, 18, 20,  1,  1,  9, 16, 23, 15, 23, 14, 23, 22, 21, 18, 21,
#        16, 16, 19, 12, 10, 22, 13, 21, 14,  5,  9, 13,  3,  7, 13, 19, 17,
#         9,  8, 24,  3, 20, 11, 19, 15, 17, 10,  8, 17,  6,  7,  5, 11, 20,
#        20, 12,  4,  5,  9,  8, 15, 18, 12, 24, 21,  1, 17, 23, 10, 24, 14,
#        19, 13, 18, 16, 24, 22, 14, 10, 22, 15,  6,  2,  4,  1,  1,  1,  1])


# feats = feats[ranks == 1]
train = True
model = []
upper_model = []
lower_model = []
if train:
	# #compile training data from all training lakes
	train_df = pd.DataFrame()

	for _, lake_id in enumerate(train_lakes):

		new_df = pd.DataFrame()
		lake_df_res = pd.read_csv("../../../results/transfer_learning/target_"+lake_id+"/results_all_source_models.csv") 
		lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']
		lake_df = pd.read_feather("../../../metadata/diff/target_"+lake_id+"_pball_July2020.feather")
		lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes)]
		lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
		lake_df = pd.merge(left=lake_df, right=lake_df_res.astype('object'), left_on='site_id', right_on='source_id')
		new_df = lake_df
		train_df = pd.concat([train_df, new_df], ignore_index=True)

	X_trn = pd.DataFrame(train_df[feats])
	y_trn = np.array([np.log(float(x)) for x in np.ravel(pd.DataFrame(train_df['rmse']))])



	model = GradientBoostingRegressor(n_estimators=3300, learning_rate=.1)
	model.fit(X_trn, y_trn)


	dump(model, "PGMTL_GBR_pball.joblib")
# model = load("PGML_RMSE_GBR_pball2.joblib")
model = load("PGMTL_GBR_pball.joblib")
# model = load("PGML_RMSE_GBR_pball_logloss.joblib")
# upper_model = load("PGML_RMSE_GBR_pball_upper90.joblib")
# lower_model = load("PGML_RMSE_GBR_pball_lower90.joblib")
# med_model = load("PGML_RMSE_GBR_pball_median.joblib")
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
# csv = []
# csv.append('target_id,pg-mtl9_rmse,glm_rmse,ens_rmse_mean_predicted,ens_rmse_lowest_predicted')
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
	lake_df_res = pd.read_csv("../../../results/transfer_learning/target_"+lake_id+"/results_all_source_models.csv") 
	lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']

	lake_df = pd.read_feather("../../../metadata/diff/target_"+lake_id+"_pball_July2020.feather")

	# lake_df = lake_df[np.isin(lake_df['source_id'], train_lakes)]
	lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes)]
	lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
	lake_df = pd.merge(left=lake_df, right=lake_df_res.astype('object'), left_on='site_id', right_on='source_id')
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
	lake_df['rmse_pred'] = y_pred
	y_act = np.array([float(x) for x in np.ravel(lake_df['rmse'].values)])
	# mat_csv.append(",".join(["nhdhr_"+target_id,]))
	# meta_rmse_per_lake[targ_ct] = np.sqrt(((y_pred - y_act) ** 2).mean()
	meta_rmse_per_lake[targ_ct] = np.median(np.sqrt(((y_pred - y_act) ** 2).mean()))
	srcorr_per_lake[targ_ct] = spearmanr(y_pred, y_act).correlation
	#get top predicted lakes
	# if lake_df[lake_df['rmse_pred'] < hard_cut].shape[0] > 0:
	# 	lake_df = lake_df[lake_df['rmse_pred'] < hard_cut]
	# else:
	# 	print("no good lakes?")
	#rel cutoff

	lake_df.sort_values(by=['rmse_pred'], inplace=True)
	rmse_pred = lake_df['rmse_pred']
	# min_rmse = lake_df.iloc[:k]['rmse_pred'].mean()

	top_ids = [str(j) for j in lake_df.iloc[:k]['site_id']]
	
	best_site = top_ids[0]




	data_dir_target = "../../data/processed/lake_data/"+target_id+"/" 
	#target agnostic model and data params
	use_gpu = True
	n_features = 8
	# n_hidden = 20
	seq_length = 350
	win_shift = 175
	begin_loss_ind = 0
	(_, _, tst_data_target, tst_dates_target, unique_tst_dates_target, all_data_target, all_phys_data_target, all_dates_target,
	_) = buildLakeDataForRNN_manylakes_finetune2(target_id, data_dir_target, seq_length, n_features,
	                                   win_shift = win_shift, begin_loss_ind = begin_loss_ind, 
	                                   outputFullTestMatrix=True, allTestSeq=True)
	

	#useful values, LSTM params
	batch_size = all_data_target.size()[0]
	u_depths_target = np.unique(all_data_target[:,0,0])
	n_depths = torch.unique(all_data_target[:,:,0]).size()[0]
	n_test_dates_target = unique_tst_dates_target.shape[0]


	#define LSTM model
	class LSTM(nn.Module):
	    def __init__(self, input_size, hidden_size, batch_size):
	        super(LSTM, self).__init__()
	        self.input_size = input_size
	        self.hidden_size = hidden_size
	        self.batch_size = batch_size
	        self.lstm = nn.LSTM(input_size = n_features, hidden_size=hidden_size, batch_first=True) 
	        self.out = nn.Linear(hidden_size, 1)
	        self.hidden = self.init_hidden()

	    def init_hidden(self, batch_size=0):
	        # initialize both hidden layers
	        if batch_size == 0:
	            batch_size = self.batch_size
	        ret = (xavier_normal_(torch.empty(1, batch_size, self.hidden_size)),
	                xavier_normal_(torch.empty(1, batch_size, self.hidden_size)))
	        if use_gpu:
	            item0 = ret[0].cuda(non_blocking=True)
	            item1 = ret[1].cuda(non_blocking=True)
	            ret = (item0,item1)
	        return ret
	    
	    def forward(self, x, hidden): #forward network propagation 
	        self.lstm.flatten_parameters()
	        x = x.float()
	        x, hidden = self.lstm(x, self.hidden)
	        self.hidden = hidden
	        x = self.out(x)
	        return x, hidden



	#output matrix
	n_lakes = len(top_ids)
	output_mats = np.empty((n_lakes, n_depths, n_test_dates_target))
	ind_rmses = np.empty((n_lakes))
	ind_rmses[:] = np.nan
	label_mats = np.empty((n_depths, n_test_dates_target)) 
	output_mats[:] = np.nan
	label_mats[:] = np.nan

	for i, source_id in enumerate(top_ids): 
		#for each top id

		#load source model
		load_path = "../../../models/single_lake_models/"+source_id+"/PGRNN_source_model"
		# load_path = "../../../models/single_lake_models/"+source_id+"/PGRNN_norm_all"
		# load_path = "../../../models/single_lake_models/"+source_id+"/PGRNN_CV"
		n_hidden = torch.load(load_path)['state_dict']['out.weight'].shape[1]
		lstm_net = LSTM(n_features, n_hidden, batch_size)
		if use_gpu:
			lstm_net = lstm_net.cuda(0)
		pretrain_dict = torch.load(load_path)['state_dict']
		model_dict = lstm_net.state_dict()
		pretrain_dict = {key: v for key, v in pretrain_dict.items() if key in model_dict}
		model_dict.update(pretrain_dict)
		lstm_net.load_state_dict(pretrain_dict)

		#things needed to predict test data
		mse_criterion = nn.MSELoss()
		testloader = torch.utils.data.DataLoader(tst_data_target, batch_size=tst_data_target.size()[0], shuffle=False, pin_memory=True)

		lstm_net.eval()
		with torch.no_grad():
			avg_mse = 0
			ct = 0
			for m, data in enumerate(testloader, 0):
				#now for mendota data
				#this loop is dated, there is now only one item in testloader

				#parse data into inputs and targets
				inputs = data[:,:,:n_features].float()
				targets = data[:,:,-1].float()
				targets = targets[:, begin_loss_ind:]
				tmp_dates = tst_dates_target[:, begin_loss_ind:]
				depths = inputs[:,:,0]

				if use_gpu:
				    inputs = inputs.cuda()
				    targets = targets.cuda()

				#run model
				h_state = None
				lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
				pred, h_state = lstm_net(inputs, h_state)
				pred = pred.view(pred.size()[0],-1)
				pred = pred[:, begin_loss_ind:]

				#calculate error
				targets = targets.cpu()
				loss_indices = np.where(~np.isnan(targets))
				if use_gpu:
					targets = targets.cuda()
				inputs = inputs[:, begin_loss_ind:, :]
				depths = depths[:, begin_loss_ind:]
				mse = mse_criterion(pred[loss_indices], targets[loss_indices])
				# print("test loss = ",mse)
				avg_mse += mse

				if mse > 0: #obsolete i think
				    ct += 1
				avg_mse = avg_mse / ct


				#save model 
				(outputm_npy, labelm_npy) = parseMatricesFromSeqs(pred.cpu().numpy(), targets.cpu().numpy(), depths, tmp_dates, n_depths, 
				                                                n_test_dates_target, u_depths_target,
				                                                unique_tst_dates_target) 
				# if target_id == '113471457':
				# 	np.save('./best_ens/source_'+source_id+'_output', outputm_npy)
				# 	np.save('./best_ens/labels', labelm_npy)
				# 	np.save('./best_ens/dates', unique_tst_dates_target)
				#store output
				output_mats[i,:,:] = outputm_npy
				if i == 0:
					#store label
					label_mats = labelm_npy
				loss_output = outputm_npy[~np.isnan(labelm_npy)]
				loss_label = labelm_npy[~np.isnan(labelm_npy)]

				mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())
				print(source_id+" rmse=", mat_rmse)
				err_per_source[i,targ_ct] = mat_rmse

				glm_rmse = float(metadata.loc[target_id].glm_uncal_rmse_full)
				mat_csv.append(",".join(["nhdhr_"+target_id,"nhdhr_"+ source_id, str(mat_rmse),str(rmse_pred[i])] + [str(x) for x in lake_df.iloc[i][feats].values]))


	#save model 
	total_output_npy = np.average(output_mats, axis=0)

	# output_df = pd.DataFrame(data=total_output_npy, index=[str(float(x/2)) for x in range(outputm_npy.shape[0])], columns=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
	# output_df.rename(columns={'index': 'depth'})
	# output_df.to_feather("good_outputs/target_nhdhr_"+target_id+"_ENSEMBLE_outputs.feather")
	loss_output = total_output_npy[~np.isnan(label_mats)]
	loss_label = label_mats[~np.isnan(label_mats)]
	mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())
	glm_rmse = float(metadata.loc[target_id].glm_uncal_rmse_full)

	# mat_csv.append(",".join(["nhdhr_"+target_id,"9 MODEL ENSEMBLE",str(glm_rmse),str(mat_rmse)] + ['NA' for x in lake_df.iloc[i][feats].values]))
	# mat_csv.append(",".join(["nhdhr_"+target_id,"9 MODEL ENSEMBLE",str(mat_rmse)] + ['NA' for x in lake_df.iloc[i][feats].values]))

	# csv.append(",".join(["nhdhr_"+target_id,str(mat_rmse),str(glm_rmse),str(mean_rmse),str(lowest_rmse)]))
	print("Total rmse=", mat_rmse)
	spcorr = srcorr_per_lake[targ_ct]
	rmse_per_lake[targ_ct] = mat_rmse

	# if ind_rmses.mean() - mat_rmse > most_imp_diff:
	# 	most_imp_diff = ind_rmses.mean() - mat_rmse
	# 	most_imp_site_id = target_id
	# 	most_imp_total_rmse = mat_rmse
	# 	most_imp_ind_rmses = ind_rmses
	# 	most_imp_source_ids = top_ids


	# csv.append(",".join([str(target_id), str(mat_rmse)]))


# with open("../../../results/transfer_learning/rf_testset.csv",'a') as file:
# 	for line in test_lake_csv:
# 		file.write(line)
# 		file.write('\n')
with open('pgmtl9_ens_results.csv','w') as file:
    for line in csv:
        file.write(line)
        file.write('\n')
with open('pgmtl9_ens_source_results.csv','w') as file:
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