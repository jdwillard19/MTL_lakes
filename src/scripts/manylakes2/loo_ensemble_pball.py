import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut
import pdb
import sys
sys.path.append('../../data')
from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune2, parseMatricesFromSeqs
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal_
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
import math
import re
# build models to predict which models to use for transfer learning
# method = 'linear'
# k = 6
# method = 'svr'
# method = 'geotemp'
use_gpu = True

#get all lake names, divide into train and test
err_cutoff = 0
#get all lake names, divide into train and test
metadata = pd.read_feather("../../../metadata/lake_metadata_2700plus.feather")
metadata.set_index('site_id', inplace=True)
ids = pd.read_csv('../../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
glm_all_f = pd.read_csv("../../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../../results/transfer_learning/glm/train_rmses_pball.feather")
train_lakes = np.array([re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)])
n_lakes = len(train_lakes)
test_lakes = ids[~np.isin(ids, train_lakes)]

feats = ['n_obs', 'obs_temp_mean', 'dif_max_depth', 'dif_surface_area', 'perc_dif_max_depth', 'perc_dif_sqrt_surface_area']

n_test_lakes = len(train_lakes)

print(train_lakes.shape[0], " training lakes")
#model params
seq_length = 350 #how long of sequences to use in model
begin_loss_ind = 175#index in sequence where we begin to calculate error or predict
n_features = 8  #number of physical drivers
win_shift = 175 #how much to slide the window on training set each time


# #compile training data from all training lakes
# train_df = pd.DataFrame()
# depth_marg_percent = .2
# min_cand = 7
# for i, lake_id in enumerate(train_lakes):
# 	new_df = pd.DataFrame()
# 	# temp_df = pd.read_csv("../../../results/transfer_learning/target_"+lake_id+"/resultsPGRNN_CV")
# 	temp_df = pd.read_csv("../../../results/transfer_learning/target_"+lake_id+"/resultsPGRNNbasic5")
	
# 	if depth_filter:	
# 		#get max depth
# 		max_d = metadata.loc[int(lake_id)].max_depth
# 		margin = depth_marg_percent*max_d

# 		#fiter by depth margin
# 		new_df = temp_df[temp_df['max_depth'] < margin]
# 		while new_df.shape[0] < min_cand:
# 			margin += .1
# 			new_df = temp_df[temp_df['max_depth'] < margin]
# 	else:
# 		new_df = temp_df

# 	train_df = pd.concat([train_df, new_df], ignore_index=True)

# model = []

# if method == 'linear':
# 	model = LinearRegression()
# elif method == 'rf':
# 	model = RandomForestRegressor(max_depth=None,random_state=1, n_estimators=10000, max_features=3, n_jobs=-1)
# elif method == 'svr':
# 	model = SVR()

# #define x and y
# # X_trn = pd.DataFrame(train_df[['geo','tempdtw','surf_area','max_depth','lat','long','k_d']])
# X_trn = pd.DataFrame(train_df[['canopy','SDF','surf_area','max_depth','lat','long', 'k_d']])
# y_trn = pd.DataFrame(train_df['rmse'])
# # print("CVscore: ",cross_val_score(model, X_trn, y_trn, cv=n_train_lakes).mean())


# #fit model
# if method != "geotemp":
# 	model.fit(X_trn, y_trn)

# 	#predict rmse
# 	y_pred_trn = model.predict(X_trn)
# 	print("params", model.get_params())

# 	print("model score " , model.score(X_trn, y_trn))

# forest = model
# import matplotlib.pyplot as plt
# importances = forest.feature_importances_
# std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]

# # Print the feature ranking
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
# sys.exit()

############################################################################
#use model to predict top 3 for every source target lake, then predict target
################################################################################
test_lakes = train_lakes

test_lake_csv = []
n_lakes = test_lakes.shape[0]
k_arr = np.arange(2,11)
# cutoff_arr = np.arange(1,12)
# cutoff_arr = np.array([5])
csv = []

err_array = np.empty((29, k_arr.shape[0]))
err_array[:] = np.nan

with open("../../../results/transfer_learning/loocv_pball.csv",'a') as file:
	for n in range(29):
		begin = int(n*5)
		end = int((n+1)*5 - 1)
		train_df = pd.DataFrame()
		temp_test_lakes = np.array([train_lakes[begin:end]]).flatten()
		temp_train_lakes = np.delete(train_lakes, np.arange(begin, end))
		print("Lakes ",begin, " thru ", end)

		# x_trn = metadata[np.isin(metadata['nhd_id'], [int(a) for a in temp_train_lakes])][['max_depth', 'surface_area', 'latitude', 'longitude','canopy','SDF', 'K_d']]
		# x_tst = metadata[np.isin(metadata['nhd_id'], [int(a) for a in temp_test_lakes])][['max_depth', 'surface_area', 'latitude', 'longitude','canopy','SDF', 'K_d']]
		# y_trn = metadata[np.isin(metadata['nhd_id'], [int(a) for a in temp_train_lakes])]['glm_uncal_rmse']
		# y_tst = metadata[np.isin(metadata['nhd_id'], [int(a) for a in temp_test_lakes])]['glm_uncal_rmse']
		# model = RandomForestRegressor(n_estimators=5000, max_depth=15)
		# model.fit(x_trn, y_trn)
		# glm_rmse_pred = pd.DataFrame()
		# glm_rmse_pred['rmse'] = model.predict(x_tst)
		# print("pred vs actual glm ", glm_rmse_pred['rmse'].values[0], " -- ", y_tst.values[0])

		# test_ids = metadata[np.isin(metadata['nhd_id'], [int(a) for a in temp_test_lakes])]['nhd_id']
		# glm_w_ids = pd.DataFrame()
		# glm_w_ids['id'] = test_ids
		# glm_rmse_pred.reset_index(drop=True, inplace=True)
		# glm_w_ids.reset_index(drop=True, inplace=True)

		# glm_pred = pd.concat([glm_rmse_pred, glm_w_ids], ignore_index=False, axis=1)
		for _, lake_id in enumerate(temp_train_lakes):
			new_df = pd.DataFrame()
			# temp_df = pd.read_csv("../../../results/transfer_learning/target_"+lake_id+"/resultsPGRNN_CV")
			# temp_df = pd.read_csv("../../../results/transfer_learning/target_"+lake_id+"/resultsPGRNNbasic5")
			lake_df_res = pd.read_csv("../../../results/transfer_learning/target_"+lake_id+"/resultsPGRNNbasic_pball") 
			lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']

			lake_df = pd.read_feather("../../../metadata/diff/target_"+lake_id+"_pball_update.feather")

			# lake_df = lake_df[np.isin(lake_df['source_id'], train_lakes)]
			lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes)]
			lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
			lake_df = pd.merge(left=lake_df, right=lake_df_res, left_on='site_id', right_on='source_id')
			# glm_uncal_rmse = float(metadata[metadata['nhd_id']==int(lake_id)].glm_uncal_rmse)
			# temp_df['rmse_improve'] = temp_df['rmse'] - glm_uncal_rmse
			new_df = lake_df
			train_df = pd.concat([train_df, new_df], ignore_index=True)
		

		model = GradientBoostingRegressor(n_estimators=1000, learning_rate=.1)
		X_trn = pd.DataFrame(train_df[feats])
		# y_trn = pd.DataFrame(train_df['rmse_improve'])
		y_trn = pd.DataFrame(train_df['rmse'])
		model.fit(X_trn, np.ravel(y_trn))

		for i_k, k in enumerate(k_arr):
			print("select= ", k)
			rmse_per_lake = np.empty(temp_test_lakes.shape[0])
			rmse_per_lake[:] = np.nan
			for targ_ct, target_id in enumerate(temp_test_lakes): #for each target lake
				# print("target lake ", targ_ct, ":", target_id)
				# lake_df_res = pd.read_csv("../../../results/transfer_learning/target_"+lake_id+"/resultsPGRNNbasic_pball") 
				# lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']
				lake_id = target_id
				lake_df = pd.read_feather("../../../metadata/diff/target_"+lake_id+"_pball_update.feather")

				# lake_df = lake_df[np.isin(lake_df['source_id'], train_lakes)]
				lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes)]
				# lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
				# lake_df = pd.merge(left=lake_df, right=lake_df_res, left_on='site_id', right_on='source_id')
				X = pd.DataFrame(lake_df[feats])
				# y = pd.DataFrame(lake_df['rmse'])

				y_pred = []
				top_ids = []

				# glm_pred_val = glm_pred[glm_pred['id']==int(target_id)]['rmse'].values[0]
				y_pred = model.predict(X)
				lake_df['rmse_pred'] = y_pred 
				# best_rmse = lake_df['rmse_pred'].values.min()
				# upper_bound = best_rmse*(1+b)
				# if best_rmse < 0:
					# upper_bound = best_rmse+.9
				# upper_bound = best_rmse+b
				# selected = lake_df['rmse_pred'] <= upper_bound 
				#get top predicted lakes
				# lake_df.sort_values(by=['rmse_pred'], inplace=True)
				top_ids = [str(j) for j in lake_df.iloc[:int(k)]['site_id']]
				# top_ids = [str(j) for j in lake_df[lake_df['rmse_pred'] < b]['source_id']]
				# top_ids = [str(j) for j in lake_df[selected]['source_id']]
				# print("top source lakes, ", top_ids)
				#define target test data to use
				data_dir_target = "../../data/processed/lake_data/"+target_id+"/" 
				#target agnostic model and data params
				use_gpu = True
				n_features = 8
				# n_hidden = 20
				seq_length = 350
				win_shift = 175
				begin_loss_ind = 175

				(_, _, tst_data_target, tst_dates_target, unique_tst_dates_target, all_data_target, all_phys_data_target, all_dates_target,
				_) = buildLakeDataForRNN_manylakes_finetune2(target_id, data_dir_target, seq_length, n_features,
				                                   win_shift = win_shift, begin_loss_ind = begin_loss_ind, 
				                                   latter_third_test=True, outputFullTestMatrix=True, 
				                                   sparseTen=False, realization='none', allTestSeq=False, oldFeat=False)

				#useful values, LSTM params
				batch_size = all_data_target.size()[0]
				u_depths_target = np.unique(tst_data_target[:,0,0])
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
				label_mats = np.empty((n_depths, n_test_dates_target)) 
				output_mats[:] = np.nan
				label_mats[:] = np.nan

				for i, source_id in enumerate(top_ids): 
					#for each top id

					#load source model
					load_path = "../../../models/single_lake_models/"+source_id+"/PGRNN_basic_normAll_pball"
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
							#store output
							output_mats[i,:,:] = outputm_npy
							if i == 0:
								#store label
								label_mats = labelm_npy
							loss_output = outputm_npy[~np.isnan(labelm_npy)]
							loss_label = labelm_npy[~np.isnan(labelm_npy)]

							mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())
							# print(source_id+" rmse=", mat_rmse)


				#save model 
				total_output_npy = np.average(output_mats,axis=0)
				loss_output = total_output_npy[~np.isnan(label_mats)]
				loss_label = label_mats[~np.isnan(label_mats)]
				mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())
				print("Total rmse=", mat_rmse)
				if math.isnan(mat_rmse):
					print("NAN RMSE ERROR")
					sys.exit()
				rmse_per_lake[targ_ct] = mat_rmse
				# rmse_per_lake[targ_ct] = i*k

			# print("mean RMSE: ",rmse_per_lake.mean())
			# err_array[i,k-1] = rmse_per_lake.mean()
			err_array[n,i_k] = rmse_per_lake.mean()
		err_str_arr = [str(err_array[n,m]) for m in range(err_array.shape[1])]
		file.write(",".join([str(n)] + err_str_arr))
		file.write('\n')

	avg_err_array = np.empty(k_arr.shape[0])
	for j in range(k_arr.shape[0]):
		avg_err_array[j] = err_array[:,j].mean()

	print(k_arr)
	print(avg_err_array)
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