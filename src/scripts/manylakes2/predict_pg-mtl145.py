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
k = len(train_lakes)

mat_csv = []
csv = []
mat_csv.append(",".join(["target_id","source_id","glm_rmse","rmse","source_observations",\
                         "mean_source_observation_temp","diff_max_depth",'diff_surface_area', 'diff_RH_mean_autumn', 'diff_lathrop_strat','dif_glm_strat_perc',\
                         'percent_diff_max_depth', 'percent_diff_surface_area','perc_dif_sqrt_surface_area']))


feats = ['n_obs', 'obs_temp_mean', 'dif_max_depth', 'dif_surface_area',
       'dif_rh_mean_au', 'dif_lathrop_strat', 'dif_glm_strat_perc',
       'perc_dif_max_depth', 'perc_dif_surface_area',
       'perc_dif_sqrt_surface_area']

train = False
model = []

if train:
	# #compile training data from all training lakes
	train_df = pd.DataFrame()

	for _, lake_id in enumerate(train_lakes):

		new_df = pd.DataFrame()
		lake_df_res = pd.read_csv("../../../results/transfer_learning/target_"+lake_id+"/results_all_source_models.csv") 
		lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']
		lake_df = pd.read_feather("../../../metadata/diff/target_"+lake_id+"_pball_Aug2020.feather")
		lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes)]
		lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
		lake_df = pd.merge(left=lake_df, right=lake_df_res.astype('object'), left_on='site_id', right_on='source_id')
		new_df = lake_df
		train_df = pd.concat([train_df, new_df], ignore_index=True)

	X_trn = pd.DataFrame(train_df[feats])
	y_trn = np.array([float(x) for x in np.ravel(pd.DataFrame(train_df['rmse']))])



	model = GradientBoostingRegressor(n_estimators=3700, learning_rate=.05)
	model.fit(X_trn, y_trn)


	dump(model, "PGMTL_GBR_pball.joblib")

model = load("PGMTL_GBR_pball_Aug21.joblib")

rmse_per_lake = np.empty(test_lakes.shape[0])
glm_rmse_per_lake = np.empty(test_lakes.shape[0])
srcorr_per_lake = np.empty(test_lakes.shape[0])

meta_rmse_per_lake = np.empty(test_lakes.shape[0])
med_meta_rmse_per_lake = np.empty(test_lakes.shape[0])
rmse_per_lake[:] = np.nan
glm_rmse_per_lake[:] = np.nan
meta_rmse_per_lake[:] = np.nan


err_per_source = np.empty((145,len(test_lakes)))
# test_lakes = np.array(['120020398'])
for targ_ct, target_id in enumerate(test_lakes): #for each target lake
	lake_df = pd.DataFrame()
	lake_id = target_id


	lake_df_res = pd.read_csv("../../../results/transfer_learning/target_"+lake_id+"/results_all_source_models2.csv") 
	lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']

	lake_df = pd.read_feather("../../../metadata/diff/target_"+lake_id+"_pball_Aug2020.feather")
	lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes)]
	lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
	lake_df = pd.merge(left=lake_df, right=lake_df_res.astype('object'), left_on='site_id', right_on='source_id')

	X = pd.DataFrame(lake_df[feats])
	y_pred = model.predict(X)
	lake_df['rmse_pred'] = y_pred
	y_act = np.array([float(x) for x in np.ravel(lake_df['rmse'].values)])
	meta_rmse_per_lake[targ_ct] = np.median(np.sqrt(((y_pred - y_act) ** 2).mean()))
	srcorr_per_lake[targ_ct] = spearmanr(y_pred, y_act).correlation

	lake_df.sort_values(by=['rmse_pred'], inplace=True)
	lowest_rmse = lake_df.iloc[0]['rmse_pred']
	mean_rmse = lake_df.iloc[:k]['rmse_pred'].mean()
	rmse_pred = lake_df['rmse_pred'].values
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
		print(str(targ_ct),'/',len(test_lakes),': ',target_id,"--- ",str(i),"/",str(len(top_ids)),": ",source_id)

		#for each top id

		#load source model
		load_path = "../../../models/single_lake_models/"+source_id+"/PGRNN_source_model_0.7"
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
				mat_csv.append(",".join(["nhdhr_"+target_id,"nhdhr_"+ source_id,str(mat_rmse),str(rmse_pred[i])] + [str(x) for x in lake_df.iloc[i][feats].values]))



# with open('pgmtl9_ens_results.csv','w') as file:
#     for line in csv:
#         file.write(line)
#         file.write('\n')
with open('pgmtl145_results_aug.csv','w') as file:
    for line in mat_csv:
        file.write(line)
        file.write('\n')


