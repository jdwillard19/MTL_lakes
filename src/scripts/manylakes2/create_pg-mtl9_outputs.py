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
import os


metadata = pd.read_feather("../../../metadata/lake_metadata_baseJune2020.feather")
metadata.set_index('site_id', inplace=True)
ids = pd.read_csv('../../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
glm_all_f = pd.read_csv("../../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../../results/transfer_learning/glm/glm_meta_train_rmses.feather")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)
test_lakes = ids[~np.isin(ids, train_lakes)]
k = 9
biases = []
# print(train_lakes.shape[0], " training lakes")



feats = ['n_obs', 'obs_temp_mean', 'dif_max_depth', 'dif_surface_area',
       'dif_rh_mean_au', 'dif_lathrop_strat', 'dif_glm_strat_perc',
       'perc_dif_max_depth', 'perc_dif_surface_area',
       'perc_dif_sqrt_surface_area']


model = load("PGMTL_GBR_pball_Aug21.joblib")





rmse_per_lake = np.empty(test_lakes.shape[0])
glm_rmse_per_lake = np.empty(test_lakes.shape[0])
srcorr_per_lake = np.empty(test_lakes.shape[0])

meta_rmse_per_lake = np.empty(test_lakes.shape[0])
med_meta_rmse_per_lake = np.empty(test_lakes.shape[0])
rmse_per_lake[:] = np.nan
glm_rmse_per_lake[:] = np.nan
meta_rmse_per_lake[:] = np.nan

test_lakes = np.array(['120018510', '120020636', '91688597', '82815984'])
err_per_source = np.empty((145,len(test_lakes)))

for targ_ct, target_id in enumerate(test_lakes): #for each target lake
	print(str(targ_ct),'/',len(test_lakes),':',target_id)
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
	# min_rmse = lake_df.iloc[:k]['rmse_pred'].mean()

	top_ids = [str(j) for j in lake_df.iloc[:k]['site_id']] + [str(j) for j in lake_df.iloc[95:100]['site_id']]
	
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
		if not os.path.exists("./mtl_outputs_for_fig/nhdhr_"+target_id):
			os.mkdir("./mtl_outputs_for_fig/nhdhr_"+target_id)
		#load source model
		load_path = "../../../models/single_lake_models/"+source_id+"/PGRNN_source_model_0.7"
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

				output_df = pd.DataFrame(data=np.transpose(outputm_npy), columns=[str(float(x/2)) for x in range(outputm_npy.shape[0])], index=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
				output_df.rename(columns={'index': 'depth'})
				label_df = pd.DataFrame(data=np.transpose(labelm_npy), columns=[str(float(x/2)) for x in range(labelm_npy.shape[0])], index=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
				label_df.rename(columns={'index': 'depth'})

				if i == 0:
					output_df.to_feather('./mtl_outputs_for_fig/nhdhr_'+target_id+'/top_source'+str(i)+'_nhdhr'+source_id+'_output.feather')
					label_df.to_feather('./mtl_outputs_for_fig/nhdhr_'+target_id+'/labels.feather')
				elif i < 9:
					output_df.to_feather('./mtl_outputs_for_fig/nhdhr_'+target_id+'/source'+str(i)+'_nhdhr'+source_id+'_output.feather')
					label_df.to_feather('./mtl_outputs_for_fig/nhdhr_'+target_id+'/labels.feather')
				else:
					output_df.to_feather('./mtl_outputs_for_fig/nhdhr_'+target_id+'/source'+str(i+86)+'_nhdhr'+source_id+'_output.feather')
					label_df.to_feather('./mtl_outputs_for_fig/nhdhr_'+target_id+'/labels.feather')
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


	#save model 
	total_output_npy = np.average(output_mats[:9,:,:], axis=0)


	output_df = pd.DataFrame(data=np.transpose(total_output_npy), columns=[str(float(x/2)) for x in range(outputm_npy.shape[0])], index=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
	output_df.rename(columns={'index': 'depth'})
	pdb.set_trace()
	output_df.to_feather('./mtl_outputs_for_fig/nhdhr_'+target_id+'/9source_ensemble_output.feather')
	# output_df = pd.DataFrame(data=total_output_npy, index=[str(float(x/2)) for x in range(outputm_npy.shape[0])], columns=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
	# output_df.rename(columns={'index': 'depth'})
	# output_df.to_feather("good_outputs/target_nhdhr_"+target_id+"_ENSEMBLE_outputs.feather")
	loss_output = total_output_npy[~np.isnan(label_mats)]
	loss_label = label_mats[~np.isnan(label_mats)]
	mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())
	glm_rmse = float(metadata.loc[target_id].glm_uncal_rmse_full)

	print("Total rmse=", mat_rmse)
	spcorr = srcorr_per_lake[targ_ct]
	rmse_per_lake[targ_ct] = mat_rmse

