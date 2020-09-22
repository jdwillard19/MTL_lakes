import pandas as pd
import numpy as np
import pdb
import sys
import os
sys.path.append('../data')
from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune2, parseMatricesFromSeqs
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal_
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr
from joblib import dump, load
import re


metadata = pd.read_feather("../../metadata/lake_metadata_full.feather")
metadata.set_index('site_id', inplace=True)
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
train_lakes_wp = np.unique(glm_all_f['target_id'].values) #with prefix

ids = pd.read_csv('../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
n_lakes = len(train_lakes)
test_lakes = ids[~np.isin(ids, train_lakes)]
assert len(test_lakes) == 305
k = 1
output_to_file = True

save_file_path = "../../results/pgmtl_results_single_source.csv"

#########################################################################################
#paste features found in "pbmtl_feature_selection.py" here
feats = ['n_obs_sp', 'n_obs_su', 'dif_max_depth', 'dif_surface_area',
       'dif_glm_strat_perc', 'perc_dif_max_depth', 'perc_dif_surface_area',
       'perc_dif_sqrt_surface_area']
###################################################################################


#load metamodel
model_path = '../../models/metamodel_pgdl_RMSE_GBR.joblib'
model = load(model_path)

#csv to write to
mat_csv = ["target_id,source_id,meta_rmse,spearman,pb0_rmse,pgmtl_rmse"]
for feat in feats:
    mat_csv[0] = mat_csv[0] + ','+str(feat)


#data structures to fill
rmse_per_lake = np.empty(test_lakes.shape[0])
glm_rmse_per_lake = np.empty(test_lakes.shape[0])
srcorr_per_lake = np.empty(test_lakes.shape[0])

meta_rmse_per_lake = np.empty(test_lakes.shape[0])
med_meta_rmse_per_lake = np.empty(test_lakes.shape[0])
rmse_per_lake[:] = np.nan
glm_rmse_per_lake[:] = np.nan
meta_rmse_per_lake[:] = np.nan
csv = []
csv.append('target_id,rmse,rmse_pred,spearman,glm_rmse')


#where to output files
output_path = "../../results/outputs/"
if not os.path.exists(output_path):
    os.mkdir(output_path)


err_per_source = np.empty((145,len(test_lakes)))
for targ_ct, target_id in enumerate(test_lakes): #for each target lake
    print(str(targ_ct),'/',len(test_lakes),':',target_id)
    lake_df = pd.DataFrame()
    lake_id = target_id

    lake_df = pd.read_feather("../../metadata/diffs/target_nhdhr_"+lake_id+".feather")
    lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes_wp)]
    X = pd.DataFrame(lake_df[feats])


    y_pred = model.predict(X)
    lake_df['rmse_pred'] = y_pred

    lake_df.sort_values(by=['rmse_pred'], inplace=True)
    lowest_rmse = lake_df.iloc[0]['rmse_pred']

    top_ids = [str(j) for j in lake_df.iloc[:k]['site_id']]
    
    best_site = top_ids[0]




    data_dir_target = "../../data/processed/"+target_id+"/" 
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
        source_id = re.search('nhdhr_(.*)', source_id).group(1)
        #load source model
        load_path = "../../models/"+source_id+"/PGRNN_source_model_0.7"
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
                #to store output
                output_mats[i,:,:] = outputm_npy
                if i == 0:
                    #store label
                    label_mats = labelm_npy
                loss_output = outputm_npy[~np.isnan(labelm_npy)]
                loss_label = labelm_npy[~np.isnan(labelm_npy)]

                mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())
                print(source_id+" rmse=", mat_rmse)
                err_per_source[i,targ_ct] = mat_rmse

                glm_rmse = float(metadata.loc["nhdhr_"+target_id].glm_uncal_rmse_full)

                mat_csv.append(",".join(["nhdhr_"+target_id,"nhdhr_"+ source_id,str(meta_rmse_per_lake[targ_ct]),str(srcorr_per_lake[targ_ct]), str(glm_rmse),str(mat_rmse)] + [str(x) for x in lake_df.iloc[i][feats].values]))


    #save model 
    total_output_npy = np.average(output_mats, axis=0)

    if output_to_file:
        outputm_npy = np.transpose(total_output_npy)
        label_mat= np.transpose(label_mats)
        output_df = pd.DataFrame(data=outputm_npy, columns=[str(float(x/2)) for x in range(outputm_npy.shape[1])], index=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
        label_df = pd.DataFrame(data=label_mat, columns=[str(float(x/2)) for x in range(label_mat.shape[1])], index=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
        output_df.rename(columns={'index': 'depth'})
        label_df.rename(columns={'index': 'depth'})

        assert np.isfinite(np.array(output_df.values[:,1:],dtype=np.float32)).all(), "nan output"
        lake_output_path = output_path+target_id
        if not os.path.exists(lake_output_path):
            os.mkdir(lake_output_path)
        output_df.to_feather(lake_output_path+"/PGMTL_outputs.feather")
        
    loss_output = total_output_npy[~np.isnan(label_mats)]
    loss_label = label_mats[~np.isnan(label_mats)]
    mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())

    print("Total rmse=", mat_rmse)
    spcorr = srcorr_per_lake[targ_ct]
    rmse_per_lake[targ_ct] = mat_rmse




with open(save_file_path,'w') as file:
    for line in mat_csv:
        file.write(line)
        file.write('\n')




print("median srcorr: ",np.median(srcorr_per_lake))
print("median meta test RMSE(med): ",np.median(med_meta_rmse_per_lake))
print("median test RMSE: ",np.median(rmse_per_lake))

