import pandas as pd
import numpy as np
import pdb
import sys
sys.path.append('../data')
from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune2, parseMatricesFromSeqs
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal_
from sklearn.ensemble import GradientBoostingRegressor
import math
import re


#######################################################################
# (Sept 2020 - Jared) - this script uses cross validation on the training lakes
#                       to estimate the best number of lakes "k" to ensemble
#############################################################################

use_gpu = True


ids = pd.read_csv('../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_lakes = np.array([re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)])[:140]
# train_lakes_wp = np.unique(glm_all_f['target_id'].values) #with prefix

n_test_lakes = len(train_lakes)

print(train_lakes.shape[0], " training lakes")
#model params
seq_length = 350 #how long of sequences to use in model
begin_loss_ind = 0#index in sequence where we begin to calculate error or predict
n_features = 8  #number of physical drivers
win_shift = 175 #how much to slide the window on training set each time


#########################################################################################
#paste features found in "pbmtl_feature_selection.py" here
feats = ['n_obs_sp', 'n_obs_su', 'dif_max_depth', 'dif_surface_area',
       'dif_glm_strat_perc', 'perc_dif_max_depth', 'perc_dif_surface_area',
       'perc_dif_sqrt_surface_area']
###################################################################################

test_lakes = train_lakes

test_lake_csv = []
n_lakes = test_lakes.shape[0]
k_arr = np.arange(2,11)
n_fold = 7 #cross validation folds
err_array = np.empty((n_fold, k_arr.shape[0]))
err_array[:] = np.nan
tst_size = int(np.round(n_lakes / n_fold))

for n in range(n_fold):
    #for each cross validation fold

    #decleare fold's train and test
    lower_ind = n*tst_size
    upper_ind = (n+1)*tst_size
    tst_inds = np.arange(lower_ind, upper_ind)

    #loo cross validation across all lakes
    train_df = pd.DataFrame()
    temp_test_lakes = np.ravel(np.array([train_lakes[tst_inds]]))
    temp_train_lakes = np.delete(train_lakes, tst_inds)
    temp_train_lakes_wp = ["nhdhr_"+str(x) for x in temp_train_lakes]
    print("fold  ", n)


    for _, lake_id in enumerate(temp_train_lakes):
            #for every lake in instance of training within CV fold
        #get performance results (metatargets), filter out target as source
        lake_df_res = pd.read_csv("../../results/transfer_learning/target_"+lake_id+"/resultsPGRNNbasic_pball",header=None,names=['source_id','rmse'])
        lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']

        #get metadata differences between target and all the sources
        lake_df = pd.read_feather("../../metadata/diffs/target_nhdhr_"+lake_id+".feather")
        lake_df = lake_df[np.isin(lake_df['site_id'], temp_train_lakes_wp)]
        lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], temp_train_lakes)]
        lake_df_res['source_id2'] = ['nhdhr_'+str(x) for x in lake_df_res['source_id'].values]
        lake_df = pd.merge(left=lake_df, right=lake_df_res.astype('object'), left_on='site_id', right_on='source_id2')
        new_df = lake_df
        train_df = pd.concat([train_df, new_df], ignore_index=True)
    
    model = GradientBoostingRegressor(n_estimators=400,learning_rate=.05)
    X_trn = pd.DataFrame(train_df[feats])
    y_trn = pd.DataFrame(train_df['rmse'])
    model.fit(X_trn, np.ravel(y_trn))

    for i_cut, cut in enumerate(k_arr):
        k = int(cut)
        print("k= ", cut)
        rmse_per_lake = np.empty(temp_test_lakes.shape[0])
        rmse_per_lake[:] = np.nan
        for targ_ct, target_id in enumerate(temp_test_lakes): #for each target lake
            print("target lake ", targ_ct, ":", target_id)

            #get performance results (metatargets), filter out target as source
            lake_df_res = pd.read_csv("../../results/transfer_learning/target_"+target_id+"/resultsPGRNNbasic_pball",header=None,names=['source_id','rmse'])
            lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']

            #get metadata differences between target and all the sources
            lake_df = pd.read_feather("../../metadata/diffs/target_nhdhr_"+lake_id+".feather")
            lake_df = lake_df[np.isin(lake_df['site_id'], temp_train_lakes_wp)]
            lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], temp_train_lakes)]
            lake_df_res['source_id2'] = ['nhdhr_'+str(x) for x in lake_df_res['source_id'].values]
            lake_df = pd.merge(left=lake_df, right=lake_df_res.astype('object'), left_on='site_id', right_on='source_id2')
            X = pd.DataFrame(lake_df[feats])


            y_pred = model.predict(X)
            lake_df['rmse_pred'] = y_pred
            y_act = lake_df['rmse']
            lake_df.sort_values(by=['rmse_pred'], inplace=True)
            lowest_rmse = lake_df.iloc[0]['rmse_pred']

            top_ids = [str(j) for j in lake_df.iloc[:int(k)]['site_id']]
            
            best_site = top_ids[0]

            #define target test data to use
            data_dir_target = "../../data/processed/lake_data/"+target_id+"/" 
            (_, _, tst_data_target, tst_dates_target, unique_tst_dates_target, all_data_target, all_phys_data_target, all_dates_target,
            hypsography_target) = buildLakeDataForRNN_manylakes_finetune2(target_id, data_dir_target, seq_length, n_features,
                                               win_shift = win_shift, begin_loss_ind = begin_loss_ind, 
                                               outputFullTestMatrix=True, allTestSeq=False, oldFeat=False)
            

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
                source_id = re.search('nhdhr_(.*)', source_id).group(1)
                #for each top id

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

        err_array[n,i_cut] = rmse_per_lake.mean()
    err_str_arr = [str(err_array[n,m]) for m in range(err_array.shape[1])]

avg_err_array = np.empty(k_arr.shape[0])
for j in range(k_arr.shape[0]):
    avg_err_array[j] = err_array[:,j].mean()

print("k array tested: ",str(k_arr))
print("err per k (select lowest): ", avg_err_array)
# 


