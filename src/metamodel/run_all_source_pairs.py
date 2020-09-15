import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.init import xavier_normal_
from datetime import date
import pandas as pd
import pdb
import random
import math
import sys
import re
import os
sys.path.append('../data')
sys.path.append('../models')
sys.path.append('../../models')
from io_operations import saveTemperatureMatrix
from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune2, parseMatricesFromSeqs


#read lake metadata file to get all the lakenames
# metadata = pd.read_feather("../../../metadata/lake_metadata_wNew2.csv")
ids = pd.read_csv('../../metadata/pball_site_ids.csv', header=None)
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
source_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
target_lakes = [str(i) for i in ids.values.flatten()] # to loop through all lakes


#target agnostic model and data params
use_gpu = True
n_features = 8
# n_hidden = 20
seq_length = 350
win_shift = 175
begin_loss_ind = 0

#define train and test lakes
# test_lakes = np.array(['1102088', '1101506', '13631637', '1099476', '120053694', '9022741', '1101864'])
# test_lakes = np.array(['13293262', '1097324', '1109052', '1109136', '1099420', '1099432', '1099450', '1099432'])


#run params
save = True









#############################################3
##for every target lake, load every source lake and predict
#########################################################3
for target_ct, target_id in enumerate(target_lakes): 
    # if target_ct < 39:
    #     continue
    nid = target_id

    data_dir_target = "../../data/processed/lake_data/"+target_id+"/" 


    (_, _, tst_data_target, tst_dates_target, unique_tst_dates_target, all_data_target, all_phys_data_target, all_dates_target,
    hypsography_target) = buildLakeDataForRNN_manylakes_finetune2(target_id, data_dir_target, seq_length, n_features,
                                       win_shift = win_shift, begin_loss_ind = begin_loss_ind, 
                                       outputFullTestMatrix=True, 
                                       allTestSeq=True, postProcessSplits=False)


    #csv to append to for each source lake
    csv_targ = []
    first_row_str_all = "source_id,rmse"
    csv_targ.append(first_row_str_all)

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


    for source_ct, source_id in enumerate(source_lakes):
        nid = source_id

        if source_id == target_id:
            continue

        print("targ (",target_ct,"/",len(target_lakes),") ",target_id, ": source (",source_ct,"/",len(source_lakes),"): ",source_id)


        #output matrix
        n_lakes = 1
        output_mats = np.empty((n_lakes, n_depths, n_test_dates_target))
        label_mats = np.empty((n_depths, n_test_dates_target)) 
        output_mats[:] = np.nan
        label_mats[:] = np.nan
        #save output path
        if not os.path.exists("../../results/transfer_learning/target_"+target_id+"/"):
            os.mkdir("../../results/transfer_learning/target_"+target_id)

        if not os.path.exists("../../results/transfer_learning/target_"+target_id+"/source_"+source_id):
            os.mkdir("../../results/transfer_learning/target_"+target_id+"/source_"+source_id)



        #load model
        # load_path = "../../models/single_lake_models/"+source_id+"/PGRNN_source_model"
        load_path = "../../models/single_lake_models/"+source_id+"/PGRNN_source_model_0.7"

        n_hidden = torch.load(load_path)['state_dict']['out.weight'].shape[1]
        lstm_net = LSTM(n_features, n_hidden, batch_size)
        if use_gpu:
            lstm_net = lstm_net.cuda(0)
        pretrain_dict = torch.load(load_path)['state_dict']
        model_dict = lstm_net.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        lstm_net.load_state_dict(pretrain_dict)

        #data loader for test data
        testloader = torch.utils.data.DataLoader(tst_data_target, batch_size=tst_data_target.size()[0], shuffle=False, pin_memory=True)

        mse_criterion = nn.MSELoss()

        lstm_net.eval()
        with torch.no_grad():
            avg_mse = 0
            for i, data in enumerate(testloader, 0):
                #this loop is dated, there is now only one item in testloader, however in the future we could reduce batch size if we want

                #parse data into inputs and targets
                inputs = data[:,:,:n_features].float()
                targets = data[:,:,-1].float()
                targets = targets[:, begin_loss_ind:]
                tmp_dates = tst_dates_target[:, begin_loss_ind:]
                depths = inputs[:,:,0]

                if use_gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                #run model predict
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

                #format prediction and labels into depths by days matrices
                (outputm_npy, labelm_npy) = parseMatricesFromSeqs(pred.cpu().numpy(), targets.cpu().numpy(), depths, tmp_dates, n_depths, 
                                                                n_test_dates_target, u_depths_target,
                                                                unique_tst_dates_target) 
                #store output
                output_mats[0,:,:] = outputm_npy
                label_mats = labelm_npy
                loss_output = outputm_npy[~np.isnan(labelm_npy)]
                loss_label = labelm_npy[~np.isnan(labelm_npy)]

                mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())
                print(source_id+" rmse=" + str(mat_rmse) + " on " + target_id)


                            #calculate energy at each timestep
                output_torch = torch.from_numpy(outputm_npy).float()
                if use_gpu:
                    output_torch = output_torch.cuda()
                avg_mse = avg_mse.cpu().numpy()


            row_vals = [source_id, mat_rmse]
            row_vals_str = [str(i) for i in row_vals]

            #append
            csv_targ.append(",".join(row_vals_str))



    with open("../../results/transfer_learning/target_"+target_id+"/results_all_source_models2.csv",'w') as file:
        # print("saving to ../../../results/transfer_learning/target_"+target_id+"/resultsPGRNNbasic6")
        for line in csv_targ:
            file.write(line)
            file.write('\n')
    


