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
import os
sys.path.append('../../data')
sys.path.append('../../models')
from io_operations import saveTemperatureMatrix
from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune2, parseMatricesFromSeqs


#read lake metadata file to get all the lakenames
# metadata = pd.read_feather("../../../metadata/lake_metadata_wNew2.csv")
meta_new = pd.read_feather("../../../metadata/lake_metadata_wNew2.feather")
metadata = meta_new
lakenames = [str(i) for i in metadata.iloc[:,0].values] # to loop through all lakes
lakenames_new = [str(i) for i in meta_new.iloc[:,0].values] # to loop through all lakes
# pdb.set_trace()
metadata.set_index("nhd_id", inplace=True)
# meta_new.set_index("nhd_id", inplace=True)
metadata.columns = [c.replace(' ', '_') for c in metadata.columns]
meta_new.columns = [c.replace(' ', '_') for c in meta_new.columns]
csv_all = [] # to write results to
first_row_str_all = "target_id,source_id,rmse"
csv_all.append(first_row_str_all)

#target agnostic model and data params
use_gpu = True
n_features = 8
# n_hidden = 20
seq_length = 350
win_shift = 175
begin_loss_ind = 175

#define train and test lakes
# test_lakes = np.array(['1102088', '1101506', '13631637', '1099476', '120053694', '9022741', '1101864'])
# test_lakes = np.array(['13293262', '1097324', '1109052', '1109136', '1099420', '1099432', '1099450', '1099432'])

train_lakes = np.array(lakenames)
# train_lakes = np.delete(lakenames, test_lakes)

#run params
save = True








ct = 0

#############################################3
##for every target lake, load every source lake and predict
#########################################################3
for target_id in lakenames_new: 
    nid = target_id
    # if nid == '120018008' or nid == '120020307' or nid == '120020636' or nid == '32671150' or nid =='58125241'or nid=='120020800' or nid=='91598525':
    #     continue
    ct += 1

    data_dir_target = "../../data/processed/WRR_69Lake/"+target_id+"/" 
    (_, _, tst_data_target, tst_dates_target, unique_tst_dates_target, all_data_target, all_phys_data_target, all_dates_target,
    hypsography_target) = buildLakeDataForRNN_manylakes_finetune2(target_id, data_dir_target, seq_length, n_features,
                                       win_shift = win_shift, begin_loss_ind = begin_loss_ind, 
                                       latter_third_test=True, outputFullTestMatrix=True, 
                                       sparseTen=False, realization='none', allTestSeq=True, oldFeat=False)


    #           row_vals = [source_id, mat_rmse, geo_diff, tempdtw_diff, surf_area_diff, max_depth_diff, lat_diff, long_diff, clar_diff,
            #           surf_area_diff2, max_depth_diff2, lat_diff2, long_diff2, clar_diff2]
            # row_vals_all = [target_id, source_id, mat_rmse, geo_diff, tempdtw_diff, surf_area_diff, max_depth_diff, lat_diff, long_diff, clar_diff,
            #           surf_area_diff2, max_depth_diff2, lat_diff2, long_diff2, clar_diff2]

    #csv to append to for each source lake
    csv_targ = []
    first_row_str = "source_id,rmse,surf_area,max_depth,lat,long,k_d,SDF,canopy,surf_area2,max_depth2,lat2,long2,clar2"
    csv_targ.append(first_row_str)


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

    print("targetLake"+str(ct)+": "+target_id)

    #geometry difference metric for each source lake of a target

    for source_id in lakenames_new:
        # if source_id in test_lakes:
        #   continue
        nid = source_id
        if nid == '120018008' or nid == '120020307' or nid == '120020636' or nid == '32671150' or nid =='58125241'or nid=='120020800' or nid=='91598525':
            continue
        if source_id == target_id:
            continue

        print("source lake "+source_id)


        #output matrix
        n_lakes = 1
        output_mats = np.empty((n_lakes, n_depths, n_test_dates_target))
        label_mats = np.empty((n_depths, n_test_dates_target)) 
        output_mats[:] = np.nan
        label_mats[:] = np.nan
        #save output path
        save_output_path = "../../../results/transfer_learning/target_"+target_id+"/source_"+source_id+"/PGRNN_basic_normAll"
        save_label_path = "../../../results/transfer_learning/target_"+target_id+"/label"

        if not os.path.exists("../../../results/transfer_learning/target_"+target_id+"/"):
            os.mkdir("../../../results/transfer_learning/target_"+target_id)

        if not os.path.exists("../../../results/transfer_learning/target_"+target_id+"/source_"+source_id):
            os.mkdir("../../../results/transfer_learning/target_"+target_id+"/source_"+source_id)



        #load model
        load_path = "../../../models/single_lake_models/"+source_id+"/PGRNN_basic_normAll"
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
                # energies = calculate_energy(output_torch, depth_areas, use_gpu)
                # energies = energies.cpu().numpy()
                avg_mse = avg_mse.cpu().numpy()
                if save: 
                    saveTemperatureMatrix(outputm_npy, labelm_npy, unique_tst_dates_target, source_id,target_id, 
                                          save_path=save_output_path, label_path = save_label_path)

            #############################
            #get metadata differences (independent vars in model selection regression)
            ##############################
            surf_area_diff = abs(metadata.loc[source_id].surface_area - meta_new.loc[target_id].surface_area)
            max_depth_diff = abs(metadata.loc[source_id].max_depth - meta_new.loc[target_id].max_depth)
            lat_diff = abs(metadata.loc[source_id].latitude - meta_new.loc[target_id].latitude)
            long_diff = abs(metadata.loc[source_id].longitude - meta_new.loc[target_id].longitude)
            clar_diff = abs(metadata.loc[source_id].K_d - meta_new.loc[target_id].K_d)
            sdf_diff = abs(metadata.loc[source_id].SDF - meta_new.loc[target_id].SDF)
            can_diff = abs(metadata.loc[source_id].canopy - meta_new.loc[target_id].canopy)
            surf_area_diff2 = metadata.loc[source_id].surface_area - meta_new.loc[target_id].surface_area
            max_depth_diff2 = metadata.loc[source_id].max_depth - meta_new.loc[target_id].max_depth
            lat_diff2 = metadata.loc[source_id].latitude - meta_new.loc[target_id].latitude
            long_diff2 = metadata.loc[source_id].longitude - meta_new.loc[target_id].longitude
            clar_diff2 = metadata.loc[source_id].K_d - meta_new.loc[target_id].K_d
            row_vals = [source_id, mat_rmse, surf_area_diff, max_depth_diff, lat_diff, long_diff, clar_diff, sdf_diff, can_diff,
                        surf_area_diff2, max_depth_diff2, lat_diff2, long_diff2, clar_diff2]
            row_vals_all = [target_id, mat_rmse, source_id, mat_rmse, surf_area_diff, max_depth_diff, lat_diff, long_diff, clar_diff,
                        surf_area_diff2, max_depth_diff2, lat_diff2, long_diff2, clar_diff2]
            row_vals_str = [str(i) for i in row_vals]
            row_vals_str_all = [str(i) for i in row_vals_all]

            #append
            csv_targ.append(",".join(row_vals_str))
            if target_id in test_lakes:
                continue
            else:
                csv_all.append(",".join(row_vals_str_all))



    with open("../../../results/transfer_learning/target_"+target_id+"/resultsPGRNNbasic_norm2_bias",'w') as file:
        # print("saving to ../../../results/transfer_learning/target_"+target_id+"/resultsPGRNNbasic6")
        for line in csv_targ:
            file.write(line)
            file.write('\n')

# with open("../../../results/transfer_learning/resultsAllTrain_oldFeat.csv",'a') as file2:
#   for line2 in csv_all:
#       file2.write(line2)
#       file2.write('\n')

