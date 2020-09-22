import numpy as np
import torch
import torch.nn as nn
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
sys.path.append('/home/invyz/workspace/Research/lake_monitoring/src/data')
from pytorch_data_operations import buildLakeDataForRNNPretrain, calculate_energy,calculate_ec_loss_manylakes, transformTempToDensity, calculate_dc_loss
from pytorch_model_operations import saveModel
import pytorch_data_operations
import datetime
#multiple dataloader wrapping?
import pdb
from torch.utils.data import DataLoader
from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune2, parseMatricesFromSeqs

####################################################################################################3
# (Sept 2020 - Jared) - this script runs all the sparse PGDL models for target lakes and records RMSE
###########################################################################################################


#script start
currentDT = datetime.datetime.now()
print(str(currentDT))

#enable/disable cuda 
use_gpu = True 
torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=10)
# sources = pd.read_csv('pgdtl_rmse_pball_sources.csv')
ids = pd.read_csv('../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")

train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
test_lakes = ids[~np.isin(ids, train_lakes)]


#to iterate through
n_profiles = [1,2,5,10,15,20,25,30,35,40,45,50]
seeds = [0,1,2,3,4]


#monitor which sites do not have "x" observations (no_x below)
no_50 = []
no_45 = []
no_40 = []
no_35 = []
no_30 = []
no_25 = []

### debug tools
debug_train = False
debug_end = False
verbose = False
pretrain = False
save = True
save_pretrain = True

#####################3
#params
###########################33

unsup_loss_cutoff = 40
dc_unsup_loss_cutoff = 1e-3
dc_unsup_loss_cutoff2 = 1e-2
n_hidden = 20 #fixed
train_epochs = 10000
pretrain_epochs = 10000


n_ep = pretrain_epochs  #number of epochs

if debug_train or debug_end:
    n_ep = 10
first_save_epoch = 0
patience = 100

#ow
seq_length = 350 #how long of sequences to use in model
begin_loss_ind = 0#index in sequence where we begin to calculate error or predict
n_features = 8  #number of physical drivers
win_shift = 175 #how much to slide the window on training set each time
save = True 

#for each test lake
for lake_ct, lakename in enumerate(test_lakes):
    if os.path.exists("../../results/"+lakename+"/sparseModelResults.csv"):
        continue
    print("(",lake_ct,"/",len(test_lakes),"): ", lakename)
    data_dir = "../../data/processed/"+lakename+"/"




    csv_targ = []
    first_row_str_all = "source_id,n_profiles,seed,rmse"
    csv_targ.append(first_row_str_all)
    (trn_data, all_data, all_phys_data, all_dates, hypsography) = buildLakeDataForRNNPretrain(lakename, data_dir, seq_length, n_features,
                                   win_shift= win_shift, begin_loss_ind=begin_loss_ind,
                                   excludeTest=False, normAll=False, normGE10=False)
    n_depths = torch.unique(all_data[:,:,0]).size()[0]
    for n_prof in n_profiles:
        avg_over_seed = np.empty((len(seeds)))
        avg_over_seed[:] = np.nan
        for seed_ct, seed in enumerate(seeds):
            if not os.path.exists("../../models/"+lakename+"/PGRNN_sparse_" + str(n_prof) + "_" + str(seed)):
                print('not enough observations')
                continue
            load_path = "../../models/"+lakename+"/PGRNN_sparse_" + str(n_prof) + "_" + str(seed)

            ###############################
            # data preprocess
            ##################################
            #create train and test sets

            (trn_data, trn_dates, tst_data, tst_dates, unique_tst_dates, all_data, all_phys_data, all_dates,
            hypsography) = buildLakeDataForRNN_manylakes_finetune2(lakename, data_dir, seq_length, n_features,
                                               win_shift = win_shift, begin_loss_ind = begin_loss_ind, 
                                               outputFullTestMatrix=False, 
                                               allTestSeq=False, sparseCustom=n_prof, randomSeed=seed) 
            #if error code is returned (trn data as int), skip and record id
            if isinstance(trn_data, int):
                target_id = lakename
                if trn_data == 25:
                    no_25.append(target_id)
                    continue
                elif trn_data == 30:
                    no_30.append(target_id)
                    continue
                elif trn_data == 35:
                    no_35.append(target_id)
                    continue
                elif trn_data == 40:
                    no_40.append(target_id)
                    continue
                elif trn_data == 45:
                    no_45.append(target_id)
                    continue
                elif trn_data == 50:
                    no_50.append(target_id)
                    continue
            u_depths = np.unique(all_data[:,0,0])
            n_test_dates = unique_tst_dates.shape[0]
            batch_size = trn_data.size()[0]

            #Dataset classes
            class TemperatureTrainDataset(Dataset):
                #training dataset class, allows Dataloader to load both input/target
                def __init__(self, trn_data):
                    # depth_data = depth_trn
                    self.len = trn_data.shape[0]
                    self.x_data = trn_data[:,:,:-1].float()
                    self.y_data = trn_data[:,:,-1].float()

                def __getitem__(self, index):
                    return self.x_data[index], self.y_data[index]

                def __len__(self):
                    return self.len

            class TotalModelOutputDataset(Dataset):
                #dataset for unsupervised input(in this case all the data)
                def __init__(self, all_data, all_phys_data,all_dates):
                    #data of all model output, and corresponding unstandardized physical quantities
                    #needed to calculate physical loss
                    self.len = all_data.shape[0]
                    self.data = all_data[:,:,:-1].float()
                    self.label = all_data[:,:,-1].float() #DO NOT USE IN MODEL
                    self.phys = all_phys_data.float()
                    helper = np.vectorize(lambda x: date.toordinal(pd.Timestamp(x).to_pydatetime()))
                    dates = helper(all_dates)
                    self.dates = dates

                def __getitem__(self, index):
                    return self.data[index], self.phys[index], self.dates[index], self.label[index]

                def __len__(self):
                    return self.len




            #format training data for loading
            train_data = TemperatureTrainDataset(trn_data)

            #get depth area percent data
            depth_areas = torch.from_numpy(hypsography).float().flatten()
            if use_gpu:
                depth_areas = depth_areas.cuda()

            #format total y-hat data for loading
            total_data = TotalModelOutputDataset(all_data, all_phys_data, all_dates)
            n_batches = math.floor(trn_data.size()[0] / batch_size)
            yhat_batch_size = n_depths

            #batch samplers used to draw samples in dataloaders
            batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)



            #load val/test data into enumerator based on batch size
            testloader = torch.utils.data.DataLoader(tst_data, batch_size=tst_data.size()[0], shuffle=False, pin_memory=True)



            #define LSTM model class
            class myLSTM_Net(nn.Module):
                def __init__(self, input_size, hidden_size, batch_size):
                    super(myLSTM_Net, self).__init__()
                    self.input_size = input_size
                    self.hidden_size = hidden_size
                    self.batch_size = batch_size
                    self.lstm = nn.LSTM(input_size = n_features, hidden_size=hidden_size, batch_first=True) #batch_first=True?
                    self.out = nn.Linear(hidden_size, 1) #1?
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
                
                def forward(self, x, hidden):

                    self.lstm.flatten_parameters()
                    x = x.float()
                    x, hidden = self.lstm(x, self.hidden)
                    self.hidden = hidden
                    x = self.out(x)
                    return x, hidden

            #method to calculate l1 norm of model
            def calculate_l1_loss(model):
                def l1_loss(x):
                    return torch.abs(x).sum()

                to_regularize = []
                # for name, p in model.named_parameters():
                for name, p in model.named_parameters():
                    if 'bias' in name:
                        continue
                    else:
                        #take absolute value of weights and sum
                        to_regularize.append(p.view(-1))
                l1_loss_val = torch.tensor(1, requires_grad=True, dtype=torch.float32)
                l1_loss_val = l1_loss(torch.cat(to_regularize))
                return l1_loss_val


            lstm_net = myLSTM_Net(n_features, n_hidden, batch_size)

            pretrain_dict = torch.load(load_path)['state_dict']
            model_dict = lstm_net.state_dict()
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            lstm_net.load_state_dict(pretrain_dict)

            #tell model to use GPU if needed
            if use_gpu:
                lstm_net = lstm_net.cuda()




              #data loader for test data
            testloader = torch.utils.data.DataLoader(tst_data, batch_size=tst_data.size()[0], shuffle=False, pin_memory=True)

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
                    tmp_dates = tst_dates[:, begin_loss_ind:]
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
                                                                    n_test_dates, u_depths,
                                                                    unique_tst_dates) 
                    #store output
                    label_mats = labelm_npy
                    loss_output = outputm_npy[~np.isnan(labelm_npy)]
                    loss_label = labelm_npy[~np.isnan(labelm_npy)]

                    mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())
                    print(n_prof," obs ",lakename+" rmse=" + str(mat_rmse))
                    avg_over_seed[seed_ct] = mat_rmse

                    row_vals = [lakename, n_prof, seed, mat_rmse]
                    row_vals_str = [str(i) for i in row_vals]

                    #append
                    csv_targ.append(",".join(row_vals_str))
        print("\n**\n",n_prof," obs ",lakename+" AVERAGE RMSE=" + str(avg_over_seed.mean()), "\n**\n")


    if not os.path.exists("../../results/"+lakename):
        os.mkdir("../../results/"+lakename)

    with open("../../results/"+lakename+"/sparseModelResults.csv",'w') as file:
        for line in csv_targ:
            file.write(line)
            file.write('\n')

print("25-50 missed")
print(no_25)
print(no_30)
print(no_35)
print(no_40)
print(no_45)
print(no_50)
