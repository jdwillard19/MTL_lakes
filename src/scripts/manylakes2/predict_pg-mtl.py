import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pdb
import sys
import os
sys.path.append('../../data')
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


metadata = pd.read_feather("../../../metadata/lake_metadata_baseJune2020.feather")
metadata.set_index('site_id', inplace=True)
ids = pd.read_csv('../../../metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
glm_all_f = pd.read_csv("../../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_df = pd.read_feather("../../../results/transfer_learning/glm/glm_meta_train_rmses.feather")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
n_lakes = len(train_lakes)
test_lakes = ids[~np.isin(ids, train_lakes)]
k = 1
output_to_file = True
biases = []
# print(train_lakes.shape[0], " training lakes")

mat_csv = []
mat_csv.append(",".join(["target_id","source_id","glm_rmse","rmse","source_observations",\
                         "mean_source_observation_temp","diff_max_depth",'diff_surface_area', 'diff_RH_mean_autumn', 'diff_lathrop_strat','dif_glm_strat_perc',\
                         'percent_diff_max_depth', 'percent_diff_surface_area','perc_dif_sqrt_surface_area']))

feats = ['n_obs', 'obs_temp_mean', 'dif_max_depth', 'dif_surface_area',
       'dif_rh_mean_au', 'dif_lathrop_strat', 'dif_glm_strat_perc',
       'perc_dif_max_depth', 'perc_dif_surface_area',
       'perc_dif_sqrt_surface_area']
# feats = feats[ranks == 1]
train = False
model = []
if train:
    # #compile training data from all training lakes
    train_df = pd.DataFrame()

    for _, lake_id in enumerate(train_lakes):

        new_df = pd.DataFrame()
        lake_df_res = pd.read_csv("../../../results/transfer_learning/target_"+lake_id+"/results_all_source_models2.csv") 
        lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']
        lake_df_res['source_id'] = [str(x) for x in lake_df_res.astype('object')['source_id']]

        lake_df = pd.read_feather("../../../metadata/diff/target_"+lake_id+"_pball_Aug2020.feather")
        lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes)]
        lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
        new_df = pd.merge(left=lake_df, right=lake_df_res.astype('object'), left_on='site_id', right_on='source_id')
        assert not new_df.empty
        train_df = pd.concat([train_df, new_df], ignore_index=True)

    X_trn = pd.DataFrame(train_df[feats])
    y_trn = np.array([float(x) for x in np.ravel(pd.DataFrame(train_df['rmse']))])



    # model = GradientBoostingRegressor(n_estimators=3700, learning_rate=.05)
    model = GradientBoostingRegressor(n_estimators=4900, learning_rate=.05)
    model.fit(X_trn, y_trn)


    dump(model, "PGMTL_GBR_pball_Aug21.joblib")
# model = load("PGML_RMSE_GBR_pball2.joblib")
model = load("PGMTL_GBR_pball_Aug21.joblib")


importances = model.feature_importances_
print("##Feature Importances#################")
print(feats)
print(importances)
print("#####################")




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
csv.append('target_id,rmse,rmse_pred,spearman,glm_rmse,site_id')


err_per_source = np.empty((145,len(test_lakes)))
# test_lakes = np.array(['120020398'])
for targ_ct, target_id in enumerate(test_lakes): #for each target lake
    print(str(targ_ct),'/',len(test_lakes),':',target_id)
    lake_df = pd.DataFrame()
    # temp_df = pd.read_csv("../../../metadata/diff/target_"+lake_id+"_wStats.csv")
    # temp_df.drop(np.where(np.isin(temp_df['nhd_id'], temp_test_lakes[0]) == True)[0][0], inplace=True)
    lake_id = target_id
    lake_df_res = pd.read_csv("../../../results/transfer_learning/target_"+lake_id+"/results_all_source_models2.csv") 
    lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']

    lake_df = pd.read_feather("../../../metadata/diff/target_"+lake_id+"_pball_Aug2020.feather")
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
    #   lake_df = lake_df[lake_df['rmse_pred'] < hard_cut]
    # else:
    #   print("no good lakes?")
    #rel cutoff

    lake_df.sort_values(by=['rmse_pred'], inplace=True)
    lowest_rmse = lake_df.iloc[0]['rmse_pred']

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
        load_path = "../../../models/single_lake_models/"+source_id+"/PGRNN_source_model_0.7"
        # if os.path.exists("../../../models/single_lake_models/"+source_id+"/PGRNN_basic_normAllGr10"):
        #   load_path = "../../../models/single_lake_models/"+source_id+"/PGRNN_basic_normAllGr10"
        # else:
        #   load_path = "../../../models/single_lake_models/"+source_id+"/PGRNN_basic_normAll_pball"
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
                #   np.save('./best_ens/source_'+source_id+'_output', outputm_npy)
                #   np.save('./best_ens/labels', labelm_npy)
                #   np.save('./best_ens/dates', unique_tst_dates_target)
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
        output_df.to_feather("mtl_outputs_July2020/target_nhdhr_"+target_id+"_PGMTL_outputs.feather")
        label_df.to_feather("labels/nhdhr_"+target_id+"_labels.feather")
        
    loss_output = total_output_npy[~np.isnan(label_mats)]
    loss_label = label_mats[~np.isnan(label_mats)]
    mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())
    # mat_csv.append(",".join(["nhdhr_"+target_id,"9 MODEL ENSEMBLE",str(mat_rmse)] + ['NA' for x in lake_df.iloc[i][feats].values]))
    # mat_csv.append(",".join(["nhdhr_"+target_id,"9 MODEL ENSEMBLE",str(mat_rmse)] + ['NA' for x in lake_df.iloc[i][feats].values]))

    print("Total rmse=", mat_rmse)
    spcorr = srcorr_per_lake[targ_ct]
    rmse_per_lake[targ_ct] = mat_rmse

    # if ind_rmses.mean() - mat_rmse > most_imp_diff:
    #   most_imp_diff = ind_rmses.mean() - mat_rmse
    #   most_imp_site_id = target_id
    #   most_imp_total_rmse = mat_rmse
    #   most_imp_ind_rmses = ind_rmses
    #   most_imp_source_ids = top_ids


    glm_rmse = float(metadata.loc[target_id].glm_uncal_rmse_full)
    csv.append(",".join([str(target_id), str(mat_rmse)]))


# with open("../../../results/transfer_learning/rf_testset.csv",'a') as file:
#   for line in test_lake_csv:
#       file.write(line)
#       file.write('\n')
# with open('pgdtl_rmse_pball_ens9_wPredStats305.csv','w') as file:
#     for line in csv:
#         file.write(line)
#         file.write('\n')
with open('pgdlmtl_singlesource_result_Aug21.csv','w') as file:
    for line in mat_csv:
        file.write(line)
        file.write('\n')


# with open('pgml_result_logloss.csv','w') as file:
#     for line in mat_csv:
#         file.write(line)
#         file.write('\n')

# print("mean meta test RMSE: ",meta_rmse_per_lake.mean())
# print("median meta test RMSE: ",np.median(meta_rmse_per_lake))
print("median srcorr: ",np.median(srcorr_per_lake))
print("median meta test RMSE(med): ",np.median(med_meta_rmse_per_lake))
# print("mean test RMSE: ",rmse_per_lake.mean())
print("median test RMSE: ",np.median(rmse_per_lake))
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
#   model.fit(X.iloc[train,:], y.iloc[train,:])
#   score = model.score(X.iloc[test,:], y.iloc[test,:])
#   scores.append(score)