import os
import re
import sys
import numpy as np
import pandas as pd
import pdb
import math
#######################################
# Oct 2019
# Jared - calculate if each lake is well mixed or not 
# stratification condition: 1 °C between the shallowest and deepest depths measured for >70% of profiless
#######################################


raw_dir = '../../../data/raw/figure3/' #unprocessed data directory

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

metadata = pd.read_csv("../../../metadata/lake_metadata_wNew.csv")
ids = metadata['nhd_id']
metadata.set_index('nhd_id', inplace=True)
csv = []
ct = 0
for lake in ids:
    nid = lake
    if nid == '120018008' or nid == '120020307' or nid == '120020636' or nid == '32671150' or nid =='58125241'or nid=='120020800' or nid=='91598525':
    	continue

    print("lake ", lake)

    #load data
    tst = np.load('../../../data/processed/WRR_69Lake/'+lake+'/test_b.npy')
    trn = np.load('../../../data/processed/WRR_69Lake/'+lake+'/train_b.npy')

    #combine into full data
    na = np.isnan(tst)
    nb = np.isnan(trn)
    tst[na] = 0
    trn[nb] = 0
    tst += trn
    na &= nb
    tst[na] = np.nan
    full = tst
    np.save('../../../data/processed/WRR_69Lake/'+lake+'/full_obs.npy', full)



    # total_profiles = np.count_nonzero(np.count_nonzero(full, axis=0))
    multi_obs_days = np.where(np.count_nonzero(np.isfinite(full), axis=0) > 1)[0]
    multi_obs_profiles = full[:,multi_obs_days]

    #get top obs
    top_obs = np.array([multi_obs_profiles[np.where(np.isfinite(multi_obs_profiles[:,i]))[0][0],i] for i in range(multi_obs_profiles.shape[1])], dtype=np.float32)
    
    #get bot obs
    bot_obs = np.array([multi_obs_profiles[np.where(np.isfinite(multi_obs_profiles[:,i]))[0][-1],i] for i in range(multi_obs_profiles.shape[1])], dtype=np.float32)
    
    #percentage of days with temp diff > 1
    perc_strat = np.sum(np.abs(top_obs - bot_obs) > 1) / multi_obs_profiles.shape[1]

    #determine stratification based on percentc
    strat = perc_strat > 0.7

    csv.append(",".join([lake, str(strat)]))


#120017989

with open('mix_bool.csv','a') as file:
    for line in csv:
        file.write(line)
        file.write('\n')