import pandas as pd
import feather
import numpy as np
import os
import sys
import re
import shutil
###################################################################################
# (Jared) June 2019 - create 50 observation realizations for sparse 69 lake experiments
###################################################################################
raw_dir = '../../../data/raw/figure3' #unprocessed data directory
proc_dir = '../../../data/processed/WRR_69Lake/'
lnames = set()
n_features = 7
n_lakes = 0
csv = []

sample_size = 50
for filename in os.listdir(raw_dir):
    #parse lakename from file
    m = re.search(r'^nhd_(\d+)_.*', filename)
    if m is None:
        continue
    name = m.group(1)
    if name not in lnames:
        #for each unique lake
        lnames.add(name)

        #load all train observations and dates
        trn = np.load(proc_dir+name+"/train_b.npy")
        dates = np.load(proc_dir+name+"/dates.npy")

        #find cols with obs
        obs_col_inds = np.argwhere(np.any(np.isfinite(trn),axis=0) == True)
        obs_col_inds = obs_col_inds[:,0] #reshape

        #collect 3 sets of sampled random days
        r1_ind = np.random.choice(obs_col_inds, sample_size)
        r2_ind = np.random.choice(obs_col_inds, sample_size)
        r3_ind = np.random.choice(obs_col_inds, sample_size)

        #get dates to save for records
        r1_dates = dates[r1_ind]
        r2_dates = dates[r2_ind]
        r3_dates = dates[r3_ind]

        #get into csv save format
        r1_str = ','.join([str(date) for date in r1_dates])
        r2_str = ','.join([str(date) for date in r2_dates])
        r3_str = ','.join([str(date) for date in r3_dates])

        #save the indidices
        with open(proc_dir+"/dates50/"+ name+"_50sparse_dates.csv",'a') as file:
            file.write(name+"_1,"+r1_str+"\n")
            file.write(name+"_2,"+r2_str+"\n")
            file.write(name+"_3,"+r3_str+"\n")

        #make new matrices for each instance of 50 sparse days
        r1_mat = np.empty_like(trn)
        r1_mat[:] = np.nan
        r1_mat[:,r1_ind] = trn[:,r1_ind]
        r2_mat = np.empty_like(trn)
        r2_mat[:] = np.nan
        r2_mat[:,r1_ind] = trn[:,r1_ind]
        r3_mat = np.empty_like(trn)
        r3_mat[:] = np.nan
        r3_mat[:,r1_ind] = trn[:,r1_ind]

        np.save(proc_dir+name+"/train50_1", r1_mat)
        np.save(proc_dir+name+"/train50_2", r2_mat)
        np.save(proc_dir+name+"/train50_3", r3_mat)

