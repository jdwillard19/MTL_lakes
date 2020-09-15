import pandas as pd
import numpy as np
import os
import pdb


#read lake metadata file to get all the lakenames
metadata = pd.read_csv("../../../metadata/lake_metadata.csv")
lakenames = [str(i) for i in metadata.iloc[:,0].values] # to loop through all lakes
csv = []
csv.append("nhd_id,n_prof,n_obs")
ct = 0
for target_id in lakenames: 
    tst = np.load("../../../data/processed/WRR_69Lake/"+target_id+"/test_b.npy")
    trn = np.load("../../../data/processed/WRR_69Lake/"+target_id+"/train_b.npy")
    full = np.nansum(np.dstack((trn,tst)),2)
    na = np.isnan(tst)
    nb = np.isnan(trn)
    tst[na] = 0
    trn[nb] = 0
    tst += trn
    na &= nb
    tst[na] = np.nan
    full = tst

    #count profiles
    n_profiles = np.nonzero(np.count_nonzero(~np.isnan(full), axis=0))[0].shape[0]
    n_obs = np.count_nonzero(~np.isnan(full))

    csv.append(",".join([target_id, str(n_profiles), str(n_obs)]))


with open('./obs_and_profiles.csv','a') as file:
    for line in csv:
        file.write(line)
        file.write('\n')
