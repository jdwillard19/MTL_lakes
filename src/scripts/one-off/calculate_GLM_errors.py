import os
import re
import sys
import numpy as np
import pandas as pd
import pdb
import math
#######################################
# Oct 2019
# Jared - calculate GLM error per lake
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
    ind_to_del = []
    print("lake ", lake)
    ct += 1
    glm = pd.read_feather(raw_dir+"nhd_"+lake+"_temperatures.feather")
    glm['DateTime'] = np.array(glm['DateTime'],dtype='datetime64[D]')
    obs = pd.read_feather(raw_dir+"nhd_"+lake+"_test_train.feather") 
    last_tst_ind = math.floor(obs.shape[0]/3)
    obs_temps = np.array(obs.values[:last_tst_ind,3])
    glm_temps = np.empty((last_tst_ind))
    glm_temps[:] = np.nan
    for t in range(last_tst_ind):
        # if len(np.where(glm['DateTime'] == pd.to_datetime(obs['date'][t]).tz_localize('Etc/GMT+6'))[0]) == 0:
        if np.datetime64(pd.to_datetime(obs['date'][t]).tz_localize('Etc/GMT+6')).astype('datetime64[D]') < glm['DateTime'][0]:
            ind_to_del.append(t)
            continue
        if len(np.where(glm['DateTime'] == np.datetime64(pd.to_datetime(obs['date'][t]).tz_localize('Etc/GMT+6')).astype('datetime64[D]'))[0]) == 0:
            pdb.set_trace()
        row_ind = np.where(glm['DateTime'] == np.datetime64(pd.to_datetime(obs['date'][t]).tz_localize('Etc/GMT+6')).astype('datetime64[D]'))[0][0]
        col_ind = int(obs.iloc[t].depth / 0.5) + 1
        if col_ind > glm.shape[1]-1:
            ind_to_del.append(t)
            continue
        elif math.isnan(glm.iloc[row_ind, col_ind]):
            ind_to_del.append(t)
        else:
            glm_temps[t] = glm.iloc[row_ind, col_ind]
    glm_temps = np.delete(glm_temps, ind_to_del, axis=0)
    obs_temps = np.delete(obs_temps, ind_to_del, axis=0)
    err = rmse(glm_temps, obs_temps)
    print(err)
    csv.append(",".join([lake, str(err)]))

with open('glm_rmse2.csv','a') as file:
    for line in csv:
        file.write(line)
        file.write('\n')
